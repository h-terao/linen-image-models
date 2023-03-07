"""Swin transformer.

Modify from https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
"""
from __future__ import annotations
import typing as tp
import math
import copy
from functools import partial

import jax.numpy as jnp
from jax.numpy import linalg as LA
from flax import linen
from flax.linen.dtypes import promote_dtype
from einops import rearrange, reduce
import chex

from limo import register_model


ModuleDef = tp.Any


def _patch_merging_pad(x: chex.Array) -> chex.Array:
    *batch_dims, H, W, C = x.shape
    x = jnp.reshape(x, (-1, H, W, C))

    x = jnp.pad(x, [(0, 0), (0, H % 2), (0, W % 2), (0, 0)])

    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = jnp.concatenate([x0, x1, x2, x3], axis=-1)

    x = jnp.reshape(x, (*batch_dims, (H + 1) // 2, (W + 1) // 2, 4 * C))
    return x


def _get_relative_position_bias(
    relative_position_bias_table: chex.Array,
    relative_position_index: chex.Array,
    window_size: tuple[int, int],
) -> chex.Array:

    size = window_size[0] * window_size[1]
    return rearrange(
        relative_position_bias_table[relative_position_index],
        "(size1 size2) head -> 1 head size1 size2",
        size1=size,
        size2=size,
    )


class PatchMerging(linen.Module):
    dense: ModuleDef
    norm_layer: ModuleDef

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        features = x.shape[-1]
        x = _patch_merging_pad(x)
        x = self.norm_layer(name="norm")(x)
        x = self.dense(2 * features, use_bias=False, name="reduction")(x)
        return x


class PatchMergingV2(linen.Module):
    dense: ModuleDef
    norm_layer: ModuleDef  # `norm` is used as actual parameter name.

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        features = x.shape[-1]
        x = _patch_merging_pad(x)
        x = self.dense(2 * features, use_bias=False, name="reduction")(x)
        x = self.norm_layer(name="norm")(x)
        return x


def shifted_window_attention(
    x: chex.Array,
    qkv_kernel: chex.Array,
    qkv_bias: chex.Array | None,
    proj_kernel: chex.Array,
    proj_bias: chex.Array | None,
    relative_position_bias: chex.Array,
    window_size: tuple[int, int],
    shift_size: tuple[int, int],
    num_heads: int,
    atten_drop_rate: float,
    drop_rate: float,
    logit_scale: chex.Array | None = None,
    dtype: chex.ArrayDType = jnp.float32,
    deterministic: bool = True,
):
    """
    Args:
        qkv_kernel: qkv weight with shape (inC, outC).
    """

    if qkv_bias is None:
        qkv_bias = jnp.zeros_like(qkv_kernel, shape=(qkv_kernel.shape[-1],))

    if proj_bias is None:
        proj_bias = jnp.zeros_like(proj_kernel, shape=(proj_kernel.shape[-1],))

    x, qkv_kernel, qkv_bias, proj_kernel, proj_bias, relative_position_bias = promote_dtype(
        x, qkv_kernel, qkv_bias, proj_kernel, proj_bias, relative_position_bias, dtype=dtype
    )

    # Flatten.
    *batch_dims, H, W, C = x.shape
    x = jnp.reshape(x, (-1, H, W, C))

    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    x = jnp.pad(x, [(0, 0), (0, pad_b), (0, pad_r), (0, 0)])
    _, pad_H, pad_W, _ = x.shape

    # If window size is larger than feature size, do not need to shift window.
    shift_size = list(copy.copy(shift_size))
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(-shift_size[0], -shift_size[1]), axis=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = rearrange(
        x,
        "b (num_h size_h) (num_w size_w) c -> (b num_h num_w) (size_h size_w) c",
        size_h=window_size[0],
        size_w=window_size[1],
    )

    # multihead attention.
    if logit_scale is not None:
        length = qkv_bias.shape[0] // 3
        qkv_bias = qkv_bias.at[length : 2 * length].set(0)
    qkv = x @ qkv_kernel + qkv_bias
    q, k, v = rearrange(qkv, "n size (qkv head dim) -> qkv n head size dim", qkv=3, head=num_heads)
    if logit_scale is not None:
        # cosine attention.
        q = q / jnp.maximum(LA.norm(q, ord=2, axis=-1, keepdims=True), 1e-12)
        k = k / jnp.maximum(LA.norm(k, ord=2, axis=-1, keepdims=True), 1e-12)
        attn = q @ rearrange(k, "n head size dim -> n head dim size")  # n, head, size, size
        attn = attn * jnp.exp(jnp.minimum(logit_scale, math.log(100)))
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q @ rearrange(k, "n head size dim -> n head dim size")
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # attn: n head size size

        # generate attention mask.
        attn_mask = jnp.zeros((pad_H, pad_W), dtype=dtype)
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], pad_H))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], pad_W))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask = attn_mask.at[h[0] : h[1], w[0] : w[1]].set(count)
                count += 1
        attn_mask = rearrange(
            attn_mask,
            "(num_h size_h) (num_w size_w) -> (num_h num_w) (size_h size_w)",
            size_h=window_size[0],
            size_w=window_size[1],
        )
        attn_mask = attn_mask[:, None, :] - attn_mask[:, :, None]  # num_windows size size
        attn_mask = jnp.where(attn_mask == 0, jnp.zeros_like(attn_mask), jnp.full_like(attn_mask, -100.0))

        num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
        attn = rearrange(attn, "(b n) head size1 size2 -> b n head size1 size2", n=num_windows)
        attn = attn + rearrange(attn_mask, "n size1 size2 -> 1 n 1 size1 size2")
        attn = rearrange(attn, "b n head size1 size2 -> (b n) head size1 size2")

    attn = linen.softmax(attn, axis=-1)
    if not deterministic:
        attn = linen.Dropout(atten_drop_rate)(attn, deterministic)

    x = rearrange(attn @ v, "n head size dim -> n size (head dim)")
    x = x @ proj_kernel + proj_bias
    if not deterministic:
        x = linen.Dropout(drop_rate)(x, deterministic)

    # reverse windows.
    x = rearrange(
        x,
        "(b num_h num_w) (size_h size_w) dim -> b (num_h size_h) (num_w size_w) dim",
        num_h=pad_H // window_size[0],
        num_w=pad_W // window_size[1],
        size_h=window_size[0],
        size_w=window_size[1],
    )

    # reverse cyclic shift.
    if sum(shift_size) > 0:
        x = jnp.roll(x, shift=(shift_size[0], shift_size[1]), axis=(1, 2))

    # unpad and unflatten features
    x = jnp.reshape(x[:, :H, :W, :], (*batch_dims, H, W, C))
    return x


class ShiftedWindowAttention(linen.Module):
    window_size: tuple[int, int]
    shift_size: tuple[int, int]
    num_heads: int
    qkv_bias: bool = True
    proj_bias: bool = True
    atten_drop_rate: float = 0
    drop_rate: float = 0

    dtype: chex.ArrayDType | None = None
    param_dtype: chex.ArrayDType = jnp.float32

    @linen.compact
    def __call__(self, x, is_training=False):
        features = x.shape[-1]

        # Initialize params.
        qkv_kernel = self.param(
            "qkv.kernel",
            linen.initializers.lecun_normal(),
            (features, 3 * features),
            self.param_dtype,
        )

        qkv_bias = None
        if self.qkv_bias:
            qkv_bias = self.param("qkv.bias", linen.initializers.zeros, (3 * features,), self.param_dtype)

        proj_kernel = self.param(
            "proj.kernel",
            linen.initializers.lecun_normal(),
            (features, features),
            self.param_dtype,
        )

        proj_bias = None
        if self.qkv_bias:
            proj_bias = self.param("proj.bias", linen.initializers.zeros, (features,), self.param_dtype)

        # define relative position bias table.
        relative_position_bias_table = self.param(
            "relative_position_bias_table",
            linen.initializers.variance_scaling(0.02, mode="fan_in", distribution="truncated_normal"),
            ((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads),
            self.param_dtype,
        )

        # relative position index.
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = rearrange(coords, "n size_h size_w -> n (size_h size_w)")
        relative_coords = rearrange(
            coords_flatten[:, :, None] - coords_flatten[:, None, :],
            "n size1 size2 -> size1 size2 n",
        )
        relative_coords = relative_coords.at[:, :, 0].add(self.window_size[0] - 1)
        relative_coords = relative_coords.at[:, :, 1].add(self.window_size[1] - 1)
        relative_coords = relative_coords.at[:, :, 0].multiply(2 * self.window_size[1] - 1)
        relative_position_index = reduce(relative_coords, "size1 size2 n -> (size1 size2)", "sum")

        # Take relative position bias.
        relative_position_bias = _get_relative_position_bias(
            relative_position_bias_table, relative_position_index, self.window_size
        )

        return shifted_window_attention(
            # self.make_rng("dropout"),
            x,
            qkv_kernel=qkv_kernel,
            qkv_bias=qkv_bias,
            proj_kernel=proj_kernel,
            proj_bias=proj_bias,
            relative_position_bias=relative_position_bias,
            window_size=self.window_size,
            shift_size=self.shift_size,
            num_heads=self.num_heads,
            atten_drop_rate=self.atten_drop_rate,
            drop_rate=self.drop_rate,
            dtype=self.dtype,
            deterministic=not is_training,
        )


class ShiftedWindowAttentionV2(linen.Module):
    window_size: tuple[int, int]
    shift_size: tuple[int, int]
    num_heads: int
    qkv_bias: bool = True
    proj_bias: bool = True
    atten_drop_rate: float = 0
    drop_rate: float = 0

    dtype: chex.ArrayDType | None = None
    param_dtype: chex.ArrayDType = jnp.float32

    @linen.compact
    def __call__(self, x, is_training=False):
        features = x.shape[-1]
        logit_scale = self.param(
            "logit_scale",
            lambda rng, shape, dtype: jnp.log(jnp.full(shape, 10, dtype=dtype)),
            (self.num_heads, 1, 1),
            self.param_dtype,
        )

        # Initialize params.
        qkv_kernel = self.param(
            "qkv.kernel",
            linen.initializers.lecun_normal(),
            (features, 3 * features),
            self.param_dtype,
        )

        qkv_bias = None
        if self.qkv_bias:
            qkv_bias = self.param("qkv.bias", linen.initializers.zeros, (3 * features,), self.param_dtype)

        proj_kernel = self.param(
            "proj.kernel",
            linen.initializers.lecun_normal(),
            (features, features),
            self.param_dtype,
        )

        proj_bias = None
        if self.qkv_bias:
            proj_bias = self.param("proj.bias", linen.initializers.zeros, (features,), self.param_dtype)

        # define relative position bias table.
        relative_coords_h = jnp.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=self.dtype)
        relative_coords_w = jnp.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=self.dtype)
        relative_coords_table = jnp.stack(jnp.meshgrid(relative_coords_h, relative_coords_w, indexing="ij"))
        relative_coords_table = rearrange(relative_coords_table, "heads h w -> h w heads")

        relative_coords_table = relative_coords_table.at[..., 0].divide(self.window_size[0] - 1)
        relative_coords_table = relative_coords_table.at[..., 1].divide(self.window_size[1] - 1)
        relative_coords_table = 8 * relative_coords_table

        relative_coords_table = jnp.sign(relative_coords_table) * jnp.log2(jnp.abs(relative_coords_table) + 1.0) / 3.0
        relative_coords_table = linen.Dense(512, dtype=self.dtype, name="cpb_mlp.0")(relative_coords_table)
        relative_coords_table = linen.Dense(self.num_heads, use_bias=False, dtype=self.dtype, name="cpb_mlp.2")(
            linen.relu(relative_coords_table)
        )
        relative_coords_table = rearrange(relative_coords_table, "h w heads -> (h w) heads")

        # relative position index.
        coords_h = jnp.arange(self.window_size[0])
        coords_w = jnp.arange(self.window_size[1])
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = rearrange(coords, "n size_h size_w -> n (size_h size_w)")
        relative_coords = rearrange(
            coords_flatten[:, :, None] - coords_flatten[:, None, :],
            "n size1 size2 -> size1 size2 n",
        )
        relative_coords = relative_coords.at[:, :, 0].add(self.window_size[0] - 1)
        relative_coords = relative_coords.at[:, :, 1].add(self.window_size[1] - 1)
        relative_coords = relative_coords.at[:, :, 0].multiply(2 * self.window_size[1] - 1)
        relative_position_index = reduce(relative_coords, "size1 size2 n -> (size1 size2)", "sum")

        # Take relative position bias.
        relative_position_bias = _get_relative_position_bias(
            relative_coords_table, relative_position_index, self.window_size
        )
        relative_position_bias = 16 * linen.sigmoid(relative_position_bias)

        return shifted_window_attention(
            x,
            qkv_kernel=qkv_kernel,
            qkv_bias=qkv_bias,
            proj_kernel=proj_kernel,
            proj_bias=proj_bias,
            relative_position_bias=relative_position_bias,
            window_size=self.window_size,
            shift_size=self.shift_size,
            num_heads=self.num_heads,
            atten_drop_rate=self.atten_drop_rate,
            drop_rate=self.drop_rate,
            logit_scale=logit_scale,
            dtype=self.dtype,
            deterministic=not is_training,
        )


class SwinTransformerBlock(linen.Module):
    num_heads: int
    window_size: tuple[int, int]
    shift_size: tuple[int, int]
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    atten_drop_rate: float = 0.0
    drop_path_rate: float = 0
    dense: ModuleDef = linen.Dense
    norm: ModuleDef = linen.LayerNorm
    dtype: chex.ArrayDType = jnp.float32

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        features = x.shape[-1]
        stochastic_depth = linen.Dropout(self.drop_path_rate, broadcast_dims=(-1, -2, -3))
        bias_init = linen.initializers.normal(1e-6)

        # Attention block.
        h = self.norm(name="norm1")(x)
        h = ShiftedWindowAttention(
            self.window_size,
            self.shift_size,
            self.num_heads,
            atten_drop_rate=self.atten_drop_rate,
            drop_rate=self.drop_rate,
            dtype=self.dtype,
            name="attn",
        )(h, is_training)
        x = x + stochastic_depth(h, not is_training)

        # MLP block.
        h = self.norm(name="norm2")(x)
        h = self.dense(int(self.mlp_ratio * features), bias_init=bias_init, name="mlp.0")(h)
        h = linen.gelu(h, approximate=False)
        h = linen.Dropout(self.drop_rate)(h, not is_training)
        h = self.dense(features, bias_init=bias_init, name="mlp.3")(h)
        h = linen.Dropout(self.drop_rate)(h, not is_training)
        x = x + stochastic_depth(h, not is_training)
        return x


class SwinTransformerBlockV2(linen.Module):
    num_heads: int
    window_size: tuple[int, int]
    shift_size: tuple[int, int]
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    atten_drop_rate: float = 0.0
    drop_path_rate: float = 0
    dense: ModuleDef = linen.Dense
    norm: ModuleDef = linen.LayerNorm
    dtype: chex.ArrayDType = jnp.float32

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        features = x.shape[-1]
        stochastic_depth = linen.Dropout(self.drop_path_rate, broadcast_dims=(-1, -2, -3))
        bias_init = linen.initializers.normal(1e-6)

        # Attention block.
        h = ShiftedWindowAttentionV2(
            self.window_size,
            self.shift_size,
            self.num_heads,
            atten_drop_rate=self.atten_drop_rate,
            drop_rate=self.drop_rate,
            dtype=self.dtype,
            name="attn",
        )(x, is_training)
        h = self.norm(name="norm1")(h)
        x = x + stochastic_depth(h, not is_training)

        # MLP block.
        h = self.dense(int(self.mlp_ratio * features), bias_init=bias_init, name="mlp.0")(x)
        h = linen.gelu(h, approximate=False)
        h = linen.Dropout(self.drop_rate)(h, not is_training)
        h = self.dense(features, bias_init=bias_init, name="mlp.3")(h)
        h = linen.Dropout(self.drop_rate)(h, not is_training)
        h = self.norm(name="norm2")(h)
        x = x + stochastic_depth(h, not is_training)
        return x


class SwinTransformer(linen.Module):
    block: ModuleDef
    downsample: ModuleDef

    patch_size: tuple[int, int]
    depths: list[int]
    num_heads: list[int]
    hidden_dim: int
    window_size: tuple[int, int]
    drop_path_rate: float = 0.1

    num_classes: int = 1000
    mlp_ratio: float = 4.0
    drop_rate: float = 0
    atten_drop_rate: float = 0
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False):
        dense = partial(linen.Dense, dtype=self.dtype)
        norm = partial(
            linen.LayerNorm,
            epsilon=1e-5,
            dtype=self.norm_dtype or self.dtype,
            axis_name=self.axis_name,
        )

        layer_idx = 0

        x = linen.Conv(
            self.hidden_dim,
            self.patch_size,
            strides=self.patch_size,
            padding=0,
            dtype=self.dtype,
            name=f"features.{layer_idx}.0",
        )(x)
        x = norm(name=f"features.{layer_idx}.2")(x)
        layer_idx += 1

        total_blocks = sum(self.depths)
        block_idx = 0
        for i, depth in enumerate(self.depths):
            for j in range(depth):
                x = self.block(
                    num_heads=self.num_heads[i],
                    window_size=self.window_size,
                    shift_size=[0 if j % 2 == 0 else w // 2 for w in self.window_size],
                    mlp_ratio=self.mlp_ratio,
                    drop_rate=self.drop_rate,
                    atten_drop_rate=self.atten_drop_rate,
                    drop_path_rate=self.drop_path_rate * block_idx / total_blocks,
                    dense=dense,
                    norm=norm,
                    dtype=self.dtype,
                    name=f"features.{layer_idx}.{j}",
                )(x, is_training)
                block_idx += 1
            layer_idx += 1
            if i < len(self.depths) - 1:
                x = self.downsample(dense, norm, name=f"features.{layer_idx}")(x)
                layer_idx += 1

        x = norm(name="norm")(x)
        x = reduce(x, "... H W C -> ... C", "mean")
        if self.num_classes > 0:
            x = dense(self.num_classes, name="head")(x)
        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        # Stochastic depth also uses dropout rng collection.
        return ["dropout"]


@register_model(
    "IMAGENET1K_V1",
    url=None,
    meta={"input_size": (224, 224)},
    default=True,
)
def swin_t(**kwargs) -> SwinTransformer:
    return SwinTransformer(
        block=SwinTransformerBlock,
        downsample=PatchMerging,
        patch_size=[4, 4],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        hidden_dim=96,
        window_size=[7, 7],
        drop_path_rate=0.2,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url=None,
    meta={"input_size": (224, 224)},
    default=True,
)
def swin_s(**kwargs) -> SwinTransformer:
    return SwinTransformer(
        block=SwinTransformerBlock,
        downsample=PatchMerging,
        patch_size=[4, 4],
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        hidden_dim=96,
        window_size=[7, 7],
        drop_path_rate=0.3,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url=None,
    meta={"input_size": (224, 224)},
    default=True,
)
def swin_b(**kwargs) -> SwinTransformer:
    return SwinTransformer(
        block=SwinTransformerBlock,
        downsample=PatchMerging,
        patch_size=[4, 4],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        hidden_dim=128,
        window_size=[7, 7],
        drop_path_rate=0.5,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url=None,
    meta={"input_size": (224, 224)},
    default=True,
)
def swin_v2_t(**kwargs) -> SwinTransformer:
    return SwinTransformer(
        block=SwinTransformerBlockV2,
        downsample=PatchMergingV2,
        patch_size=[4, 4],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        hidden_dim=96,
        window_size=[8, 8],
        drop_path_rate=0.2,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url=None,
    meta={"input_size": (224, 224)},
    default=True,
)
def swin_v2_s(**kwargs) -> SwinTransformer:
    return SwinTransformer(
        block=SwinTransformerBlockV2,
        downsample=PatchMergingV2,
        patch_size=[4, 4],
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        hidden_dim=96,
        window_size=[8, 8],
        drop_path_rate=0.3,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url=None,
    meta={"input_size": (224, 224)},
    default=True,
)
def swin_v2_b(**kwargs) -> SwinTransformer:
    return SwinTransformer(
        block=SwinTransformerBlockV2,
        downsample=PatchMergingV2,
        patch_size=[4, 4],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        hidden_dim=128,
        window_size=[8, 8],
        drop_path_rate=0.5,
        **kwargs,
    )
