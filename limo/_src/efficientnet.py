from __future__ import annotations
import typing as tp
from functools import partial
import math

import jax.numpy as jnp
from flax import linen
import chex

from limo import register_model


ModuleDef = tp.Any


def make_divisible(
    v: int, divisor: int = 8, min_value: int | None = None, round_limit: float = 0.9
):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite(linen.Module):
    hidden_dim: int
    act: tp.Callable = linen.relu
    scale_act: tp.Callable = linen.sigmoid
    conv: ModuleDef = linen.Conv

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        h = jnp.mean(x, axis=(-2, -3), keepdims=True)
        h = self.conv(self.hidden_dim, (1, 1), padding=0, name="fc1")(h)
        h = self.act(h)
        h = self.conv(x.shape[-1], (1, 1), padding=0, name="fc2")(h)
        return x * self.scale_act(h)


class MBConv(linen.Module):
    features: int
    expand_ratio: float
    kernel_size: int
    stride: int
    drop_path_rate: float
    conv: ModuleDef  # partial(use_bias=False, dtype)
    norm: ModuleDef  # partial(use_running_average, dtype, axis_name)
    squeeze_excite: ModuleDef  # partial(act, dtype)
    stochastic_depth: ModuleDef

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        h = x

        features = x.shape[-1]
        expanded_features = make_divisible(features * self.expand_ratio)

        block_idx = 0
        if features != expanded_features:
            h = self.conv(expanded_features, (1, 1), padding=0, name=f"block.{block_idx}.0")(h)
            h = self.norm(name=f"block.{block_idx}.1")(h)
            h = linen.silu(h)
            block_idx += 1

        # depthwise.
        h = self.conv(
            expanded_features,
            (self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=((self.stride - 1) + (self.kernel_size - 1)) // 2,
            feature_group_count=expanded_features,
            name=f"block.{block_idx}.0",
        )(h)
        h = self.norm(name=f"block.{block_idx}.1")(h)
        h = linen.silu(h)
        block_idx += 1

        # squeeze and excitation.
        h = self.squeeze_excite(max(1, features // 4), name=f"block.{block_idx}")(h)
        block_idx += 1

        # project
        h = self.conv(self.features, (1, 1), padding=0, name=f"block.{block_idx}.0")(h)
        h = self.norm(name=f"block.{block_idx}.1")(h)
        block_idx += 1

        if x.shape == h.shape:
            # residual connection.
            h = x + self.stochastic_depth(rate=self.drop_path_rate)(h)

        return h


class FusedMBConv(linen.Module):
    features: int
    expand_ratio: float
    kernel_size: int
    stride: int
    drop_path_rate: float
    conv: ModuleDef  # partial(use_bias=False, dtype)
    norm: ModuleDef  # partial(use_running_average, dtype, axis_name)
    squeeze_excite: ModuleDef  # partial(act, dtype)
    stochastic_depth: ModuleDef

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        h = x

        features = x.shape[-1]
        expanded_features = make_divisible(features * self.expand_ratio)

        block_idx = 0
        if features != expanded_features:
            # fused expand.
            h = self.conv(
                expanded_features,
                (self.kernel_size, self.kernel_size),
                strides=self.stride,
                padding=((self.stride - 1) + (self.kernel_size - 1)) // 2,
                name=f"block.{block_idx}.0",
            )(h)
            h = self.norm(name=f"block.{block_idx}.1")(h)
            h = linen.silu(h)
            block_idx += 1

            # project.
            h = self.conv(self.features, (1, 1), padding=0, name=f"block.{block_idx}.0")(h)
            h = self.norm(name=f"block.{block_idx}.1")(h)
            block_idx += 1
        else:
            # fused expand.
            h = self.conv(
                self.features,
                (self.kernel_size, self.kernel_size),
                strides=self.stride,
                padding=((self.stride - 1) + (self.kernel_size - 1)) // 2,
                name=f"block.{block_idx}.0",
            )(h)
            h = self.norm(name=f"block.{block_idx}.1")(h)
            h = linen.silu(h)
            block_idx += 1

        if x.shape == h.shape:
            # residual connection.
            h = x + self.stochastic_depth(rate=self.drop_path_rate)(h)

        return h


class BlockSpec(tp.NamedTuple):
    block: ModuleDef
    features: int
    expand_ratio: float
    kernel_size: int
    stride: int
    num_layers: int


class EfficientNet(linen.Module):
    stem_size: int
    block_specs: tp.Sequence[BlockSpec]
    features: int | None = None
    drop_rate: float = 0

    num_classes: int = 1000
    drop_path_rate: float = 0.2
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    norm_momentum: float = 0.9
    norm_epsilon: float = 1e-5
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        conv = partial(linen.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(
            linen.BatchNorm,
            use_running_average=not is_training,
            momentum=self.norm_momentum,
            epsilon=self.norm_epsilon,
            dtype=self.norm_dtype or self.dtype,
            axis_name=self.axis_name,
        )
        squeeze_excite = partial(SqueezeExcite, act=linen.silu, conv=conv)
        stochastic_depth = partial(
            linen.Dropout, broadcast_dims=(-1, -2, -3), deterministic=not is_training
        )
        layer_idx = 0

        # stem
        x = conv(self.stem_size, (3, 3), 2, padding=1, name=f"features.{layer_idx}.0")(x)
        x = norm(name=f"features.{layer_idx}.1")(x)
        x = linen.silu(x)
        layer_idx += 1

        total_blocks = sum([block_spec.num_layers for block_spec in self.block_specs])
        block_idx = 0
        for block_spec in self.block_specs:
            for i in range(block_spec.num_layers):
                x = block_spec.block(
                    features=block_spec.features,
                    expand_ratio=block_spec.expand_ratio,
                    kernel_size=block_spec.kernel_size,
                    stride=block_spec.stride if i == 0 else 1,
                    drop_path_rate=self.drop_path_rate * block_idx / total_blocks,
                    conv=conv,
                    norm=norm,
                    squeeze_excite=squeeze_excite,
                    stochastic_depth=stochastic_depth,
                    name=f"features.{layer_idx}.{i}",
                )(x)
                block_idx += 1
            layer_idx += 1

        # Last several layers.
        x = conv(
            self.features or 4 * block_spec.features,
            (1, 1),
            padding=0,
            name=f"features.{layer_idx}.0",
        )(x)
        x = norm(name=f"features.{layer_idx}.1")(x)
        x = linen.silu(x)

        # global average pooling.
        x = jnp.mean(x, axis=(-2, -3))

        if self.num_classes > 0:
            x = linen.Dropout(self.drop_rate)(x, not is_training)
            x = linen.Dense(self.num_classes, dtype=self.dtype, name="classifier.1")(x)

        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        # Stochastic depth also uses dropout rng collection.
        return ["dropout"]


def _efficientnet_v1_config(width_mult, depth_mult):
    block_specs = [
        BlockSpec(MBConv, 16, 1, 3, 1, 1),
        BlockSpec(MBConv, 24, 6, 3, 2, 2),
        BlockSpec(MBConv, 40, 6, 5, 2, 2),
        BlockSpec(MBConv, 80, 6, 3, 2, 3),
        BlockSpec(MBConv, 112, 6, 5, 1, 3),
        BlockSpec(MBConv, 192, 6, 5, 2, 4),
        BlockSpec(MBConv, 320, 6, 3, 1, 1),
    ]

    new_block_specs = []
    for block_spec in block_specs:
        features = make_divisible(block_spec.features * width_mult)
        num_layers = int(math.ceil(block_spec.num_layers * depth_mult))
        new_block_spec = block_spec._replace(features=features, num_layers=num_layers)
        new_block_specs.append(new_block_spec)

    stem_size = make_divisible(32 * width_mult)
    return stem_size, new_block_specs, None


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118929&authkey=AMM6d8g3gpozW7M",  # noqa: E501
    default=True,
)
def efficientnet_b0(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.0, 1.0)
    kwargs.setdefault("drop_rate", 0.2)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118924&authkey=AM_ZAxIF6bceIfY",  # noqa: E501
    default=True,
)
def efficientnet_b1(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.0, 1.1)
    kwargs.setdefault("drop_rate", 0.2)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118930&authkey=AO_txgmS0-4cjxo",  # noqa: E501
    default=True,
)
def efficientnet_b2(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.1, 1.2)
    kwargs.setdefault("drop_rate", 0.3)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118923&authkey=ADFUQR9Rt3dokGc",  # noqa: E501
    default=True,
)
def efficientnet_b3(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.2, 1.4)
    kwargs.setdefault("drop_rate", 0.3)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118927&authkey=ADHllSwTMeKCwi4",  # noqa: E501
    default=True,
)
def efficientnet_b4(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.4, 1.8)
    kwargs.setdefault("drop_rate", 0.4)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118921&authkey=AMHpPcHlf6jQHK8",  # noqa:E501
    default=True,
)
def efficientnet_b5(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.6, 2.2)
    kwargs.setdefault("drop_rate", 0.4)
    kwargs.setdefault("norm_epsilon", 1e-3)
    kwargs.setdefault("norm_momentum", 0.99)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118928&authkey=AMY-SRgG9nngMbM",  # noqa:E501
    default=True,
)
def efficientnet_b6(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(1.8, 2.6)
    kwargs.setdefault("drop_rate", 0.5)
    kwargs.setdefault("norm_epsilon", 1e-3)
    kwargs.setdefault("norm_momentum", 0.99)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118933&authkey=AMT3FroLOieY4Ds",  # noqa: E501
    default=True,
)
def efficientnet_b7(**kwargs):
    stem_size, block_specs, features = _efficientnet_v1_config(2.0, 3.1)
    kwargs.setdefault("drop_rate", 0.5)
    kwargs.setdefault("norm_epsilon", 1e-3)
    kwargs.setdefault("norm_momentum", 0.99)
    return EfficientNet(stem_size, block_specs, features, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118925&authkey=AEKLdkU8H3mtDnY",  # noqa:E501
    default=True,
)
def efficientnet_v2_s(**kwargs):
    block_specs = [
        BlockSpec(FusedMBConv, 24, 1, 3, 1, 2),
        BlockSpec(FusedMBConv, 48, 4, 3, 2, 4),
        BlockSpec(FusedMBConv, 64, 4, 3, 2, 4),
        BlockSpec(MBConv, 128, 4, 3, 2, 6),
        BlockSpec(MBConv, 160, 6, 3, 1, 9),
        BlockSpec(MBConv, 256, 6, 3, 2, 15),
    ]
    kwargs.setdefault("drop_rate", 0.2)
    kwargs.setdefault("norm_epsilon", 1e-3)
    return EfficientNet(24, block_specs, 1280, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118926&authkey=AOhvlaJKq7U00ls",  # noqa:E501
    default=True,
)
def efficientnet_v2_m(**kwargs):
    block_specs = [
        BlockSpec(FusedMBConv, 24, 1, 3, 1, 3),
        BlockSpec(FusedMBConv, 48, 4, 3, 2, 5),
        BlockSpec(FusedMBConv, 80, 4, 3, 2, 5),
        BlockSpec(MBConv, 160, 4, 3, 2, 7),
        BlockSpec(MBConv, 176, 6, 3, 1, 14),
        BlockSpec(MBConv, 304, 6, 3, 2, 18),
        BlockSpec(MBConv, 512, 6, 3, 1, 5),
    ]
    kwargs.setdefault("drop_rate", 0.3)
    kwargs.setdefault("norm_epsilon", 1e-3)
    return EfficientNet(24, block_specs, 1280, **kwargs)


@register_model(
    pretrained="IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118934&authkey=AG_Rxb4CNwOFBCE",  # noqa:E501
    default=True,
)
def efficientnet_v2_l(**kwargs):
    block_specs = [
        BlockSpec(FusedMBConv, 32, 1, 3, 1, 4),
        BlockSpec(FusedMBConv, 64, 4, 3, 2, 7),
        BlockSpec(FusedMBConv, 96, 4, 3, 2, 7),
        BlockSpec(MBConv, 192, 4, 3, 2, 10),
        BlockSpec(MBConv, 224, 6, 3, 1, 19),
        BlockSpec(MBConv, 384, 6, 3, 2, 25),
        BlockSpec(MBConv, 640, 6, 3, 1, 7),
    ]
    kwargs.setdefault("drop_rate", 0.4)
    kwargs.setdefault("norm_epsilon", 1e-3)
    return EfficientNet(32, block_specs, 1280, **kwargs)
