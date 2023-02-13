from __future__ import annotations
import typing as tp
from functools import partial

import jax.numpy as jnp
from flax import linen
import chex

from limo import register_model


ModuleDef = tp.Any


class ConvNeXtBlock(linen.Module):
    features: int
    init_layer_scale: float
    drop_path_rate: float

    dense: ModuleDef
    conv: ModuleDef
    norm: ModuleDef
    stochastic_depth: ModuleDef

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        h = self.conv(
            self.features,
            (7, 7),
            padding=3,
            feature_group_count=self.features,
            name="block.0",
        )(x)
        h = self.norm(name="block.2")(h)
        h = self.dense(4 * self.features, name="block.3")(h)
        h = linen.gelu(h, approximate=False)
        h = self.dense(self.features, name="block.5")(h)

        h = h * self.param(
            "layer_scale", linen.initializers.constant(self.init_layer_scale), (self.features,)
        )
        h = self.stochastic_depth(self.drop_path_rate)(h)
        return x + h


class ConvNeXt(linen.Module):
    widths: tp.Sequence[int]
    depths: tp.Sequence[int]

    drop_rate: float = 0
    drop_path_rate: float = 0
    init_layer_scale: float = 1e-6
    num_classes: int = 1000
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        dense = partial(
            linen.Dense,
            dtype=self.dtype,
            kernel_init=linen.initializers.variance_scaling(
                scale=0.02, mode="fan_in", distribution="truncated_normal"
            ),
        )
        conv = partial(
            linen.Conv,
            dtype=self.dtype,
            kernel_init=linen.initializers.variance_scaling(
                scale=0.02, mode="fan_in", distribution="truncated_normal"
            ),
        )
        norm = partial(
            linen.LayerNorm, dtype=self.norm_dtype or self.dtype, axis_name=self.axis_name
        )
        stochastic_depth = partial(
            linen.Dropout, broadcast_dims=(-1, -2, -3), deterministic=not is_training
        )

        assert len(self.widths) == len(self.depths)
        layer_idx = 0

        # stem
        x = conv(
            self.widths[0],
            (4, 4),
            strides=4,
            padding=0,
            name=f"features.{layer_idx}.0",
        )(x)
        x = norm(name=f"features.{layer_idx}.1")(x)
        layer_idx += 1

        total_blocks = sum(self.depths)
        for i, (width, depth) in enumerate(zip(self.widths, self.depths)):
            # bottlenecks
            for j in range(depth):
                x = ConvNeXtBlock(
                    width,
                    self.init_layer_scale,
                    self.drop_path_rate * (layer_idx - 1) / (total_blocks - 1),
                    dense=dense,
                    conv=conv,
                    norm=norm,
                    stochastic_depth=stochastic_depth,
                    name=f"features.{layer_idx}.{j}",
                )(x)
            layer_idx += 1

            if i + 1 < len(self.widths):
                # downsampling.
                x = norm(name=f"features.{layer_idx}.0")(x)
                x = conv(
                    self.widths[i + 1],
                    (2, 2),
                    2,
                    padding=0,
                    name=f"features.{layer_idx}.1",
                )(x)
                layer_idx += 1

        # global average pooling
        x = jnp.mean(x, axis=(-2, -3))

        if self.num_classes > 0:
            x = norm(name="classifier.0")(x)
            x = linen.Dropout(self.drop_rate, deterministic=not is_training)(x)
            x = dense(self.num_classes, name="classifier.2")(x)

        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        # Stochastic depth also uses dropout rng collection.
        return ["dropout"]


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118919&authkey=ACjW5LU-jm8JTnA",  # noqa: E501
    default=True,
)
def convnext_tiny(**kwargs) -> ConvNeXt:
    kwargs.setdefault("drop_path_rate", 0.1)
    return ConvNeXt([96, 192, 384, 768], [3, 3, 9, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118920&authkey=AG62QxOQTpUPPV4",  # noqa: E501
    default=True,
)
def convnext_small(**kwargs) -> ConvNeXt:
    kwargs.setdefault("drop_path_rate", 0.4)
    return ConvNeXt([96, 192, 384, 768], [3, 3, 27, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118922&authkey=APdfeRZrgcg873Y",  # noqa: E501
    default=True,
)
def convnext_base(**kwargs) -> ConvNeXt:
    kwargs.setdefault("drop_path_rate", 0.5)
    return ConvNeXt([128, 256, 512, 1024], [3, 3, 27, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118931&authkey=AIaH9uWOCPCIA_I",  # noqa: E501
    default=True,
)
def convnext_large(**kwargs) -> ConvNeXt:
    kwargs.setdefault("drop_path_rate", 0.5)
    return ConvNeXt([192, 384, 768, 1536], [3, 3, 27, 3], **kwargs)
