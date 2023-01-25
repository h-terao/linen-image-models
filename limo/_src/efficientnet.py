from __future__ import annotations
import typing as tp
import dataclasses
import math

import jax.numpy as jnp
from flax import linen
import chex

from limo import layers
from limo import ModuleDef


def make_divisible(
    v: int, divisor: int = 8, min_value: int | None = None, round_limit: float = 0.9
):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SqueezeExcite(linen.Module):
    rd_ratio: float = 0.25
    rd_features: int | None = None
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.ReLU
    gate_layer: ModuleDef = layers.Sigmoid
    round_fn: tp.Callable = round

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        features = x.shape[-1]
        rd_features = self.rd_features or round(self.rd_ratio * features)

        x_se = jnp.mean(x, axis=(-2, -3), keepdims=True)
        x_se = self.conv_layer(rd_features, 1, bias=True, name="conv_reduce")(x_se)
        x_se = self.act_layer(name="act1")(x_se)
        x_se = self.conv_layer(features, 1, bias=True, name="conv_expand")(x_se)
        return x * self.gate_layer(name="gate")(x_se)


class ConvNormAct(linen.Module):
    features: int
    kernel_size: int
    stride: int = 1
    dilation: int = 1
    group_size: int = 0

    no_skip: bool = True
    conv_layer: ModuleDef = layers.Conv
    norm_layer: ModuleDef = layers.BatchNorm
    act_layer: ModuleDef = layers.ReLU
    drop_path_rate: float = 0

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        in_features = x.shape[-1]
        if self.group_size:
            groups, mod = divmod(in_features, self.group_size)
            assert mod == 0
        else:
            groups = 1

        identity = x

        x = self.conv_layer(
            self.features, self.kernel_size, self.stride, self.dilation, groups, name="conv"
        )(x)
        x = self.norm_layer(name="bn1")(x)
        x = self.act_layer(name="bn1.act")(x)

        if x.shape == identity.shape and not self.no_skip:
            x = layers.DropPath(self.drop_path_rate, name="drop_path")(x) + identity

        return x


class DepthwiseSeparableConv(linen.Module):
    features: int
    dw_kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    group_size: int = 1
    no_skip: bool = False
    pw_kernel_size: int = 1
    pw_act: bool = False

    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.ReLU
    norm_layer: ModuleDef = layers.BatchNorm
    se_ratio: float = 0
    drop_path_rate: float = 0

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        in_features = x.shape[-1]
        if self.group_size:
            groups, mod = divmod(in_features, self.group_size)
            assert mod == 0
        else:
            groups = 1

        identity = x

        x = self.conv_layer(
            in_features,
            self.dw_kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            name="conv_dw",
        )(x)
        x = self.norm_layer(name="bn1")(x)
        x = self.act_layer(name="bn1.act")(x)

        # squeeze-and-excitation.
        if self.se_ratio > 0:
            x = SqueezeExcite(
                self.se_ratio,
                conv_layer=self.conv_layer,
                act_layer=self.act_layer,
                name="se",
            )(x)

        x = self.conv_layer(self.features, self.pw_kernel_size, name="conv_pw")(x)
        x = self.norm_layer(name="bn2")(x)
        if self.pw_act:
            x = self.act_layer(name="bn2.act")(x)

        if x.shape == identity.shape and not self.no_skip:
            x = layers.DropPath(self.drop_path_rate, name="drop_path")(x) + identity

        return x


class InvertedResidual(linen.Module):
    features: int
    dw_kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    group_size: int = 1
    no_skip: bool = False
    exp_ratio: float = 1.0
    exp_kernel_size: int = 1
    pw_kernel_size: int = 1

    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.ReLU
    norm_layer: ModuleDef = layers.BatchNorm
    se_ratio: float = 0
    drop_path_rate: float = 0

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        in_features = x.shape[-1]
        mid_features = make_divisible(in_features * self.exp_ratio)
        if self.group_size:
            groups, mod = divmod(mid_features, self.group_size)
            assert mod == 0
        else:
            groups = 1

        identity = x

        # Point-wise expansion
        x = self.conv_layer(mid_features, self.exp_kernel_size, name="conv_pw")(x)
        x = self.norm_layer(name="bn1")(x)
        x = self.act_layer(name="bn1.act")(x)

        # Depth-wise convolution
        x = self.conv_layer(
            mid_features,
            self.dw_kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=groups,
            name="conv_dw",
        )(x)
        x = self.norm_layer(name="bn2")(x)
        x = self.act_layer(name="bn2.act")(x)

        # squeeze-and-excitation.
        if self.se_ratio > 0:
            x = SqueezeExcite(
                self.se_ratio / self.exp_ratio,
                conv_layer=self.conv_layer,
                act_layer=self.act_layer,
                name="se",
            )(x)

        # Point-wise linear projection
        x = self.conv_layer(self.features, self.pw_kernel_size, name="conv_pwl")(x)
        x = self.norm_layer(name="bn3")(x)

        if x.shape == identity.shape and not self.no_skip:
            x = layers.DropPath(self.drop_path_rate, name="drop_path")(x) + identity

        return x


class EdgeResidual(linen.Module):
    features: int
    exp_kernel_size: int = 3
    stride: int = 1
    dilation: int = 1
    group_size: int = 0
    force_in_features: int = 0
    no_skip: bool = False
    exp_ratio: float = 1.0
    pw_kernel_size: int = 1
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.ReLU
    norm_layer: ModuleDef = layers.BatchNorm
    se_ratio: float = 0
    drop_path_rate: float = 0

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        in_features = self.force_in_features if self.force_in_features > 0 else x.shape[-1]
        mid_features = make_divisible(in_features * self.exp_ratio)
        if self.group_size:
            groups, mod = divmod(x.shape[-1], self.group_size)
            assert mod == 0
        else:
            groups = 1

        identity = x

        # Expansion.
        x = self.conv_layer(
            mid_features, self.exp_kernel_size, self.stride, self.dilation, groups, name="conv_exp"
        )(x)
        x = self.norm_layer(name="bn1")(x)
        x = self.act_layer(name="bn1.act")(x)

        # Squeeze-and-excitation
        if self.se_ratio > 0:
            x = SqueezeExcite(self.se_ratio, None, self.conv_layer, self.act_layer)(x)

        # Point-wise linear projection.
        x = self.conv_layer(self.features, self.pw_kernel_size, name="conv_pwl")(x)
        x = self.norm_layer(name="bn2")(x)
        if x.shape == identity.shape and not self.no_skip:
            x = layers.DropPath(self.drop_path_rate, name="drop_path")(x) + identity

        return x


@dataclasses.dataclass
class StageSpec:
    block: ModuleDef
    num_blocks: int
    kernel_size: int
    stride: int
    exp_ratio: float
    features: int
    se_ratio: float = 0
    no_skip: bool = False


class EfficientNet(linen.Module):
    stage_specs: tp.Sequence[StageSpec]
    stem_size: int
    features: int

    num_classes: int = 1000
    drop_rate: float = 0
    drop_path_rate: float = 0.2

    conv_layer: ModuleDef = layers.Conv
    norm_layer: ModuleDef = layers.BatchNorm
    act_layer: ModuleDef = layers.SiLU

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.conv_layer(self.stem_size, 3, 2, name="conv_stem")(x)
        x = self.norm_layer(name="bn1")(x)
        x = self.act_layer(name="bn1.act")(x)

        total_blocks = sum(x.num_blocks for x in self.stage_specs)
        block_idx = 0
        for i, stage_spec in enumerate(self.stage_specs):
            kwargs = {
                "features": stage_spec.features,
                "dw_kernel_size": stage_spec.kernel_size,
                "se_ratio": stage_spec.se_ratio,
                "conv_layer": self.conv_layer,
                "norm_layer": self.norm_layer,
                "act_layer": self.act_layer,
                "no_skip": stage_spec.no_skip,
            }

            # Some blocks need to modify args.
            if stage_spec.block == ConvNormAct:
                kwargs["kernel_size"] = kwargs.pop("dw_kernel_size")
                del kwargs["se_ratio"]
            if stage_spec.block == InvertedResidual:
                kwargs["exp_ratio"] = stage_spec.exp_ratio
            if stage_spec.block == EdgeResidual:
                kwargs["exp_ratio"] = stage_spec.exp_ratio
                kwargs["exp_kernel_size"] = kwargs.pop("dw_kernel_size")

            for j in range(stage_spec.num_blocks):
                x = stage_spec.block(
                    stride=stage_spec.stride if j == 0 else 1,
                    drop_path_rate=self.drop_path_rate * block_idx / total_blocks,
                    **kwargs,
                    name=f"blocks.{i}.{j}",
                )(x)
                block_idx += 1

        x = self.conv_layer(self.features, 1, name="conv_head")(x)
        x = self.norm_layer(name="bn2")(x)
        x = self.act_layer(name="bn2.act")(x)

        x = jnp.mean(x, axis=(-2, -3))  # GAP
        if self.num_classes > 0:
            x = layers.Dropout(self.drop_rate)(x)
            x = layers.Dense(self.num_classes, name="classifier")(x)
        return x


def _efficientnet(feature_multiplier, depth_multiplier, drop_rate):

    stage_specs = [
        StageSpec(DepthwiseSeparableConv, 1, 3, 1, 1, 16, 0.25),
        StageSpec(InvertedResidual, 2, 3, 2, 6, 24, 0.25),
        StageSpec(InvertedResidual, 2, 5, 2, 6, 40, 0.25),
        StageSpec(InvertedResidual, 3, 3, 2, 6, 80, 0.25),
        StageSpec(InvertedResidual, 3, 5, 1, 6, 112, 0.25),
        StageSpec(InvertedResidual, 4, 5, 2, 6, 192, 0.25),
        StageSpec(InvertedResidual, 1, 3, 1, 6, 320, 0.25),
    ]

    # scale depth and width.
    stage_specs = [
        dataclasses.replace(
            x,
            features=make_divisible(feature_multiplier * x.features),
            num_blocks=int(math.ceil(depth_multiplier * x.num_blocks)),
        )
        for x in stage_specs
    ]

    stem_size = make_divisible(feature_multiplier * 32)
    features = 4 * stage_specs[-1].features

    def model_maker(
        num_classes: int = 1000, drop_rate: float = drop_rate, drop_path_rate=0.2, **kwargs
    ):
        return EfficientNet(
            stage_specs=stage_specs,
            stem_size=stem_size,
            features=features,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    return model_maker


def _tinynet(feature_multiplier, depth_multiplier):
    stage_specs = [
        StageSpec(DepthwiseSeparableConv, 1, 3, 1, 1, 16, 0.25),
        StageSpec(InvertedResidual, 2, 3, 2, 6, 24, 0.25),
        StageSpec(InvertedResidual, 2, 5, 2, 6, 40, 0.25),
        StageSpec(InvertedResidual, 3, 3, 2, 6, 80, 0.25),
        StageSpec(InvertedResidual, 3, 5, 1, 6, 112, 0.25),
        StageSpec(InvertedResidual, 4, 5, 2, 6, 192, 0.25),
        StageSpec(InvertedResidual, 1, 3, 1, 6, 320, 0.25),
    ]

    # scale depth and width.
    stage_specs = [
        dataclasses.replace(
            x,
            features=make_divisible(feature_multiplier * x.features),
            num_blocks=max(1, round(depth_multiplier * x.num_blocks)),
        )
        for x in stage_specs
    ]

    features = max(1280, make_divisible(1280 * feature_multiplier))

    def model_maker(num_classes: int = 1000, drop_rate: float = 0, drop_path_rate=0.2, **kwargs):
        return EfficientNet(
            stage_specs=stage_specs,
            stem_size=32,
            features=features,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    return model_maker


efficientnet_b0 = _efficientnet(feature_multiplier=1.0, depth_multiplier=1.0, drop_rate=0.2)
efficientnet_b1 = _efficientnet(feature_multiplier=1.0, depth_multiplier=1.1, drop_rate=0.2)
efficientnet_b2 = _efficientnet(feature_multiplier=1.1, depth_multiplier=1.2, drop_rate=0.3)
efficientnet_b3 = _efficientnet(feature_multiplier=1.2, depth_multiplier=1.4, drop_rate=0.3)
efficientnet_b4 = _efficientnet(feature_multiplier=1.4, depth_multiplier=1.8, drop_rate=0.4)
efficientnet_b5 = _efficientnet(feature_multiplier=1.6, depth_multiplier=2.2, drop_rate=0.4)
efficientnet_b6 = _efficientnet(feature_multiplier=1.8, depth_multiplier=2.6, drop_rate=0.5)
efficientnet_b7 = _efficientnet(feature_multiplier=2.0, depth_multiplier=3.1, drop_rate=0.5)
efficientnet_b8 = _efficientnet(feature_multiplier=2.2, depth_multiplier=3.6, drop_rate=0.5)


tinynet_a = _tinynet(feature_multiplier=1.0, depth_multiplier=1.2)
tinynet_b = _tinynet(feature_multiplier=0.75, depth_multiplier=1.1)
tinynet_c = _tinynet(feature_multiplier=0.54, depth_multiplier=0.85)
tinynet_d = _tinynet(feature_multiplier=0.54, depth_multiplier=0.695)
tinynet_e = _tinynet(feature_multiplier=0.51, depth_multiplier=0.6)


def efficientnetv2_s(
    num_classes: int = 1000, drop_rate: float = 0, drop_path_rate: float = 0, **kwargs
) -> EfficientNet:
    stage_specs = [
        StageSpec(ConvNormAct, 2, 3, 1, 1, 24),
        StageSpec(EdgeResidual, 4, 3, 2, 4, 48),
        StageSpec(EdgeResidual, 4, 3, 2, 4, 64),
        StageSpec(InvertedResidual, 6, 3, 2, 4, 128, 0.25),
        StageSpec(InvertedResidual, 9, 3, 1, 6, 160, 0.25),
        StageSpec(InvertedResidual, 15, 3, 2, 6, 256, 0.25),
    ]

    return EfficientNet(
        stage_specs=stage_specs,
        stem_size=24,
        features=1280,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def efficientnetv2_m(
    num_classes: int = 1000, drop_rate: float = 0, drop_path_rate: float = 0, **kwargs
) -> EfficientNet:
    stage_specs = [
        StageSpec(ConvNormAct, 3, 3, 1, 1, 24),
        StageSpec(EdgeResidual, 5, 3, 2, 4, 48),
        StageSpec(EdgeResidual, 5, 3, 2, 4, 80),
        StageSpec(InvertedResidual, 7, 3, 2, 4, 160, 0.25),
        StageSpec(InvertedResidual, 14, 3, 1, 6, 176, 0.25),
        StageSpec(InvertedResidual, 18, 3, 2, 6, 304, 0.25),
        StageSpec(InvertedResidual, 5, 3, 1, 6, 512, 0.25),
    ]

    return EfficientNet(
        stage_specs=stage_specs,
        stem_size=24,
        features=1280,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def efficientnetv2_l(
    num_classes: int = 1000, drop_rate: float = 0, drop_path_rate: float = 0, **kwargs
) -> EfficientNet:
    stage_specs = [
        StageSpec(ConvNormAct, 4, 3, 1, 1, 32),
        StageSpec(EdgeResidual, 7, 3, 2, 4, 64),
        StageSpec(EdgeResidual, 7, 3, 2, 4, 96),
        StageSpec(InvertedResidual, 10, 3, 2, 4, 192, 0.25),
        StageSpec(InvertedResidual, 19, 3, 1, 6, 224, 0.25),
        StageSpec(InvertedResidual, 25, 3, 2, 6, 384, 0.25),
        StageSpec(InvertedResidual, 7, 3, 1, 6, 640, 0.25),
    ]

    return EfficientNet(
        stage_specs=stage_specs,
        stem_size=32,
        features=1280,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )


def efficientnetv2_xl(
    num_classes: int = 1000, drop_rate: float = 0, drop_path_rate=0, **kwargs
) -> EfficientNet:
    stage_specs = [
        StageSpec(ConvNormAct, 4, 3, 1, 1, 32),
        StageSpec(EdgeResidual, 8, 3, 2, 4, 64),
        StageSpec(EdgeResidual, 8, 3, 2, 4, 96),
        StageSpec(InvertedResidual, 16, 3, 2, 4, 192, 0.25),
        StageSpec(InvertedResidual, 24, 3, 1, 6, 256, 0.25),
        StageSpec(InvertedResidual, 32, 3, 2, 6, 512, 0.25),
        StageSpec(InvertedResidual, 8, 3, 1, 6, 640, 0.25),
    ]

    return EfficientNet(
        stage_specs=stage_specs,
        stem_size=32,
        features=1280,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        **kwargs,
    )
