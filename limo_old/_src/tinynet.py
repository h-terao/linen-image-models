from __future__ import annotations
import typing as tp
import dataclasses

import jax.numpy as jnp
from flax import linen
import chex

from limo import (
    layers,
    register_model,
    using_config,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)

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


class TinyNet(linen.Module):
    stage_specs: tp.Sequence[StageSpec]
    stem_size: int
    features: int

    num_classes: int = 1000
    drop_rate: float = 0
    drop_path_rate: float = 0.2

    conv_layer: ModuleDef = layers.Conv
    norm_layer: ModuleDef = layers.BatchNorm
    act_layer: ModuleDef = layers.SiLU
    torch_like: bool = True

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        with using_config(torch_like=self.torch_like):
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
                if stage_spec.block == InvertedResidual:
                    kwargs["exp_ratio"] = stage_spec.exp_ratio

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
        return TinyNet(
            stage_specs=stage_specs,
            stem_size=32,
            features=features,
            num_classes=num_classes,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            **kwargs,
        )

    return model_maker


tinynet_a = _tinynet(feature_multiplier=1.0, depth_multiplier=1.2)
tinynet_b = _tinynet(feature_multiplier=0.75, depth_multiplier=1.1)
tinynet_c = _tinynet(feature_multiplier=0.54, depth_multiplier=0.85)
tinynet_d = _tinynet(feature_multiplier=0.54, depth_multiplier=0.695)
tinynet_e = _tinynet(feature_multiplier=0.51, depth_multiplier=0.6)


_cfg = {
    "input_size": (224, 224, 3),
    "crop_mode": None,
    "crop_pct": 0.875,
    "interpolation": "bicubic",
    "mean": IMAGENET_DEFAULT_MEAN,
    "std": IMAGENET_DEFAULT_STD,
}

register_model(
    "tinynet_a",
    tinynet_a,
    checkpoint_name="in1k",
    default_cfg=dict(_cfg, input_size=(192, 192, 3)),
    default_checkpoint=True,
)

register_model(
    "tinynet_b",
    tinynet_b,
    checkpoint_name="in1k",
    default_cfg=dict(_cfg, input_size=(188, 188, 3)),
    default_checkpoint=True,
)

register_model(
    "tinynet_c",
    tinynet_c,
    checkpoint_name="in1k",
    default_cfg=dict(_cfg, input_size=(184, 184, 3)),
    default_checkpoint=True,
)

register_model(
    "tinynet_d",
    tinynet_d,
    checkpoint_name="in1k",
    default_cfg=dict(_cfg, input_size=(152, 152, 3)),
    default_checkpoint=True,
)

register_model(
    "tinynet_e",
    tinynet_e,
    checkpoint_name="in1k",
    default_cfg=dict(_cfg, input_size=(106, 106, 3)),
    default_checkpoint=True,
)
