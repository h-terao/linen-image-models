from __future__ import annotations
import typing as tp
import dataclasses
from functools import partial
import inspect
import math

import jax.numpy as jnp
from flax import linen
import chex

from limo import layers
from limo import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from limo import register_model, register_pretrained

# Uncomment below two lines if you folk this file.
# register_model = lambda x, y: y
# register_variables = lambda *args, **kwargs: None


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
    se_ratio: float = 0
    drop_path_rate: float = 0
    torch_like: bool = True
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.ReLU
    norm_layer: ModuleDef = layers.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        conv_layer = partial(self.conv_layer, torch_like=self.torch_like)

        in_features = x.shape[-1]
        if self.group_size:
            groups, mod = divmod(in_features, self.group_size)
            assert mod == 0
        else:
            groups = 1

        identity = x

        x = conv_layer(
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

        x = conv_layer(self.features, self.pw_kernel_size, name="conv_pw")(x)
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
    se_ratio: float = 0
    drop_path_rate: float = 0
    torch_like: bool = True
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.ReLU
    norm_layer: ModuleDef = layers.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        conv_layer = partial(self.conv_layer, torch_like=self.torch_like)

        in_features = x.shape[-1]
        mid_features = make_divisible(in_features * self.exp_ratio)
        if self.group_size:
            groups, mod = divmod(mid_features, self.group_size)
            assert mod == 0
        else:
            groups = 1

        identity = x

        # Point-wise expansion
        x = conv_layer(mid_features, self.exp_kernel_size, name="conv_pw")(x)
        x = self.norm_layer(name="bn1")(x)
        x = self.act_layer(name="bn1.act")(x)

        # Depth-wise convolution
        x = conv_layer(
            mid_features,
            self.dw_kernel_size,
            self.stride,
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
        x = conv_layer(self.features, self.pw_kernel_size, 1, name="conv_pwl")(x)
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


class EfficientNet(linen.Module):
    """
    Args:
        torch_like: If True, use PyTorch-like padding approach
            in conv and pooling layers.
    """

    stage_specs: tp.Sequence[StageSpec]
    stem_size: int
    features: int
    num_classes: int = 1000
    drop_rate: float = 0
    drop_path_rate: float = 0.2
    torch_like: bool = True
    conv_layer: ModuleDef = layers.Conv
    norm_layer: ModuleDef = layers.BatchNorm
    act_layer: ModuleDef = layers.SiLU

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        conv_layer = partial(self.conv_layer, torch_like=self.torch_like)

        x = conv_layer(self.stem_size, 3, 2, name="conv_stem")(x)
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
                "torch_like": self.torch_like,
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

        x = conv_layer(self.features, 1, torch_like=self.torch_like, name="conv_head")(x)
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

    def model_maker(**kwargs):
        if "drop_rate" not in kwargs:
            kwargs["drop_rate"] = drop_rate

        # filter.
        all_keys = inspect.signature(EfficientNet).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in all_keys}

        return EfficientNet(
            stage_specs=stage_specs,
            stem_size=stem_size,
            features=features,
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


#
#  Remove the below section if you folk this file.
#

_cfg = {
    "num_classes": 1000,
    "input_size": (224, 224, 3),
    "crop_mode": None,
    "crop_pct": 0.875,
    "interpolation": "bicubic",
    "mean": IMAGENET_DEFAULT_MEAN,
    "std": IMAGENET_DEFAULT_STD,
    "torch_like": True,
}


register_model("efficientnet_b0", efficientnet_b0, _cfg)
register_model(
    "efficientnet_b1", efficientnet_b1, dict(_cfg, test_input_size=(256, 256, 3), crop_pct=1.0)
)
register_model(
    "efficientnet_b2",
    efficientnet_b2,
    dict(_cfg, input_size=(256, 256, 3), test_input_size=(288, 288, 3), crop_pct=1.0),
)
register_model(
    "efficientnet_b3",
    efficientnet_b3,
    dict(_cfg, input_size=(288, 288, 3), test_input_size=(320, 320, 3), crop_pct=1.0),
)
register_model(
    "efficientnet_b4",
    efficientnet_b4,
    dict(_cfg, input_size=(320, 320, 3), test_input_size=(384, 384, 3), crop_pct=1.0),
)
register_model(
    "efficientnet_b5", efficientnet_b5, dict(_cfg, input_size=(456, 456, 3), crop_pct=0.934)
)
register_model(
    "efficientnet_b6", efficientnet_b6, dict(_cfg, input_size=(528, 528, 3), crop_pct=0.942)
)
register_model(
    "efficientnet_b7", efficientnet_b7, dict(_cfg, input_size=(600, 600, 3), crop_pct=0.949)
)
register_model(
    "efficientnet_b8", efficientnet_b8, dict(_cfg, input_size=(672, 672, 3), crop_pct=0.954)
)


# Pretrained models come from TIMM.
# https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/efficientnet.py
register_pretrained(
    "efficientnet_b0",
    "ra_in1k",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118878&authkey=AG9gcJIqTDdSWnU",  # noqa: E501
    default=True,
)

register_pretrained(
    "efficientnet_b1",
    "ft_in1k",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118880&authkey=ABVOTHwmZyMNApU",  # noqa: E501
    default=True,
)

register_pretrained(
    "efficientnet_b2",
    "ra_in1k",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118879&authkey=AG98vf1U24xjh4w",  # noqa: E501
    default=True,
)

register_pretrained(
    "efficientnet_b3",
    "ra2_in1k",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118881&authkey=AJEJzzKVegOaI1Q",  # noqa: E501
    default=True,
)

register_pretrained(
    "efficientnet_b4",
    "ra2_in1k",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118886&authkey=AKGpCVHgPtpdGhc",  # noqa: E501
    default=True,
)
