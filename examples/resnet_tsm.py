"""Example of model customization.

Specifically, take the following steps:
1. Copy and paste `limo._src.resnet.py`
2. Replace `from limo import register_model` with
    `from limo import fake_register_model as register_model`.
3. Add or remove modules. (In this example, implement and insert temporal shift modules.)
"""
from __future__ import annotations
import typing as tp
from functools import partial

import jax.numpy as jnp
from flax import linen
import chex

# from limo import register_model
from limo import fake_register_model as register_model


ModuleDef = tp.Any


def temporal_shift(x: chex.Array, fold_div: int = 8) -> chex.Array:
    """
    Args:
        x: Input array with shape (...,T,H,W,C)
    """
    *batch_dims, T, H, W, C = x.shape
    x = jnp.reshape(x, (-1, T, H, W, C))  # flatten.

    fold = C // fold_div
    new_x = jnp.pad(x, [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0]], mode="constant", constant_values=0)
    new_x = jnp.concatenate(
        [
            new_x[:, 2:, ..., :fold],  # shift left.
            new_x[:, :-2, ..., fold : 2 * fold],  # shift right
            new_x[:, 1:-1, ..., 2 * fold :],  # not shift
        ],
        axis=-1,
    )

    new_x = jnp.reshape(new_x, (*batch_dims, T, H, W, C))
    return new_x


class BasicBlock(linen.Module):
    features: int
    stride: int = 1
    temporal_shift: bool = False
    groups: int = 1
    base_width: int = 64
    expansion: int = 1
    dilation: int = 1

    conv: tp.Type[linen.Conv] = None
    norm: tp.Type[linen.BatchNorm] = None

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        assert self.groups == 1
        assert self.base_width == 64
        assert self.expansion == 1
        assert self.dilation == 1, "dilation > 1 is not supported in BasicBlock."

        h = x
        if self.temporal_shift:
            h = temporal_shift(h)

        h = self.conv(
            self.features,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=self.dilation,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            name="conv1",
        )(h)
        h = self.norm(name="bn1")(h)
        h = linen.relu(h)

        h = self.conv(
            self.features,
            kernel_size=(3, 3),
            padding=self.dilation,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            name="conv2",
        )(h)
        h = self.norm(scale_init=linen.initializers.constant(0), name="bn2")(h)

        if x.shape != h.shape:
            x = self.conv(
                self.features,
                kernel_size=(1, 1),
                strides=self.stride,
                name="downsample.0",
            )(x)
            x = self.norm(name="downsample.1")(x)

        return linen.relu(x + h)


class Bottleneck(linen.Module):
    features: int
    stride: int = 1
    temporal_shift: bool = False
    groups: int = 1
    base_width: int = 64
    expansion: int = 4
    dilation: int = 1

    conv: tp.Type[linen.Conv] = None
    norm: tp.Type[linen.BatchNorm] = None

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        width = int(self.features * self.base_width / 64.0) * self.groups

        h = x
        if self.temporal_shift:
            h = temporal_shift(h)

        h = self.conv(width, kernel_size=(1, 1), name="conv1")(h)
        h = self.norm(name="bn1")(h)
        h = linen.relu(h)

        h = self.conv(
            width,
            kernel_size=(3, 3),
            strides=self.stride,
            padding=self.dilation,
            kernel_dilation=self.dilation,
            feature_group_count=self.groups,
            name="conv2",
        )(h)
        h = self.norm(name="bn2")(h)
        h = linen.relu(h)

        h = self.conv(self.features * self.expansion, kernel_size=(1, 1), name="conv3")(h)
        h = self.norm(scale_init=linen.initializers.constant(0), name="bn3")(h)

        if x.shape != h.shape:
            x = self.conv(
                self.features * self.expansion,
                kernel_size=(1, 1),
                strides=self.stride,
                padding="VALID",
                name="downsample.0",
            )(x)
            x = self.norm(name="downsample.1")(x)

        return linen.relu(x + h)


class ResNet(linen.Module):
    block: tp.Type[BasicBlock | Bottleneck]
    stage_sizes: tp.Sequence[int]

    num_classes: int = 1000
    drop_rate: float = 0
    groups: int = 1
    width_per_group: int = 64
    replace_stride_with_dilation: tp.Sequence[bool] | None = None
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        conv: tp.Type[linen.Conv] = partial(
            linen.Conv, padding="VALID", dtype=self.dtype, use_bias=False
        )
        norm: tp.Type[linen.BatchNorm] = partial(
            linen.BatchNorm,
            use_running_average=not is_training,
            dtype=self.norm_dtype or self.dtype,
            axis_name=self.axis_name,
        )

        replace_stride_with_dilation = self.replace_stride_with_dilation
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False] * len(self.stage_sizes)

        x = conv(64, (7, 7), strides=2, padding=3, use_bias=False, name="conv1")(x)
        x = norm(name="bn1")(x)
        x = linen.relu(x)
        x = linen.max_pool(x, (3, 3), (2, 2), ((1, 1), (1, 1)))

        n_round = 1 if self.stage_sizes[3] < 23 else 2
        block_idx = 0

        dilation = 1
        for i, block_size in enumerate(self.stage_sizes):
            stride = 1 if i == 0 else 2
            if replace_stride_with_dilation[i]:
                dilation *= stride
                stride = 1

            for j in range(block_size):
                x = self.block(
                    64 * (2**i),
                    stride=stride if j == 0 else 1,
                    temporal_shift=block_idx % n_round == 0,
                    groups=self.groups,
                    base_width=self.width_per_group,
                    dilation=dilation,
                    conv=conv,
                    norm=norm,
                    name=f"layer{i+1}.{j}",
                )(x)
                n_round += 1

        x = jnp.mean(x, axis=(-2, -3))  # global average pooling.
        if self.num_classes > 0:
            x = linen.Dropout(self.drop_rate, deterministic=not is_training)(x)
            x = linen.Dense(self.num_classes, dtype=self.dtype, name="fc")(x)
        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        return ["dropout"]


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118937&authkey=AN97JGd5PA0cGdg",  # noqa: E501
    default=True,
)
def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118941&authkey=AJ3QE127yQB3lCw",  # noqa:E501
    default=True,
)
def resnet34(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118954&authkey=AND8t6gMX6Ib5pI",  # noqa: E501
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118940&authkey=AOVVlnid3m7kWLM",  # noqa:E501
    default=True,
)
def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118955&authkey=AJpYfmfmG-mfuC0",  # noqa: E501,
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118935&authkey=ABWThemG90mtqq8",  # noqa:E501
    default=True,
)
def resnet101(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118957&authkey=ALmD7XMgEV0GHlg",  # noqa: E501
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118936&authkey=ABE6gEj87AYtuII",  # noqa:E501
    default=True,
)
def resnet152(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118956&authkey=AHL0v4Je5L3iGPE",  # noqa: E501
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118932&authkey=AHCGsd4W5NRAysc",  # noqa:E501
    default=True,
)
def resnext50_32x4d(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118959&authkey=ALXk9rcFhMo85rU",  # noqa: E501
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118943&authkey=AF0-Cq0gGhKUFPE",  # noqa:E501
    default=True,
)
def resnext101_32x8d(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118938&authkey=AJtfdrIb1fFl6A0",  # noqa:E501
    default=True,
)
def resnext101_64x4d(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], groups=64, width_per_group=4, **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118958&authkey=AEPQVkGP-5KzF-k",  # noqa: E501
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118939&authkey=AH-vHXStFHXE8uc",  # noqa:E501
    default=True,
)
def wide_resnet50_2(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], width_per_group=64 * 2, **kwargs)


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118960&authkey=AKcX4_2Xgf3VyVI",  # noqa: E501
)
@register_model(
    "IMAGENET1K_V2",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118942&authkey=ANa8PmwYcOWi8GQ",  # noqa:E501
    default=True,
)
def wide_resnet101_2(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], width_per_group=64 * 2, **kwargs)
