"""ConvNext"""
from __future__ import annotations
import typing as tp
import inspect

import jax.numpy as jnp
from flax import linen
import chex

from limo import layers
from limo import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from limo import register_model, register_pretrained


ModuleDef = tp.Any


class ConvNextBlock(linen.Module):
    features: int
    kernel_size: int = 7
    stride: int = 1
    dilation: int = 1
    mlp_ratio: int = 4
    conv_mlp: bool = False
    conv_bias: bool = True
    ls_init_value: float | None = 1e-6
    drop_path_ratio: float = 0
    torch_like: bool = True
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.GELU
    norm_layer: ModuleDef = layers.LayerNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        identity = x
        x = self.conv_layer(
            self.features,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=x.shape[-1],
            bias=self.conv_bias,
            torch_like=self.torch_like,
            name="conv_dw",
        )(x)

        x = self.norm_layer(name="norm")(x)
        x = self.conv_layer(
            int(self.features * self.mlp_ratio),
            kernel_size=1,
            bias=True,
            torch_like=self.torch_like,
            name="mlp.fc1",
        )(x)
        x = self.act_layer(name="act")(x)
        x = self.conv_layer(
            self.features, kernel_size=1, bias=True, torch_like=self.torch_like, name="mlp.fc2"
        )(x)

        if self.ls_init_value is not None:
            x *= self.param(
                "gamma",
                linen.initializers.constant(self.ls_init_value),
                (self.features,),
            )

        x = layers.DropPath(self.drop_path_ratio)(x) + identity
        return x


class ConvNextStage(linen.Module):
    features: int
    kernel_size: int = 7
    stride: int = 2
    depth: int = 2
    dilation: int = (1, 1)
    drop_path_rates: tp.Sequence = None
    ls_init_value: float = 1.0
    conv_bias: bool = True
    torch_like: bool = True
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.GELU
    norm_layer: ModuleDef = layers.LayerNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if x.shape[-1] != self.features or self.stride > 1 or self.dilation[0] != self.dilation[1]:
            ds_ks = 2 if self.stride > 1 or self.dilation[0] != self.dilation[1] else 1
            x = self.norm_layer(name="downsample.0")(x)
            x = self.conv_layer(
                self.features,
                kernel_size=(ds_ks, ds_ks),
                stride=self.stride,
                dilation=self.dilation[0],
                bias=self.conv_bias,
                padding="SAME" if self.dilation[1] > 1 else "VALID",
                name="downsample.1",
            )(x)

        drop_path_rates = self.drop_path_rates
        if drop_path_rates is None:
            drop_path_rates = [0.0] * self.depth

        for i in range(self.depth):
            x = ConvNextBlock(
                features=self.features,
                kernel_size=self.kernel_size,
                dilation=self.dilation[1],
                conv_bias=self.conv_bias,
                ls_init_value=self.ls_init_value,
                drop_path_ratio=drop_path_rates[i],
                torch_like=self.torch_like,
                conv_layer=self.conv_layer,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                name=f"blocks.{i}",
            )(x)
        return x


class ConvNeXt(linen.Module):
    num_classes: int = 1000
    output_stride: int = 32
    depths: tp.Sequence[int] = (3, 3, 9, 3)
    dims: tp.Sequence[int] = (96, 192, 384, 768)
    kernel_size: int | tp.Sequence[int] = 7
    ls_init_value: float = 1e-6
    patch_size: int = 4
    drop_rate: float = 0
    drop_path_rate: float = 0
    conv_bias: bool = True
    torch_like: bool = True
    conv_layer: ModuleDef = layers.Conv
    act_layer: ModuleDef = layers.GELU
    norm_layer: ModuleDef = layers.LayerNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        kernel_sizes = (
            (self.kernel_size,) * 4 if isinstance(self.kernel_size, int) else self.kernel_size
        )

        x = self.conv_layer(
            self.dims[0],
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=self.conv_bias,
            name="stem.0",
        )(x)
        x = self.norm_layer(name="stem.1")(x)

        print("STEM SIZE:", x.shape)

        drop_path_rates = jnp.split(
            jnp.linspace(0, self.drop_path_rate, sum(self.depths)),
            jnp.cumsum(jnp.array(self.depths))[:-1],
        )

        dilation = 1
        curr_stride = self.patch_size
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= self.output_stride and self.stride > 1:
                dilation *= stride
                stride = 1
            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            print(stride, curr_stride, first_dilation, dilation)
            x = ConvNextStage(
                self.dims[i],
                kernel_size=kernel_sizes[i],
                stride=stride,
                depth=self.depths[i],
                dilation=(first_dilation, dilation),
                drop_path_rates=drop_path_rates[i],
                ls_init_value=self.ls_init_value,
                conv_bias=self.conv_bias,
                torch_like=self.torch_like,
                conv_layer=self.conv_layer,
                act_layer=self.act_layer,
                norm_layer=self.norm_layer,
                name=f"stages.{i}",
            )(x)
            print("STAGE SIZE:", x.shape)

        print("Pooling shape:", x.shape)

        x = jnp.mean(x, axis=(-2, -3))
        x = self.norm_layer(name="head.norm")(x)
        x = layers.Dense(self.num_classes, name="head.fc")(x)
        return x


def _convnext(
    depths: tp.Sequence[int] = (3, 3, 9, 3),
    dims: tp.Sequence[int] = (96, 192, 384, 768),
):
    def model_maker(**kwargs):
        all_keys = inspect.signature(ConvNeXt).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in all_keys}
        return ConvNeXt(depths=depths, dims=dims, **kwargs)

    return model_maker


convnext_atto = _convnext(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320))
convnext_femto = _convnext(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384))
convnext_pico = _convnext(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512))
convnext_nano = _convnext(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640))
convnext_tiny = _convnext(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
convnext_small = _convnext(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768))
convnext_base = _convnext(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
convnext_large = _convnext(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536))

_cfg = {
    "num_classes": 1000,
    "input_size": (224, 224, 3),
    "crop_pct": 0.875,
    "interpolation": "bicubic",
    "mean": IMAGENET_DEFAULT_MEAN,
    "std": IMAGENET_DEFAULT_STD,
}


register_model("convnext_tiny", convnext_tiny, _cfg)
register_pretrained(
    "convnext_tiny",
    "d2_in1k",
    # url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118882&authkey=AF_ciaUf0a6D-kI",  # noqa: E501
    default=True,
)
