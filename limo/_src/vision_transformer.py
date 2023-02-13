from __future__ import annotations
import typing as tp
from functools import partial

import jax.numpy as jnp
from flax import linen
import chex

from limo import register_model


ModuleDef = tp.Any


class EncoderBlock(linen.Module):
    num_heads: int
    mlp_dim: int
    drop_rate: float
    atten_drop_rate: float

    dense: ModuleDef
    norm: ModuleDef
    attention: ModuleDef
    dropout: ModuleDef

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        features = x.shape[-1]

        # Attention block.
        h = self.norm(name="ln_1")(x)
        h = self.attention(
            self.num_heads,
            dropout_rate=self.atten_drop_rate,
            name="self_attention",
        )(h, h)
        h = self.dropout(self.drop_rate)(h)
        x = x + h

        # MLP block.
        y = self.norm(name="ln_2")(x)
        y = self.dense(self.mlp_dim, name="mlp.0")(y)
        y = linen.gelu(y, approximate=False)
        y = self.dropout(self.drop_rate)(y)
        y = self.dense(features, name="mlp.3")(y)
        y = self.dropout(self.drop_rate)(y)
        return x + y


class VisionTransformer(linen.Module):
    patch_size: int
    num_layers: int
    num_heads: int
    hidden_dim: int
    mlp_dim: int

    num_classes: int = 1000
    drop_rate: float = 0
    atten_drop_rate: float = 0
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        *batch_dims, h, w, _ = x.shape
        assert h % self.patch_size == 0, "Input height should be divisible by patch size."
        assert w % self.patch_size == 0, "Input width should be divisible by patch size."

        dense = partial(linen.Dense, dtype=self.dtype)
        norm = partial(
            linen.LayerNorm, dtype=self.norm_dtype or self.dtype, axis_name=self.axis_name
        )
        attention = partial(
            linen.MultiHeadDotProductAttention, dtype=self.dtype, deterministic=not is_training
        )
        dropout = partial(linen.Dropout, deterministic=not is_training)

        # Stem
        x = linen.Conv(
            self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding=0,
            dtype=self.dtype,
            name="conv_proj",
        )(x)
        x = jnp.reshape(x, (*batch_dims, -1, self.hidden_dim))

        # Add class token along `seq` dimension.
        class_token = self.param("class_token", linen.initializers.zeros, (self.hidden_dim,))
        class_token += jnp.zeros_like(x)[..., :1, :]  # broadcast to (*batch_dims, 1, hidden_dim)
        x = jnp.concatenate([class_token, x], axis=-2)

        # Encode.
        x = x + self.param("encoder.pos_embedding", linen.initializers.normal(0.02), x.shape[-2:])
        x = dropout(self.drop_rate)(x)
        for i in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                drop_rate=self.drop_rate,
                atten_drop_rate=self.atten_drop_rate,
                dense=dense,
                norm=norm,
                attention=attention,
                dropout=dropout,
                name=f"encoder.layers.encoder_layer_{i}",
            )(x)
        x = norm(name="encoder.ln")(x)

        x = x[..., 0, :]  # Take a token
        if self.num_classes > 0:
            x = dense(self.num_classes, name="heads.head")(x)
        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        # Stochastic depth also uses dropout rng collection.
        return ["dropout"]


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118948&authkey=AD389GL7k4I1i1s",  # noqa: E501
    default=True,
)
@register_model(
    "IMAGENET1K_SWAG_E2E_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118950&authkey=ALwqBkoCHhnBYNI",  # noqa: E501
    meta={"input_size": (384, 384)},
)
@register_model(
    "IMAGENET1K_SWAG_LINEAR_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118949&authkey=AB4wu5ggh73hkTw",  # noqa: E501
)
def vit_b_16(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118944&authkey=ALgEeEI0wpwO8lQ",  # noqa: E501
    default=True,
)
def vit_b_32(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118947&authkey=ADG0EJw7ctBMBNc",  # noqa: E501
    default=True,
)
@register_model(
    "IMAGENET1K_SWAG_E2E_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118951&authkey=AIhATrEjwHjXEmM",  # noqa: E501
    meta={"input_size": (512, 512)},
)
@register_model(
    "IMAGENET1K_SWAG_LINEAR_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118946&authkey=AEyjxdfdiV6zniA",  # noqa: E501
)
def vit_l_16(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118945&authkey=AEjJM_5fHKXgC7E",  # noqa: E501
    default=True,
)
def vit_l_32(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        **kwargs,
    )


@register_model(
    "IMAGENET1K_SWAG_E2E_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118953&authkey=ADfBxYKPk3859MY",  # noqa: E501
    meta={"input_size": (518, 518)},
    default=True,
)
@register_model(
    "IMAGENET1K_SWAG_LINEAR_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118952&authkey=APYmGtKfhzElVDY",  # noqa: E501
)
def vit_h_14(**kwargs) -> VisionTransformer:
    return VisionTransformer(
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        **kwargs,
    )
