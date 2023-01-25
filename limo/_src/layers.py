"""Configurable layer modules."""
from __future__ import annotations
import typing as tp

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import linen
import chex

from limo import get_config


def Dense(
    features: int,
    use_bias: bool = True,
    kernel_init: tp.Callable = linen.initializers.lecun_normal(),
    bias_init: tp.Callable = linen.initializers.zeros,
    name: str | None = None,
) -> linen.Dense:
    return linen.Dense(
        features,
        use_bias=use_bias,
        dtype=get_config("dtype"),
        kernel_init=kernel_init,
        bias_init=bias_init,
        name=name,
    )


def Conv(
    features: int,
    kernel_size: int | tp.Sequence[int],
    stride: int | tp.Sequence[int] = 1,
    dilation: int | tp.Sequence[int] = 1,
    groups: int = 1,
    bias: bool = False,
    name: str | None = None,
) -> linen.Conv:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    num_spatial_dims = len(kernel_size)

    if isinstance(stride, int):
        stride = (stride,) * num_spatial_dims
    assert len(stride) == num_spatial_dims

    if isinstance(dilation, int):
        dilation = (dilation,) * num_spatial_dims
    assert len(dilation) == num_spatial_dims

    if get_config("torch_like"):
        padding = []
        for k, s, d in zip(kernel_size, stride, dilation):
            pad_size = ((s - 1) + d * (k - 1)) // 2
            padding.append((pad_size, pad_size))
    else:
        padding = "SAME"

    return linen.Conv(
        features,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        kernel_dilation=dilation,
        feature_group_count=groups,
        use_bias=bias,
        dtype=get_config("dtype"),
        name=name,
    )


def BatchNorm(
    axis: int = -1,
    momentum: float = 0.99,
    epsilon: float = 1e-5,
    use_bias: bool = True,
    use_scale: bool = True,
    bias_init: tp.Callable = linen.initializers.zeros,
    scale_init: tp.Callable = linen.initializers.ones,
    name: str | None = None,
) -> linen.BatchNorm:
    return linen.BatchNorm(
        use_running_average=not get_config("train"),
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        dtype=get_config("norm_dtype"),
        use_bias=use_bias,
        use_scale=use_scale,
        bias_init=bias_init,
        scale_init=scale_init,
        axis_name=get_config("axis_name"),
        name=name,
    )


class MaxPool(linen.Module):
    window_shape: tp.Sequence[int]
    strides: int | tp.Sequence[int] = 1
    padding: str = "VALID"

    @linen.compact
    def __call__(self, x):
        window_shape = tuple(self.window_shape)

        strides = self.strides
        if isinstance(strides, int):
            strides = (strides,) * len(window_shape)
        strides = tuple(strides)
        assert len(strides) == len(window_shape)

        padding = self.padding
        if padding == "SAME" and get_config("torch_like"):
            padding = []
            for k, s in zip(window_shape, strides):
                pad_size = ((s - 1) + (k - 1)) // 2
                padding.append((pad_size, pad_size))
            padding = tuple(padding)

        return linen.max_pool(x, window_shape, strides, padding)


class Dropout(linen.Module):
    """DropPath module modified from official Dropout impl.
    The masked output is scaled by `1/(1-rate)`.

    Args:
        rate: Drop rate.
        num_spatial_dims: Number of spatial dims. Required to support `batch-free` inputs.
        rng_collection: PRNG key name.
    """

    rate: float
    broadcast_dims: tp.Sequence[int] = ()
    rng_collection: str = "dropout"

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if (self.rate == 0.0) or not get_config("train"):
            return x

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(x)

        keep_prob = 1.0 - self.rate
        rng = self.make_rng(self.rng_collection)
        broadcast_shape = list(x.shape)
        for dim in self.broadcast_dims:
            broadcast_shape[dim] = 1
        mask = jr.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, x.shape)
        return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class DropPath(linen.Module):
    """DropPath module modified from official Dropout impl.
    The masked output is scaled by `1/(1-rate)`.

    Args:
        rate: Drop rate.
        num_spatial_dims: Number of spatial dims. Required to support `batch-free` inputs.
        rng_collection: PRNG key name.
    """

    rate: float
    num_spatial_dims: int = 2
    rng_collection: str = "dropout"

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if (self.rate == 0.0) or not get_config("train"):
            return x

        # Prevent gradient NaNs in 1.0 edge-case.
        if self.rate == 1.0:
            return jnp.zeros_like(x)

        dims = self.num_spatial_dims + 1
        broadcast_shape = tuple(x.shape[: x.ndim - dims]) + (1,) * dims
        assert len(broadcast_shape) == x.ndim

        keep_prob = 1.0 - self.rate
        rng = self.make_rng(self.rng_collection)

        mask = jr.bernoulli(rng, p=keep_prob, shape=broadcast_shape)
        mask = jnp.broadcast_to(mask, x.shape)
        return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


#
#  Activation layers.
#
class Identity(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return x


class Sigmoid(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.sigmoid(x)


class HardSigmoid(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.hard_sigmoid(x)


class Tanh(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.tanh(x)


class HardTanh(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.hard_tanh(x)


class ReLU(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.relu(x)


class ReLU6(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.relu6(x)


class LeakyReLU(linen.Module):
    negative_slope: float = 0.01

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.leaky_relu(x, self.negative_slope)


class SiLU(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.silu(x)


class HardSiLU(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.hard_silu(x)


class Mish(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return x * linen.tanh(linen.softplus(x))


class HardMish(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return 0.5 * x * jnp.clip(x + 2, 0, 2)


class GELU(linen.Module):
    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.gelu(x, approximate=False)


class GELUTanh(linen.Module):
    def __call__(self, x: chex.Array) -> chex.Array:
        return linen.gelu(x, approximate=True)


#
#  Experimental layers.
#
class TanhExp(linen.Module):
    """TanhExp layer.

    Paper: TanhExp: A smooth activation function with high convergence speed for
        lightweight neural networks - arXiv:2003.09855
    """

    def __call__(self, x: chex.Array) -> chex.Array:
        return x * jnp.tanh(jnp.exp(x))


class FReLU(linen.Module):
    """Funnel activation or FReLU layer.

    Paper: Funnel Activation for Visual Recognition - arxiv:2007.11824

    Args:
        kernel_size: Kernel size of funnel condition.
    """

    kernel_size: tp.Sequence[int]

    def __call__(self, x: chex.Array) -> chex.Array:
        features = x.shape[-1]
        tx = Conv(features, self.kernel_size, groups=features, name="conv")(x)
        tx = BatchNorm(name="bn")(tx)
        return jnp.maximum(x, tx)
