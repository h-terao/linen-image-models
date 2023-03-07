"""Test utilities.

This module will not include in the public API of flaxnets to drop PyTorch dependency.
"""
from __future__ import annotations
import numpy
import jax.numpy as jnp
import jax.random as jr
from flax import linen
import torch

from limo._src.torch_utils import load_weights_from_torch_model


def assert_computable(
    flax_model: linen.Module,
    is_training: bool = False,
    input_size: tuple[int, int, int] = (224, 224, 3),
    batch_size: int = 4,
):
    param_rng, array_rng = jr.split(jr.PRNGKey(0))
    if hasattr(flax_model, "rng_keys"):
        rng_keys = ["params", *flax_model.rng_keys]
        rngs = dict(zip(rng_keys, jr.split(param_rng, len(rng_keys))))
    else:
        rngs = {"params": param_rng}
    variables = flax_model.init(rngs, jnp.zeros(input_size))

    rngs = None
    if hasattr(flax_model, "rng_keys"):
        rngs = dict(zip(flax_model.rng_keys, jr.split(array_rng, len(flax_model.rng_keys))))
    flax_model.apply(
        variables,
        jnp.zeros((batch_size, *input_size)),
        rngs=rngs,
        is_training=is_training,
        mutable=True,
    )


def assert_close_outputs(
    rng: jr.PRNGKey,
    flax_model: linen.Module,
    torch_model: torch.nn.Module,
    input_size: tuple[int, int, int] = (224, 224, 3),
    batch_size: int = 4,
    tensor_transpose_axes: tuple[int, ...] | None = (2, 0, 1),
) -> None:
    """Initialize variables of `flax_model`,
    and compare the outputs of `flax_model` with that of `torch_model`.

    """
    param_rng, array_rng = jr.split(rng)
    if hasattr(flax_model, "rng_keys"):
        rng_keys = ["params", *flax_model.rng_keys]
        rngs = dict(zip(rng_keys, jr.split(param_rng, len(rng_keys))))
    else:
        rngs = {"params": param_rng}
    variables = flax_model.init(rngs, jnp.zeros(input_size))
    variables = load_weights_from_torch_model(variables, torch_model)

    x = jr.normal(array_rng, shape=(batch_size, *input_size))
    y_flax = flax_model.apply(variables, x.copy())

    with torch.no_grad():
        device = "cuda" if torch.cuda.is_available() and False else "cpu"

        torch_model = torch_model.eval().to(device)

        if tensor_transpose_axes is not None:
            tensor_transpose_axes = [0] + list(map(lambda x: x + 1, tensor_transpose_axes))
            x = jnp.transpose(x, tensor_transpose_axes)  # NHWC -> NCHW

        x = torch.from_numpy(numpy.array(x)).to(device=device)
        y_torch = torch_model(x)
        y_torch = jnp.array(y_torch.detach().cpu().numpy())

        if tensor_transpose_axes is not None and y_torch.ndim == 1 + len(input_size):
            axes = [tensor_transpose_axes.index(x) for x in range(len(tensor_transpose_axes))]
            y_torch = jnp.transpose(y_torch, axes)  # NCHW -> NHWC

    # d = jnp.abs(y_flax - y_torch)
    # msg = f"Average distance: {jnp.mean(d)}, Maximum distance: {jnp.max(d)}"
    numpy.testing.assert_allclose(y_flax, y_torch, atol=5e-5)
