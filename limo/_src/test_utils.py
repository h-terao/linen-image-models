from __future__ import annotations
import os

import numpy
import jax.random as jr
import jax.numpy as jnp
import chex

import torch
import torch.nn as nn
from .helpers import load_weights_from_torch_model

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def assert_close_outputs(
    rng: chex.PRNGKey,
    flax_model,
    torch_model: nn.Module,
    shape: chex.Shape = (224, 224, 3),
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    shape = (8,) + tuple(shape)  # canonicalize.

    torch_model = torch_model.eval()
    torch_model = torch_model.to("cuda:0")

    arr_rng, init_rng = jr.split(rng)
    array = jr.normal(arr_rng, shape, dtype=jnp.float32)

    variables = flax_model.init(init_rng, array)
    variables = load_weights_from_torch_model(variables, torch_model)
    out_flax = flax_model.apply(variables, array.copy())

    tensor = torch.from_numpy(numpy.array(array.transpose(0, 3, 1, 2)))
    tensor = tensor.to(device="cuda:0")

    with torch.no_grad():
        out_torch = torch_model(tensor)
        out_torch = out_torch.detach().cpu().numpy()
        out_torch = out_torch.transpose(0, *list(range(2, out_torch.ndim)), 1)

    if not numpy.allclose(out_flax, out_torch, rtol, atol):
        print("Different outputs.")
        print(numpy.abs(out_flax - out_torch).mean())
