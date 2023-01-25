from __future__ import annotations
import os

import numpy
import jax.random as jr
import jax.numpy as jnp
import chex

import torch
import torch.nn as nn
from limo._src.model_utils import load_weights_from_torch_model

os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def assert_close_outputs(
    rng: chex.PRNGKey,
    flax_model,
    torch_model: nn.Module,
    shape: chex.Shape = (224, 224, 3),
):
    shape = (1,) + tuple(shape)  # canonicalize.
    torch_model = torch_model.eval()

    arr_rng, init_rng = jr.split(rng)
    array = jr.normal(arr_rng, shape, dtype=jnp.float32)

    variables = flax_model.init(init_rng, array)
    variables = load_weights_from_torch_model(variables, torch_model)
    out_flax = flax_model.apply(variables, array.copy())

    tensor = torch.from_numpy(numpy.array(array.transpose(0, 3, 1, 2)))
    with torch.no_grad():
        out_torch = torch_model(tensor)
        out_torch = out_torch.detach().cpu().numpy()
        out_torch = out_torch.transpose(0, *list(range(2, out_torch.ndim)), 1)

    assert numpy.allclose(out_flax, out_torch, rtol=1.0, atol=1e-6)
