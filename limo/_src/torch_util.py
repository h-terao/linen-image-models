"""A utilities to translate PyTorch parameters into linen-image-models

Note: This module should not be installed
"""
from __future__ import annotations
import os
import typing as tp
import warnings
from functools import partial

import numpy
import torch
import torch.nn as nn
import jax.numpy as jnp
from flax import core, traverse_util
import chex

from timm.models.convnext import ConvNeXtBlock

import limo


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
SEP = "/"  # sep char that is not used in layer name of flax and pytorch.


def _translate(module: nn.Module, tensors: tp.Mapping[str, torch.Tensor]) -> chex.ArrayTree:
    params, batch_stats = {}, {}
    if isinstance(module, nn.Linear):
        kernel = tensors["weight"].detach().cpu().numpy()
        params["kernel"] = jnp.transpose(kernel, (1, 0))

        if "bias" in tensors:
            params["bias"] = tensors["bias"].detach().cpu().numpy()
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        kernel = tensors["weight"].detach().cpu().numpy()
        axes = tuple(range(2, kernel.ndim)) + (1, 0)
        params["kernel"] = jnp.transpose(kernel, axes)

        if "bias" in tensors:
            params["bias"] = tensors["bias"].detach().cpu().numpy()
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if "weight" in tensors:
            params["scale"] = tensors["weight"].detach().cpu().numpy()
        if "bias" in tensors:
            params["bias"] = tensors["bias"].detach().cpu().numpy()

        if "running_mean" in tensors:
            batch_stats["mean"] = tensors["running_mean"].detach().cpu().numpy()
        if "running_var" in tensors:
            batch_stats["var"] = tensors["running_var"].detach().cpu().numpy()
    elif isinstance(module, nn.LayerNorm):
        if "weight" in tensors:
            params["scale"] = tensors["weight"].detach().cpu().numpy()
        if "bias" in tensors:
            params["bias"] = tensors["bias"].detach().cpu().numpy()

    elif isinstance(module, ConvNeXtBlock):
        print("*** LOAD GAMMA ***")
        params["gamma"] = tensors["gamma"].detach().cpu().numpy()
    else:
        raise ValueError(f"Cannot convert module type {type(module)}.")

    variables = {}
    if params:
        variables["params"] = params
    if batch_stats:
        variables["batch_stats"] = batch_stats

    return variables


def torch2flax(variables: tp.Mapping, torch_model: nn.Module) -> tp.Mapping:
    """Overwrite Flax parameters using PyTorch parameters.

    Example:
        ::

            flax_model = flaxmodels.create_model(...)
            torch_model = timm.create_model(...)

            variables = flax_model.init(...)
            variables = torch_to_flax(torch_model, variables)
    """
    cast_to = type(variables)
    if isinstance(variables, core.FrozenDict):
        variables = variables.unfreeze()

    flax_params = {}
    variable_flat = traverse_util.flatten_dict(variables, sep=SEP)
    for key, array in variable_flat.items():
        col, key = key.split(SEP, maxsplit=1)  # col: params, batch_stats, ...
        layer_name, param_name = key.rsplit(SEP, maxsplit=1)
        flax_params.setdefault(layer_name, {})
        flax_params[layer_name].setdefault(col, {})
        flax_params[layer_name][col][param_name] = array

    torch_params = {}
    for key, value in torch_model.state_dict().items():
        layer_name, param_name = key.rsplit(".", maxsplit=1)
        torch_params.setdefault(layer_name, {})
        torch_params[layer_name][param_name] = value

    new_variables = dict()
    unused_torch_param_keys = list(torch_params)
    for flax_key, flax_param in flax_params.items():
        torch_key = flax_key.replace(SEP, ".")
        if torch_key in torch_params:
            torch_module = torch_model
            for k in torch_key.split("."):
                torch_module = getattr(torch_module, k)
            new_variable = _translate(torch_module, torch_params[torch_key])
            unused_torch_param_keys.remove(torch_key)

            new_variable_flat = traverse_util.flatten_dict(new_variable)
            flax_param_flat = traverse_util.flatten_dict(flax_param)
            for k in new_variable_flat:
                if k not in flax_param_flat:
                    msg = f"{k} is not found."
                    warnings.warn(msg)

                if (
                    new_variable_flat[k].ndim == 2
                    and flax_param_flat[k].ndim == 4
                    and tuple(flax_param_flat[k].shape[:2]) == (1, 1)
                ):
                    # Convert Linear -> Conv1x1.
                    inC, outC = new_variable_flat[k].shape
                    new_variable_flat[k] = new_variable_flat[k].reshape(1, 1, inC, outC)

                if new_variable_flat[k].shape != flax_param_flat[k].shape:
                    msg = (
                        f"Shape mismatch is found in {k} of {torch_key}. "
                        f"Expected: {new_variable_flat[k].shape}, "
                        f"Actual: {flax_param_flat[k].shape}."
                    )

                    warnings.warn(msg)
                    new_variable_flat[k] = flax_param_flat[k]
            new_variable = traverse_util.unflatten_dict(new_variable_flat)
        else:
            new_variable = flax_param
            warnings.warn(
                f"{torch_key} is not found in the PyTorch model. Use the initialized parameters."
            )

        for col, arrays in new_variable.items():
            for suffix, array in arrays.items():
                new_variables[SEP.join((col, flax_key, suffix))] = array

    if unused_torch_param_keys:
        warnings.warn(f"{unused_torch_param_keys} are not used.")

    new_variables = traverse_util.unflatten_dict(new_variables, SEP)
    new_variables = cast_to(new_variables)
    return new_variables


@torch.no_grad()
@partial(limo.configure, train=False)
def assert_equal_preds(flax_model, torch_model, variables, array):
    # Forward Flax model.
    logits_flax = flax_model.apply(variables, array)

    # Forward PyTorch model.
    tensor = torch.from_numpy(numpy.array(array))
    tensor = tensor.permute(0, 3, 1, 2)  # NHWC -> NCHW
    torch_model.eval()
    logits_torch = torch_model(tensor)
    logits_torch = jnp.array(logits_torch.detach().cpu().numpy())
    if logits_torch.ndim == 4:
        # This is image.
        logits_torch = jnp.transpose(logits_torch, (0, 2, 3, 1))

    distance = float(jnp.sum(jnp.abs(logits_torch - logits_flax), axis=-1).mean())
    assert jnp.allclose(logits_flax, logits_torch, rtol=1.0, atol=1e-5), f"Difference: {distance}"


# def download_model(save_dir, model_name, pretrained, force: bool = False):
#     """"""
#     save_dir_path = Path(save_dir)
#     save_dir_path.mkdir(parents=True, exist_ok=True)
#     model_path = save_dir_path / f"{model_name}.{pretrained}.ckpt"
#     if force or not model_path.exists():
#         print(f"Downloading {model_name}.{pretrained}")
#         torch_model = timm.create_model(model_name, pretrained)
#         flax_model, model_cfg = limo.create_model(model_name, pretrained, True)

#         # TODO: check cfg is correct.
#         input_size = (4, *model_cfg.get("input_size", (224, 224, 3)))

#         input = jnp.zeros(input_size, dtype=jnp.float32)
#         variables = flax_model.init(jr.PRNGKey(0), input)
#         variables = load_torch_state(variables, torch_model)

#         assert_equal(flax_model, torch_model, variables, input)
#         model_path.write_bytes(pickle.dumps(variables))
