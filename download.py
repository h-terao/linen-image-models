"""Download pre-trained weights from timm."""
from __future__ import annotations
import typing as tp
import argparse
import warnings

import torch
import torch.nn as nn
import timm
import jax.numpy as jnp
from flax import core, traverse_util
import chex

from limo import list_models


SEP = "/"  # sep char that is not used in layer name of flax and pytorch.


def translate(module: nn.Module, tensors: tp.Mapping[str, torch.Tensor]) -> chex.ArrayTree:
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
    else:
        raise ValueError(f"Cannot convert module type {type(module)}.")

    variables = {}
    if params:
        variables["params"] = params
    if batch_stats:
        variables["batch_stats"] = batch_stats

    return variables


def load_weights_from_torch_model(variables: tp.Mapping, torch_model: nn.Module) -> tp.Mapping:
    """Overwrite Flax parameters using PyTorch parameters.

    TODO:
        - Checkpoint saving option.

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
            new_variable = translate(torch_module, torch_params[torch_key])
            unused_torch_param_keys.remove(torch_key)

            new_variable_flat = traverse_util.flatten_dict(new_variable)
            flax_param_flat = traverse_util.flatten_dict(flax_param)
            for k in new_variable_flat:
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


def worker(pattern):
    """Download models, and save"""
    models = list_models(pattern)
    for model in models:
        for pretrained in x:
            checkpoint_path = f"weights/{model}.{pretrained}.ckpt"

            torch_model = timm.create_model(model, pretrained=pretrained)
            flax_model = limo.create_model(model)

            #
            variables = flax_model.init()
            variables = load_weights_from_torch_model(variables, torch_model)

            # In actual,
            # variables = limo.load_model(model, pretrained)

            if torch_features == flax_features:
                # save.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", default=None, type=str, required=False, help="")
    parser.add_argument("-o", "--out", default="weights", help="Output directory.")
    parser.add_argument("-f", "--force", action="store_true", help="Reinstall download.")
    args = parser.parse_args()
