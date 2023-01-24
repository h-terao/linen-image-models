"""Helper functions to use limo models."""
from __future__ import annotations
import typing as tp
import warnings

import torch
import torch.nn as nn
from flax import linen

import jax.numpy as jnp
from flax import traverse_util, core
import chex


SEP = "/"  # sep char that is not used in layer name of flax and pytorch.
_translator_registry = {}
_model_registry = {}


def register_model(model_name: str, create_torch_model: tp.Callable | None = None):
    def deco(fun):
        if model_name in _model_registry:
            raise RuntimeError(f"{model_name} is already registered.")
        _model_registry[model_name] = (fun, create_torch_model)
        return fun

    return deco


def register_model_from_tv(model_name: str, weight_name: str):
    """Register a model converted from torchvision."""

    def load_weights(variables):
        from torchvision import models

        weights = models.get_weight(weight_name)
        torch_model = getattr(models, model_name)(weights=weights)
        return load_weights_from_torch_model(variables, torch_model)

    def deco(fun):
        if model_name in _model_registry:
            raise RuntimeError(f"{model_name} is already registered.")
        _model_registry[model_name] = (fun, load_weights)
        return fun

    return deco


def register_model_from_hub(model_name, repo_or_dir: str, source: str = "github", **kwargs):
    """Register a model converted from a PyTorch model puplished by `torch.hub`."""

    def load_weights(variables):
        torch_model = torch.hub.load(repo_or_dir, model_name, source=source, **kwargs)
        return load_weights_from_torch_model(variables, torch_model)

    def deco(fun):
        if model_name in _model_registry:
            raise RuntimeError(f"{model_name} is already registered.")
        _model_registry[model_name] = (fun, load_weights)
        return fun

    return deco


def register_translator(*torch_modules) -> tp.Callable:
    """Register translate functions.

    Args:
        torch_modules: PyTorch modules to convert layer parameters.

    Returns:
    """

    def deco(f):
        for torch_module in torch_modules:
            _translator_registry[torch_module] = f
        return f

    return deco


def create_model(model_name: str, **kwargs) -> linen.Module:
    create_flax_model, _ = _model_registry[model_name]
    flax_model = create_flax_model(**kwargs)
    return flax_model


def list_model(pretrained: bool = False) -> tp.Sequence[str]:
    models = _model_registry
    if pretrained:
        # v[1] is `create_torch_model` fun.
        models = [k for k, v in models.items() if v[1] is not None]
    return list(models)


def load_weights(model_name: str, variables: chex.ArrayTree) -> chex.ArrayTree:
    """Load pre-trained variables from torchvision."""
    _, load_fun = _model_registry[model_name]
    if load_fun is None:
        raise RuntimeError(f"Cannot load pretrained weights of {model_name}.")
    else:
        return load_fun(variables)


#
#  Register translate functions.
#
@register_translator(nn.Linear)
def linear_translator(tensors: tp.Mapping[str, torch.Tensor]):
    params = {}

    kernel = tensors["weight"].detach().cpu().numpy()
    params["kernel"] = jnp.transpose(kernel, (1, 0))

    if "bias" in tensors:
        params["bias"] = tensors["bias"].detach().cpu().numpy()

    return {"params": params}


@register_translator(nn.Conv1d, nn.Conv2d, nn.Conv3d)
def conv_translator(tensors: tp.Mapping[str, torch.Tensor]) -> tp.Mapping[str, chex.Array]:
    params = {}

    kernel = tensors["weight"].detach().cpu().numpy()
    axes = tuple(range(2, kernel.ndim)) + (1, 0)
    params["kernel"] = jnp.transpose(kernel, axes)

    if "bias" in tensors:
        params["bias"] = tensors["bias"].detach().cpu().numpy()

    return {"params": params}


@register_translator(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
def batch_norm_translator(tensors: tp.Mapping[str, torch.Tensor]) -> tp.Mapping[str, chex.Array]:
    params, batch_stats = {}, {}

    if "weight" in tensors:
        params["scale"] = tensors["weight"].detach().cpu().numpy()
    if "bias" in tensors:
        params["bias"] = tensors["bias"].detach().cpu().numpy()

    if "running_mean" in tensors:
        batch_stats["mean"] = tensors["running_mean"].detach().cpu().numpy()
    if "running_var" in tensors:
        batch_stats["var"] = tensors["running_var"].detach().cpu().numpy()

    return {"params": params, "batch_stats": batch_stats}


#
#  Core functions.
#
def load_weights_from_torch_model(variables: tp.Mapping, torch_model: nn.Module) -> tp.Mapping:
    """Overwrite Flax parameters using PyTorch parameters.

    TODO:
        - Support stem operation.

    Example:
        ::

            flax_model = flaxmodels.create_model(...)
            torch_model = timm.create_model(...)

            variables = flax_model.init(...)
            variables = torch_to_flax(torch_model, variables)
    """

    def translate(torch_key, tensors):
        m = torch_model
        for key in torch_key.split("."):
            m = getattr(m, key)

        module_type = type(m)
        for module_type, translator in _translator_registry.items():
            if isinstance(m, module_type):
                return translator(tensors)

        raise ValueError(
            (
                "Failed to translate PyTorch parameters into that of Flax. "
                f"{module_type} is not registered."
            )
        )

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
            new_variable = translate(torch_key, torch_params[torch_key])
            unused_torch_param_keys.remove(torch_key)

            new_variable_flat = traverse_util.flatten_dict(new_variable)
            flax_param_flat = traverse_util.flatten_dict(flax_param)
            for k in new_variable_flat:
                if new_variable_flat[k].shape != flax_param_flat[k].shape:
                    warnings.warn(f"Shape mismatch is found: {k} in {torch_key}")
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
