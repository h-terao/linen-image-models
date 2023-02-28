from __future__ import annotations
import typing as tp
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision.models.convnext import CNBlock as ConvNeXtBlock
from torchvision.models.vision_transformer import (
    VisionTransformer,
    Encoder as VisionTransformerEncoder,
)
from jax import tree_util
import chex
from einops import rearrange

from limo._src.helpers import maybe_overwrite_variables


SEP = "/"
_converter_registry = {}


def to_arrays(tensors):
    def to_array(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        else:
            return tensor

    return tree_util.tree_map(to_array, tensors)


def register_converter(*torch_modules):
    def deco(f):
        for m in torch_modules:
            _converter_registry[m] = f
        return f

    return deco


def children(m: nn.Module):
    return m._modules


def make_variables_from_torch_model(torch_model: nn.Module) -> tp.Mapping:
    """Parses pytorch module and returns flax-like variables."""
    variables = defaultdict(dict)

    for module_type, convert in _converter_registry.items():
        if isinstance(torch_model, module_type):
            state = to_arrays(torch_model.state_dict())
            variables, named_modules = convert(torch_model, state)
            break
    else:
        # If `torch_model` is not converted, convert all modules.
        named_modules = children(torch_model)

    all_params = {k: v for k, v in torch_model.named_parameters() if "." not in k}
    variables["params"] = dict(to_arrays(all_params), **variables["params"])

    for name, module in named_modules.items():
        # child_variables: {"params": ..., "batch_stats": ...}
        child_variables = make_variables_from_torch_model(module)
        for col, arrays in child_variables.items():
            assert name not in variables[col]
            variables[col][name] = arrays

    return variables


def load_weights_from_torch_model(variables: chex.ArrayTree, torch_model: nn.Module) -> chex.ArrayTree:
    # convert PyTorch model and get flatten state.
    to_load = make_variables_from_torch_model(torch_model)
    return maybe_overwrite_variables(variables, to_load)


#
#  Converters of `torch.nn`
#
@register_converter(nn.Linear)
def convert_dense(m, state):
    params = {"kernel": rearrange(state["weight"], "outC inC -> inC outC")}
    if "bias" in state:
        params["bias"] = state["bias"]
    return {"params": params}, {}


@register_converter(nn.Conv1d, nn.Conv2d, nn.Conv3d)
def convert_conv(m, state):
    params = {"kernel": rearrange(state["weight"], "outC inC ... -> ... inC outC")}
    if "bias" in state:
        params["bias"] = state["bias"]
    return {"params": params}, {}


@register_converter(nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
def convert_batch_norm(m, state):
    variables = defaultdict(dict)
    if "weight" in state:
        variables["params"]["scale"] = state["weight"]
    if "bias" in state:
        variables["params"]["bias"] = state["bias"]
    if "running_mean" in state:
        variables["batch_stats"]["mean"] = state["running_mean"]
    if "running_var" in state:
        variables["batch_stats"]["var"] = state["running_var"]
    return dict(variables), {}


@register_converter(nn.LayerNorm)
def convert_layer_norm(m, state):
    params = {}
    if "weight" in state:
        params["scale"] = state["weight"]
    if "bias" in state:
        params["bias"] = state["bias"]
    return {"params": params}, {}


@register_converter(nn.MultiheadAttention)
def convert_multihead_attention(m, state):
    params = defaultdict(lambda: defaultdict(dict))
    if "in_proj_weight" in state:
        pattern = "(qkv head outC) inC -> qkv inC head outC"
        kernels = rearrange(state["in_proj_weight"], pattern, qkv=3, head=m.num_heads)
        params["query"]["kernel"], params["key"]["kernel"], params["value"]["kernel"] = kernels

    # qkv_same_embed_dim=False case.
    pattern = "(head outC) inC -> inC head outC"
    if "q_proj_weight" in state:
        # TODO: Test
        warnings.warn("Converting q_proj_weight of MultiheadAttention. This is not tested yet.")
        params["query"]["kernel"] = rearrange(params["q_proj_weight"], pattern, head=m.num_heads)
    if "k_proj_weight" in state:
        warnings.warn("Converting k_proj_weight of MultiheadAttention. This is not tested yet.")
        params["query"]["kernel"] = rearrange(params["k_proj_weight"], pattern, head=m.num_heads)
    if "v_proj_weight" in state:
        warnings.warn("Converting v_proj_weight of MultiheadAttention. This is not tested yet.")
        params["query"]["kernel"] = rearrange(params["v_proj_weight"], pattern, head=m.num_heads)

    if "in_proj_bias" in state:
        pattern = "(qkv head outC) -> qkv head outC"
        biases = rearrange(state["in_proj_bias"], pattern, qkv=3, head=m.num_heads)
        params["query"]["bias"], params["key"]["bias"], params["value"]["bias"] = biases

    # convert out_proj.
    state = to_arrays(m.out_proj.state_dict())
    params["out"]["kernel"] = rearrange(state["weight"], "outC (head inC) -> head inC outC", head=m.num_heads)
    if "bias" in state:
        params["out"]["bias"] = state["bias"]

    return {"params": params}, {}


#
#  Converters of specific models or modules.
#
@register_converter(ConvNeXtBlock)
def convert_convnext_block(m, state):
    variables = {
        "params": {"layer_scale": rearrange(state["layer_scale"], "dim 1 1 -> dim")},
    }
    return variables, children(m)


@register_converter(VisionTransformer)
def convert_vision_transformer(m, state):
    variables = {"params": {"class_token": rearrange(state["class_token"], "1 1 dim -> dim")}}
    return variables, children(m)


@register_converter(VisionTransformerEncoder)
def convert_vision_transformer_encoder(m, state):
    variables = {"params": {"pos_embedding": rearrange(state["pos_embedding"], "1 time dim -> time dim")}}
    return variables, children(m)
