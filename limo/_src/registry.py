from __future__ import annotations
import typing as tp
from collections import defaultdict


_model_registry = defaultdict(dict)


def register_model(
    model_name: str,
    model_fun: tp.Callable,
    checkpoint_name: str | None = None,
    default_cfg=None,
    default_checkpoint: bool = False,
):
    """
    Args:
        model_name: Model name.
        model_fun: Model builder.
        checkpoint_name: Checkpoint name.
        default_cfg: Default config of checkpoint.
        default_checkpoint: If True, this checkpoint is used
            as a default checkpoint.

    """
    default_cfg = default_cfg or dict()
    if checkpoint_name is not None:
        _model_registry[model_name][checkpoint_name] = (model_fun, default_cfg)

    if default_checkpoint:
        assert "_DEFAULT" not in _model_registry[model_name]
        _model_registry[model_name]["_DEFAULT"] = (model_fun, default_cfg)


def create_model(model_name: str, pretrained=False, **kwargs):
    if isinstance(pretrained, bool):
        pretrained = "_DEFAULT"
    model_fun, _ = _model_registry[model_name][pretrained]
    return model_fun(**kwargs)


# def list_models(pattern: str, pretrained: bool = False):
#     for model_name in _model_registry:

#         for checkpoint_name in model_dict:
#             if checkpoint_name != "_DEFAULT":
#                 return model_name


def get_model_cfg(model_name: str, pretrained=False):
    if isinstance(pretrained, bool):
        pretrained = "_DEFAULT"
    _, default_cfg = _model_registry[model_name][pretrained]
    return default_cfg
