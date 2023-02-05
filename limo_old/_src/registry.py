from __future__ import annotations
import typing as tp
from collections import defaultdict


_model_registry = defaultdict(dict)


def register_model(
    model_name: str,
    model_fun: tp.Callable,
    default_cfg=None,
    checkpoint_name: str | None = None,
    default: bool = False,
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

    if default:
        assert False not in _model_registry[model_name]
        _model_registry[model_name][False] = (model_fun, default_cfg)

        if checkpoint_name is not None:
            assert True not in _model_registry[model_name]
            _model_registry[model_name][True] = (model_fun, default_cfg)


def create_model(model_name: str, pretrained=False, with_cfg: bool = False, **kwargs):
    """

    NOTE:
        There are some special arguments.

        torch_like: Whether PyTorch like convolution and pooling are used.
    """
    if isinstance(pretrained, bool):
        pretrained = "_DEFAULT"
    model_builder, default_cfg = _model_registry[model_name][pretrained]
    if "num_classes" not in kwargs and "num_classes" in default_cfg:
        kwargs["num_classes"] = default_cfg["num_classes"]
    if "torch_like" not in kwargs and "torch_like" in default_cfg:
        kwargs["torch_like"] = default_cfg["torch_like"]

    model = model_builder(**kwargs)
    if with_cfg:
        return model, default_cfg
    else:
        return model


def list_models(name: str | None = None, pretrained: bool = False) -> tp.Sequence[str]:
    print(_model_registry)

    model_list = []
    for model_name, model_dict in filter(
        lambda item: name is None or name in item[0],
        _model_registry.items(),
    ):
        # If there are no pretrained weights, model_dict only has False
        if not pretrained or len(model_dict) > 1:
            model_list.append(model_name)
    return sorted(model_list)


def list_pretrained(model_name: str) -> tp.Sequence[str]:
    return list(filter(lambda k: isinstance(k, str), _model_registry[model_name]))


def get_default_cfg(model_name: str, pretrained=False):
    if isinstance(pretrained, bool):
        pretrained = "_DEFAULT"
    _, default_cfg = _model_registry[model_name][pretrained]
    return default_cfg
