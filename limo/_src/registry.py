from __future__ import annotations
import typing as tp
from collections import defaultdict


_registry = defaultdict(dict)


class ModelSpec(tp.NamedTuple):
    builder: tp.Callable
    pretrained: str
    default: bool


def register_model(
    model_name: str, model_builder: tp.Callable, pretrained: str, default: bool = False, **kwargs
):
    """
    Args:
        name: Model name to register.
        model_builder: A callable that creates model.

    """
    _registry[model_name] = ModelSpec(model_builder, pretrained, default)
