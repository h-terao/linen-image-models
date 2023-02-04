from __future__ import annotations
import typing as tp


_registry = {}


def register_model(fun: tp.Callable):
    """
    Args:
        name: Model name to register.
        model_builder: A callable that creates model.

    """
    _registry[fun.__name__] = fun
