from __future__ import annotations
import typing as tp


_registry = {}


def register_model(fun: tp.Callable | None = None, /, **kwargs):
    """
    Args:
        name: Model name to register.
        model_builder: A callable that creates model.

    """
    if callable(fun):
        _registry[fun.__name__] = fun

    else:

        def deco(fun):
            register_model(fun, **kwargs)
            return fun

        return deco
