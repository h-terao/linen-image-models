"""Configuration utility."""
from __future__ import annotations
import typing as tp
from dataclasses import dataclass
import functools
import contextlib
import threading

import jax.numpy as jnp
import chex


@dataclass
class LocalConfig(threading.local):
    train: bool = False
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    torch_like: bool = True
    axis_name: str | None = None

    def get(self, name: str) -> tp.Any:
        v = getattr(self, name)
        if name == "norm_dtype" and v is None:
            v = self.dtype
        return v


_local_config = LocalConfig()


@contextlib.contextmanager
def using_config(**configs) -> None:
    """Context manager to temporarily change the limo configuration.

    Args:
        train (bool): Whether models should work in training mode. default: False.
        dtype: Data type of computation. default: `jax.numpy.float32`
        norm_dtype: Data type of computation in norm layers. If None, use `dtype`.
            default: None.
        torch_like (bool): Whether to use PyTorch-like padding in convolution and pooling.
        axis_name (str): Axis name to sync batch stats. default: None.

    Note:
        All arguments are optional keyword only arguments.
    """
    prev_config = {}
    for key, value in configs.items():
        if hasattr(_local_config, key):
            prev_config[key] = _local_config.get(key)
            setattr(_local_config, key, value)
        else:
            raise RuntimeError(f"Unknown parameter {key} is configured.")

    try:
        yield

    finally:
        # Restore the previous states.
        for key, prev_value in prev_config.items():
            setattr(_local_config, key, prev_value)


def configure(fun: tp.Callable, **configs) -> tp.Callable:
    """Creates a function that configure `fun` using the specified arguments.
    In the configured function, configs are available via `get_config`.

    Args:
        fun: A function to configure.
        **configs: Configurations.

    Returns:
        A wrapped version of `fun`.

    """

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        with using_config(**configs):
            return fun(*args, **kwargs)

    return wrapped


get_config = _local_config.get
