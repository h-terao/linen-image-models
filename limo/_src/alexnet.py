from __future__ import annotations
import typing as tp

import jax.numpy as jnp
from flax import linen
from einops import rearrange
import chex

from limo import register_model


ModuleDef = tp.Any


class AlexNet(linen.Module):
    num_classes: int = 1000
    drop_rate: float = 0
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        x = linen.Conv(64, (11, 11), 4, 2, dtype=self.dtype, name="features.0")(x)
        x = linen.relu(x)
        x = linen.max_pool(x, (3, 3), (2, 2))
        x = linen.Conv(192, (5, 5), 1, 2, dtype=self.dtype, name="features.3")(x)
        x = linen.relu(x)
        x = linen.max_pool(x, (3, 3), (2, 2))
        x = linen.Conv(384, (3, 3), 1, 1, dtype=self.dtype, name="features.6")(x)
        x = linen.relu(x)
        x = linen.Conv(256, (3, 3), 1, 1, dtype=self.dtype, name="features.8")(x)
        x = linen.relu(x)
        x = linen.Conv(256, (3, 3), 1, 1, dtype=self.dtype, name="features.10")(x)
        x = linen.relu(x)
        x = linen.max_pool(x, (3, 3), (2, 2))
        x = rearrange(x, "... H W C -> ... (C H W)")
        if self.num_classes > 0:
            x = linen.Dropout(self.drop_rate)(x, not is_training)
            x = linen.Dense(4096, dtype=self.dtype, name="classifier.1")(x)
            x = linen.relu(x)
            x = linen.Dropout(self.drop_rate)(x, not is_training)
            x = linen.Dense(4096, dtype=self.dtype, name="classifier.4")(x)
            x = linen.relu(x)
            x = linen.Dense(self.num_classes, dtype=self.dtype, name="classifier.6")(x)
        return x

    @property
    def rng_keys(self) -> tp.Sequence[str]:
        return ["dropout"]


@register_model(
    "IMAGENET1K_V1",
    url="https://onedrive.live.com/download?cid=A750EE44BB6AE6CF&resid=A750EE44BB6AE6CF%2118963&authkey=AHkmZrZEb5ez2Zk",  # noqa:E501
    default=True,
)
def alexnet(**kwargs) -> AlexNet:
    return AlexNet(**kwargs)
