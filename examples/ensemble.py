"""Example of loading variables into modules of a model."""
from __future__ import annotations
import typing as tp

import jax.numpy as jnp
import jax.random as jr
from flax import linen
import limo
import chex


class Ensemble(linen.Module):
    model_names: tp.Sequence[str]

    num_classes: int = 1000
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x, is_training=False):
        kwargs = {
            "num_classes": self.num_classes,
            "dtype": self.dtype,
            "norm_dtype": self.norm_dtype,
            "axis_name": self.axis_name,
        }

        logits = {}
        for i, model_name in enumerate(self.model_names):
            model = limo.create_model(model_name, **kwargs, name=str(i))
            logits[model_name] = model(x, is_training)

        logit_main = jnp.sum(jnp.stack(list(logits.values()), axis=0), axis=0)
        return logit_main, logits


class DeepEnsemble(linen.Module):
    """Equivarent to `Ensemble`, but `Ensemble` is a submodule."""

    model_names: tp.Sequence[str]

    num_classes: int = 1000
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType | None = None
    axis_name: str | None = None

    @linen.compact
    def __call__(self, x, is_training=False):
        kwargs = {
            "model_names": self.model_names,
            "num_classes": self.num_classes,
            "dtype": self.dtype,
            "norm_dtype": self.norm_dtype,
            "axis_name": self.axis_name,
        }

        ensemble = Ensemble(**kwargs, name="ensemble")
        return ensemble(x, is_training)


def main():
    model_names = ["convnext_tiny", "resnet18", "efficientnet_b0"]
    model = Ensemble(model_names)

    rng = jr.PRNGKey(1234)
    rng, init_rng = jr.split(rng)

    x = jnp.zeros((224, 224, 3))
    variables = model.init(init_rng, x)
    for i, model_name in enumerate(model_names):
        variables = limo.load_pretrained(
            variables, model_name, pretrained=True, module_name=str(i), save_dir="weights"
        )

    logit_main, logits = model.apply(variables, x, is_training=False)
    print("Output shape:", logit_main.shape)
    for model_name in model_names:
        model = limo.create_model(model_name)
        variables = model.init(init_rng, x)
        variables = limo.load_pretrained(variables, model_name, pretrained=True, save_dir="weights")
        logit = model.apply(variables, x, is_training=False)
        print(f"MAE of {model_name} outputs:", jnp.mean(jnp.abs(logit - logits[model_name])))

    # Example of loading variables into a module that is not directly under the model.
    model = DeepEnsemble(model_names)
    variables = model.init(init_rng, x)
    for i, model_name in enumerate(model_names):
        variables = limo.load_pretrained(
            variables, model_name, pretrained=True, module_name=f"ensemble.{i}", save_dir="weights"
        )

    logit_main_v2, logits = model.apply(variables, x, is_training=False)
    print("MAE of main logits:", jnp.mean(jnp.abs(logit_main - logit_main_v2)))


if __name__ == "__main__":
    main()
