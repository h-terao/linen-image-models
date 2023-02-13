""""""
from __future__ import annotations
import sys
from pathlib import Path
import warnings

import jax.numpy as jnp
import jax.random as jr
from torchvision import models
from absl import app, flags

sys.path.append(".")

import limo  # noqa: E402
from limo._src.helpers import _model_registry  # noqa: E402
from limo._src.torch_utils import load_weights_from_torch_model  # noqa: E402


FLAGS = flags.FLAGS
flags.DEFINE_string("save_dir", "weights", help="Directory to save weights.")
flags.DEFINE_bool("force", False, "Whether to download weights that is already downloaded.")


def main(_):
    for model_name, entries in _model_registry.items():
        for weight_name, entry in entries.items():
            if entry.url is not None:
                continue  # URL of this weight is already registered.

            if isinstance(weight_name, bool):
                continue  # This is alias.

            weight_enum = models.get_model_weights(model_name)
            if hasattr(weight_enum, weight_name):
                save_path = Path(
                    FLAGS.save_dir, entry.family_name, f"{model_name}.{weight_name}.pkl"
                )
                if save_path.exists() and not FLAGS.force:
                    continue  # This weight is already downloaded.

                print(f"Download: {model_name}.{weight_name} >> {save_path}")
                weight = getattr(weight_enum, weight_name)
                torch_model = getattr(models, model_name)(weights=weight)

                # Initialize variables.
                H, W = map(lambda x: max(x, 224), weight.meta.get("min_size", (1, 1)))
                flax_model = limo.create_model(model_name, pretrained=weight_name)
                variables = flax_model.init(jr.PRNGKey(0), jnp.zeros((H, W, 3)))

                # Load variables from pretrained PyTorch model.
                variables = load_weights_from_torch_model(variables, torch_model)

                if not save_path.parent.exists():
                    save_path.parent.mkdir(parents=True)
                limo.save(save_path, variables, exist_ok=True)

            else:
                warnings.warn(f"{model_name}.{weight_name} is not provided from torchvision.")


if __name__ == "__main__":
    app.run(main)
