# Linen Image Models

limo is an easily customizable image model implementation with pre-trained variables. The pre-trained variables are loaded correctly even if you add or remove modules. Currently, this project would like to reimplement `torchvision.models`.

*Due to the limitation of my onedrive storage, pretrained variables are not available now. Sorry!*


## Installation

1. Install JAX for your environment. See details in [the installation guide of JAX](https://github.com/google/jax#installation).
2. Install limo via pip:
```bash
$ pip install git+https://github.com/h-terao/linen-image-models
```

## Usage

### Basic usage

To use builtin models and their pretrained variables, take the following steps.
1. Create model via `limo.create_model`.
2. Initialize varaiables in the standard flax manner.
3. Overwrite initalized variables with pretrained variables using `limo.load_pretrained`.

```python
import jax
import limo

x =  jax.numpy.zeros((224, 224, 3))

model = limo.create_model("convnext_tiny", num_classes=100)
variables = model.init(jax.random.PRNGKey(0), x)
variables = limo.load_pretrained(variables, "convnext_tiny", pretrained=True)
state, params = variables.pop("params")

# inference mode.
out = model.apply({"params": params, **state}, x)

# train mode.
out, new_state = model.apply(
    {"params": params, **state},
    x,
    rngs={"dropout": jax.random.PRNGKey(0)}
    is_training=True,
    mutable=True,
)
```

### Use builtin models as modules of your model


Call `limo.create_model` in your model to use builtin models iside your model. To load pretrained variables, name the created model and specify the name as `module_name` when calling `limo.load_pretrained`. If you would like to load variables to deeper modules, specify module names joined by dot (e.g., f1.f1_child.f1_grandchild).

```python
import jax
from flax import linen
import limo


class Model(linen.Module):

    @linen.compact
    def __call__(self, x, is_training):
        f1 = limo.create_model("convnext_tiny", name="f1")  # Pass name to load variables.
        f2 = limo.create_model("efficientnet_b0", name="f2")
        y = f1(x, is_training) + f2(x, is_training)
        return y

x =  jax.numpy.zeros((224, 224, 3))

model = limo.create_model("convnext_tiny", num_classes=100)
variables = model.init(jax.random.PRNGKey(0), x)
variables = limo.load_pretrained(variables, "convnext_tiny", pretrained=True, module_name="f1")
variables = limo.load_pretrained(variables, "efficientnet_b0", pretrained=True, module_name="f2")

# inference mode.
out = model.apply(variables, x, is_training=False)
```


### Load your own variables

To load your own variables, `limo.maybe_overwrite_variables` is useful. This method also supports `module_name` option to load variables to modules like `limo.load_pretrained`.

```python
to_load = ...  # your own variables.
variables = limo.maybe_overwrite_variables(variables, to_load)
variables = limo.maybe_overwrite_variables(variables, to_load, module_name="f1")  # load variables to `f1` module.
```


## Examples

In `examples/`, some examples are implemented.

- ensemble.py: Example of how to use builtin models as modules of your model, and how to load variables into modules of a model.
- resnet_tsm.py: Example of model customization.
