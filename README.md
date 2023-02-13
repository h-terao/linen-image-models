# Linen Image Models

limo is an easily customizable image model implementation with pre-trained variables. The pre-trained variables are loaded correctly even if you add or remove modules. Currently, this project would like to reimplement `torchvision.models`.

## Usage

### Create models

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
out, new_state = model.apply({"params": params, **state}, x, rngs={"dropout": jax.random.PRNGKey(0)} is_training=True, mutable=True)
```
