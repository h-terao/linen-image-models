# Linen Image Models

Linen Image Models (limo) aims to port various image models and their pre-trained weights from timm.

## Example

```python
import limo

model, cfg = limo.create_model("efficientnet_b0")
model_names = limo.list_models(pretrained=True)
# model_names = limo.list_models(pretrained="in12k")

variables = limo.load_weights(variables, "efficientnet_b0", pretrained=True)
```


## Philosophy
- Easy to use pre-trained model
- Easy to folk and modify models

## Supported Models

- EfficientNet
- EfficientNetV2 (experimental)
- TinyNet

## Modules Overview

### Configuration

Configuration is a core component of `limo`. To switch model behaviour (e.g., training mode, torch-like padding, and half precision training), you must transform functions via `limo.configure`.

If you prefer more pythonic way, use a context manager `limo.using_config`.

### Layers

`limo.layers` provide