# Linen Image Models

Linen Image Models (limo) aims to port various image models and their pre-trained weights from timm.

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