# Linen Image Models

limo is a easy-to-customize image model implementations with pretrained weights.
If you add or remove any modules, the pretrained weights will be loaded correctly.


## Prerequests

- jax, jaxlib
- flax
- chex
- einops
- requests
- tqdm

In addition, pytests, torch, and torchvision are required to test this project.

## Usage

### Create off-the-shelf models

`limo.create_model`

### Load pretrained parameters.

### Customize models

All model implementations are almost self-contained in a Python file, and it is easy to copy and paste model definitions into your project. If you add or remove modules, `limo.load_pretrained` also works well because all implemented modules are manually named, and `limo.load_pretrained` uses their names to determine where to load weights.

Specifically, you can customize `limo`'s models as follows:
1. Copy limo/_src/*.py into your project.
2. Replace the line `from limo import register_model` with `from limo import fake_register_model as register_model` to avoid registration error.