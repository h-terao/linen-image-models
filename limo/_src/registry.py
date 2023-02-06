from __future__ import annotations
import typing as tp
from pathlib import Path
import pickle
import collections
import warnings
import requests

from flax import linen, core
from flax.traverse_util import flatten_dict, unflatten_dict


SEP = "/"

_model_registry = dict()
_checkpoint_registry = collections.defaultdict(dict)
_checkpoint_cache = {}


def list_models(model_name=None, pretrained=False):
    model_list = []
    for name in _model_registry:
        if model_name is not None and model_name not in name:
            continue
        if pretrained and pretrained not in _checkpoint_registry.get(name, list()):
            continue
        model_list.append(name)
    return model_list


def create_model(model_name, pretrained=False, **kwargs) -> linen.Module:
    model_builder, default_cfg = _model_registry[model_name]
    if pretrained:
        _, default_cfg = _checkpoint_registry[model_name][pretrained]
        kwargs = dict(default_cfg, **kwargs)
    return model_builder(**kwargs), default_cfg


def load_pretrained(
    variables, model_name, pretrained, module_name=None, ckpt_dir=None, cache: bool = False
):
    """Download pre-trained parameters from online,
        and overwrite them into models.

    Args:
        model_name: Model name.
        pretrained: Pretrained name.

        cache: If True, cache weights on memory.

    Example:
        >>> new_variables = load_variable(variables, model_name, pretrained)
    """
    url, _ = _checkpoint_registry[model_name][pretrained]
    unique_id = f"{model_name}.{pretrained}.ckpt"
    if unique_id in _checkpoint_cache:
        to_load = _checkpoint_cache[unique_id]
    elif ckpt_dir is not None:
        model_path = Path(ckpt_dir, unique_id)
        if not model_path.exists():
            content = requests.get(url).content
            to_load = pickle.loads(content)
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(content)
        else:
            to_load = pickle.loads(model_path.read_bytes())
    else:
        content = requests.get(url).content
        to_load = pickle.loads(content)

    if cache:
        _checkpoint_cache[unique_id] = to_load

    cast_to = type(variables)
    if isinstance(variables, core.FrozenDict):
        variables = variables.unfreeze()

    new_variables = {}
    to_load_flat = flatten_dict(to_load, sep=".")
    for key, array in flatten_dict(variables).items():
        join_key = ".".join(key)
        if module_name is not None:
            join_key = join_key.replace(f".{module_name}")

        if join_key in to_load_flat:
            new_array = to_load_flat[join_key]
            if array.shape == new_array.shape:
                array = new_array
            else:
                msg = f"Failed to load {join_key}. Shape is mismatch."
                warnings.warn(msg)
        else:
            msg = f"Failed to load {join_key}. This is not found in pretrained models."
            warnings.warn(msg)
        new_variables[key] = array

    new_variables = unflatten_dict(new_variables)
    return cast_to(new_variables)


def register_model(
    model_name: str, model_builder: tp.Callable[..., linen.Module], default_cfg=None
):
    assert model_name not in _model_registry, f"{model_name} is already registered."
    default_cfg = default_cfg or dict()
    _model_registry[model_name] = (model_builder, default_cfg)
    return model_builder


def register_pretrained(
    model_name: str,
    tag: str,
    url: str | None = None,
    default_cfg: tp.Mapping | None = None,
    default: bool = False,
) -> None:
    default_cfg = default_cfg or dict()

    if model_name not in _model_registry:
        msg = (
            f"Failed to register new pretrained variables for {model_name}. "
            "This model is not registered in the model registry."
        )
        warnings.warn(msg)
    else:
        item = (url, dict(_model_registry[model_name][1], **default_cfg))
        _checkpoint_registry[model_name][tag] = item
        if default:
            _checkpoint_registry[model_name][True] = item


# #
# #  Register all models under `limo/_src/models`
# #
# model_dir_path = Path(__file__).parent / "models"
# for model_path in filter(lambda x: "__init__" not in str(x), model_dir_path.iterdir()):
#     module = importlib.import_module(f"limo._src.models.{model_path.stem}")
#     for model_name in module.__all__:
#         model_builder = getattr(module, model_name)
#         register_model(model_name, model_builder)

# #
# #  Register pretrained models here...
# #
