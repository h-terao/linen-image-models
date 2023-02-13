"""

TODO: version -> pretrained.
"""
from __future__ import annotations
import typing as tp
from pathlib import Path
from collections import defaultdict
import warnings
import io
import pickle
import inspect
import zlib

import requests
from tqdm import tqdm
from flax import linen, traverse_util, core
import chex

ModelFun = tp.Callable[..., linen.Module]
SEP = "/"


class ModelEntry(tp.NamedTuple):
    name: str
    weight_name: str
    family_name: str
    model_fun: ModelFun
    url: str | None
    defaults: tp.Mapping


_model_registry: tp.Mapping[str, tp.Mapping[str, ModelEntry]] = defaultdict(dict)
_weight_cache = dict()


def loads(data):
    """compress + serialize."""
    return pickle.loads(zlib.decompress(data))


def dumps(obj):
    """deserialize + decompress."""
    return zlib.compress(pickle.dumps(obj))


def register_model(pretrained: str, url: str | None = None, default: bool = False, **kwargs):
    """

    Args:
        pretrained: Name of pretrained weights.
        url: URL of the pretrained weights.
        default: If True, set this pretrained weights as default.
        **kwargs: Default parameters.
    """
    frame = inspect.currentframe().f_back
    frameinfo = inspect.getframeinfo(frame)
    family_name = Path(frameinfo.filename).stem

    def register(model_fun: ModelFun):
        name = model_fun.__name__
        new_entry = ModelEntry(
            name=name,
            weight_name=pretrained,
            family_name=family_name,
            model_fun=model_fun,
            url=url,
            defaults=kwargs,
        )
        _model_registry[name][pretrained] = new_entry
        if default:
            _model_registry[name][False] = new_entry
            _model_registry[name][True] = new_entry
        return model_fun

    return register


def fake_register_model(pretrained: str, url: str | None = None, default: bool = False, **defaults):
    """Have the same interfance of `register_model`, but do nothing."""

    def register(model_fun: ModelFun):
        return model_fun

    return register


def create_model(name: str, pretrained: bool | str = False, **kwargs) -> linen.Module:
    """Instantiate flax model.

    Args:
        name: Model name.
        pretrained: Pretrained weight name. Used to determine default parameters.
        **kwargs: Arguments of the specified model.

    Returns:
        Instantiated flax model.
    """
    assert name in _model_registry, f"The specified model {name} is not registered yet."

    msg = f"The specified pretrained version {pretrained} is not registered yet for {name}."
    assert pretrained in _model_registry[name], msg

    entry = _model_registry[name][pretrained]
    new_kwargs = dict(entry.defaults, **kwargs)
    return entry.model_fun(**new_kwargs)


def list_models(name: str | None = None, pretrained: str | bool = False) -> tp.Sequence[str]:
    """List available models.

    Args:
        name: Model name to filter available models.
        pretrained: Pretrained weight name to filter available models.

    Returns:
        List of model names filtered by `name` and `pretrained`.
    """
    model_names = []
    for model_name, entries in _model_registry.items():
        if name is None or name in model_name:
            for entry_name in entries:
                if pretrained is None or pretrained in entry_name:
                    model_names.append(model_name)
    return model_names


def maybe_overwrite_variables(
    variables: chex.ArrayTree, to_load: chex.ArrayTree, module_name: str | None = None
) -> chex.ArrayTree:
    """Create a new variables that holds `to_load` values as much as possible.

    Args:
        variables: Variables.
        to_load: Another variables to load.
        module_name: Module name to load partially.

    Returns:
        variables.
    """

    def load(variables, to_load):
        # convert PyTorch model and get flatten state.
        to_load = traverse_util.flatten_dict(to_load, sep=".")
        checked = {name: False for name in to_load}

        new_variables = {}
        for name, array in traverse_util.flatten_dict(variables, sep=SEP).items():
            new_name = name.replace(SEP, ".")
            if new_name not in to_load:
                msg = (
                    f"{new_name} is not found in PyTorch model. "
                    "This value is not overwritten and the initialized value is used."
                )
                warnings.warn(msg)
            elif array.shape != to_load[new_name].shape:
                msg = (
                    f"The shape of {new_name} is different. This value is not overwritten and "
                    f"the initialized value is used. (Variable shape is {tuple(array.shape)}, but "
                    f"PyTorch model has a tensor with shape {tuple(to_load[new_name].shape)}.)"
                )
                warnings.warn(msg)
                checked[new_name] = True
            else:
                array = to_load[new_name]
                checked[new_name] = True
            new_variables[name] = array

        unchecked = [k for k, v in checked.items() if not v]
        if unchecked:
            msg = f"The following parameters of PyTorch model are not loaded: {unchecked}."
            warnings.warn(msg)

        new_variables = traverse_util.unflatten_dict(new_variables, sep=SEP)
        return new_variables

    cast_to = type(variables)
    if isinstance(variables, core.FrozenDict):
        variables = variables.unfreeze()

    if module_name is None:
        new_variables = load(variables, to_load)
    else:
        # canonicalize module name
        if module_name[-1] == ".":
            module_name = module_name[:-1]

        # Partial load.
        tmp = {}
        for col, xs in variables.items():
            tmp.setdefault(col, dict())
            xs_flat: tp.Mapping[str, chex.Array] = traverse_util.flatten_dict(xs, sep=".")
            for k, x in xs_flat.items():
                if k.startswith(module_name):
                    tmp[col][k.removeprefix(module_name + ".")] = x

        tmp = load(tmp, to_load)

        new_variables = {}
        for col, xs in variables.items():
            xs_flat = traverse_util.flatten_dict(xs)
            for k, x in xs_flat.items():
                if ".".join(k).startswith(module_name):
                    new_variables[(col, *k)] = tmp[col][".".join(k).removeprefix(module_name + ".")]
                else:
                    new_variables[(col, *k)] = x
        new_variables = traverse_util.unflatten_dict(new_variables)

    new_variables = cast_to(new_variables)
    return new_variables


def download_variables_from_url(
    name: str, pretrained: str | bool, save_dir: str | None, cache: bool = False
) -> chex.ArrayTree:
    entry = _model_registry[name][pretrained]
    weight_name = f"{name}.{entry.weight_name}.pkl"

    if weight_name in _weight_cache:
        return _weight_cache[weight_name]

    def download():
        # FIXME: Progressbar is not shown well because `total_size` is always zero.

        assert entry.url is not None, f"URL of {weight_name} is not registered."
        total_size = int(requests.head(entry.url).headers["content-length"])
        stream = requests.get(entry.url, stream=True)

        pbar = tqdm(total=total_size, unit="B", unit_scale=True)
        variable_bytes = io.BytesIO()
        for chunk in stream.iter_content(chunk_size=1024):
            variable_bytes.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
        return loads(variable_bytes.getvalue())

    if save_dir is not None:
        file_path = Path(save_dir, entry.family_name, weight_name)
        if file_path.exists():
            weights = load(file_path)
        else:
            # download from url and save.
            weights = download()
            file_path.parent.mkdir(parents=True, exist_ok=True)
            save(file_path, weights)
    else:
        weights = download()

    if cache:
        _weight_cache[weight_name] = weights

    return weights


def load_pretrained(
    variables: chex.ArrayTree,
    name: str,
    pretrained: str | bool = False,
    module_name: str | None = None,
    save_dir: str | None = None,
    cache: bool = False,
) -> chex.ArrayTree:
    """Load pretrained weights.

    Args:
        variables: Initialized weights.
        name: Model name.
        pretrained: Name of pretrained weights. If True, use default weights.
        module_name: Module name.
        save_dir: Path to the directory to save weights.
        cache: If True, cache weights on memory. If you load the same weights
            several times, caching will reduce loading time.

    Returns:
        New weights.
    """
    if pretrained:
        to_load = download_variables_from_url(name, pretrained, save_dir, cache)
        variables = maybe_overwrite_variables(variables, to_load, module_name)
    return variables


def save(file: str | Path, obj: tp.Any, exist_ok: bool = False) -> None:
    """Save object with zlib compression in a safe manner.

    Args:
        file: Filename to save `obj`.
        obj: Object to save.
        exist_ok: Whether save `obj` when `file` is already exists.

    Raises:
        FileExistsError: Raised when `file` is already exists and `exist_ok` is False.
    """
    file_path = Path(file)
    if file_path.exists() and not exist_ok:
        raise FileExistsError(f"{file_path} is already exists.")

    tmp_file_path = file_path.with_suffix(".tmp")
    tmp_file_path.write_bytes(dumps(obj))
    tmp_file_path.rename(file_path)


def load(file: str | Path) -> tp.Any:
    """Load object compressed by zlib."""
    return loads(Path(file).read_bytes())
