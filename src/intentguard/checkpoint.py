"""Pickle-free parameter artifacts and strict shape/identity checks."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util
from safetensors.numpy import load_file, save_file

from .io import read_json, sha256, write_json
from .model import IntentEncoder, ModelConfig


def save_weights(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    flat = traverse_util.flatten_dict(pure)
    values = {json.dumps(path): np.asarray(jax.device_get(v)) for path, v in flat.items()}
    if not all(np.isfinite(v).all() for v in values.values()):
        raise ValueError("Refusing non-finite model parameters")
    temporary = directory / "weights.tmp"
    save_file(values, str(temporary))
    temporary.replace(directory / "weights.safetensors")
    write_json(
        directory / "model.json",
        {
            "config": model.config.to_dict(),
            "weights_sha256": sha256(directory / "weights.safetensors"),
        },
    )


def load_weights(directory):
    directory = Path(directory)
    metadata = read_json(directory / "model.json")
    if sha256(directory / "weights.safetensors") != metadata["weights_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    model = IntentEncoder(ModelConfig(**metadata["config"]))
    state = nnx.state(model, nnx.Param)
    expected = traverse_util.flatten_dict(nnx.to_pure_dict(state))
    loaded = load_file(str(directory / "weights.safetensors"))
    if set(loaded) != {json.dumps(path) for path in expected}:
        raise ValueError("Checkpoint parameter keys do not match the architecture")
    restored = {}
    for path, target in expected.items():
        value = loaded[json.dumps(path)]
        if value.shape != target.shape or value.dtype != np.dtype(target.dtype):
            raise ValueError("Checkpoint parameter shape or dtype mismatch")
        if not np.isfinite(value).all():
            raise ValueError("Checkpoint contains non-finite parameters")
        restored[path] = jnp.asarray(value)
    nnx.replace_by_pure_dict(state, traverse_util.unflatten_dict(restored))
    nnx.update(model, state)
    return model
