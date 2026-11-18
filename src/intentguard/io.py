"""Small, strict artifact boundaries; credentials and ambient environment are never logged."""

import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Artifact identity must not depend on the host OS. Preserve the CRLF bytes
    # used by the original study, explicitly on every platform.
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\r\n"
    )
    temporary.replace(path)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(
        Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=invalid
    )


def environment():
    import psutil

    packages = [
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "transformers",
        "tokenizers",
        "scikit-learn",
        "numpy",
        "scipy",
        "safetensors",
        "fastapi",
    ]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": psutil.Process().cpu_affinity()
        if hasattr(psutil.Process, "cpu_affinity")
        else None,
        "packages": {name: importlib.metadata.version(name) for name in packages},
    }
