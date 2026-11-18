import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import psutil
import pytest

process = psutil.Process()
if hasattr(process, "cpu_affinity"):
    process.cpu_affinity(process.cpu_affinity()[:4])


@pytest.fixture
def raw_dataset():
    return {
        "train": [
            [f"{label} training request {i}", label]
            for label in ("weather", "music")
            for i in range(24)
        ],
        "val": [
            [f"{label} validation request {i}", label]
            for label in ("weather", "music")
            for i in range(20)
        ],
        "test": [
            [f"{label} testing request {i}", label]
            for label in ("weather", "music")
            for i in range(5)
        ],
        "oos_train": [["unrelated unused training", "oos"]],
        "oos_val": [[f"unrelated policy example {i}", "oos"] for i in range(10)],
        "oos_test": [[f"unrelated testing example {i}", "oos"] for i in range(5)],
    }


@pytest.fixture
def dataset(tmp_path, raw_dataset):
    from intentguard.data import partition
    from intentguard.io import sha256, write_json

    root = tmp_path / "data"
    names, roles, audit = partition(raw_dataset)
    files = {**roles, "labels": names, "audit": audit}
    for name, value in files.items():
        write_json(root / f"{name}.json", value)
    write_json(
        root / "manifest.json",
        {"files": {f"{name}.json": sha256(root / f"{name}.json") for name in files}},
    )
    return root


@pytest.fixture
def tiny_bundle(dataset, tmp_path):
    from intentguard.evaluation import calibrate
    from intentguard.training import train

    bundle = tmp_path / "bundle"
    train(
        dataset,
        bundle,
        epochs=2,
        batch_size=8,
        width=16,
        layers=1,
        max_length=12,
        vocab_size=48,
        patience=2,
    )
    calibrate(dataset, bundle, alpha=0.2)
    return bundle
