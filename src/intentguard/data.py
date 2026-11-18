"""Pinned CLINC150 source, text-identity audit, and disjoint development roles."""

import hashlib
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import read_json, sha256, write_json

SOURCE_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/clinc/oos-eval/{SOURCE_REVISION}/data/data_full.json"
)
SOURCE_SHA256 = "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0"
ROLES = ("train", "selection", "temperature", "conformal", "policy", "test")


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def identity(text):
    return hashlib.sha256(canonical(text).encode("utf-8")).hexdigest()


def partition(raw, seed=2026):
    required = {"train", "val", "test", "oos_train", "oos_val", "oos_test"}
    if set(raw) != required:
        raise ValueError("Expected the six original CLINC150 splits")
    labels = sorted({label for _, label in raw["train"]})
    if "oos" in labels or len(labels) < 2:
        raise ValueError("Training must contain at least two in-scope classes")
    occurrences = defaultdict(list)
    for split, rows in raw.items():
        for index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("Expected [text, label] records")
            text, label = row
            if not isinstance(text, str) or not canonical(text) or not isinstance(label, str):
                raise ValueError("Texts and labels must be nonempty strings")
            if label not in labels and label != "oos":
                raise ValueError("Unknown label")
            occurrences[identity(text)].append((split, index, label))
    # This rule uses text identity, not correctness or model outputs: quarantine every
    # occurrence of every duplicate before training, irrespective of label agreement.
    duplicates = {key: values for key, values in occurrences.items() if len(values) > 1}
    cleaned = {}
    for split, rows in raw.items():
        cleaned[split] = [
            {
                "id": identity(text),
                "text": text,
                "label": label,
                "source": split,
                "source_index": index,
            }
            for index, (text, label) in enumerate(rows)
            if identity(text) not in duplicates
        ]
    result = {role: [] for role in ROLES}
    result["train"] = cleaned["train"]
    result["test"] = cleaned["test"] + cleaned["oos_test"]
    for label in labels:
        rows = sorted((r for r in cleaned["val"] if r["label"] == label), key=lambda r: r["id"])
        if len(rows) < 5:
            raise ValueError("Need at least five validation records per class")
        # Five near-equal groups: 2 for checkpoint selection, 1 each for T, q, policy.
        class_seed = int(hashlib.sha256(f"{seed}:{label}".encode()).hexdigest()[:8], 16)
        order = np.random.default_rng(class_seed).permutation(len(rows))
        groups = np.array_split(order, 5)
        for role, positions in zip(
            ("selection", "temperature", "conformal", "policy"),
            (np.concatenate(groups[:2]), *groups[2:]),
        ):
            result[role].extend(rows[int(i)] for i in positions)
    result["policy"].extend(cleaned["oos_val"])
    ids = [r["id"] for rows in result.values() for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Text identity crossed an experiment boundary")
    audit = {
        "rule": "NFKC + casefold + whitespace collapse; quarantine every duplicate occurrence",
        "duplicate_groups": [
            {"id": key, "occurrences": value, "conflicting_labels": len({r[2] for r in value}) > 1}
            for key, value in sorted(duplicates.items())
        ],
        "original_counts": {k: len(v) for k, v in raw.items()},
        "cleaned_counts": {k: len(v) for k, v in cleaned.items()},
        "roles": {
            k: {
                "total": len(v),
                "in_scope": sum(r["label"] != "oos" for r in v),
                "out_of_scope": sum(r["label"] == "oos" for r in v),
            }
            for k, v in result.items()
        },
        "unused_oos_training_rows": len(cleaned["oos_train"]),
        "limitations": "Exact canonical duplicates only; semantic and paraphrase overlap unverified",
    }
    return labels, result, audit


def prepare(output, source=None, seed=2026):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Dataset output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "source.json"
    if source is None:
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "intentguard-jax/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(10_000_001)
        if len(data) > 10_000_000:
            raise ValueError("Dataset exceeds the expected source size")
        destination.write_bytes(data)
    else:
        destination.write_bytes(Path(source).read_bytes())
    if sha256(destination) != SOURCE_SHA256:
        raise ValueError("Source SHA-256 mismatch; no split artifacts accepted")
    labels, rows, audit = partition(read_json(destination), seed)
    for role, records in rows.items():
        write_json(output / f"{role}.json", records)
    write_json(output / "labels.json", labels)
    write_json(output / "audit.json", audit)
    manifest = {
        "schema_version": 1,
        "source_url": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "source_revision": SOURCE_REVISION,
        "split_seed": seed,
        "license": "CC-BY-3.0; original CLINC150 attribution applies",
        "files": {
            f"{name}.json": sha256(output / f"{name}.json") for name in (*ROLES, "labels", "audit")
        },
    }
    write_json(output / "manifest.json", manifest)
    return audit


def load_role(root, role):
    if role not in ROLES:
        raise ValueError("Unknown data role")
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    for name in (f"{role}.json", "labels.json"):
        if sha256(root / name) != manifest["files"][name]:
            raise ValueError(f"Dataset changed: {name}")
    labels = read_json(root / "labels.json")
    mapping = {label: i for i, label in enumerate(labels)}
    rows = read_json(root / f"{role}.json")
    targets = np.array([mapping.get(row["label"], -1) for row in rows], dtype=np.int32)
    return rows, targets, labels
