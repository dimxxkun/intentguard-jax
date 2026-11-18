import pytest

from intentguard.data import canonical, load_role, partition
from intentguard.io import read_json, write_json


def test_duplicate_quarantine_uses_text_identity_even_when_labels_conflict(raw_dataset):
    raw_dataset["train"].append(["  SAME Text ", "weather"])
    raw_dataset["test"].append(["same text", "music"])
    _, roles, audit = partition(raw_dataset)
    assert len(audit["duplicate_groups"]) == 1
    assert audit["duplicate_groups"][0]["conflicting_labels"]
    assert not any(canonical(row["text"]) == "same text" for rows in roles.values() for row in rows)


def test_roles_are_disjoint_deterministic_and_oos_not_in_fitting(raw_dataset):
    labels, roles, audit = partition(raw_dataset)
    assert (labels, roles, audit) == partition(raw_dataset)
    all_ids = [r["id"] for rows in roles.values() for r in rows]
    assert len(all_ids) == len(set(all_ids))
    for role in ("train", "selection", "temperature", "conformal"):
        assert all(r["label"] != "oos" for r in roles[role])
    assert audit["unused_oos_training_rows"] == 1


def test_source_provenance_and_modified_artifact_rejected(dataset):
    rows, labels, classes = load_role(dataset, "train")
    assert classes == ["music", "weather"]
    assert labels.shape == (48,)
    assert rows[0]["source"] == "train"
    (dataset / "train.json").write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        load_role(dataset, "train")


def test_strict_json_rejects_duplicate_keys_and_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"value":1,"value":2}', '{"value":NaN}'):
        path.write_text(text)
        with pytest.raises(ValueError):
            read_json(path)


def test_artifact_bytes_are_platform_independent(tmp_path):
    path = tmp_path / "artifact.json"
    write_json(path, {"label": "weather", "score": 0.5})
    assert path.read_bytes() == b'{\r\n  "label": "weather",\r\n  "score": 0.5\r\n}\r\n'
    assert read_json(path) == {"label": "weather", "score": 0.5}


def test_insufficient_validation_fails(raw_dataset):
    raw_dataset["val"] = raw_dataset["val"][:2]
    with pytest.raises(ValueError, match="five"):
        partition(raw_dataset)
