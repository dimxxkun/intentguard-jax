import json
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from intentguard.api import create_app
from intentguard.evaluation import evaluate_bundle, verify_bundle
from intentguard.io import read_json
from intentguard.predictor import Predictor


def test_real_train_calibrate_cli_api_evaluate_and_tamper(dataset, tiny_bundle, tmp_path):
    # One integrated workflow owns the expensive tiny training fixture.
    engine = Predictor(tiny_bundle)
    direct = engine.predict(["weather forecast", "music please"])
    assert len(direct["results"]) == 2
    with TestClient(create_app(predictor=engine)) as client:
        assert client.get("/health").json()["bundle_id"] == engine.bundle_id
        response = client.post("/predict", json={"texts": ["weather forecast", "music please"]})
        assert response.status_code == 200
        assert response.json() == direct
        for body in (
            {"texts": []},
            {"texts": ["  "]},
            {"texts": [123]},
            {"texts": ["x"] * 33},
            {"texts": ["x" * 501]},
            {"texts": ["hi"], "top_k": True},
            {"texts": ["hi"], "unsupported_option": True},
        ):
            assert client.post("/predict", json=body).status_code == 422
    report = evaluate_bundle(dataset, tiny_bundle, tmp_path / "evaluation.json")
    assert report["uncalibrated"]["samples"] == 10
    assert report["out_of_scope"]["samples"] == 5
    assert report["decision_rule"]["test_used_for_fitting"] is False
    result = subprocess.run(
        [sys.executable, "-m", "intentguard", "verify", "--bundle", str(tiny_bundle)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified"]
    (tiny_bundle / "calibration.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_bundle(tiny_bundle)


def test_untrained_validation_does_not_touch_test_role(dataset, tmp_path):
    import numpy as np

    from intentguard.evaluation import fit_decision_rule

    (dataset / "test.json").unlink()
    requested = []

    def logits(rows):
        requested.extend(row["source"] for row in rows)
        return np.tile([1.0, 0.0], (len(rows), 1))

    rule = fit_decision_rule(logits, dataset)
    assert rule["test_used_for_fitting"] is False
    assert set(requested) == {"val", "oos_val"}


def test_nonempty_output_not_overwritten(dataset, tmp_path):
    from intentguard.training import train

    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("evidence")
    with pytest.raises(ValueError, match="empty"):
        train(dataset, output)
    assert (output / "keep.txt").read_text() == "evidence"


def test_train_tokenizer_has_no_validation_only_word(dataset, tmp_path):
    from intentguard.data import load_role
    from intentguard.tokenization import fit_tokenizer

    rows, _, _ = load_role(dataset, "train")
    tokenizer = fit_tokenizer([r["text"] for r in rows], tmp_path / "tokenizer", vocab_size=100)
    assert "validation" not in tokenizer.get_vocab()
    assert "testing" not in tokenizer.get_vocab()
    assert read_json(tmp_path / "tokenizer" / "tokenizer.json")["model"]["type"] == "WordPiece"
