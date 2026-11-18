"""Separate calibration, frozen-bundle evaluation, and a strong lexical baseline."""

from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .checkpoint import load_weights
from .data import load_role
from .io import read_json, sha256, write_json
from .tokenization import encode, load_tokenizer
from .training import predict_logits
from .uncertainty import conformal_quantile, evaluate, fit_policy, fit_temperature, probabilities


def same_dataset(data, bundle):
    protocol = read_json(Path(bundle) / "protocol.json")
    if protocol["manifest_sha256"] != sha256(Path(data) / "manifest.json"):
        raise ValueError("This model was trained against a different data manifest")


def fit_decision_rule(get_logits, data, alpha=0.1, max_oos_accept=0.05):
    # The function is intentionally unable to request the test role.
    temp_rows, temp_y, _ = load_role(data, "temperature")
    temperature = fit_temperature(get_logits(temp_rows), temp_y)
    conf_rows, conf_y, _ = load_role(data, "conformal")
    quantile = conformal_quantile(probabilities(get_logits(conf_rows), temperature), conf_y, alpha)
    policy_rows, policy_y, _ = load_role(data, "policy")
    policy = fit_policy(
        probabilities(get_logits(policy_rows), temperature), policy_y, quantile, max_oos_accept
    )
    return {
        "temperature": temperature,
        "conformal_quantile": quantile,
        "alpha": alpha,
        "conformal_score": "1 - probability of true class (LAC)",
        "conformal_samples": len(conf_y),
        "temperature_samples": len(temp_y),
        "policy": policy,
        "test_used_for_fitting": False,
    }


def calibrate(data, bundle, alpha=0.1, max_oos_accept=0.05):
    bundle = Path(bundle)
    if (bundle / "bundle.json").exists():
        raise ValueError("Bundle already sealed; use a separate experiment directory")
    same_dataset(data, bundle)
    model, tokenizer = load_weights(bundle), load_tokenizer(bundle / "tokenizer")

    def get_logits(rows):
        return predict_logits(
            model, encode(tokenizer, [r["text"] for r in rows], model.config.max_length)
        )

    rule = fit_decision_rule(get_logits, data, alpha, max_oos_accept)
    rule["manifest_sha256"] = sha256(Path(data) / "manifest.json")
    write_json(bundle / "calibration.json", rule)
    names = [
        "weights.safetensors",
        "model.json",
        "labels.json",
        "calibration.json",
        "protocol.json",
    ]
    names += [
        str(p.relative_to(bundle)).replace("\\", "/")
        for p in sorted((bundle / "tokenizer").iterdir())
        if p.is_file()
    ]
    write_json(
        bundle / "bundle.json",
        {
            "schema_version": 1,
            "files": {name: sha256(bundle / name) for name in names},
            "purpose": "Frozen inference bundle; weights and decision rule selected without test evaluation",
        },
    )
    return rule


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = read_json(bundle / "bundle.json")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported bundle version")
    files = manifest["files"]
    required = {
        "weights.safetensors",
        "model.json",
        "labels.json",
        "calibration.json",
        "protocol.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    }
    if not required.issubset(files):
        raise ValueError("Incomplete inference bundle")
    for name, expected in files.items():
        target = (bundle / name).resolve()
        if not target.is_relative_to(bundle.resolve()) or not target.is_file():
            raise ValueError("Invalid bundle artifact path")
        if sha256(target) != expected:
            raise ValueError(f"Bundle artifact changed: {name}")
    return manifest


def error_examples(rows, labels, logits, class_names, temperature, limit=15):
    probs = probabilities(logits, temperature)
    prediction = probs.argmax(1)
    wrong = np.flatnonzero((labels >= 0) & (prediction != labels))
    order = wrong[np.argsort(-probs.max(1)[wrong], kind="stable")][:limit]
    return [
        {
            "id": rows[i]["id"],
            "source_index": rows[i]["source_index"],
            "true": class_names[int(labels[i])],
            "predicted": class_names[int(prediction[i])],
            "confidence": float(probs[i].max()),
        }
        for i in order
    ]


def evaluate_bundle(data, bundle, output):
    verify_bundle(bundle)
    same_dataset(data, bundle)
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation report already exists")
    model, tokenizer = load_weights(bundle), load_tokenizer(Path(bundle) / "tokenizer")
    rule = read_json(Path(bundle) / "calibration.json")
    rows, labels, names = load_role(data, "test")
    logits = predict_logits(
        model, encode(tokenizer, [r["text"] for r in rows], model.config.max_length)
    )
    report = evaluate(
        logits, labels, rule["temperature"], rule["conformal_quantile"], rule["policy"]
    )
    report.update(
        {
            "model": "flax_transformer",
            "manifest_sha256": sha256(Path(data) / "manifest.json"),
            "bundle_sha256": sha256(Path(bundle) / "bundle.json"),
            "decision_rule": rule,
            "errors": error_examples(rows, labels, logits, names, rule["temperature"]),
        }
    )
    write_json(output, report)
    return report


def baseline(data, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Baseline report already exists")
    rows, labels, names = load_role(data, "train")
    estimator = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20_000),
        LogisticRegression(C=4.0, max_iter=400, solver="lbfgs"),
    )
    estimator.fit([r["text"] for r in rows], labels)

    def get_logits(records):
        return estimator.decision_function([r["text"] for r in records])

    rule = fit_decision_rule(get_logits, data)
    test_rows, test_y, _ = load_role(data, "test")
    logits = get_logits(test_rows)
    report = evaluate(
        logits, test_y, rule["temperature"], rule["conformal_quantile"], rule["policy"]
    )
    report.update(
        {
            "model": "tfidf_logistic_regression",
            "decision_rule": rule,
            "manifest_sha256": sha256(Path(data) / "manifest.json"),
            "config": {
                "ngram_range": [1, 2],
                "min_df": 2,
                "sublinear_tf": True,
                "max_features": 20_000,
                "C": 4.0,
                "max_iter": 400,
            },
            "features": len(estimator[0].vocabulary_),
            "iterations": estimator[1].n_iter_.tolist(),
            "errors": error_examples(test_rows, test_y, logits, names, rule["temperature"]),
        }
    )
    write_json(output, report)
    return report
