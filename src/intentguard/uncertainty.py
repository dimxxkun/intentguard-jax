"""Temperature, split-conformal sets, and an empirical out-of-scope routing policy."""

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def checked_logits(logits, labels=None):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or min(logits.shape) < 1 or logits.shape[1] < 2:
        raise ValueError("Expected nonempty [samples, classes] logits")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape != (len(logits),) or labels.dtype.kind not in "iu":
            raise ValueError("Expected one integer label per sample")
        if np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("Class index out of range")
    return logits


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return softmax(checked_logits(logits) / temperature, axis=1)


def nll(logits, labels, temperature=1.0):
    logits = checked_logits(logits, labels)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    scaled = logits / temperature
    return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))


def fit_temperature(logits, labels):
    checked_logits(logits, labels)
    result = minimize_scalar(
        lambda log_t: nll(logits, labels, math.exp(log_t)),
        bounds=(math.log(0.05), math.log(20)),
        method="bounded",
    )
    if not result.success:
        raise ValueError("Temperature optimization failed")
    fitted = math.exp(float(result.x))
    return fitted if nll(logits, labels, fitted) < nll(logits, labels) else 1.0


def checked_probabilities(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 1 or values.shape[1] < 2:
        raise ValueError("Expected nonempty probability matrix")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Invalid probability values")
    if not np.allclose(values.sum(axis=1), 1, rtol=1e-7, atol=1e-9):
        raise ValueError("Probability rows must sum to one")
    return values


def conformal_quantile(probs, labels, alpha=0.1):
    probs = checked_probabilities(probs)
    checked_logits(probs, labels)
    if not 0 < alpha < 1:
        raise ValueError("Alpha must be in (0,1)")
    scores = 1 - probs[np.arange(len(labels)), labels]
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    # The finite-sample +1 correction can require the full label set.
    return 1.0 if rank > len(scores) else float(np.sort(scores)[rank - 1])


def prediction_sets(probs, quantile):
    probs = checked_probabilities(probs)
    if not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("Invalid conformal quantile")
    return (1 - probs) <= quantile


def route_mask(probs, quantile, policy):
    singleton = prediction_sets(probs, quantile).sum(axis=1) == 1
    if policy["reject_all"]:
        return np.zeros(len(probs), dtype=bool)
    threshold = policy["threshold"]
    if threshold is None or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid routing threshold")
    return singleton & (probs.max(axis=1) >= threshold)


def fit_policy(probs, labels, quantile, max_oos_accept=0.05):
    probs = checked_probabilities(probs)
    labels = np.asarray(labels)
    if labels.shape != (len(probs),) or labels.dtype.kind not in "iu":
        raise ValueError("Expected policy labels")
    inside, outside = labels >= 0, labels == -1
    if not inside.any() or not outside.any() or np.any(labels < -1):
        raise ValueError("Policy calibration requires both in-scope and out-of-scope examples")
    if np.any(labels >= probs.shape[1]) or not 0 <= max_oos_accept <= 1:
        raise ValueError("Invalid policy labels or false-accept budget")
    confidence = probs.max(axis=1)
    singleton = prediction_sets(probs, quantile).sum(axis=1) == 1
    correct = probs.argmax(axis=1) == labels
    best = {"reject_all": True, "threshold": None}
    best_key = (0, 0, 0, 0.0)
    for threshold in np.unique(confidence[singleton]):
        accepted = singleton & (confidence >= threshold)
        oos_count = int((accepted & outside).sum())
        if oos_count / int(outside.sum()) > max_oos_accept:
            continue
        good = int((accepted & inside & correct).sum())
        bad = int((accepted & inside & ~correct).sum())
        key = (good, -bad, -oos_count, float(threshold))
        if key > best_key:
            best_key = key
            best = {"reject_all": False, "threshold": float(threshold)}
    accepted = route_mask(probs, quantile, best)
    return {
        **best,
        "target_max_calibration_oos_accept": max_oos_accept,
        "calibration_oos_count": int(outside.sum()),
        "calibration_oos_accepted": int((accepted & outside).sum()),
        "calibration_in_scope_correct_accepted": int((accepted & inside & correct).sum()),
        "objective": "max correct ID accepted, then fewer ID errors, fewer OOS accepted, higher threshold",
        "scope": "empirical calibration constraint, not a population false-accept guarantee",
    }


def classification_report(logits, labels, temperature=1.0, bins=15):
    checked_logits(logits, labels)
    probs = probabilities(logits, temperature)
    prediction = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = prediction == labels
    bucket = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for i in range(bins):
        mask = bucket == i
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        average = float(confidence[mask].mean()) if count else None
        if count:
            ece += count / len(labels) * abs(accuracy - average)
        reliability.append(
            {
                "lower": i / bins,
                "upper": (i + 1) / bins,
                "count": count,
                "accuracy": accuracy,
                "confidence": average,
            }
        )
    target = np.eye(probs.shape[1])[labels]
    return {
        "samples": len(labels),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=np.arange(probs.shape[1]),
                average="macro",
                zero_division=0,
            )
        ),
        "nll": nll(logits, labels, temperature),
        "ece": float(ece),
        "brier": float(np.mean(np.sum((probs - target) ** 2, axis=1))),
        "reliability": reliability,
    }


def evaluate(logits, labels, temperature, quantile, policy):
    logits = checked_logits(logits)
    labels = np.asarray(labels)
    if labels.shape != (len(logits),) or labels.dtype.kind not in "iu":
        raise ValueError("Invalid evaluation labels")
    inside, outside = labels >= 0, labels == -1
    if not inside.any() or not outside.any() or np.any(labels < -1):
        raise ValueError("Evaluation requires in-scope and out-of-scope examples")
    if np.any(labels >= logits.shape[1]):
        raise ValueError("Invalid evaluation class")
    probs = probabilities(logits, temperature)
    sets = prediction_sets(probs, quantile)
    accepted = route_mask(probs, quantile, policy)
    correct = probs.argmax(axis=1) == labels
    accepted_id = accepted & inside
    id_sets = sets[inside]
    confidence = probs.max(axis=1)
    curve = []
    for threshold in np.linspace(0, 1, 51):
        keep = inside & (confidence >= threshold)
        curve.append(
            {
                "threshold": float(threshold),
                "coverage": float(keep.sum() / inside.sum()),
                "risk": float((~correct[keep]).mean()) if keep.any() else None,
                "oos_accept": float((confidence[outside] >= threshold).mean()),
            }
        )
    return {
        "uncalibrated": classification_report(logits[inside], labels[inside]),
        "calibrated": classification_report(logits[inside], labels[inside], temperature),
        "out_of_scope": {
            "samples": int(outside.sum()),
            "auroc": float(roc_auc_score(outside, 1 - confidence)),
            "average_precision": float(average_precision_score(outside, 1 - confidence)),
            "score": "1 - maximum calibrated probability; higher means more uncertain",
        },
        "sets": {
            "in_scope_coverage": float(id_sets[np.arange(inside.sum()), labels[inside]].mean()),
            "mean_in_scope_size": float(id_sets.sum(1).mean()),
            "in_scope_singleton_fraction": float((id_sets.sum(1) == 1).mean()),
        },
        "routing": {
            "in_scope_total": int(inside.sum()),
            "in_scope_accepted": int(accepted_id.sum()),
            "in_scope_coverage": float(accepted_id.sum() / inside.sum()),
            "accepted_in_scope_accuracy": float(correct[accepted_id].mean())
            if accepted_id.any()
            else None,
            "in_scope_correct_accepted": int((accepted_id & correct).sum()),
            "out_of_scope_accepted": int((accepted & outside).sum()),
            "out_of_scope_false_accept_rate": float(accepted[outside].mean()),
            "total_accepted": int(accepted.sum()),
        },
        "confidence_only_curve": curve,
    }
