import numpy as np
import pytest

from intentguard.uncertainty import (
    conformal_quantile,
    evaluate,
    fit_policy,
    fit_temperature,
    nll,
    prediction_sets,
    probabilities,
    route_mask,
)


def test_temperature_fits_finite_loss_without_changing_argmax():
    logits = np.array([[8.0, 0.0], [0.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
    labels = np.array([0, 1, 1, 0])
    temperature = fit_temperature(logits, labels)
    assert temperature > 1
    assert nll(logits, labels, temperature) < nll(logits, labels)
    np.testing.assert_array_equal(probabilities(logits, temperature).argmax(1), logits.argmax(1))


def test_logsumexp_handles_large_logits():
    assert np.isfinite(nll(np.array([[10000.0, -10000.0]]), np.array([1])))
    assert nll(np.array([[10000.0, -10000.0]]), np.array([1])) == 20000


def test_conformal_rank_uses_finite_sample_correction():
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2]])
    labels = np.zeros(4, dtype=int)
    assert conformal_quantile(probs, labels, alpha=0.4) == pytest.approx(0.3)
    assert conformal_quantile(probs, labels, alpha=0.01) == 1
    assert prediction_sets(probs, 1).all()


def test_empirical_policy_respects_oos_budget_and_can_reject_everything():
    probs = np.array([[0.98, 0.02], [0.6, 0.4], [0.8, 0.2], [0.52, 0.48]])
    labels = np.array([0, 0, -1, -1])
    policy = fit_policy(probs, labels, 0.45, max_oos_accept=0)
    np.testing.assert_array_equal(route_mask(probs, 0.45, policy), [True, False, False, False])
    impossible = np.array([[0.9, 0.1], [1.0, 0.0]])
    policy = fit_policy(impossible, np.array([0, -1]), 0.2, max_oos_accept=0)
    assert policy["reject_all"]
    assert policy["threshold"] is None
    assert not route_mask(impossible, 0.2, policy).any()


def test_empty_and_ambiguous_sets_abstain_even_with_zero_threshold():
    probs = np.array([[0.5, 0.5], [0.6, 0.4]])
    policy = {"reject_all": False, "threshold": 0}
    assert not route_mask(probs, 0.1, policy).any()
    assert not route_mask(probs, 1, policy).any()


def test_metrics_keep_zero_acceptance_explicit():
    logits = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    report = evaluate(logits, np.array([0, 1, -1]), 1, 1, {"reject_all": True, "threshold": None})
    assert report["calibrated"]["accuracy"] == 1
    assert report["routing"]["accepted_in_scope_accuracy"] is None
    assert report["routing"]["in_scope_coverage"] == 0


@pytest.mark.parametrize("values", [[[np.nan, 1]], [[np.inf, 0]], [[1, 2, 3], [1, 2, 3]]])
def test_invalid_probabilities_rejected(values):
    with pytest.raises(ValueError):
        prediction_sets(values, 0.5)


def test_invalid_labels_temperature_and_missing_oos_rejected():
    with pytest.raises(ValueError):
        nll(np.ones((2, 2)), np.array([-1, 0]))
    with pytest.raises(ValueError):
        probabilities(np.ones((2, 2)), 0)
    with pytest.raises(ValueError, match="both"):
        fit_policy(np.full((2, 2), 0.5), np.array([0, 1]), 0.5)
