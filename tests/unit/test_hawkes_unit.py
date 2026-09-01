import numpy as np

from detectors.sequential.hawkes_detector import HawkesDetector

N_FEATURES = 24


def _fit_detector(rng=None, count=60):
    rng = rng or np.random.default_rng(0)
    hourly = rng.poisson(5, size=(count, N_FEATURES)).astype(float)
    return HawkesDetector().fit(hourly), hourly


def test_conditional_loglik_negated_higher_for_burst():
    """A high-activity (burst) day must yield a higher negated log-likelihood."""
    rng = np.random.default_rng(0)
    det, hourly = _fit_detector(rng)
    burst = rng.poisson(40, size=(2, N_FEATURES)).astype(float)
    X = np.vstack([hourly, burst])
    neg_ll = det._conditional_loglik(X)
    assert neg_ll[60:].mean() > neg_ll[:60].mean()


def test_scores_normalized_to_unit_range():
    det, hourly = _fit_detector()
    _, scores = det.predict(np.vstack([hourly, hourly]))
    det._assert_unit_range(scores)
