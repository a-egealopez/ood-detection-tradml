import numpy as np

from detectors.sequential.hmm_detector import HMMDetector


def test_regime_change_is_flagged():
    """Acceptance: a predictive HMM flags an abrupt regime change (tail > head)."""
    rng = np.random.default_rng(0)
    n_regime_a = 40
    X_normal = rng.normal(loc=0.0, scale=1.0, size=(n_regime_a, 3))
    X_shift = rng.normal(loc=4.0, scale=1.0, size=(20, 3))
    X = np.vstack([X_normal, X_shift])

    detector = HMMDetector(n_components=2)
    detector.fit(X_normal)
    _, scores = detector.predict(X)

    tail_mean = float(scores[n_regime_a:].mean())
    head_mean = float(scores[:n_regime_a].mean())
    assert tail_mean > head_mean


def test_scores_in_unit_range():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(40, 3))
    X_test = rng.normal(size=(10, 3))
    detector = HMMDetector(n_components=2).fit(X_train)
    _, scores = detector.predict(X_test)
    detector._assert_unit_range(scores)
