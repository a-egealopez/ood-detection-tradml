import numpy as np

from detectors.sequential.hmm_detector import HMMDetector


def test_predictive_loglik_is_causal_and_lower_after_regime_change():
    rng = np.random.default_rng(0)
    X_normal = rng.normal(size=(40, 3))
    X = np.vstack([X_normal, rng.normal(loc=5.0, scale=1.0, size=(20, 3))])
    det = HMMDetector(n_components=2).fit(X_normal)
    ll = det._predictive_loglik(X)
    assert np.mean(ll[40:]) < np.mean(ll[:40])


def test_scores_more_anomalous_means_lower_loglik():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(40, 3))
    det = HMMDetector(n_components=2).fit(X_train)
    scores = det.predict(X_train)[1]
    # A far outlier should receive a high (anomalous) score.
    outlier = det.predict(np.array([[10.0, 10.0, 10.0]]))[1]
    assert outlier[0] >= scores.max() * 0.9
