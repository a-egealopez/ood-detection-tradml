import numpy as np

from detectors.vectorial.zscore_detector import ZScoreDetector


def _shifted_test(normal_rng_seed: int = 42):
    rng = np.random.default_rng(normal_rng_seed)
    normal = rng.standard_normal((100, 5))
    test = np.vstack(
        [rng.standard_normal((80, 5)), rng.standard_normal((20, 5)) + 6]
    )
    return normal, test


def test_scores_in_unit_range():
    normal, test = _shifted_test()
    det = ZScoreDetector(threshold=3.0)
    det.fit(normal)
    _, scores = det.predict(test)
    det._assert_unit_range(scores)


def test_detects_shifted_anomalies():
    normal, test = _shifted_test()
    det = ZScoreDetector(threshold=3.0)
    det.fit(normal)
    anomalies, _ = det.predict(test)
    assert int(anomalies[80:].sum()) > 10


def test_clean_points_low_score():
    rng = np.random.default_rng(0)
    normal = rng.standard_normal((100, 5))
    det = ZScoreDetector().fit(normal)
    _, scores = det.predict(normal[:10])
    assert scores.max() < 0.99
