import numpy as np

from detectors.vectorial.iforest_detector import IsolationForestDetector


def _shifted_test():
    rng = np.random.default_rng(42)
    normal = rng.standard_normal((100, 5))
    test = np.vstack(
        [rng.standard_normal((80, 5)), rng.standard_normal((20, 5)) + 6]
    )
    return normal, test


def test_scores_in_unit_range():
    normal, test = _shifted_test()
    det = IsolationForestDetector(contamination=0.05)
    det.fit(normal)
    _, scores = det.predict(test)
    det._assert_unit_range(scores)


def test_single_row_score_in_unit_range():
    normal, test = _shifted_test()
    det = IsolationForestDetector(contamination=0.05)
    det.fit(normal)
    _, single_score = det.predict(test[:1])
    det._assert_unit_range(single_score)


def test_binary_anomalies():
    normal, test = _shifted_test()
    det = IsolationForestDetector(contamination=0.05)
    det.fit(normal)
    anomalies, _ = det.predict(test)
    assert set(np.unique(anomalies)).issubset({0, 1})
