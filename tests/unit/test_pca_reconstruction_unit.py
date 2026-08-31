import numpy as np

from detectors.vectorial.pca_reconstruction_detector import PCAReconstructionDetector


def _shifted_test():
    rng = np.random.default_rng(42)
    normal = rng.standard_normal((100, 5))
    test = np.vstack(
        [rng.standard_normal((80, 5)), rng.standard_normal((20, 5)) + 6]
    )
    return normal, test


def test_scores_in_unit_range():
    normal, test = _shifted_test()
    det = PCAReconstructionDetector(n_components=3, threshold_percentile=95)
    det.fit(normal)
    _, scores = det.predict(test)
    det._assert_unit_range(scores)


def test_detects_shifted_anomalies():
    normal, test = _shifted_test()
    det = PCAReconstructionDetector(n_components=3, threshold_percentile=95)
    det.fit(normal)
    anomalies, _ = det.predict(test)
    assert int(anomalies[80:].sum()) > 0


def test_projection_to_low_dim_keeps_positive_error():
    normal, _ = _shifted_test()
    det = PCAReconstructionDetector(n_components=3, threshold_percentile=95)
    det.fit(normal)
    errors = det._reconstruction_errors(normal)
    assert (errors >= 0).all()
