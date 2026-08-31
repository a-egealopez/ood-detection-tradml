import numpy as np
import pytest

from detectors.base import BaseDetector, _as_float_array
from detectors.constants import MAX_SCORE_TOLERANCE, minmax_normalize


class _MinimalDetector(BaseDetector):
    """Concrete detector for testing the shared BaseDetector behaviour."""

    def __init__(self):
        self.raw_scores = None
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "_MinimalDetector":
        X = self._to_float(X)
        self.raw_scores = X.sum(axis=1)
        self.score_min = float(self.raw_scores.min())
        self.score_max = float(self.raw_scores.max())
        self.threshold = float(np.percentile(self.raw_scores, 90))
        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.raw_scores is None:
            self._check_fitted("raw_scores")
        X = self._to_float(X)
        raw = X.sum(axis=1)
        scores = self._scores_to_unit(raw, self.score_min, self.score_max)
        return self._above_threshold(scores, self.threshold), scores


def test_scores_normalized_to_unit_range():
    det = _MinimalDetector()
    det.fit(np.array([[0.0], [2.0], [4.0]]))
    _, scores = det.predict(np.array([[0.0], [3.0], [8.0]]))
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0 + MAX_SCORE_TOLERANCE


def test_scores_clipped_to_unit_range():
    _, scores = _MinimalDetector().fit(np.arange(5)[:, None]).predict(np.arange(5, 10)[:, None])
    assert scores.max() <= 1.0


def test_predict_before_fit_raises_runtime_error():
    det = _MinimalDetector()
    with pytest.raises(RuntimeError):
        det.predict(np.array([[0.0], [1.0]]))


def test_finite_input_required():
    with pytest.raises(ValueError):
        _as_float_array(np.array([1.0, np.nan]))


def test_to_float_rejects_non_finite():
    det = _MinimalDetector()
    with pytest.raises(ValueError):
        det.fit(np.array([[1.0, np.inf]]))


def test_above_threshold_is_binary():
    result = _MinimalDetector._above_threshold(np.array([0.1, 0.9]), 0.5)
    assert (result == np.array([0, 1])).all()
    assert result.dtype in (np.int64, np.int32)


def test_minmax_normalize_scales_linearly():
    out = minmax_normalize(np.array([0.0, 5.0, 10.0]), 0.0, 10.0)
    assert np.allclose(out, np.array([0.0, 0.5, 1.0]))


def test_minmax_normalize_degenerate_returns_zeros():
    out = minmax_normalize(np.array([3.0, 3.0]), 3.0, 3.0)
    assert (out == 0.0).all()


def test_assert_unit_range_rejects_out_of_range():
    det = _MinimalDetector()
    with pytest.raises(ValueError):
        det._assert_unit_range(np.array([0.0, 1.5]))


def test_assert_unit_range_rejects_non_finite():
    det = _MinimalDetector()
    with pytest.raises(ValueError):
        det._assert_unit_range(np.array([0.0, np.nan]))


def test_to_binary_from_labels():
    result = _MinimalDetector._to_binary_from_labels(np.array([1, -1, 5, -1]))
    assert (result == np.array([0, 1, 0, 1])).all()
