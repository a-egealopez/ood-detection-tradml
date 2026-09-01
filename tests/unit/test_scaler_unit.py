import numpy as np
import pytest

from features.common import FeatureScaler


def test_fit_transform_zero_mean_unit_variance():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5)) * 10 + 50
    X_scaled = FeatureScaler().fit_transform(X)
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-6)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        FeatureScaler().transform(np.zeros((5, 5)))


def test_transform_applies_train_stats():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 3)) * 2 + 10
    scaler = FeatureScaler().fit(X_train)
    out = scaler.transform(X_train)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-6)
