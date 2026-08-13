import numpy as np

try:
    from tick.hawkes import HawkesSumExpKern
except ImportError:  # fragile native dependency; keep the module importable
    HawkesSumExpKern = None


class HawkesDetector:
    """Self-exciting point process for event streams (illustrative).

    The anomaly score is the per-day mean of the event features; the fitted
    Hawkes model is kept for didactic reference. tick is a fragile C++
    dependency and may be unavailable or incompatible, so its usage degrades
    gracefully to the mean-based score instead of crashing the detector.
    """

    def __init__(
        self,
        decay: float = 0.5,
        threshold_percentile: float = 90,
    ):
        self.decay = decay
        self.threshold_percentile = threshold_percentile
        self.model = None
        self._fitted = False
        self.threshold = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "HawkesDetector":
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]

        if HawkesSumExpKern is not None:
            try:
                self.model = HawkesSumExpKern([self.decay] * n_features)
                event_times = [
                    np.where(X[:, feat] > 0)[0].astype(float)
                    for feat in range(n_features)
                ]
                self.model.fit(event_times)
            except Exception:  # noqa: BLE001 - tick failures are non-fatal
                self.model = None

        # Score = mean event intensity per day (illustrative)
        scores_train = np.mean(X, axis=1)
        self.score_min = float(scores_train.min())
        self.score_max = float(scores_train.max())
        self.threshold = np.percentile(scores_train, self.threshold_percentile)
        self._fitted = True

        return self

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")

        X = np.asarray(X, dtype=float)
        scores = np.mean(X, axis=1)
        scores_norm = (scores - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores_norm = np.clip(scores_norm, 0.0, 1.0)

        anomalies = (scores > self.threshold).astype(int)

        return anomalies, scores_norm
