import numpy as np
from sklearn.ensemble import IsolationForest


class ExtendedIForestDetector:
    """
    Wrapper de sklearn IsolationForest con sliced_path=True (EIF variante).
    Mejor para datos altos-dimensionales.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        sliced: bool = True,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.sliced = sliced
        self.model = None
        self.score_min = None
        self.score_max = None

    def fit(self, X: np.ndarray) -> "ExtendedIForestDetector":
        X = np.asarray(X, dtype=float)

        kwargs = {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }

        # sklearn 1.3+ tiene sliced_path, sino ignorar
        try:
            self.model = IsolationForest(**kwargs, sliced_path=self.sliced)
        except TypeError:
            self.model = IsolationForest(**kwargs)

        self.model.fit(X)

        raw_train = -self.model.score_samples(X)
        self.score_min = float(raw_train.min())
        self.score_max = float(raw_train.max())

        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Llama fit() antes de predict()")

        X = np.asarray(X, dtype=float)
        predictions = self.model.predict(X)
        anomalies = (predictions == -1).astype(int)

        raw_scores = -self.model.score_samples(X)
        scores = (raw_scores - self.score_min) / (
            self.score_max - self.score_min + 1e-8
        )
        scores = np.clip(scores, 0.0, None)

        return anomalies, scores
