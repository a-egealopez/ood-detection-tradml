import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import compute_metrics


def inject_synthetic_anomalies(
    X: np.ndarray,
    contamination: float = 0.1,
    magnitude: float = 6.0,
    random_state: int = 42,
):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape

    n_synthetic = max(1, int(round(n_samples * contamination)))
    synthetic_idx = rng.choice(n_samples, size=n_synthetic, replace=False)

    X_eval = X.copy()
    feature_std = X.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    for idx in synthetic_idx:
        n_to_perturb = rng.integers(1, n_features + 1)
        features_to_perturb = rng.choice(n_features, size=n_to_perturb, replace=False)
        signs = rng.choice([-1.0, 1.0], size=n_to_perturb)
        X_eval[idx, features_to_perturb] += signs * magnitude * feature_std[features_to_perturb]

    y_synthetic = np.zeros(n_samples, dtype=int)
    y_synthetic[synthetic_idx] = 1

    return X_eval, y_synthetic


def evaluate_with_synthetic_anomalies(
    ensemble,
    X_holdout: np.ndarray,
    contamination: float = 0.1,
    magnitude: float = 6.0,
    random_state: int = 42,
) -> dict:
    if len(X_holdout) < 5:
        raise ValueError("Se necesitan al menos 5 muestras de holdout para evaluar de forma estable")

    X_eval, y_synthetic = inject_synthetic_anomalies(
        X_holdout, contamination=contamination, magnitude=magnitude, random_state=random_state
    )

    y_pred, scores, _ = ensemble.predict(X_eval)

    metrics = compute_metrics(y_synthetic, y_pred)
    metrics["auroc"] = float(roc_auc_score(y_synthetic, scores))
    metrics["n_holdout"] = len(X_holdout)
    metrics["n_injected"] = int(y_synthetic.sum())
    metrics["contamination_injected"] = contamination
    metrics["magnitude_injected"] = magnitude

    return metrics


def describe_scores(scores: np.ndarray, anomalies: np.ndarray) -> dict:
    return {
        "n_samples": len(scores),
        "anomaly_rate": float(anomalies.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p99": float(np.percentile(scores, 99)),
    }


if __name__ == "__main__":
    from detectors import ZScoreDetector, IsolationForestDetector, PCAReconstructionDetector, EnsembleDetector

    np.random.seed(0)
    X_train = np.random.randn(120, 5)
    X_holdout = np.random.randn(40, 5)

    ensemble = EnsembleDetector(detectors=[
        ZScoreDetector(), IsolationForestDetector(), PCAReconstructionDetector(),
    ])
    ensemble.fit(X_train)

    metrics_weak = evaluate_with_synthetic_anomalies(ensemble, X_holdout, contamination=0.15, magnitude=1.0)
    metrics_strong = evaluate_with_synthetic_anomalies(ensemble, X_holdout, contamination=0.15, magnitude=8.0)

    print(f" AUROC anomalía sutil (magnitude=1.0): {metrics_weak['auroc']:.3f}")
    print(f" AUROC anomalía fuerte (magnitude=8.0): {metrics_strong['auroc']:.3f}")

    assert metrics_strong["auroc"] >= metrics_weak["auroc"]
    print(" Validación OK")