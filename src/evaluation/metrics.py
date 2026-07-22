import numpy as np
import pandas as pd


def precision(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp + fn == 0:
        return 0.0
    return float(tp / (tp + fn))


def f1(y_true, y_pred) -> float:
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def metrics_to_dataframe(metrics: dict) -> pd.DataFrame:
    row = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    return pd.DataFrame([row])


if __name__ == "__main__":
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])

    metrics = compute_metrics(y_true, y_pred)

    print(f" Precision: {metrics['precision']:.3f}")
    print(f" Recall:    {metrics['recall']:.3f}")
    print(f" F1:        {metrics['f1']:.3f}")
    print(f" Accuracy:  {metrics['accuracy']:.3f}")
    print(f" Confusion Matrix:\n{metrics['confusion_matrix']}")

    df = metrics_to_dataframe(metrics)
    print(f"\n✓ DataFrame:\n{df}")