import numpy as np


def _counts(y_true, y_pred) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for binary 0/1 labels, computed once."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return tp, fp, fn, tn


def precision(y_true, y_pred) -> float:
    tp, fp, _, _ = _counts(y_true, y_pred)
    if tp + fp == 0:
        return 0.0
    return float(tp / (tp + fp))


def recall(y_true, y_pred) -> float:
    tp, _, fn, _ = _counts(y_true, y_pred)
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
    tp, fp, fn, tn = _counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]])


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
