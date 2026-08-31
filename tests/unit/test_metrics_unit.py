import numpy as np
import pytest

from evaluation.metrics import (
    accuracy,
    compute_metrics,
    confusion_matrix,
    f1,
    precision,
    recall,
)

Y_TRUE = np.array([0, 0, 0, 1, 1, 0, 1, 0, 0, 1])
Y_PRED = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 1])


def test_compute_metrics_matches_manual():
    m = compute_metrics(Y_TRUE, Y_PRED)
    tp = int(np.sum((Y_TRUE == 1) & (Y_PRED == 1)))
    fp = int(np.sum((Y_TRUE == 0) & (Y_PRED == 1)))
    fn = int(np.sum((Y_TRUE == 1) & (Y_PRED == 0)))
    tn = int(np.sum((Y_TRUE == 0) & (Y_PRED == 0)))
    assert m["precision"] == pytest.approx(tp / (tp + fp))
    assert m["recall"] == pytest.approx(tp / (tp + fn))
    assert m["accuracy"] == pytest.approx((tp + tn) / len(Y_TRUE))
    assert m["f1"] == pytest.approx(2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"]))
    assert (m["confusion_matrix"] == np.array([[tn, fp], [fn, tp]])).all()


def test_confusion_matrix_layout():
    cm = confusion_matrix(Y_TRUE, Y_PRED)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == int(np.sum((Y_TRUE == 0) & (Y_PRED == 0)))  # tn
    assert cm[1, 1] == int(np.sum((Y_TRUE == 1) & (Y_PRED == 1)))  # tp


def test_precision_zero_when_no_positives_predicted():
    assert precision(Y_TRUE, np.zeros_like(Y_TRUE)) == 0.0


def test_recall_zero_when_no_positives():
    assert recall(Y_TRUE, np.zeros_like(Y_TRUE)) == 0.0


def test_f1_zero_when_no_predicted_positives():
    assert f1(Y_TRUE, np.zeros_like(Y_TRUE)) == 0.0


def test_accuracy_perfect_is_one():
    assert accuracy(Y_TRUE, Y_TRUE) == 1.0
