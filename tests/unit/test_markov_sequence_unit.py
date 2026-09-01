import numpy as np

from detectors.sequential.markov_sequence_detector import MarkovSequenceDetector


def test_raw_scores_negate_mean_logprob_column():
    det = MarkovSequenceDetector()
    X = np.array([[-1.0, -2.0, 0.0], [-0.5, -1.0, 0.0]])
    raw = det._raw_scores(X)
    # Negative log-probability column is negated so the score is positive.
    assert np.allclose(raw, np.array([1.0, 0.5]))


def test_fit_predict_unit_range():
    det = MarkovSequenceDetector()
    X = np.array([[-1.0, -2.0, 0.0], [-0.2, -0.3, 0.0], [-0.1, -0.2, 0.0]])
    det.fit(X)
    _, scores = det.predict(X)
    det._assert_unit_range(scores)


def test_predict_before_fit_raises():
    det = MarkovSequenceDetector()
    try:
        det.predict(np.array([[-1.0, -2.0, 0.0]]))
    except RuntimeError:
        return
    raise AssertionError("predict before fit should raise")
