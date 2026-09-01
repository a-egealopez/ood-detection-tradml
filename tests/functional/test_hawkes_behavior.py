import numpy as np

from detectors.sequential.hawkes_detector import HawkesDetector

NORMAL_COUNT = 60
N_FEATURES = 24


def test_burst_onset_spikes_score():
    """Acceptance: a sustained burst must spike the score at its onset."""
    rng = np.random.default_rng(0)
    hourly = rng.poisson(5, size=(NORMAL_COUNT, N_FEATURES)).astype(float)
    burst = rng.poisson(40, size=(2, N_FEATURES)).astype(float)
    tail = rng.poisson(5, size=(20, N_FEATURES)).astype(float)
    X = np.vstack([hourly, burst, tail])

    detector = HawkesDetector().fit(hourly)
    _, scores = detector.predict(X)

    burst_window = slice(NORMAL_COUNT, NORMAL_COUNT + len(burst))
    quiet_window = slice(NORMAL_COUNT + len(burst), len(X))
    assert scores[burst_window].max() >= scores[quiet_window].max()
    detector._assert_unit_range(scores)


def test_temporal_displacement_raises_score():
    """Acceptance: events moved from a busy hour into a quiet night hour must score higher."""
    rng = np.random.default_rng(0)
    # Reproduce the original module's single shared rng: the burst block advances
    # the generator so the displacement block sees the values it was validated with.
    n_normal = 60
    rng.poisson(5, size=(n_normal, N_FEATURES)).astype(float)
    rng.poisson(40, size=(2, N_FEATURES)).astype(float)
    rng.poisson(5, size=(20, N_FEATURES)).astype(float)

    normal_day = rng.poisson(5, size=(N_FEATURES,)).astype(float)
    displaced_day = normal_day.copy()
    moved = min(6, int(displaced_day[12]))
    displaced_day[12] -= moved
    displaced_day[3] += moved

    hourly = rng.poisson(5, size=(NORMAL_COUNT, N_FEATURES)).astype(float)
    X = np.vstack([hourly, displaced_day[None, :], hourly[:3]])
    detector = HawkesDetector().fit(hourly)
    _, scores = detector.predict(X)

    displaced_score = float(scores[NORMAL_COUNT])
    baseline_score = float(scores[NORMAL_COUNT + 1 :].mean())
    assert displaced_score > baseline_score
    detector._assert_unit_range(scores)
