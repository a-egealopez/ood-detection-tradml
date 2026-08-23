# Anomaly taxonomy (operational definitions)

This document gives a *code-verifiable* definition of the three anomaly types used in
the synthetic-injection evaluation. Each definition states what makes the anomaly
detectable *in principle*, which marginal statistics it must leave untouched, and which
family of methods should (and should not) catch it.

The evaluation has no real labels, so every anomaly type is injected. The injectors must
therefore respect the invariants below; otherwise a detector can win "by accident" (e.g.
a distance detector catching a sequence anomaly because the injection also moved a
feature it happens to watch) and the experiment loses validity.

## Definitions

### point

A single day whose aggregate activity is "loud" — a classic volume anomaly. Injected on
the *raw event stream*: a burst of extra events from one sensor is added at a fixed,
normally quiet hour band (3-4 AM) on each anomalous day.

- **Operational definition**: one day's aggregate features (`n_events`,
  `activity_hours`, `peak_hour`, `night_activity`) move several standard deviations
  from the training distribution, because the day's event count is inflated by a
  night burst.
- **Invariant**: none — changing the marginal values *is* the anomaly, so there is
  nothing to preserve (unlike contextual/collective).
- **Detectable by**: distance/vectorial detectors (Z-Score, Mahalanobis, Isolation
  Forest, PCA Reconstruction, ...), which is the point of the family. Because the
  anomaly exists in the raw stream, count/volume models (Hawkes per-hour, HMM on the
  count features) and even the next-event Markov model also react — there is no
  "must stay blind" constraint on point by design.
- **Injection**: `inject_point_events` (event-level night burst, intensity = extra
  events as a fraction of the day's own count: `POINT_INTENSITIES`).

### contextual

Values that are individually normal but anomalous *given the temporal context*
(time-of-day band, latent day regime). The context is encoded in the features
(`night_activity`, `peak_hour`, hourly counts), so conditioning on time is what makes
the anomaly visible.

- **Operational definition**: a day whose *total* activity, per-sensor counts and
  sensor *sequence* match the training distribution, but whose *clock time* is wrong —
  the resident's whole routine happens `S` hours later than usual (a circular shift of
  the day's timeline within the calendar date).
- **Invariants** (must hold for the injector, `inject_contextual_events`):
  - total events per day: unchanged;
  - per-sensor daily counts: unchanged;
  - sensor sequence: unchanged (the chronological order only *rotates*, so exactly one
    wrap transition differs) — this is what keeps the order detectors blind;
  - hourly distribution: **allowed** to change — this change *is* the anomaly.
- **Detectable by**: time-context features and models that condition on them —
  Z-Score on `night_activity`, the predictive HMM, Hawkes over per-hour counts.
- **Should NOT be caught by**: pure transition/order models (next-event Markov,
  n-gram), whose input sequence is effectively unchanged.
- **Measured**: full-vector distance detectors land well above the originally hoped
  0.6–0.7 (see the measured table below) — daily aggregate features encode context
  (`peak_hour`, `night_activity`, `entropy_hourly`) and those have tiny day-to-day
  variance, so even a 1 h shift is a large z-deviation. That is an expected consequence
  of encoding context as features, not a bug; the designated context winners
  (Z-Score, HMM, Hawkes) still lead on this type.

### collective

Values individually normal, but the joint pattern/order is anomalous. Requires modeling
the sequential dependency to be visible.

- **Operational definition**: a day whose per-hour and per-sensor counts are unchanged,
  but whose *temporal order of sensors* deviates from the learned transition structure
  (e.g. the intra-day sensor sequence is reversed or partially shuffled).
- **Invariants** (must hold for the injector):
  - per-sensor daily counts: unchanged;
  - per-hour counts: unchanged;
  - transition structure (order): **allowed** to change — this change *is* the anomaly.
- **Detectable by**: sequence/transition models — next-event Markov (DeepLog-style),
  n-gram pattern entropy, `MarkovSequenceDetector`.
- **Should NOT be caught by**: distance detectors (their 9 daily features are all
  invariant to intra-day order), the predictive HMM on those same features, and the
  Hawkes volume/timing model (its per-hour counts are unchanged). Expected ≈ 0.5 AUROC.
  Note that "reversing" the order only produces rare transitions when the movement
  graph is strongly asymmetric — this is why the synthetic generator (Fase 1) mandates
  asymmetric transition weights.

## Who should win what

| Type | Should win | Should be blind (≈ 0.5) |
|------|------------|--------------------------|
| point | distance/vectorial | — |
| contextual | Z-Score(`night_activity`), HMM (predictive), Hawkes (per-hour) | next-event / Markov (≈ 0.5); full-vector distance is *partial* (measured well above 0.5 — see below) |
| collective | next-event / MarkovSequence / n-gram | distance, HMM, Hawkes |

## Detector ↔ input view

| Detector | Consumes | Sensitive to |
|----------|----------|--------------|
| vectorial (9 daily features) | z-scored 9-feature matrix | point; partial contextual (features encode context) |
| HMM (predictive) | z-scored 9-feature matrix, causally conditioned | contextual, point |
| Hawkes | raw per-hour counts (24-dim) | contextual (time profile), point |
| MarkovSequence | per-day next-event log-probabilities | collective (order), point-in-sequence |

## Measured matrix (synthetic houses, 70/30 split, mean AUROC over 5 seeds)

From `scripts/run_matrix.py` on the 4 synthetic houses (aruba, cairo, milan, tulum),
pooled across houses. The Fase-4 gates (`scripts/verify_pipeline.py`) require the
"should win" columns to be ≥ 0.75 (point ≥ 0.85 at high intensity), the "should be
blind" columns to stay ≤ 0.65 (order family on contextual within 0.5 ± 0.15), the null
control to stay within 0.5 ± 0.12, and the winners to rise monotonically low → high.

| type / intensity | Z-Score | Mahalanobis | IForest | PCA | HMM | Hawkes | MarkovSeq |
|------------------|---------|-------------|---------|-----|-----|--------|-----------|
| point low        | 1.000   | 0.979       | 0.995   | 0.962 | 0.971 | 0.633  | 0.962     |
| point medium     | 1.000   | 0.979       | 1.000   | 0.962 | 0.971 | 0.647  | 0.999     |
| point high       | 1.000   | 0.979       | 1.000   | 0.962 | 0.971 | 0.662  | 1.000     |
| contextual low   | 0.717   | 0.853       | 0.706   | 0.885 | 0.674 | 0.552  | 0.518     |
| contextual med   | 1.000   | 0.979       | 0.846   | 0.962 | 0.960 | 0.707  | 0.520     |
| contextual high  | 1.000   | 0.979       | 0.845   | 0.962 | 0.971 | 0.790  | 0.529     |
| collective low   | 0.459   | 0.506       | 0.491   | 0.533 | 0.543 | 0.455  | 1.000     |
| collective med   | 0.459   | 0.506       | 0.491   | 0.533 | 0.543 | 0.455  | 1.000     |
| collective high  | 0.459   | 0.506       | 0.491   | 0.533 | 0.543 | 0.455  | 1.000     |
| control (null)   | 0.450   | 0.463       | 0.477   | 0.465 | 0.505 | 0.454  | 0.467     |

Reading: point is an aggregate-volume anomaly, so the distance family is perfect
(≥ 0.96) and HMM/MarkovSeq also see it (a 3 AM burst is loud in any view; Hawkes reacts
partially with graded intensity). Collective is only visible to MarkovSequence (everyone
else stays ~0.5 — the identical rows across intensities confirm the injector is truly
marginal-preserving); contextual is visible to the context family with graded intensity,
order models stay blind. The control row confirms the labels carry no residual signal.

## Measured on the real CASAS dataset (70/30 split, mean AUROC over 5 seeds)

Same matrix on the real downloaded CASAS houses (aruba 60 d, cairo 45 d, milan 50 d,
tulum 40 d; 4 sensors; 30-18-day holdouts). All 35 Fase-4 gates pass; the collective
gate is **informational** here (see below).

| type / intensity | Z-Score | Mahalanobis | IForest | PCA | HMM | Hawkes | MarkovSeq |
|------------------|---------|-------------|---------|-----|-----|--------|-----------|
| point low        | 0.967   | 0.859       | 0.919   | 0.869 | 0.863 | 0.613  | 0.324     |
| point medium     | 0.967   | 0.859       | 0.933   | 0.869 | 0.863 | 0.625  | 0.373     |
| point high       | 0.967   | 0.859       | 0.949   | 0.869 | 0.863 | 0.636  | 0.398     |
| contextual low   | 0.606   | 0.830       | 0.586   | 0.842 | 0.713 | 0.566  | 0.479     |
| contextual med   | 0.836   | 0.859       | 0.686   | 0.869 | 0.853 | 0.716  | 0.479     |
| contextual high  | 0.884   | 0.859       | 0.712   | 0.869 | 0.863 | 0.750  | 0.461     |
| collective low   | 0.431   | 0.438       | 0.427   | 0.456 | 0.440 | 0.518  | 0.488     |
| collective med   | 0.431   | 0.438       | 0.427   | 0.456 | 0.440 | 0.518  | 0.512     |
| collective high  | 0.431   | 0.438       | 0.427   | 0.456 | 0.440 | 0.518  | 0.497     |
| control (null)   | 0.474   | 0.478       | 0.434   | 0.528 | 0.448 | 0.502  | 0.531     |

Reading (real): the overall pattern survives on real data — distance wins point
(Z-Score/Mahalanobis/PCARecon ≥ 0.85; IForest 0.95, well past the 0.85 gate that the
old feature-level injection failed on the short tulum holdout) and contextual is caught
by Z-Score / HMM / Hawkes with graded intensity (Hawkes = 0.57 → 0.75, its whole design
range). The sequence model's *negative* AUROC on point (< 0.5) is an artifact of the
short holdout, not a designed axis: point carries no "must stay blind" invariant. Two
caveats:

- **Collective → MarkovSequence is blind on real data (0.50).** This is expected, not
  a regression: real CASAS transitions are near-symmetric — `transition_asymmetry`
  (mean per-pair min/max, self-loops excluded) is 0.97 aruba / 0.93 cairo / 0.99 milan
  / 0.95 tulum vs 0.35 synthetic. With marginal-preserving reversal and symmetric
  transitions, the reversed order produces no *rare* transitions, so any first-order
  model is provably unable to see the anomaly. The reversal injector only has signal
  when the movement graph is directional (the synthetic generator's deliberate design).
  `scripts/verify_pipeline.py` measures the asymmetry and downgrades the collective
  gates to informational when the stream is symmetric.
- **Monotonicity on real data is noisier** (short holdouts): the winners are monotone
  at n_seeds = 5 — HMM [0.713, 0.853, 0.863], Z-Score [0.606, 0.836, 0.884], Hawkes
  [0.566, 0.716, 0.750]. The HMM is seeded (`random_state`), so the matrix is
  deterministic run to run.

## Circularity statement

The `contextual` and `collective` anomalies are constructed to exploit structure (graph
asymmetry, latent day regime, hour bands) that was **deliberately introduced** into the
synthetic generator. The evaluation therefore demonstrates internal consistency of the
pipeline — each detector does what it is designed to do given the data-generation
design — but does **not** demonstrate generalization to unseen real behavioral
anomalies in real CASAS data.