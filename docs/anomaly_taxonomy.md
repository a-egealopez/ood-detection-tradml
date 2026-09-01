# Anomaly Taxonomy and the Coherent-Evaluation Protocol

This document gives the **operational definitions** of the anomaly types used
throughout the project, the injection protocol that realizes them, and the
evaluation protocol that measures them. It accompanies the summary in the
[README](../README.md).

## 1. Why a taxonomy

The CASAS smart-home streams carry **no anomaly ground truth**. To evaluate an
unsupervised detector we must *generate* labeled anomalies. The generator is
deliberately **coherent** with the body of anomaly-detection literature (see the
Chandola et al. taxonomy in the [bibliography](../README.md#bibliography)): each
anomaly type targets a distinct invariant of a normal daily routine, so the
experiments show *which detector family senses which type of deviation*.

Three types are defined — **point**, **contextual**, **collective** — plus a
**control** (null) scenario.

## 2. Operational definitions

| Type | Injection | What changes | What is preserved |
|---|---|---|---|
| `point` | add a **night burst** (03:00–04:00) from one sensor on an anomalous day | daily volume (n_events), night activity | — (a volume anomaly has no invariant) |
| `contextual` | **circularly shift** the whole day's routine forward by `S` hours | the hourly distribution of events (the *context*) | total and per-sensor counts; the within-day sensor sequence |
| `collective` | **partially reverse** the intra-day sensor order | the transition structure (pairwise order statistics) | per-sensor counts **and** per-hour counts |
| `control` | inject nothing | — | — |

Each real type is **intensity-graded**:

- `point`: the burst size is a fraction of the day's event count
  (`POINT_INTENSITIES`), low / medium / high.
- `contextual`: the shift `S` in hours increases with intensity
  (`CONTEXTUAL_INTENSITIES`), low / medium / high.
- `collective`: the fraction of reversed events increases with intensity.

### Design invariant

**There is no feature-level injection.** Every anomaly is applied to the *raw event
stream* (`src/evaluation/event_injection.py`, exposed in the app via
`app/data_access.apply_injection`), and the daily features are **re-extracted** from the
injected stream. This keeps the evaluation honest: a detector can only exploit signals
that survive feature extraction from genuinely anomalous events.

## 3. Why these types

- **Point** — the *loud* day: an aggregate/volume signal. No invariant exists; the
  winning detectors are the distance/reconstruction family (Z-Score, Mahalanobis,
  Isolation Forest, PCA-Reconstruction). Order-based models should stay blind.
- **Contextual** — the *routine at the wrong hour*: aggregate statistics are untouched,
  only the temporal placement changes. The context carriers (hourly-distribution
  features) and the sequential models (HMM regime change, Hawkes intensity) react.
- **Collective** — the *sequence broken while counts stay fixed*: the same events, at
  the same hours, in a partially reversed order. Only the transition structure differs.
  A first-order Markov detector on transitions (here **Markov Sequence**) is the unique
  sensitive family.

## 4. Source-awareness constraint (collective)

The collective anomaly is only detectable by a first-order model if the reversed
transitions are **rare**, i.e. if the normal stream is *directional*:

- `transition_asymmetry` (`src/evaluation/event_injection.py`, mean per-pair
  min/max ratio over distinct sensor pairs, self-loops excluded):
  - synthetic houses ≈ **0.35** → directional, collective fully detectable;
  - real CASAS homes ≈ **0.95** → near-symmetric room-to-room movement, so a
    marginal-preserving reversal produces no rare transitions and is provably
    undetectable by first-order models.

Consequently the collective gate in `scripts/verify_pipeline.py` is **source-aware**:
it is a hard gate only when the transitions are directional; on symmetric data it is
reported as informational instead of failing.

## 5. The evaluation protocol

`src/evaluation/matrix_evaluation.py` runs the full **type × intensity × detector ×
seed** matrix:

- fixed **70/30** temporal holdout (detectors train only on clean head days);
- scaler (`FeatureScaler`) fit on the train days only — no look-ahead;
- anomalies injected on the raw stream of the holdout days (contiguous block);
- metric: **AUROC** against the injected labels, averaged over seeds;
- the HMM detector is seeded (`random_state`) so the matrix is deterministic.

Entry points:

| Purpose | Command |
|---|---|
| Per-house evaluation report | `python scripts/run_evaluation.py --source real` |
| Full coherence matrix | `python scripts/run_matrix.py --source synthetic` |
| All gates (pytest + matrix) | `python scripts/verify_pipeline.py --source synthetic` |

See the [README results section](../README.md#evaluation-results) for the measured
matrix and the verified winner structure.