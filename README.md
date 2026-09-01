# Unsupervised Out-of-Distribution Detection on Event-Driven Smart-Home Time Series

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ood-detection-tradml.streamlit.app/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-68%20pytest-green.svg)](scripts/verify_pipeline.py)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Research exploration of **unsupervised anomaly / out-of-distribution (OOD) detection**
with classical statistics and machine-learning methods, applied to **discrete,
event-driven time series** typical of the CASAS smart-home datasets used in dementia
research. The project combines an ensemble of *vectorial* (distance-, density- and
reconstruction-based) and *sequential* (order- and intensity-based) detectors, and
evaluates them without ground-truth labels through a **coherent synthetic-anomaly
injection protocol**.

An interactive Streamlit application exposes both a **teaching track** (2-D decision
boundaries) and a full **CASAS track** (daily-feature extraction, ensemble scoring,
per-detector AUROC).

## Contents

- [Quick links](#quick-links)
- [Research objectives](#research-objectives)
- [Two tracks in the app](#two-tracks-in-the-app)
- [Methodology and design decisions](#methodology-and-design-decisions)
- [Detector library](#detector-library)
- [Evaluation results](#evaluation-results)
- [Repository layout](#repository-layout)
- [Installation and usage](#installation-and-usage)
- [Validation and quality gates](#validation-and-quality-gates)
- [Bibliography](#bibliography)
- [License](#license)

## Quick links

| Resource | Link |
|---|---|
| Interactive app | [Streamlit Community Cloud](#deploying-on-streamlit-community-cloud) |
| Contribution guidelines | [AGENTS.md](AGENTS.md) |
| Anomaly taxonomy & evaluation protocol | [`docs/anomaly_taxonomy.md`](docs/anomaly_taxonomy.md) |
| Pipelines and detectors | [`src/`](src/) |
| Test suite | [`tests/`](tests/) |

## Research objectives

1. **Unsupervised anomaly detection without labels.** The CASAS streams carry no anomaly
   ground truth, so standard supervised evaluation is impossible. This project instead
   injects **event-level anomalies** (not feature-level) on the raw stream and re-extracts
   daily features afterward, scoring the holdout with precision / recall / F1 / AUROC.
2. **Cover the full Chandola taxonomy.** The injection protocol spans **point** (night
   volume bursts), **contextual** (whole-routine temporal shift) and **collective** (order
   reversal with preserved marginals) anomalies, plus a **null control** that must keep
   AUROC ≈ 0.5.
3. **Didactic visualization of detector behavior.** Each detector's real decision boundary
   is drawn over a dense mesh on the 2-D teaching track, so the geometry behind each
   classical method is explicit.
4. **Back the demo with a reproducible, verifiable protocol.** A fixed 70/30 holdout, a
   seed-aware pipeline, and an end-to-end verification script (`scripts/verify_pipeline.py`)
   make every reported number reproducible.

## Two tracks in the app

### Teaching track (2-D geometry)

Synthetic 2-D datasets generated with scikit-learn (`make_blobs`, `make_moons`,
`make_circles`, `make_swiss_roll`) with a known contamination rate. The playground plots
each detector's **actual decision boundary** on a mesh alongside the training points, so
the inductive bias of every method (axis-aligned bands for Z-Score, ellipses for MCD,
principal axes for PCA-reconstruction, iso-lines for IForest, radial regions for OC-SVM)
is directly observable.

### CASAS track (event-driven time series)

Pipeline over the WSU CASAS houses (`aruba`, `cairo`, `milan`, `tulum`):

```
raw event stream (timestamp, sensor_id, event_type, value)
  → daily feature extraction (9 features/day)
  → z-score scaling (fit on train days only)
  → ensemble of vectorial + sequential detectors
  → anomaly injection (point / contextual / collective / control)
  → re-extraction, holdout scoring: precision / recall / F1 / AUROC
```

The Data step lets the user pick the anomaly scenario and intensity interactively and
applies it to the synthetic CASAS fixtures in the running app.

## Methodology and design decisions

### 1. Unified detector contract

Every detector implements the same interface:

```python
def fit(self, X: np.ndarray) -> "Self": ...
def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # (anomaly_mask_0_1, anomaly_scores_normalized_to_[0,1])
```

- `predict` *always* returns scores normalized to **[0, 1]** (higher = more anomalous),
  using `score_min` / `score_max` captured during `fit`. This is what makes a
  **score-summing ensemble** well-behaved: an unbounded score (e.g. raw z-score) would
  dominate the vote.
- Detectors raise `RuntimeError` on `predict` before `fit`.
- `EnsembleDetector` sums per-detector scores (soft) or votes hard (majority) with
  configurable weights and a percentile threshold.

### 2. Event-level anomaly injection (no feature-level cheating)

Every anomaly type is injected on the **raw event stream**, then the daily features are
re-extracted from the injected stream. Nothing is perturbed at feature level.

- `point` — adds a night burst (3–4 AM) from one sensor on an anomalous day (**volume**
  signal; no invariant exists).
- `contextual` — circularly shifts a whole day's routine by `S` hours (total and
  per-sensor counts, and the sensor sequence, are preserved; only the **hourly context**
  changes).
- `collective` — partially reverses the intra-day sensor order (**per-sensor and per-hour
  counts are preserved**; only the transition structure changes).
- `control` — injects nothing; a correct detector must stay at AUROC ≈ 0.5.

### 3. Structural integrity of the evaluation

- **Scaler fit on train only**: `FeatureScaler` (z-score in `features/common.py`) learns
  `mu`/`sigma` on the 70% clean days — no look-ahead.
- **Fixed 70/30 holdout** and **seeded HMM** make the evaluation deterministic run to run.
- **Per-detector inputs**: the ensemble routes each detector through a `detector_inputs`
  specification. Notably the **Hawkes detector receives only the raw daily count features**
  (`n_events`, `n_sensors`, `activity_hours`) since a Poisson/Hawkes intensity is only
  valid for counts, never for z-scored continuous values.
- **Source-aware gates**: the collective (order-reversal) gate is only enforced when the
  stream's transitions are *directional*. `transition_asymmetry` ≈ 0.35 on synthetic
  houses but ≈ 0.95 on the real CASAS homes (near-symmetric room-to-room movement), where
  a marginal-preserving reversal produces no rare transitions and is provably undetectable
  by first-order models.

### 4. Feature-extraction design

Four event-driven extractors in `src/features/event_driven_extractors.py`, each targeting
a different anomaly family (interface `extract(df) -> (X, dates)` + `diagnostics(group)`):

| Extractor | Anomaly family targeted |
|---|---|
| `TemporalFeatureExtractor` | contextual + collective (day level, 9 features) |
| `IntervalStatisticsExtractor` | point (raw intervals, CV / Fano factor) + collective |
| `NGramTransitionExtractor` | collective / sequence (first-order Markov, entropy) |
| `NextEventTransitionExtractor` | point (single event) + collective — predicts the next sensor and scores each transition by log-likelihood (DeepLog-style) |

The didactic generator `generate_synthetic_events` draws sensors from an **asymmetric
first-order Markov chain** (directed cycle `i → i+1` dominant) so a collective reversal
produces *rare* transitions and the order detectors have structure to learn.

### 5. Deterministic, reproducible science

- `DEFAULT_RANDOM_STATE = 42` and `EPSILON` are single-sourced in `src/config.py`.
- Percentile thresholds, contamination and train-split are centralized in
  `src/detectors/constants.py`.
- `hmmlearn` and `pyod` imports are lazy (`HMMDetector.fit`, `LOFDetector.fit`), so the
  core package imports without them; the heavy `tick` dependency is optional and reserved
  for the sequential Hawkes detector.

## Detector library

| Detector | Family | Theoretical basis |
|---|---|---|
| Z-Score | vectorial, statistical | thresholded univariate z-scoring per feature |
| Isolation Forest | vectorial, isolation | Liu, Ting & Zhou (2008) |
| Extended IForest | vectorial, isolation | sliced-path extension |
| Mahalanobis | vectorial, distance | Mahalanobis distance to centroid |
| Elliptic Envelope | vectorial, covariance | MCD / robust covariance (Rousseeuw & Van Driessen, 1999) |
| Robust Covariance | vectorial, covariance | classical MCD; Gaussian fallback on degenerate input |
| K Nearest Neighbors | vectorial, distance | distance to k-th neighbor |
| One-Class SVM | vectorial, kernel | Schölkopf et al. (2001) support estimation |
| Local Outlier Factor | vectorial, density | Breunig et al. (2000) local density deviation |
| PCA Reconstruction | vectorial, reconstruction | reconstruction error on principal subspace |
| Hidden Markov Model | sequential | Rabiner (1989) HMM; regime-change signal |
| Hawkes | sequential, point process | exponential-kernel Hawkes intensity (Ogata forward recursion) |
| Markov Sequence | sequential, order | first-order transition log-probability |

## Evaluation results

Measured with `scripts/run_matrix.py --source synthetic --n-seeds 3` (AUROC on the fixed
70/30 holdout; 210 cell-rows per house). Consistent winner structure across intensity
levels:

| Anomaly type | Expected winners | Control (blind) |
|---|---|---|
| **point** | Z-Score, Mahalanobis, Isolation Forest, PCA-Reconstruction ≈ **0.96–1.00** at all intensities | Hawkes moderately blind (≈ 0.68–0.69) |
| **contextual** | Z-Score 1.00, PCA 0.96, Mahalanobis 0.98, HMM 0.97, Hawkes 0.83 at high | Markov Sequence ~ 0.49 (blind) |
| **collective** | **Markov Sequence 1.00** at all intensities | all others ≤ 0.56 (blind) |
| **control** | all detectors ≈ **0.43–0.49** (null) | — |

Key readings:
- **Point** anomalies are volume signals, so distance/reconstruction/reconstruction
  detectors all perform; order-based models stay blind.
- **Contextual** anomalies move the routine in time; Z-Score on the hourly-distribution
  features and the sequential HMM/Hawkes capture the shift.
- **Collective** anomalies keep all counts identical and only break transitions —
  **Markov Sequence** is the unique winner, confirming the first-order order-signal is
  the right lever.
- The **control** stays ≈ 0.5, confirming the pipeline is not leaking signal into the
  null scenario.

`scripts/verify_pipeline.py --source synthetic` confirms **31 / 31 gates pass**
(monotonicity of expected winners low→high, blind detectors near 0.5, validation panel,
matrix gates).

## Repository layout

```
● app/          Streamlit UI (thin layer; no ML logic)
● src/          Detectors, features, evaluation, ingestion, teaching datasets
  ├─ detectors/    12 vectorial + sequential detectors, factory, ensemble
  ├─ features/     FeatureScaler, daily aggregates, event-driven extractors
  ├─ evaluation/   metrics, event injection, matrix evaluation
  ├─ ingestion/    CASAS loader, SQLite manager, Markov stream generator
  └─ teaching/     synthetic 2-D dataset generator
● scripts/      run.sh / run.bat, evaluation CLI, matrix CLI, verify gates
● tests/        pytest suite (unit + functional)
```

## Installation and usage

### Local

```bash
scripts/run.sh                 # venv bootstrap + data load + launch app (Linux/WSL)
scripts/run.bat                # same (Windows)
```

With dependencies already ready:

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

### CLI evaluation

```bash
# Per-house evaluation report (real CASAS data)
python scripts/run_evaluation.py --source real

# Full anomaly-type × intensity × detector matrix (synthetic houses)
python scripts/run_matrix.py --source synthetic

# End-to-end verification gates (exit ≠ 0 on failure)
python scripts/verify_pipeline.py --source synthetic
```

### Tests

```bash
venv/bin/python -m pytest            # unit + functional suites
```

### Deploying on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **Create app** → select the repo → main file **`app/streamlit_app.py`** → **Deploy**.
4. The app self-provisions its synthetic CASAS fixtures on first run (see
   `ensure_synthetic_db` in `app/data_access.py`), so the CASAS track works without
   committing any data.
5. Your app is live at `https://ood-detection-tradml.streamlit.app/` (customize the
   subdomain in App settings).

## Validation and quality gates

- `venv/bin/ruff check . --fix` — lint.
- `venv/bin/pyright` — strict type checking.
- `venv/bin/vulture src/ app/ scripts/ --min-confidence 80` — dead code.
- `venv/bin/pip-audit` — dependency security audit.
- `scripts/verify_pipeline.py` — runs the pytest gates and the coherent-anomaly matrix,
  exiting non-zero on any failure.

## Bibliography

1. Chandola, V., Banerjee, A., and Kumar, V. **Anomaly Detection: A Survey.** *ACM
   Computing Surveys* 41(3), Article 15, 2009. DOI: [10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882).
2. Cook, D. J., Crandall, A. S., Thomas, B. L., and Krishnan, N. C. **CASAS: A Smart Home
   in a Box.** *IEEE Computer* 46(7):62–69, 2013. DOI: [10.1109/MC.2012.328](https://doi.org/10.1109/MC.2012.328).
3. Liu, F. T., Ting, K. M., and Zhou, Z.-H. **Isolation Forest.** *Proc. 8th IEEE
   International Conference on Data Mining (ICDM)*, 413–422, 2008. DOI:
   [10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
4. Rousseeuw, P. J., and Van Driessen, K. **A Fast Algorithm for the Minimum Covariance
   Determinant Estimator.** *Technometrics* 41(3):212–223, 1999.
5. Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., and Williamson, R. C.
   **Estimating the Support of a High-Dimensional Distribution.** *Neural Computation*
   13(7):1443–1471, 2001.
6. Breiman, L. **Random Forests.** *Machine Learning* 45(1):5–32, 2001.
7. Hawkes, A. G. **Spectra of Some Self-Exciting and Mutually Exciting Point Processes.**
   *Biometrika* 58(1):83–90, 1971.
8. Rabiner, L. R. **A Tutorial on Hidden Markov Models and Selected Applications in Speech
   Recognition.** *Proceedings of the IEEE* 77(2):257–286, 1989.
9. Du, M., Li, F., Zheng, G., and Srikumar, V. **DeepLog: Anomaly Detection and Diagnosis
   from System Logs through Deep Learning.** *Proc. ACM SIGSAC CCS*, 1285–1298, 2017.
   DOI: [10.1145/3133956.3134015](https://doi.org/10.1145/3133956.3134015).
10. Breunig, M. M., Kriegel, H.-P., Ng, R. T., and Sander, J. **LOF: Identifying Density-
    Based Local Outliers.** *ACM SIGMOD Record* 29(2):93–104, 2000.

## License

MIT. See [LICENSE](LICENSE).