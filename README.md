# Unsupervised Out-of-Distribution Detection on Smart-Home Event Streams

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/tests-68%20pytest-green.svg)](scripts/verify_pipeline.py)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An exploration of unsupervised anomaly / out-of-distribution (OOD) detection
with classical statistics and machine learning, applied to discrete,
event-driven time series from the WSU CASAS smart-home datasets.

The project evaluates an ensemble of vectorial and sequential detectors in a
domain with **no anomaly labels**, using a coherent synthetic-anomaly injection
protocol. It runs as a Streamlit application with a guided walkthrough: a 2-D
teaching track that plots each detector's real decision boundary, and a CASAS
track that scores daily feature vectors with the full ensemble.

## Quick start

```bash
git clone https://github.com/a-egealopez/ood-detection-tradml.git
cd ood-detection-tradml
scripts/run.sh          # bootstrap venv, load data, launch the app (Linux/WSL)
```

`run.sh` / `run.bat` auto-detect the data source: if real CASAS CSVs are present
under `data/real/` they are used; on a fresh clone (no `data/`) the app falls back
to the synthetic fixtures, so the clone works out of the box. Pass `--synthetic`
or `--real` to force either source explicitly.

## Quick links

| Resource | Link |
|---|---|
| Live app | [ood-detection-tradml.streamlit.app](https://ood-detection-tradml.streamlit.app/) |
| Development guide | [AGENTS.md](AGENTS.md) |
| Tests | [`tests/`](tests/) |
| License | [LICENSE](LICENSE) |

## Motivation

Previous work in image-based OOD detection typically relies on deep generative
models — diffusion, variational autoencoders, foundation-model embeddings. This
project stands on the opposite premise: it explores the **established classical
detectors** that those methods build on, and studies how they behave on a domain
that images do not have — **event-driven time series**.

Smart-home streams are a natural test bed: the sensor events carry no label of
"anomalous day", so every detector must be evaluated on its own. The CASAS
datasets, used for years in activity recognition and dementia research, provide
the structure; a Markov stream generator provides synthetic houses in which the
order-based detectors have something to learn.

## Two tracks in the app

### Teaching track (2-D geometry)

Scikit-learn datasets (`make_blobs`, `make_moons`, `make_circles`,
`make_swiss_roll`) with a known contamination rate. The playground draws each
detector's actual decision boundary on a mesh alongside the training points, so
the inductive bias of every method — axis-aligned bands for Z-Score, ellipses
for MCD, principal axes for PCA-reconstruction, iso-lines for Isolation Forest,
radial regions for OC-SVM — is directly observable.

### CASAS track (event-driven time series)

```
raw event stream (timestamp, sensor_id, event_type, value)
  → daily feature extraction (9 features/day)
  → z-score scaling (fit on train days only)
  → ensemble of vectorial + sequential detectors
  → anomaly injection (point / contextual / collective / control)
  → re-extraction, holdout scoring: precision / recall / F1 / AUROC
```

The four CASAS houses (`aruba`, `cairo`, `milan`, `tulum`) are available with
synthetic fixtures that the app self-provisions on first launch.

**Using the real CASAS data** (optional): the four houses
(`aruba`, `cairo`, `milan`, `tulum`) are published by the WSU CASAS project under
CC-BY-4.0 (Cook, 2025, DOI: [10.5281/zenodo.17180309](https://doi.org/10.5281/zenodo.17180309)).
`scripts/fetch_casas_data.py` downloads `new_labeled_data.zip`, converts the raw
motion/door streams into `data/real/casas_{aruba,cairo,milan,tulum}_raw.csv` in the
loader's schema, and drops activity labels and temperature readings. `data/real/` is
gitignored, so a clone never ships the data; `run.sh`/`run.bat` pick it up
automatically, or load it manually with `python src/ingestion/casas_loader.py --source real`.
The real homes cover 2009–2011 with 27–34 motion/door sensors per house.

## Feature extractors

`src/features/event_driven_extractors.py` exposes four extractors, each
targeting a different anomaly family:

| Extractor | Detects |
|---|---|
| `TemporalFeatureExtractor` | contextual + collective (day level, 9 features) |
| `IntervalStatisticsExtractor` | point (raw intervals, CV / Fano factor) + collective |
| `NGramTransitionExtractor` | collective / sequence (first-order Markov, entropy) |
| `NextEventTransitionExtractor` | point + collective — predicts the next sensor and scores each transition by log-likelihood (DeepLog-style) |

## Evaluation without labels

Real activity data carries no anomaly ground truth, so evaluation relies on
injection. Three anomaly types are injected on the **raw event stream** (never
on the features), and the daily features are re-extracted afterwards:

- **point** — a night burst of extra events from one sensor (a "loud" day);
- **contextual** — a whole day's routine circularly shifted by some hours (the
  resident behaved normally, just at the wrong time of day);
- **collective** — the intra-day sensor order partially reversed (same events,
  same hours, different sequence);
- **control** — nothing injected; AUROC must stay ≈ 0.5.

This follows the point / contextual / collective taxonomy of Chandola et al.
(2009) and is the reason the sequential detectors (HMM, Hawkes, Markov) are part
of the ensemble: a vectorial detector only sees marginal features, while a
sequential one catches order changes.

## Methodology

- **One contract for every detector**: `fit(X)` and
  `predict(X) -> (anomalies_0_1, scores_0_1)`; scores are normalized to [0, 1]
  (higher = more anomalous) so the score-summing ensemble stays well-behaved.
- **Scaler fit on train only**: `FeatureScaler` computes z-score statistics on
  the 70% clean training days — no look-ahead.
- **Fixed holdout, seeded pipeline**: 70/30 temporal split and a seeded HMM make
  the evaluation deterministic run to run.
- **Per-detector inputs**: each detector receives the features it is built for.
  The Hawkes detector, for instance, is routed only the raw daily count features
  (`n_events`, `n_sensors`, `activity_hours`), because a Poisson-intensity score
  is valid for counts, not for z-scored continuous values.
- **Source-aware gates**: the collective (order-reversal) gate is enforced only
  when the stream's transitions are directional. `transition_asymmetry` is ≈ 0.35
  on the synthetic houses and 0.68–0.83 on the real CASAS homes (aruba 0.68, cairo
  0.83, milan 0.73, tulum 0.80) — below the 0.85 directionality threshold, so the
  reversal injector does produce rare transitions on these real streams. On
  near-symmetric data the collective gates are reported as informational instead
  of failing.

## Results

Measured with `scripts/run_matrix.py --source synthetic --n-seeds 3` (AUROC on
the fixed 70/30 holdout; 210 cell-rows per house). The winner structure is
consistent across intensity levels:

| Anomaly type | Winners | Intuition |
|---|---|---|
| **point** | Z-Score, Mahalanobis, Isolation Forest, PCA ≈ **0.96–1.00** | a volume anomaly; distance and reconstruction families see it directly |
| **contextual** | Z-Score 1.00, Mahalanobis 0.98, PCA 0.96, HMM 0.97, Hawkes 0.83 | the routine moved in time; hourly-distribution and sequential models catch it |
| **collective** | **Markov Sequence 1.00**, rest ≤ 0.56 | counts preserved, only the transition structure changes — the order model is the unique winner |
| **control** | all ≈ **0.43–0.49** | no signal to leak; the pipeline is behaving |

`scripts/verify_pipeline.py --source synthetic` runs 31 gates (pytest
self-validation plus the matrix gates) and fails the run if any break.

## Detectors

13 detector classes compose the public API (15 registered variants in the factory —
OC-SVM is registered per kernel and Isolation Forest has an `sliced_path` extended
variant).

| Detector | Family | Basis |
|---|---|---|
| Z-Score | vectorial | thresholded univariate z-scoring |
| Mahalanobis | vectorial | covariance-aware distance to centroid |
| Elliptic Envelope / Robust Covariance | vectorial | minimum covariance determinant; Gaussian fallback on degenerate input |
| Isolation Forest / Extended IForest | vectorial | random-partition isolation; sliced paths for the extended variant |
| K Nearest Neighbors | vectorial | distance to k-th neighbor |
| OC-SVM (RBF / Linear / Poly) | vectorial | kernel support estimation |
| Local Outlier Factor | vectorial | local vs. neighbor density |
| PCA Reconstruction | vectorial | reconstruction error on the principal subspace |
| Hidden Markov Model | sequential | regime-change signal (seeded) |
| Hawkes | sequential | exponential-kernel self-exciting process (Ogata forward recursion in numpy) |
| Markov Sequence | sequential | first-order transition log-probability |

## Repository layout

```
app/        Streamlit UI (thin layer; no ML logic)
src/        detectors, features, evaluation, ingestion, teaching datasets
scripts/    run.sh / run.bat, data fetch, evaluation CLI, matrix CLI, verification gates
tests/      pytest suite (unit + functional)
```

## CLI

| Task | Command |
|---|---|
| Download real CASAS data | `python scripts/fetch_casas_data.py` (Zenodo, CC-BY-4.0) |
| Per-house evaluation report (real data) | `python scripts/run_evaluation.py --source real` |
| Type × intensity × detector matrix | `python scripts/run_matrix.py --source synthetic` |
| Verification gates (exit ≠ 0 on failure) | `python scripts/verify_pipeline.py --source synthetic` |
| Run tests | `venv/bin/python -m pytest` |

## Quality gates

- `venv/bin/ruff check . --fix` — lint
- `venv/bin/pyright` — strict type checking
- `venv/bin/vulture src/ app/ scripts/ --min-confidence 80` — dead code
- `venv/bin/pip-audit` — dependency security audit

## References

1. Chandola, V., Banerjee, A., and Kumar, V. **Anomaly Detection: A Survey.** *ACM Computing Surveys* 41(3), Article 15, 2009. doi:[10.1145/1541880.1541882](https://doi.org/10.1145/1541880.1541882).
2. Cook, D. J., Crandall, A. S., Thomas, B. L., and Krishnan, N. C. **CASAS: A Smart Home in a Box.** *IEEE Computer* 46(7):62–69, 2013. doi:[10.1109/MC.2012.328](https://doi.org/10.1109/MC.2012.328).
3. Cook, D. J. **CASAS Smart Home dataset (aruba, cairo, milan, tulum).** Zenodo, 2025. doi:[10.5281/zenodo.17180309](https://doi.org/10.5281/zenodo.17180309), CC-BY-4.0.
4. Liu, F. T., Ting, K. M., and Zhou, Z.-H. **Isolation Forest.** *Proc. 8th IEEE ICDM*, 413–422, 2008. doi:[10.1109/ICDM.2008.17](https://doi.org/10.1109/ICDM.2008.17).
5. Hawkes, A. G. **Spectra of Some Self-Exciting and Mutually Exciting Point Processes.** *Biometrika* 58(1):83–90, 1971. See also Ogata, Y., *Journal of the American Statistical Association* 83(401):9–27, 1988, for the forward-recursion likelihood this project implements in numpy.
6. Rabiner, L. R. **A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition.** *Proceedings of the IEEE* 77(2):257–286, 1989.
7. Du, M., Li, F., Zheng, G., and Srikumar, V. **DeepLog: Anomaly Detection and Diagnosis from System Logs through Deep Learning.** *Proc. ACM SIGSAC CCS*, 1285–1298, 2017. doi:[10.1145/3133956.3134015](https://doi.org/10.1145/3133956.3134015).

## License

MIT — see [LICENSE](LICENSE). Authors: [a-egealopez](https://github.com/a-egealopez).