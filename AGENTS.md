# AGENTS.md — OOD Detection with Classical ML (CASAS / dementia)

## Purpose

Didactic project to learn classical statistics & ML techniques for **unsupervised anomaly
/ out-of-distribution (OOD) detection**, built as career training for ML roles focused on
**discrete / event-driven time series** and **dementia research**.

Two learning tracks, both exposed in a single Streamlit app:

1. **Teaching track**: detect anomalies on synthetic 2-D datasets from scikit-learn
   (`make_blobs`, `make_moons`, `make_circles`, `make_swiss_roll`) and draw each
   detector's *real* decision boundary over a mesh.
2. **CASAS track**: ingest event streams (timestamp, sensor_id, event_type, value) from
   the WSU CASAS smart-home datasets (houses `aruba`, `cairo`, `milan`, `tulum`), extract
   daily feature vectors, and score them with an ensemble of vectorial + sequential
   detectors. There are **no real anomaly labels**, so evaluation uses **synthetic anomaly
   injection** on a holdout split (precision / recall / F1 / AUROC).

## Language

- All new code, comments, docstrings, and docs are written in **English**.
- Identifiers/APIs stay English. Legacy code still contains Spanish comments; translating
  it is an accepted, incremental refactor task.

## Stack & dependencies

- Python ≥3.10, Streamlit UI, Plotly charts (always use Plotly in Streamlit), scikit-learn,
  numpy, pandas, scipy, `hmmlearn` (sequential), `tick` (Hawkes — heavy/C++ build, fragile
  to install), `pyod` (LOF).
- Dependencies live in **both** `requirements.txt` and `pyproject.toml` (Poetry). Keep the
  two files in sync when adding/removing a dependency. No `yfinance`, no `python-dotenv`
  (removed as dead deps — do not re-add).
- There is no `README.md` yet (the original `pyproject.toml` referenced one; it was
  removed). Write one when a README is requested.

## Project layout

```
ood-detection-tradml/
├── app/                        # Streamlit UI (thin layer; no ML logic here)
│   ├── streamlit_app.py        # Entry point: guided 3-step workflow (Data->Features->Detect)
│   ├── streamlit_config.py     # Unified detector registry + UI defaults (single source)
│   ├── theme.py                # Shared Plotly theme (palette, family colors, cards/badges)
│   ├── components.py           # Reusable UI blocks (stepper, detector_card, badges, metrics)
│   ├── data_access.py          # Cached DB access helpers
│   └── views/                  # One module per view
│       ├── playground_view.py          # 2D Playground: decision-boundary visualizations
│       ├── feature_extraction_view.py  # Didactic view of the 3 event-driven extractors
│       ├── casas_view.py               # CASAS track: sidebar config + auto-run + result tabs
│       └── documentation_view.py       # Documentation content
├── src/                        # Library code (importable as top-level package via src/ on sys.path)
│   ├── config.py               # Paths, house/source constants, logging setup
│   ├── pipeline.py             # CASAS anomaly pipeline (extract, scale, ensemble, evaluate)
│   ├── detectors/
│   │   ├── __init__.py         # Public API: 12 detectors + EnsembleDetector
│   │   ├── factory.py          # Detector factory: name -> class, build_detector(s)
│   │   ├── ensemble.py         # EnsembleDetector (soft / hard voting)
│   │   ├── vectorial/          # ZScore, IsolationForest, ExtendedIForest, Mahalanobis,
│   │   │                       # EllipticEnvelope, RobustCovariance, KNN, OC-SVM, LOF,
│   │   │                       # PCAReconstruction
│   │   └── sequential/         # HMMDetector, HawkesDetector
│   ├── features/               # scaler.py, temporal_features.py (pipeline), event_driven_extractors.py (didactic)
│   ├── evaluation/             # metrics.py, synthetic_injection.py
│   ├── ingestion/              # casas_loader.py (CLI CSV->SQLite), sqlite_manager.py
│   └── teaching/               # datasets.py (synthetic 2-D), visualization.py (plotly helpers)
├── scripts/
│   ├── run.sh / run.bat        # venv bootstrap + load data + launch app
│   ├── run_evaluation.py       # CLI: evaluate detectors per house -> CSV report
│   └── generate_test_fixtures.py  # Generate synthetic CASAS-style CSVs into data/synthetic/
├── data/                       # gitignored: real/, synthetic/, *.db (generated at runtime)
├── logs/                       # gitignored: app.log
├── .gitattributes              # *.py text eol=lf — keep line endings normalized
└── .opencode/                  # opencode skills (see below)
```

`data/` and `logs/` are gitignored. A fresh clone must generate fixtures and load data
before running the app (see the `generate-fixtures` and `run-app` skills).

## Detector interface (important)

Every detector implements the same contract:

```python
def fit(self, X: np.ndarray) -> "Self": ...
def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # returns (anomalies_binary_0_1, anomaly_scores_in_[0,1])
```

- `predict` **always returns scores normalized to [0, 1]** (higher = more anomalous). The
  ensemble sums these across detectors, so unbounded scores (e.g. raw z-score) break it.
- Detectors raise `RuntimeError` if `predict` is called before `fit`.
- Scores are min-max normalized using `score_min` / `score_max` captured during `fit`.
- `EnsembleDetector` is exported from `src/detectors/__init__.py` (used by the evaluation
  CLI and the app via `from detectors import ...`).

## Feature extraction (CASAS track)

- **Pipeline** uses `TemporalFeatureExtractor` (`src/features/temporal_features.py`): 9
  daily features (n_events, n_sensors, activity_hours, avg_event_gap_minutes, peak_hour,
  night_activity, event_frequency_std, entropy_hourly, entropy_sensor), then
  `FeatureScaler` (z-score, fit on train only).
- **Didactic tab** uses the 3 event-driven extractors in
  `src/features/event_driven_extractors.py`:
  `WindowAggregationExtractor` (contextual + collective), `IntervalStatisticsExtractor`
  (point + collective, CV/Fano factor), `NGramTransitionExtractor` (first-order Markov,
  sequence collective). Each exposes `extract(df) -> (X, dates)` and `diagnostics(group)`.
- Known overlap: `WindowAggregationExtractor` largely duplicates `TemporalFeatureExtractor`.
  Consolidating them is a roadmap candidate.

## Evaluation without labels

`src/evaluation/synthetic_injection.py`:
1. Train on ~70% of scaled daily features, keep ~30% holdout.
2. Inject synthetic anomalies (±`magnitude` std on a subset of features, on ~`contamination`
   of holdout rows).
3. `precision` / `recall` / `f1` / `accuracy` (custom `src/evaluation/metrics.py`) + AUROC
   (sklearn) vs synthetic labels.
4. AUROC < 0.70 → the ensemble is failing on synthetic anomalies; tune parameters.

## Commands

| Task | Command |
|---|---|
| Launch the app | `scripts/run.sh` (Linux/WSL) or `scripts/run.bat` (Windows) |
| Launch app only (deps ready) | `PYTHONPATH=src streamlit run app/streamlit_app.py` |
| CLI evaluation | `python scripts/run_evaluation.py --source real` |
| Generate synthetic data | `python scripts/generate_test_fixtures.py` |
| Load CASAS data to SQLite | `python src/ingestion/casas_loader.py --source real\|synthetic` |

The root `venv/` exists and is **populated** (all dependencies installed, including
`tick`). If a fresh clone lacks it, run `scripts/run.sh` (bootstraps venv + deps) or
`pip install -r requirements.txt` inside the venv. `tick` is the fragile dependency
(native build); if it fails, the sequential Hawkes detector is the only consumer — install
it last and consider making sequential imports lazy.

## Communication & token efficiency

- **No greetings, sign-offs, or long intros/explanations.** Go straight to the code,
  command, or technical solution.
- **Return only the modified code block or a diff format**, never the whole file unless
  strictly necessary.
- Keep prose minimal unless the user explicitly asks for an explanation.

## Conventions

- **Modularity**: UI lives in `app/`, ML/analysis logic in `src/`. Do not put model logic
  in Streamlit widgets; keep detectors/features/evaluation importable outside Streamlit.
- **Didactic clarity** over one-liners: readable, commented code that explains the method.
- **Plotly only** for Streamlit charts.
- **Package imports**: modules add `sys.path.insert(0, <project>/src)` at the top
  (legacy pattern). Prefer that the `src` layout stay consistent; installing the package
  with `pip install -e .` is a roadmap candidate.
- **Git hygiene**: `.gitattributes` enforces LF for Python files. Never commit venvs,
  `data/`, `logs/`, or `.env`. Never commit `notes.txt` / `proyecto_completo.txt`.

## Refactoring roadmap (agreed with user)

Priorities are indicative; update this list as work progresses.

1. **Packaging**: replace `sys.path.insert` hacks with a proper package layout
   (`pyproject.toml` + `pip install -e .`), add `pytest`, `ruff`/`flake8`, type hints
   (`typing`/`py.typed`).
2. **Consolidate feature extraction**: merge `temporal_features.py` +
   `event_driven_extractors.py` into one coherent `features` API (the didactic 3-method
   split is worth keeping, but the pipeline/tutorial duplication should go).
3. **Sequential detectors**: `HawkesDetector` currently scores `mean(X, axis=1)` and does
   not use the fitted Hawkes model — either implement a real intensity-based score or
   document it clearly as illustrative. Make `tick`/`hmmlearn`/`pyod` imports lazy so the
   core package imports without them.
4. **CLI/UX parity**: done — ZScore and PCAReconstruction exposed in the UI via the
   unified `DETECTOR_REGISTRY` (`app/streamlit_config.py` + `src/detectors/factory.py`).
5. **Tests**: add unit tests for metrics, synthetic injection, feature extractors, and each
   detector; wire into `pytest` and CI.
6. **README**: write a proper project README (currently missing) once the refactor
   stabilizes.
7. **Cleanup**: done — `DETECTOR_*` config unified into a single `DETECTOR_REGISTRY`
   (`app/streamlit_config.py`) shared by the sidebar and the teaching track; dead code
   `src/teaching/visualization.py` is still pending removal.
8. **Language pass**: convert remaining Spanish comments/docstrings to English.

## OpenCode setup

- Context: this file (AGENTS.md) is the single source of truth for project context.
- Skills live in `.opencode/skills/` and are loaded on demand by matching their
  `description`. Current skills: `run-app`, `evaluate-models`, `generate-fixtures`,
  `code-conventions`.
- Config: `.opencode` project config is declared in `opencode.json` at the repo root.
- After editing skills, agents, or `opencode.json`, restart opencode for changes to apply.
