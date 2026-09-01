# AGENTS.md — OOD Detection with Classical ML (CASAS / dementia)

## Purpose

Research project on **unsupervised anomaly / out-of-distribution (OOD) detection** with
classical statistics & ML, targeting **discrete / event-driven time series** and
**dementia research**. It builds on the WSU CASAS smart-home datasets and evaluates an
ensemble of vectorial + sequential detectors.

Two learning tracks, both exposed in a single Streamlit app:

1. **Teaching track**: detect anomalies on synthetic 2-D datasets from scikit-learn
   (`make_blobs`, `make_moons`, `make_circles`, `make_swiss_roll`) and draw each
   detector's *real* decision boundary over a mesh.
2. **CASAS track**: ingest event streams (timestamp, sensor_id, event_type, value) from
   the WSU CASAS smart-home datasets (houses `aruba`, `cairo`, `milan`, `tulum`), extract
   daily feature vectors, and score them with an ensemble of vectorial + sequential
   detectors. There are **no real anomaly labels**, so evaluation uses **synthetic anomaly
   injection** on a holdout split (precision / recall / F1 / AUROC). The app exposes
   the same injection interactively on the synthetic CASAS track (Data step: anomaly
   scenario `control`/`point`/`contextual`/`collective` + intensity low/med/high,
   applied via `app/data_access.apply_injection` to the DB-fixture houses; the
   Ensemble results then show the injected days and per-detector AUROC).

## Language

- All code, comments, docstrings, and docs are written in **English**.
- Identifiers/APIs stay English. Legacy code still contains Spanish comments; translating
  it is an accepted, incremental refactor task.

## Stack & dependencies

- Python ≥3.10, Streamlit UI, Plotly charts (always use Plotly in Streamlit), scikit-learn,
  numpy, pandas, scipy, `hmmlearn` (sequential), `tick` (Hawkes — heavy/C++ build, fragile
  to install), `pyod` (LOF).
- Dependencies live in **both** `requirements.txt` and `pyproject.toml` (Poetry). Keep the
  two files in sync when adding/removing a dependency. No `yfinance`, no `python-dotenv`
  (removed as dead deps — do not re-add).

## Project layout

```
ood-detection-tradml/
├── app/                        # Streamlit UI (thin layer; no ML logic here)
│   ├── streamlit_app.py        # Entry point: guided 3-step workflow (Data->Features->Detect)
│   ├── streamlit_config.py     # Unified detector registry + UI defaults (single source)
│   ├── theme.py                # Shared Plotly theme (palette, family colors, cards/badges)
│   ├── components.py           # Reusable UI blocks (stepper, detector_card, badges, metrics)
│   ├── references.py           # Documentation & concepts sidebar content
│   ├── mesh.py                 # 2-D decision-boundary mesh helpers
│   ├── data_access.py          # Cached DB access helpers
│   └── views/                  # One module per view
│       ├── playground_view.py          # 2D Playground: decision-boundary visualizations
│       ├── feature_extraction_view.py  # Didactic view of the event-driven extractors
│       ├── casas_view.py               # CASAS track: sidebar config + auto-run + result tabs
│       └── data_view.py                # Data step: source selection + anomaly injection
├── src/                        # Library code (importable as top-level package via src/ on sys.path)
│   ├── config.py               # Paths, house/source constants, logging setup; also the single
│   │                           #   source for EPSILON+DEFAULT_RANDOM_STATE (no-dep layer so both
│   │                           #   features and detectors import them without a circular import)
│   ├── pipeline.py             # CASAS anomaly pipeline (extract, scale, ensemble, evaluate)
│   ├── detectors/
│   │   ├── __init__.py         # Public API: 12 detectors + EnsembleDetector
│   │   ├── factory.py          # Detector factory: name -> class, build_detector(s)
│   │   ├── ensemble.py         # EnsembleDetector (soft / hard voting)
│   │   ├── base.py             # BaseDetector: shared fit/predict boilerplate
│   │   ├── constants.py        # Shared detector constants (labels, threshold percentiles,
│   │   │                       #   train_split, contamination); re-exports EPSILON/DEFAULT_RANDOM_STATE
│   │   │                       #   from config.py (single source)
│   │   ├── vectorial/          # ZScore, IsolationForest (incl. Extended variant), Mahalanobis,
│   │   │                       # EllipticEnvelope, RobustCovariance, KNN, OC-SVM, LOF,
│   │   │                       # PCAReconstruction, classical_gaussian.py (MCD fallback)
│   │   └── sequential/         # HMMDetector, HawkesDetector, MarkovSequenceDetector
│   ├── features/               # common.py (FeatureScaler, entropy, daily_aggregates, extract_by_date;
│   │                           #   EPSILON re-exported from config), event_driven_extractors.py
│   ├── evaluation/             # metrics.py, event_injection.py (point/contextual/collective + control),
│   │                           #   matrix_evaluation.py
│   ├── ingestion/              # casas_loader.py (CLI CSV->SQLite), sqlite_manager.py, markov_generator.py (synthetic streams)
│   └── teaching/               # datasets.py (synthetic 2-D)
├── scripts/
│   ├── run.sh / run.bat        # venv bootstrap + load data + launch app
│   ├── run_evaluation.py       # CLI: evaluate detectors per house -> CSV report
│   ├── run_matrix.py           # CLI: anomaly-type x intensity x detector matrix (coherent-anomaly evaluation)
│   ├── verify_pipeline.py      # end-to-end verification gates (runs pytest + matrix, exit != 0 on failure)
│   └── generate_test_fixtures.py  # Generate synthetic CASAS-style CSVs into data/synthetic/
├── tests/                      # pytest suite
│   ├── conftest.py             # shared fixtures (Markov stream, house stream, samples)
│   ├── unit/                   # fast property tests (detectors, injectors, generator, scaler, ...)
│   └── functional/             # acceptance criteria / behavior (burst, regime change, reversal, smoke)
├── docs/                       # anomaly taxonnomy & coherent-evaluation protocol (anomaly_taxonomy.md)
├── data/                       # gitignored: real/, synthetic/, *.db (generated at runtime)
├── logs/                       # gitignored: app.log
├── .gitattributes              # *.py text eol=lf — keep line endings normalized
└── .opencode/skills/           # project workflow skills (run-app, evaluate-models, ...)
```

`data/` and `logs/` are gitignored. A fresh clone must generate fixtures and load data
before running the app (`scripts/run.sh` does both).

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

- **Pipeline** uses `TemporalFeatureExtractor` (`src/features/event_driven_extractors.py`):
  9 daily features (n_events, n_sensors, activity_hours, avg_event_gap_minutes, peak_hour,
  night_activity, event_frequency_std, entropy_hourly, entropy_sensor), then
  `FeatureScaler` (z-score, `src/features/common.py`, fit on train only).
- **Didactic tab** uses the event-driven extractors in
  `src/features/event_driven_extractors.py`:
  `TemporalFeatureExtractor`, `IntervalStatisticsExtractor` (point + collective,
  CV/Fano factor), `NGramTransitionExtractor` (first-order Markov, sequence collective),
  and `NextEventTransitionExtractor` (first-order Markov *prediction* — learns the normal
  transition probabilities and scores each transition by log-likelihood, so a single
  unlikely next-sensor flags a point anomaly; the "predict the next event and flag
  deviations" family from DeepLog / Chandola et al.). Each exposes `extract(df) -> (X, dates)`
  and `diagnostics(group)`.
- The didactic generator `generate_synthetic_events` draws sensors from an **asymmetric
  first-order Markov chain** (`sensor_chain_probabilities`, directed cycle `i → i+1`
  dominant, backward edge rare), so a collective reversal produces rare transitions and
  the order detectors have structure to learn. Caveat: `NGramTransitionExtractor`'s
  day-level features are invariant to an *exact* intra-day reversal (they depend only on
  the multiset of per-pair counts); the collective contrast shows in
  `NextEventTransitionExtractor` instead.

## Evaluation without labels

There is **no feature-level injection anywhere**: every anomaly type is injected on the
*raw event stream* in the Data step / `data_access.apply_injection`, and the daily
features are re-extracted afterwards.

**Coherent-anomaly evaluation** (see `docs/anomaly_taxonomy.md` for the operational
definitions and the measured matrix):
- `src/evaluation/event_injection.py` injects the three event-level anomaly types,
  each intensity-graded (low/medium/high):
  `inject_point_events` *adds* a night burst (3-4 AM) from one sensor on an anomalous
  day — a "loud" day whose aggregate features deviate (a volume anomaly has no
  invariants); `inject_contextual_events` circularly shifts an anomalous day's whole
  routine by `S` hours (total/per-sensor counts and the sensor sequence preserved;
  hourly distribution changes — the context signal); `inject_collective_events`
  partially reverses the intra-day sensor order (per-sensor AND per-hour counts
  preserved; only the transition structure changes). The null **control** injects
  nothing (AUROC must stay ~0.5).
- `src/evaluation/matrix_evaluation.py` runs the type × intensity × detector × seed matrix
  (AUROC on the fixed 70/30 holdout, shared views prepared once per cell): point → distance
  wins, contextual → Z-Score/HMM/Hawkes, collective → MarkovSequence, plus the `control`
  null. Entry points: `scripts/run_matrix.py` (report) and `scripts/verify_pipeline.py`
  (all gates, exit ≠ 0 on failure).
- The collective gate in `verify_pipeline.py` is **source-aware**: it is a hard gate only
  when the stream's transitions are directional. `transition_asymmetry`
  (`event_injection.py`, mean per-pair min/max ratio over *distinct* sensor pairs,
  self-loops excluded) is ~0.35 on the synthetic houses but ~0.95 on the real CASAS data;
  real homes move to/from each room nearly symmetrically, so a marginal-preserving
  reversal produces no rare transitions and is provably undetectable by first-order
  models. On symmetric data the collective gates are reported as informational instead of
  failing. The HMM detector is seeded (`random_state`) so the matrix is deterministic run
  to run.
- The synthetic fixtures are generated by `ingestion/markov_generator.py` (asymmetric
  directed-cycle movement graph + sticky latent day regime) precisely so the
  sequence/order detectors have structure to learn and the injectors produce rare
  transitions.

## Methodology notes

- **HawkesDetector** uses a real intensity-based score (negative conditional
  log-likelihood under an exponential-kernel Hawkes model, Ogata's forward recursion in
  numpy; `tick` is not required). The Hawkes Poisson intensity is valid only for counts,
  so the pipeline routes it the raw daily count features (`n_events`, `n_sensors`,
  `activity_hours`) via the ensemble's per-detector `detector_inputs`, never the z-scored
  continuous matrix.
- **hmmlearn** / **pyod** imports are lazy (`HMMDetector.fit`, `LOFDetector.fit`), so the
  core package imports without them.
- **MCD detectors** (EllipticEnvelope, RobustCovariance) fall back to a classical Gaussian
  on degenerate input (`vectorial/classical_gaussian.py`).
- **Percentile family** is centralized in `detectors/constants.py`
  (`DEFAULT_DETECTOR_THRESHOLD_PERCENTILE`, `DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE`,
  `contamination_percentile`, `DEFAULT_TRAIN_SPLIT`, `DEFAULT_CONTAMINATION`); EPSILON +
  DEFAULT_RANDOM_STATE are single-sourced in `config.py`.

## Commands

| Task | Command |
|---|---|
| Launch the app | `scripts/run.sh` (Linux/WSL) or `scripts/run.bat` (Windows) |
| Launch app only (deps ready) | `PYTHONPATH=src streamlit run app/streamlit_app.py` |
| CLI evaluation | `python scripts/run_evaluation.py --source real` (add `--extractor next_event` to use the Markov next-event extractor instead of the 9-feature temporal one) |
| Matrix evaluation | `python scripts/run_matrix.py --source synthetic` (writes per-seed matrix + aggregated pivot to CSV) |
| Verify pipeline gates | `python scripts/verify_pipeline.py` (runs pytest over generator/injector/detector tests + matrix; exit ≠ 0 on failure) |
| Run tests | `venv/bin/python -m pytest` (unit + functional suites under `tests/`) |
| Generate synthetic data | `python scripts/generate_test_fixtures.py` |
| Load CASAS data to SQLite | `python src/ingestion/casas_loader.py --source real\|synthetic` |

The root `venv/` is gitignored; **populate it on a fresh clone** with `scripts/run.sh`
(bootstraps venv + deps) or `pip install -r requirements.txt` inside the venv. `tick` is
the fragile dependency (native build); if it fails, the sequential Hawkes detector is the
only consumer — install it last.

## Conventions

- **Modularity**: UI lives in `app/`, ML/analysis logic in `src/`. Do not put model logic
  in Streamlit widgets; keep detectors/features/evaluation importable outside Streamlit.
- **Didactic clarity** over one-liners: readable, commented code that explains the method.
- **Plotly only** for Streamlit charts.
- **Package imports**: modules add `sys.path.insert(0, <project>/src)` at the top
  (legacy pattern). A proper package layout (`pip install -e .`) is a roadmap item.
- **Git hygiene**: `.gitattributes` enforces LF for Python files. Never commit venvs,
  `data/`, `logs/`, `.env`, or `audit.json`. Never commit `notes.txt` / `proyecto_completo.txt`.

## Validation workflow

After each code `edit`/`write`, run on the touched files:
```bash
venv/bin/ruff check <archivo> --fix && venv/bin/pyright <archivo>
```
If either fails, fix and re-run before continuing.

Full suite on demand:
```bash
venv/bin/ruff check . --fix       # lint + auto-fix
venv/bin/pyright                   # type check
venv/bin/vulture src/ app/ scripts/ --min-confidence 80  # dead code
venv/bin/pip-audit                 # security audit
```
