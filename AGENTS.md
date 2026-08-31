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
   injection** on a holdout split (precision / recall / F1 / AUROC). The app exposes
   the same injection interactively on the synthetic CASAS track (Data step: anomaly
   scenario `control`/`point`/`contextual`/`collective` + intensity low/med/high,
   applied via `app/data_access.apply_injection` to the DB-fixture houses; the
   Ensemble results then show the injected days and per-detector AUROC).

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
│       └── casas_view.py               # CASAS track: sidebar config + auto-run + result tabs
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
│   │   ├── vectorial/          # ZScore, IsolationForest, ExtendedIForest, Mahalanobis,
│   │   │                       # EllipticEnvelope, RobustCovariance, KNN, OC-SVM, LOF,
│   │   │                       # PCAReconstruction, classical_gaussian.py (MCD fallback)
│   │   └── sequential/         # HMMDetector, HawkesDetector, MarkovSequenceDetector
│   ├── features/               # common.py (entropy, daily_aggregates; EPSILON re-exported from config), scaler.py, temporal_features.py (pipeline), event_driven_extractors.py (didactic)
│   ├── evaluation/             # metrics.py, event_injection.py (point/contextual/collective + control), matrix_evaluation.py
│   ├── ingestion/              # casas_loader.py (CLI CSV->SQLite), sqlite_manager.py, markov_generator.py (synthetic streams)
│   └── teaching/               # datasets.py (synthetic 2-D), visualization.py (plotly helpers)
├── scripts/
│   ├── run.sh / run.bat        # venv bootstrap + load data + launch app
│   ├── run_evaluation.py       # CLI: evaluate detectors per house -> CSV report
│   ├── run_matrix.py           # CLI: anomaly-type x intensity x detector matrix (coherent-anomaly evaluation)
│   ├── verify_pipeline.py      # end-to-end DoD gates (runs pytest + matrix, exit != 0 on failure)
│   └── generate_test_fixtures.py  # Generate synthetic CASAS-style CSVs into data/synthetic/
├── tests/                      # pytest suite (extracted from src/ `__main__` blocks)
│   ├── conftest.py             # shared fixtures (Markov stream, house stream, samples)
│   ├── unit/                   # fast property tests (detectors, injectors, generator, scaler, ...)
│   └── functional/             # acceptance criteria / behavior (burst, regime change, reversal, smoke)
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
- **Didactic tab** uses the event-driven extractors in
  `src/features/event_driven_extractors.py`:
  `WindowAggregationExtractor` (contextual + collective), `IntervalStatisticsExtractor`
  (point + collective, CV/Fano factor), `NGramTransitionExtractor` (first-order Markov,
  sequence collective), and `NextEventTransitionExtractor` (first-order Markov
  *prediction* — learns the normal transition probabilities and scores each transition
  by log-likelihood, so a single unlikely next-sensor flags a point anomaly; the
  "predict the next event and flag deviations" family from DeepLog / Chandola et al.).
  Each exposes `extract(df) -> (X, dates)` and `diagnostics(group)`.
- The didactic generator `generate_synthetic_events` now draws sensors from an **asymmetric
  first-order Markov chain** (`sensor_chain_probabilities`, directed cycle `i → i+1`
  dominant, backward edge rare), so a collective reversal produces rare transitions and
  the order detectors have structure to learn. Caveat: `NGramTransitionExtractor`'s
  day-level features are invariant to an *exact* intra-day reversal (they depend only on
  the multiset of per-pair counts); the collective contrast shows in
  `NextEventTransitionExtractor` instead.
- Known overlap: `WindowAggregationExtractor` largely duplicates `TemporalFeatureExtractor`.
  Consolidating them is a roadmap candidate.

## Evaluation without labels

There is **no feature-level injection anywhere**: every anomaly type is injected on the
*raw event stream* in the Data step / `data_access.apply_injection`, and the daily
features are re-extracted afterwards.

**Coherent-anomaly evaluation** (see `docs/anomaly_taxonomy.md` for the operational
definitions and the measured matrix):
- `src/evaluation/event_injection.py` injects the three event-level anomaly types,
  each intensity-graded (low/medium/high) with DoD proxies:
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
  (all DoD gates, exit ≠ 0 on failure).
- The collective gate in `verify_pipeline.py` is **source-aware**: it is a hard gate only
  when the stream's transitions are directional. `transition_asymmetry`
  (`event_injection.py`, mean per-pair min/max ratio over *distinct* sensor pairs,
  self-loops excluded) is ~0.35 on the synthetic houses but ~0.95 on the real CASAS data;
  real homes move to/from each room nearly symmetrically, so a marginal-preserving
  reversal produces no rare transitions and is provably undetectable by first-order
  models. On symmetric data the collective gates are reported as informational instead of
  failing. Measured: synthetic 35/35 gates pass; real 35/35 pass at `--n-seeds 5`
  (the event-level point burst raises IForest well past the 0.85 gate that the old
  feature-level injection failed on real data). The HMM detector is seeded
  (`random_state`) so the matrix is deterministic run to run.
- The synthetic fixtures (Fase 1) are generated by `ingestion/markov_generator.py`
  (asymmetric directed-cycle movement graph + sticky latent day regime) precisely so the
  sequence/order detectors have structure to learn and the injectors produce rare
  transitions.

## Commands

| Task | Command |
|---|---|
| Launch the app | `scripts/run.sh` (Linux/WSL) or `scripts/run.bat` (Windows) |
| Launch app only (deps ready) | `PYTHONPATH=src streamlit run app/streamlit_app.py` |
| CLI evaluation | `python scripts/run_evaluation.py --source real` (add `--extractor next_event` to use the Markov next-event extractor instead of the 9-feature temporal one) |
| Matrix evaluation | `python scripts/run_matrix.py --source synthetic` (writes per-seed matrix + aggregated pivot to CSV) |
| Verify pipeline gates | `python scripts/verify_pipeline.py` (all DoD gates: runs pytest over generator/injector/detector tests + matrix; exit ≠ 0 on failure) |
| Run tests | `venv/bin/python -m pytest` (unit + functional suites under `tests/`) |
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
2. **Consolidate feature extraction**: done — the daily-aggregation logic shared by
   `TemporalFeatureExtractor` (pipeline) and `WindowAggregationExtractor` (didactic) now
   lives in a single helper `daily_aggregates`/`extract_by_date` in `features/common.py`;
   the didactic 3-method split (`Window`, `Interval`, `NGram`) is preserved.
3. **Sequential detectors**: done — `HawkesDetector` uses a real intensity-based score
   (negative conditional log-likelihood under an exponential-kernel Hawkes model, Ogata's
   forward recursion in numpy; `tick` is no longer required) and `hmmlearn`/`pyod` imports
   are lazy (`HMMDetector.fit`, `LOFDetector.fit`), so the core package imports without them.
   The Hawkes Poisson intensity is valid only for counts, so the pipeline routes it the raw
   daily count features (`n_events`, `n_sensors`, `activity_hours`; see
   `TemporalFeatureExtractor.COUNT_FEATURE_NAMES` / `count_columns`) via the ensemble's
   per-detector `detector_inputs`, never the z-scored continuous matrix. Its UI card shows
   only the timeline (no PCA-plane score map) with an explanatory caption.
4. **CLI/UX parity**: done — ZScore and PCAReconstruction exposed in the UI via the
   unified `DETECTOR_REGISTRY` (`app/streamlit_config.py` + `src/detectors/factory.py`).
5. **Tests**: done — the self-validation `__main__` blocks (previously DoD gates) and the
   smoke/demo blocks were extracted from `src/` into a pytest suite under `tests/`
   (`unit/` for fast property tests, `functional/` for behavioral acceptance criteria),
   with shared fixtures in `conftest.py`. `pytest` is a dev dependency; `tests` are
   excluded from pyright's strict checking (tests deliberately poke private helpers like
   `_assert_unit_range`) and have `S101` disabled via `[tool.ruff.lint.per-file-ignores]`.
   `scripts/verify_pipeline.py` now runs the generator/injector/detector gates by invoking
   pytest over `tests/…/test_{markov_generator,injectors,hmm,hawkes,markov_sequence}*.py`
   instead of running each module's `__main__` as a subprocess; the matrix gates are
   unchanged. CI wiring is still pending.
6. **README**: write a proper project README (currently missing) once the refactor
   stabilizes.
7. **Cleanup**: done — `DETECTOR_*` config unified into a single `DETECTOR_REGISTRY`
   (`app/streamlit_config.py`) shared by the sidebar and the teaching track; dead code
   `src/teaching/visualization.py` removed; EIForest folded into `IsolationForestDetector`
   via a `sliced_path` param (fixed per registry variant).
8. **Language pass**: convert remaining Spanish comments/docstrings to English.
9. **Detector-stack consolidation**: done — HMM score normalization unified to the canonical
   `(score_min, score_max)` convention; NaN guard added at the `as_float_array` choke point;
   percentile family centralized (`DEFAULT_DETECTOR_THRESHOLD_PERCENTILE`,
   `DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE`, `contamination_percentile`, plus
   `DEFAULT_TRAIN_SPLIT`/`DEFAULT_CONTAMINATION`; EPSILON + DEFAULT_RANDOM_STATE
   single-sourced in `config.py`);
   `hmmlearn`/`pyod` made lazy; MCD detectors (EllipticEnvelope, RobustCovariance) fall back
   to a classical Gaussian on degenerate input (`vectorial/classical_gaussian.py`); teaching
   track gained overlays for OC-SVM (decision boundary), Z-Score (axis-aligned band),
   PCA Reconstruction (principal axes) and IForest (iso-lines).
10. **Coherent-anomaly evaluation**: done — the anomaly-type × intensity × detector matrix
      (`src/evaluation/event_injection.py` + `matrix_evaluation.py`, `docs/anomaly_taxonomy.md`,
      `scripts/run_matrix.py`, `scripts/verify_pipeline.py`). All three types are injected on
      the raw event stream (no feature-level injection anywhere): point is a nightly burst
      (`POINT_INTENSITIES` as a fraction of the day's count), contextual a whole-day circular
      time shift (`CONTEXTUAL_INTENSITIES` in hours), collective a partial intra-day reversal;
      the null scenario is the `control`. All 35 DoD gates pass on the synthetic houses
      (point → distance ≥ 0.85 at high with monotonic intensity, contextual →
      Z-Score/HMM/Hawkes ≥ 0.75 with monotonic intensity, collective → MarkovSequence ≥ 0.75
      with the rest ~0.5, control ~0.5). On real data the collective gate is informational
      (symmetric transitions, `transition_asymmetry` ≈ 0.95) and 35/35 gates pass
      (`--source real`, HMM seeded so the matrix is deterministic).

## Validation workflow (auto-check after edits)

After each `edit`/`write`, the agent runs:
```bash
venv/bin/ruff check <archivo> --fix && venv/bin/pyright <archivo>
```
If either fails, the agent fixes and re-runs before continuing.

Full suite on demand:
```bash
venv/bin/ruff check . --fix       # lint + auto-fix
venv/bin/pyright                   # type check
venv/bin/vulture src/ app/ scripts/ --min-confidence 80  # dead code (ocasional)
venv/bin/pip-audit                 # security audit (semanal / pre-release)
```

## OpenCode setup

- Context: this file (AGENTS.md) is the single source of truth for project context.
- Skills live in `.opencode/skills/` and are loaded on demand by matching their
  `description`. Current skills: `run-app`, `evaluate-models`, `generate-fixtures`,
  `code-conventions`.
- Config: `.opencode` project config is declared in `opencode.json` at the repo root.
- After editing skills, agents, or `opencode.json`, restart opencode for changes to apply.
