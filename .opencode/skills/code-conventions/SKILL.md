---
name: code-conventions
description: Use when writing, editing, or refactoring any code in this project (Python, Streamlit, tests). Encodes language, modularity, the detector contract, the features API, plotting rules, packaging, and git hygiene. Read it before making code changes.
---

# Code Conventions

## Language

- All new code, comments, docstrings, and docs in **English**. Identifiers/APIs English.
- Translating legacy Spanish comments/docstrings is an accepted incremental refactor.

## Modularity

- UI lives in `app/`, ML/analysis logic in `src/`. Never put model/feature/eval logic
  inside Streamlit widgets; it must stay importable outside Streamlit.
- `app/streamlit_config.py` centralizes UI defaults, param ranges, and display order.
- Didactic clarity beats clever one-liners: readable, commented code that explains the
  method is a project goal.

## Detector contract (all detectors in `src/detectors/`)

```python
def fit(self, X: np.ndarray) -> "Self": ...
def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # returns (anomalies_binary_0_1, anomaly_scores_in_[0,1])
```

- `predict` **must** return scores normalized to [0, 1] (higher = more anomalous) using
  `score_min`/`score_max` captured in `fit`; the ensemble sums them, so unbounded scores
  break it. Clip to [0, 1].
- Raise `RuntimeError` if `predict` is called before `fit`.
- Export new detectors from `src/detectors/vectorial/__init__.py` and
  `src/detectors/__init__.py` (the public API used by the CLI and app).

## Features API (`src/features/`)

- Pipeline extractor: `TemporalFeatureExtractor.extract(df) -> (X, dates)` (9 daily
  features) + `FeatureScaler` (z-score, fit on train only).
- Didactic extractors (`event_driven_extractors.py`): each exposes `extract(df) ->
  (X, dates)` and `diagnostics(group)`. Do not let the two APIs drift further apart.

## Plotting

- **Plotly only** for Streamlit charts (interactivity). Use `st.plotly_chart`.
- Teaching/feature tabs use consistent templates (`plotly_white`, muted backgrounds) —
  reuse the existing helpers instead of restyling each chart.

## Packaging & dependencies

- Dependencies must be declared in **both** `requirements.txt` and `pyproject.toml`
  (Poetry) — keep them in sync.
- Modules currently use `sys.path.insert(0, <project>/src)` (legacy pattern). Keep the
  `src` layout consistent; replacing with `pip install -e .` is the roadmap target.
- Do not re-add `yfinance` or `python-dotenv` (removed as dead deps).

## Git hygiene

- `.gitattributes` enforces `*.py text eol=lf` — keep files LF; do not reintroduce CRLF
  noise (a bulk CRLF diff polluted the working tree before `.gitattributes` was added).
- Never commit venvs, `data/`, `logs/`, `.env`, `notes.txt`, or `proyecto_completo.txt`.

## Data & evaluation

- Real and synthetic data live in separate databases (`data/sensor_data.db` vs
  `data/synthetic/sensor_data.db`), never mixed.
- No real labels exist → always evaluate with synthetic anomaly injection
  (`src/evaluation/synthetic_injection.py`), train on train split only, scale on train
  stats only.
