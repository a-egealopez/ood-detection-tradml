---
name: run-app
description: Use when the user wants to launch, run, or test the Streamlit app (e.g. "run the app", "launch streamlit", "start the dashboard", "open the UI", "test the interface"). Covers venv bootstrap, data loading, flags, and app structure.
---

# Run the Streamlit App

## Quick start

On Linux/WSL:

```bash
scripts/run.sh
```

On Windows:

```bat
scripts\run.bat
```

Both scripts: (1) create/activate `venv/` if missing, (2) `pip install -r requirements.txt`,
(3) generate synthetic fixtures and/or load CASAS CSVs into SQLite, (4)
`streamlit run app/streamlit_app.py`.

Flags:

- `--synthetic`: generate + load the synthetic test data first, and default the app to it.
- `--skip-setup`: skip venv creation and pip install (when deps are already installed).

## Environment notes

- The root `venv/` exists and is **populated** (all dependencies installed, including
  `tick`). Use `venv/bin/python` / `venv/bin/streamlit` (or `scripts/run.sh`, which
  activates the venv). `streamlit` is **not** on the system PATH — never call it bare.
- `tick` is the fragile dependency (native C++ build). If `pip install` fails on it,
  install everything else first, then `pip install tick` last; if it still fails, the
  sequential Hawkes detector is the only consumer (skip it for UI work).
- A stale Windows venv was removed from git (`scripts/venv/`) — never recreate/commit it.

## App structure (entry point: `app/streamlit_app.py`)

The app is a guided 3-step workflow (`1 · Data → 2 · Features → 3 · Detect`):

- Step 1 (Data): choose the track — `2D Playground` (2-D decision boundaries on
  synthetic datasets) or `CASAS Smart Home` (with a Synthetic/Real toggle).
- Step 2 (Features): for CASAS, a guided walkthrough (origin → method → diagnostics)
  of the 3 event-driven extractors; for 2D data it is a trivial no-op.
- Step 3 (Detect): the 2D Playground detector grid, or the CASAS anomaly pipeline
  (`app/views/casas_view.py`) with auto-run and Guided/Advanced modes.
- Documentation & Concepts is always available as a sidebar expander.

The anomaly pipeline reads houses from the SQLite DB (`data/sensor_data.db` for real,
`data/sensor_data_synthetic.db` for synthetic), trains an ensemble on ~70% of daily
features, and evaluates with synthetic anomaly injection.

## Manual launch (deps ready)

```bash
PYTHONPATH=src streamlit run app/streamlit_app.py
```

Data must be loaded first: `python src/ingestion/casas_loader.py --source real|synthetic`.

## Troubleshooting

- "No data loaded" → run the loader (`scripts/run.sh` does this automatically).
- Detectors missing in the UI → only 10 of the 12 detectors are registered in
  `streamlit_config.py` + `build_detectors` (ZScore/PCAReconstruction are CLI-only).
- Theme is dark (`app` default from `.streamlit/config.toml`).
