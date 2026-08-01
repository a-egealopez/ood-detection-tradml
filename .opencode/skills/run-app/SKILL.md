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

- The root `venv/` exists but is currently **empty**. The scripts install dependencies.
- `tick` is the fragile dependency (native C++ build). If `pip install` fails on it,
  install everything else first, then `pip install tick` last; if it still fails, the
  sequential Hawkes detector is the only consumer (skip it for UI work).
- A stale Windows venv was removed from git (`scripts/venv/`) — never recreate/commit it.

## App structure (entry point: `app/streamlit_app.py`)

Top-level radio `data_mode` selects between:

- `🎓 Teaching: Synthetic Datasets` → `teaching_tab.py` (2-D decision boundaries).
- `🏠 Synthetic CASAS Data` / `🏠 Real CASAS Data` → sub-radio between
  "Feature Extraction Tutorial" (`feature_extraction_tab.py`) and the anomaly pipeline
  (`_run_anomaly_pipeline(source)` in `streamlit_app.py`).
- `📚 Documentation` → static markdown.

The anomaly pipeline reads houses from the SQLite DB (`data/sensor_data.db` for real,
`data/synthetic/sensor_data.db` for synthetic), trains an ensemble on ~70% of daily
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
