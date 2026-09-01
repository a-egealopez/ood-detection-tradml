---
name: generate-fixtures
description: Use when the user wants to generate or regenerate the synthetic CASAS test data (e.g. "generate fixtures", "regenerate synthetic data", "create test data", "make sample data"). Covers the fixture generator and how fixtures feed the app/CLI.
---

# Generate Synthetic CASAS Fixtures

Produces realistic CASAS-style event streams (timestamp, sensor_id, reading) into
`data/synthetic/`, one CSV per house, so the app and CLI work without the real datasets.

## Command

```bash
PYTHONPATH=src python scripts/generate_test_fixtures.py
```

`data/synthetic/` is gitignored; regenerate after a fresh clone.

## What it creates

Four CSVs named `casas_{house}_raw.csv`, one per house profile defined at the top of
`scripts/generate_test_fixtures.py`:

| House  | ~events/day | night ratio | days | seed |
|--------|-------------|-------------|------|------|
| aruba  | 180         | 0.08        | 60   | 1    |
| cairo  | 90          | 0.05        | 45   | 2    |
| milan  | 260         | 0.15        | 50   | 3    |
| tulum  | 130         | 0.10        | 40   | 4    |

Every profile also injects ~1/20 anomalous days (many more events + much higher night
activity) as an internal sanity signal.

## Loading the fixtures

Fixtures are just CSVs; they become usable by the app/CLI only after the loader runs:

```bash
python src/ingestion/casas_loader.py --source synthetic
```

This writes to `data/sensor_data_synthetic.db`. `scripts/run.sh --synthetic` does
generate + load + launch in one step.

## Schema expectations

The raw CSV is headerless with columns `date,time,sensor_id,reading` where `reading` is
`ON/OFF/OPEN/CLOSE`. `src/ingestion/casas_loader.py` converts readings to
`event_type` + `value` (1.0/0.0) and stores events in the `sensor_events` table.
