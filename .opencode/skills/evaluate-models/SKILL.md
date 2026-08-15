---
name: evaluate-models
description: Use when the user wants to run the CLI model evaluation or inspect detector metrics (e.g. "run evaluation", "evaluate the detectors", "check AUROC", "validate the models", "compare houses"). Covers the evaluation CLI, its arguments, and metric interpretation.
---

# Evaluate Models (CLI)

Runs the anomaly-detection ensemble per house and reports synthetic-injection metrics.

## Command

```bash
PYTHONPATH=src python scripts/run_evaluation.py [options]
```

Requires dependencies installed (`pip install -r requirements.txt`) and data loaded
(`python src/ingestion/casas_loader.py --source real`).

## Useful options

| Option | Default | Purpose |
|---|---|---|
| `--source real\|synthetic` | `real` | Which database to evaluate |
| `--extractor temporal\|next_event` | `temporal` | Feature extractor: `temporal` (9 daily features) or `next_event` (first-order Markov: log-probability of predicting the next sensor) |
| `--houses aruba cairo ...` | all | Houses to evaluate independently |
| `--train-split` | `0.7` | Fraction of days used for training (rest = holdout) |
| `--zscore-threshold` | `3.0` | Z-score cutoff for `ZScoreDetector` |
| `--iforest-contamination` | `0.05` | Contamination for `IsolationForestDetector` |
| `--pca-components` | `5` | Kept components for `PCAReconstructionDetector` |
| `--ensemble-threshold-percentile` | `90` | Percentile that separates normal vs anomaly |
| `--contamination` | `0.15` | Fraction of holdout rows perturbed |
| `--magnitude` | `6.0` | Anomaly magnitude in std deviations |
| `--output out.csv` | — | Save the report to a CSV |

Example:

```bash
PYTHONPATH=src python scripts/run_evaluation.py --source real --houses aruba cairo --output reports/real.csv
```

## What the CLI evaluates

Per house, with `--extractor temporal` (default): `TemporalFeatureExtractor` (9 daily
features) → `FeatureScaler` (z-score, fit on train) → `EnsembleDetector` over
`[ZScore, IsolationForest, PCAReconstruction]` → synthetic anomaly injection on the
holdout → precision / recall / F1 / AUROC.

With `--extractor next_event`: `NextEventTransitionExtractor` learns the normal
transition probabilities from the **train days only** (honest holdout), then scores
every day's sequence by log-likelihood (3 features: `mean_logprob`, `min_logprob`,
`rare_transition_rate`) before the same scaling / ensemble / injection steps.

Note: `next_event` targets *sequence* anomalies, so it will not match `temporal` on
feature-magnitude injection (which shifts numeric columns); a lower AUROC there is
expected and reflects a different anomaly axis, not a bug.

## Interpreting AUROC

- `> 0.75` good discrimination; `0.70–0.75` acceptable; `< 0.70` the ensemble is failing
  on the injected anomalies — tune contamination / components / magnitude.

## Note

- Houses with `< 10` days or holdouts `< 5` days are skipped (status column explains why).
- The sequential detectors (HMM, Hawkes) are **not** part of this CLI ensemble.
- ZScore and PCAReconstruction are exposed here and via the package API but are not yet
  in the Streamlit UI (see AGENTS.md roadmap).
