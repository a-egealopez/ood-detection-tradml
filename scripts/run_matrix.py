"""CLI: run the full anomaly-type x intensity x detector evaluation matrix.

The matrix evaluation (``evaluation.matrix_evaluation``) checks the coherence
story of ``docs/anomaly_taxonomy.md`` across every house:

- point        -> event-level night burst; distance family wins
- contextual   -> HMM / Hawkes / Z-Score win, order family blind
- collective   -> Markov Sequence wins, everything else blind
- control      -> null control, AUROC must stay ~0.5

All three anomaly types are injected on the raw event stream (intensity
low/medium/high) and the features are re-extracted afterwards; there is no
feature-level injection. Scores are AUROC against the synthetic labels on the
holdout (split fixed 70/30, no look-ahead), repeated over ``n_seeds``. Writes
the per-seed matrix and the aggregated pivot to CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import db_path, setup_logging
from detectors.constants import DEFAULT_RANDOM_STATE
from evaluation.matrix_evaluation import (
    aggregate_matrix,
    monotonicity_check,
    run_matrix,
)
from ingestion.sqlite_manager import SQLiteDataManager

logger = setup_logging()

EXPECTED_WINNERS = {
    "point": ["Z-Score", "Mahalanobis", "Isolation Forest", "PCA Reconstruction"],
    "contextual": ["Z-Score", "HMM", "Hawkes"],
    "collective": ["Markov Sequence"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        choices=["real", "synthetic"],
        default="synthetic",
        help="Data source: 'real' (data/real/) or 'synthetic' (data/synthetic/)",
    )
    parser.add_argument(
        "--houses",
        nargs="+",
        default=None,
        help="Houses to evaluate (default: every house in the DB)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=10,
        help="Seeds per (type, intensity) cell; AUROC reported as mean over seeds",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="First seed; consecutive seeds seed_base .. seed_base+n_seeds-1 used",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fraction of days used for training (rest = holdout)",
    )
    parser.add_argument("--contamination", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="CSV path for the per-seed matrix (default: data/matrix_evaluation.csv)",
    )
    return parser.parse_args()


def pivot_table(matrix: pd.DataFrame) -> pd.DataFrame:
    agg = aggregate_matrix(matrix)
    return agg.pivot_table(
        index=["anomaly_type", "intensity"], columns="detector", values="auroc_mean"
    ).round(3)


def main():
    args = parse_args()

    db = SQLiteDataManager(str(db_path(args.source)))
    db.connect()
    available = db.list_houses()
    if not available:
        print(
            f"Database for source '{args.source}' is empty. Load data first:\n"
            f"  python src/ingestion/casas_loader.py --source {args.source}"
        )
        db.close()
        sys.exit(1)

    houses = args.houses or available
    unknown = set(houses) - set(available)
    if unknown:
        print(f"Skipping houses not in DB: {sorted(unknown)}")
        houses = [h for h in houses if h in available]

    print(f"Source: {args.source} ({db_path(args.source)})")
    print(f"Houses: {houses}, n_seeds={args.n_seeds}\n")

    frames = []
    for house_id in houses:
        df_house = db.query_house(house_id)
        if df_house.empty:
            print(f"[{house_id}] no data, skipped")
            continue
        matrix = run_matrix(
            df_house,
            house_id,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            train_split=args.train_split,
            contamination=args.contamination,
        )
        frames.append(matrix)
        print(f"[{house_id}] {len(matrix)} cell-rows done")
    db.close()

    if not frames:
        sys.exit(1)
    full = pd.concat(frames, ignore_index=True)

    pd.set_option("display.width", 220)
    print(pivot_table(full).to_string())

    print("\n== Monotonicity of expected winners (low -> medium -> high) ==")
    mono = monotonicity_check(aggregate_matrix(full), EXPECTED_WINNERS)
    all_ok = True
    for key, (ok, values) in mono.items():
        if ok is None:
            print(f"  {key}: skipped (not intensity-graded)")
        else:
            print(f"  {key}: {'OK' if ok else 'NOT MONOTONIC'} {values}")
            all_ok &= ok
    print(f"\nMonotonicity: {'all OK' if all_ok else 'FAILURES'}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(out_path, index=False)
        agg = aggregate_matrix(full)
        agg.to_csv(out_path.with_name(out_path.stem + "_agg.csv"), index=False)
        print(f"\nMatrix saved to {out_path} (+ _agg.csv)")


if __name__ == "__main__":
    main()