import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.metrics import roc_auc_score

from config import MIN_DAYS, db_path, setup_logging
from detectors import (
    EnsembleDetector,
    IsolationForestDetector,
    PCAReconstructionDetector,
    ZScoreDetector,
)
from detectors.constants import (
    DEFAULT_CONTAMINATION,
    DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TRAIN_SPLIT,
)
from evaluation.event_injection import (
    INTENSITY_PRESETS,
    inject_collective_events,
    inject_contextual_events,
    inject_point_events,
    select_anomaly_dates,
)
from evaluation.metrics import compute_metrics
from features import (
    FeatureScaler,
    NextEventTransitionExtractor,
    TemporalFeatureExtractor,
)
from features.common import truncate_stream_to_days
from ingestion.sqlite_manager import SQLiteDataManager

logger = setup_logging()

INJECTORS = {
    "point": inject_point_events,
    "contextual": inject_contextual_events,
    "collective": inject_collective_events,
}


def describe_scores(scores: np.ndarray, anomalies: np.ndarray) -> dict:
    return {
        "n_samples": len(scores),
        "anomaly_rate": float(anomalies.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "score_p90": float(np.percentile(scores, 90)),
        "score_p99": float(np.percentile(scores, 99)),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        choices=["real", "synthetic"],
        default="real",
        help="Qué fuente de datos evaluar: 'real' (data/real/) o 'synthetic' (data/synthetic/)",
    )
    parser.add_argument(
        "--extractor",
        choices=["temporal", "next_event"],
        default="temporal",
        help="Extractor de features: 'temporal' (9 features diarias, default) o "
        "'next_event' (Markov de primer orden: 3 features de log-probabilidad de "
        "predicción del siguiente sensor).",
    )
    parser.add_argument(
        "--houses",
        nargs="+",
        default=None,
        help="Casas a evaluar (por defecto, todas las que haya en la BD)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=None,
        help="Utilizar solo los primeros N días cronológicos de cada casa "
        "(las casas reales abarcan hasta ~235 días y ralentizan el pipeline). "
        "Por defecto: todos los días.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help="Fracción de días usados para entrenar (resto = holdout)",
    )
    parser.add_argument("--zscore-threshold", type=float, default=3.0)
    parser.add_argument("--iforest-contamination", type=float, default=0.05)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--ensemble-threshold-percentile", type=float, default=DEFAULT_ENSEMBLE_THRESHOLD_PERCENTILE)
    parser.add_argument(
        "--scenario",
        choices=["point", "contextual", "collective"],
        default="point",
        help="Anomaly scenario injected on the raw event stream: 'point' (night "
        "burst, default), 'contextual' (whole-day circular shift), or 'collective' "
        "(intra-day reordering).",
    )
    parser.add_argument(
        "--intensity",
        choices=["low", "medium", "high"],
        default="high",
        help="Intensity preset for the injected scenario (low/medium/high).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
        help="Fraction of holdout days with an injected anomaly",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Número de semillas para repetir la inyección sintética y reportar "
        "media ± desviación estándar de las métricas (1 = evaluación única)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Semilla inicial; se usan las N semillas consecutivas "
        "(seed_base ... seed_base + n_seeds - 1). El split temporal 70/30 se "
        "mantiene fijo (sin look-ahead); cada semilla varía la inyección "
        "sintética y el random_state de los detectores estocásticos.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta CSV donde guardar el reporte (opcional)",
    )
    return parser.parse_args()


def build_ensemble(args, random_state: int) -> EnsembleDetector:
    """The three base detectors, fitted with a given random_state for the
    stochastic ones (Isolation Forest), so each seed is a fresh draw."""
    return EnsembleDetector(
        detectors=[
            ZScoreDetector(threshold=args.zscore_threshold),
            IsolationForestDetector(
                contamination=args.iforest_contamination,
                random_state=random_state,
            ),
            PCAReconstructionDetector(n_components=args.pca_components),
        ],
        ensemble_threshold_percentile=args.ensemble_threshold_percentile,
    )


def aggregate_synthetic(seed_metrics: list[dict]) -> dict:
    """Collapse per-seed metrics into mean ± std columns (std over N-1)."""
    if not seed_metrics:
        return {}
    metrics = ("precision", "recall", "f1", "auroc")
    agg: dict[str, float | int] = {"n_seeds": len(seed_metrics)}
    for metric in metrics:
        arr = np.asarray([r[metric] for r in seed_metrics], dtype=float)
        agg[f"synthetic_{metric}_mean"] = float(arr.mean())
        agg[f"synthetic_{metric}_std"] = float(arr.std(ddof=1))
    return agg


def run_house(db: SQLiteDataManager, house_id: str, args) -> dict:
    df_house = db.query_house(house_id)
    if df_house.empty:
        return {"house_id": house_id, "status": "sin datos"}

    df_house = truncate_stream_to_days(df_house, args.max_days)
    df_house = df_house.copy()
    df_house["timestamp"] = pd.to_datetime(df_house["timestamp"])
    df_house["date"] = df_house["timestamp"].dt.date
    dates = sorted(df_house["date"].unique())
    if len(dates) < MIN_DAYS:
        return {
            "house_id": house_id,
            "status": f"muy pocos días ({len(dates)}), se omite",
        }

    split_idx = max(1, int(len(dates) * args.train_split))
    train_dates = set(dates[:split_idx])
    df_train = df_house[df_house["date"].isin(train_dates)]
    df_full = df_house

    if args.extractor == "next_event":
        extractor = NextEventTransitionExtractor()
        # Learn the normal transition probabilities from the train days only, so
        # the holdout is scored against a model that never saw it.
        extractor.fit(df_train)
        X_full, _dates = extractor.extract(df_full)
    else:
        extractor = TemporalFeatureExtractor()
        X_full, _dates = extractor.extract(df_full)

    # Fit the scaler on the training days only (no look-ahead), then transform
    # the full stream so the eval tail is scored against the train distribution.
    scaler = FeatureScaler()
    scaler.fit(X_full[:split_idx])
    X_scaled = scaler.transform(X_full)
    X_train, X_holdout = X_scaled[:split_idx], X_scaled[split_idx:]

    # Reference fit (default seed) for the score-distribution stats; the
    # scenario metrics below are recomputed per seed so stochastic detectors
    # and the injection both vary across runs.
    ensemble = build_ensemble(args, random_state=DEFAULT_RANDOM_STATE)
    ensemble.fit(X_train)
    anomalies, scores, _ = ensemble.predict(X_scaled)

    row = {
        "house_id": house_id,
        "status": "ok",
        "n_days": len(X_full),
        "n_train": len(X_train),
        "n_holdout": len(X_holdout),
    }
    row.update(describe_scores(scores, anomalies))

    if len(X_holdout) >= 5:
        holdout_dates = dates[split_idx:]
        seed_metrics = []
        for seed in range(args.seed_base, args.seed_base + args.n_seeds):
            rng = np.random.default_rng(seed)
            n_anomaly = max(1, round(len(holdout_dates) * args.contamination))
            anomaly_dates = select_anomaly_dates(holdout_dates, rng, n_anomaly)
            fraction = INTENSITY_PRESETS[args.scenario][args.intensity]
            if args.scenario == "contextual":
                fraction = int(fraction)
            # contextual injector expects int hours; others accept float fraction
            df_inj = INJECTORS[args.scenario](df_full, rng, fraction, anomaly_dates)  # type: ignore[arg-type]

            # Re-extract features from the injected stream, scale with the same
            # train-only scaler, and score with a fresh ensemble for this seed.
            if args.extractor == "next_event":
                X_inj, _dates = extractor.extract(df_inj)
            else:
                X_inj, _dates = TemporalFeatureExtractor().extract(df_inj)
            X_inj_scaled = scaler.transform(X_inj)

            ens = build_ensemble(args, random_state=seed)
            ens.fit(X_train)
            y_pred, scores_inj, _ = ens.predict(X_inj_scaled)

            y = np.zeros(len(dates), dtype=int)
            for i, d in enumerate(dates):
                if d in anomaly_dates:
                    y[i] = 1
            y_holdout = y[split_idx:]
            scores_holdout = scores_inj[split_idx:]

            metrics = compute_metrics(y_holdout, y_pred[split_idx:])
            metrics["auroc"] = float(roc_auc_score(y_holdout, scores_holdout))
            metrics["n_holdout"] = len(y_holdout)
            metrics["n_injected"] = int(y_holdout.sum())
            metrics["scenario"] = args.scenario
            metrics["intensity"] = args.intensity
            seed_metrics.append(metrics)
        row.update(aggregate_synthetic(seed_metrics))
    else:
        row["status"] = (
            f"holdout demasiado chico ({len(X_holdout)} días) para métricas sintéticas"
        )

    return row


def main():
    args = parse_args()

    db = SQLiteDataManager(str(db_path(args.source)))
    db.connect()

    available = db.list_houses()
    if not available:
        print(
            f"La base de datos de la fuente '{args.source}' está vacía. Carga los datos primero con:\n"
            f"  python src/ingestion/casas_loader.py --source {args.source}"
        )
        db.close()
        sys.exit(1)

    houses = args.houses or available
    unknown = set(houses) - set(available)
    if unknown:
        print(f"Estas casas no están en la BD y se omiten: {sorted(unknown)}")
        houses = [h for h in houses if h in available]

    print(f"Fuente de datos: {args.source} ({db_path(args.source)})")
    print(f"Evaluando {len(houses)} casa(s) de forma independiente: {houses}\n")

    rows = [run_house(db, house_id, args) for house_id in houses]
    db.close()

    report = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 25)
    print(report.to_string(index=False))

    ok_rows = report[report["status"] == "ok"]
    if not ok_rows.empty and "synthetic_auroc_mean" in ok_rows.columns:
        avg_auroc = ok_rows["synthetic_auroc_mean"].mean()
        n_seeds = int(np.asarray(ok_rows["n_seeds"])[0])
        print(f"\nAUROC sintético promedio entre casas: {avg_auroc:.3f} "
              f"(n_seeds={n_seeds}, split 70/30 fijo)")
        spread = ok_rows["synthetic_auroc_std"].mean()
        if avg_auroc < 0.7:
            print(
                "AUROC bajo: el ensemble apenas distingue las anomalías sintéticas "
                "inyectadas del comportamiento normal. Prueba bajar "
                "--iforest-contamination, subir --pca-components, o revisar si hay "
                "suficientes días de holdout."
            )
        else:
            print(
                "El ensemble detecta de forma consistente las anomalías sintéticas inyectadas."
            )
        if spread >= 0.15:
            print(
                f"Nota: alta variabilidad entre semillas (std media ≈ {spread:.3f}): "
                "las diferencias entre casas/detectores pueden ser ruido. Compara con "
                "--n-seeds más alto o revisa el tamaño del holdout."
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        print(f"\nReporte guardado en {out_path}")


if __name__ == "__main__":
    main()
