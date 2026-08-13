import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import db_path, setup_logging
from detectors import (
    EnsembleDetector,
    IsolationForestDetector,
    PCAReconstructionDetector,
    ZScoreDetector,
)
from evaluation.synthetic_injection import (
    describe_scores,
    evaluate_with_synthetic_anomalies,
)
from features import FeatureScaler, TemporalFeatureExtractor
from ingestion.sqlite_manager import SQLiteDataManager

logger = setup_logging()


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
        "--houses",
        nargs="+",
        default=None,
        help="Casas a evaluar (por defecto, todas las que haya en la BD)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Fracción de días usados para entrenar (resto = holdout)",
    )
    parser.add_argument("--zscore-threshold", type=float, default=3.0)
    parser.add_argument("--iforest-contamination", type=float, default=0.05)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--ensemble-threshold-percentile", type=float, default=90)
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.15,
        help="%% de días del holdout con anomalía sintética inyectada",
    )
    parser.add_argument(
        "--magnitude",
        type=float,
        default=6.0,
        help="Magnitud (en desv. estándar) de la anomalía sintética",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta CSV donde guardar el reporte (opcional)",
    )
    return parser.parse_args()


def run_house(db: SQLiteDataManager, house_id: str, args) -> dict:
    df_house = db.query_house(house_id)
    if df_house.empty:
        return {"house_id": house_id, "status": "sin datos"}

    extractor = TemporalFeatureExtractor()
    X, _dates = extractor.extract(df_house)

    if len(X) < 10:
        return {"house_id": house_id, "status": f"muy pocos días ({len(X)}), se omite"}

    scaler = FeatureScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = max(1, int(len(X_scaled) * args.train_split))
    X_train, X_holdout = X_scaled[:split_idx], X_scaled[split_idx:]

    ensemble = EnsembleDetector(
        detectors=[
            ZScoreDetector(threshold=args.zscore_threshold),
            IsolationForestDetector(contamination=args.iforest_contamination),
            PCAReconstructionDetector(n_components=args.pca_components),
        ],
        ensemble_threshold_percentile=args.ensemble_threshold_percentile,
    )
    ensemble.fit(X_train)
    anomalies, scores, _ = ensemble.predict(X_scaled)

    row = {
        "house_id": house_id,
        "status": "ok",
        "n_days": len(X),
        "n_train": len(X_train),
        "n_holdout": len(X_holdout),
    }
    row.update(describe_scores(scores, anomalies))

    if len(X_holdout) >= 5:
        synth = evaluate_with_synthetic_anomalies(
            ensemble,
            X_holdout,
            contamination=args.contamination,
            magnitude=args.magnitude,
        )
        row.update(
            {
                "synthetic_precision": synth["precision"],
                "synthetic_recall": synth["recall"],
                "synthetic_f1": synth["f1"],
                "synthetic_auroc": synth["auroc"],
            }
        )
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
            f"⚠️  La base de datos de la fuente '{args.source}' está vacía. Carga los datos primero con:\n"
            f"  python src/ingestion/casas_loader.py --source {args.source}"
        )
        db.close()
        sys.exit(1)

    houses = args.houses or available
    unknown = set(houses) - set(available)
    if unknown:
        print(f"⚠️  Estas casas no están en la BD y se omiten: {sorted(unknown)}")
        houses = [h for h in houses if h in available]

    print(f"Fuente de datos: {args.source} ({db_path(args.source)})")
    print(f"Evaluando {len(houses)} casa(s) de forma independiente: {houses}\n")

    rows = [run_house(db, house_id, args) for house_id in houses]
    db.close()

    report = pd.DataFrame(rows)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(report.to_string(index=False))

    ok_rows = report[report["status"] == "ok"]
    if not ok_rows.empty and "synthetic_auroc" in ok_rows.columns:
        avg_auroc = ok_rows["synthetic_auroc"].mean()
        print(f"\nAUROC sintético promedio entre casas: {avg_auroc:.3f}")
        if avg_auroc < 0.7:
            print(
                "⚠️  AUROC bajo: el ensemble apenas distingue las anomalías sintéticas inyectadas "
                "del comportamiento normal. Prueba bajar --iforest-contamination, subir "
                "--pca-components, o revisar si hay suficientes días de holdout."
            )
        else:
            print(
                "✓ El ensemble detecta de forma consistente las anomalías sintéticas inyectadas."
            )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(out_path, index=False)
        print(f"\n✓ Reporte guardado en {out_path}")


if __name__ == "__main__":
    main()
