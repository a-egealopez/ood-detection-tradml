import argparse
import sys
import logging
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion.sqlite_manager import SQLiteDataManager
from config import HOUSES, raw_csv_path, db_path, setup_logging

logger = logging.getLogger(__name__)


def _convert_reading(reading: str):
    reading = str(reading).strip()

    if reading == "ON":
        return "ON", 1.0
    if reading == "OFF":
        return "OFF", 0.0
    if reading == "OPEN":
        return "OPEN", 1.0
    if reading == "CLOSE":
        return "CLOSE", 0.0

    logger.warning(f"Valor de sensor no reconocido (se descarta la fila): '{reading}'")
    return None, None


def load_casas_csv(csv_path: Path, house_id: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {csv_path}. Descárgalo desde "
            f"https://casas.wsu.edu/datasets/ y guárdalo como {csv_path}, o genera "
            f"datos de prueba con: python scripts/generate_test_fixtures.py"
        )

    logger.info(f"[{house_id}] Leyendo dataset crudo desde {csv_path}")

    df = pd.read_csv(
        csv_path,
        header=None,
        names=["date", "time", "sensor_id", "reading"],
        dtype=str,
        on_bad_lines="warn",
    )

    logger.info(f"[{house_id}] Filas leídas: {len(df)}")

    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="mixed", errors="coerce"
    )

    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts > 0:
        logger.warning(f"[{house_id}] {n_bad_ts} filas con timestamp inválido, se descartan")
        df = df.dropna(subset=["timestamp"])

    converted = df["reading"].apply(_convert_reading)
    df["event_type"] = converted.apply(lambda t: t[0])
    df["value"] = converted.apply(lambda t: t[1])

    n_bad_reading = df["event_type"].isna().sum()
    if n_bad_reading > 0:
        logger.warning(f"[{house_id}] {n_bad_reading} filas con valor no reconocido, se descartan")
        df = df.dropna(subset=["event_type"])

    df_clean = df[["timestamp", "sensor_id", "event_type", "value"]].copy()
    df_clean["house_id"] = house_id
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)

    logger.info(f"[{house_id}] Dataset transformado: {len(df_clean)} eventos listos para insertar")
    return df_clean


def load_house(db: SQLiteDataManager, house_id: str, source: str = "real") -> int:
    csv_path = raw_csv_path(house_id, source=source)
    if not csv_path.exists():
        logger.warning(f"[{house_id}] No se encontró {csv_path} (fuente '{source}'), se omite esta casa")
        return 0

    df = load_casas_csv(csv_path, house_id)
    inserted = db.insert_events(df, house_id=house_id)
    logger.info(f"[{house_id}] {inserted} eventos nuevos insertados")
    return inserted


def load_all_houses(db: SQLiteDataManager, source: str = "real") -> dict:
    results = {}
    for house_id in HOUSES:
        results[house_id] = load_house(db, house_id, source=source)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source", choices=["real", "synthetic"], default="real",
        help="Qué fuente de datos cargar: 'real' (data/real/) o 'synthetic' (data/synthetic/)",
    )
    args = parser.parse_args()

    setup_logging()

    db = SQLiteDataManager(str(db_path(args.source)))
    db.connect()
    db.create_tables()

    logger.info(f"Cargando fuente '{args.source}' -> {db_path(args.source)}")
    results = load_all_houses(db, source=args.source)

    for house_id, n_inserted in results.items():
        stats = db.get_stats(house_id=house_id)
        logger.info(f" [{house_id}] {n_inserted} eventos nuevos, stats: {stats}")

    db.close()
    logger.info(f" Proceso completado (fuente: {args.source})")


if __name__ == "__main__":
    main()