import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class SQLiteDataManager:
    def __init__(self, db_path: str = "data/sensor_data.db"):
        self.db_path = db_path
        self.conn: sqlite3.Connection | None = None
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            logger.info(f"Conectado a base de datos: {self.db_path}")
            return self.conn
        except sqlite3.Error as e:
            logger.error(f"Error al conectar a la BD: {e}")
            raise

    def create_tables(self) -> None:
        if self.conn is None:
            raise RuntimeError("You must call connect() before create_tables()")

        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    house_id TEXT NOT NULL DEFAULT 'aruba',
                    timestamp DATETIME NOT NULL,
                    sensor_id TEXT,
                    event_type TEXT,
                    value REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(house_id, timestamp, sensor_id, event_type)
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON sensor_events(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_house
                ON sensor_events(house_id)
            """)
            self.conn.commit()
            logger.info("Tablas e índices creados correctamente")
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Error al crear tablas: {e}")
            raise

    def insert_events(self, df: pd.DataFrame, house_id: str = "aruba") -> int:
        if self.conn is None:
            raise RuntimeError("You must call connect() before insert_events()")

        if "house_id" not in df.columns:
            df = df.copy()
            df["house_id"] = house_id

        records = df.to_dict("records")
        return self.insert_batch(records)

    def insert_batch(self, list_of_dicts: list[dict]) -> int:
        if self.conn is None:
            raise RuntimeError("You must call connect() before insert_batch()")

        if not list_of_dicts:
            logger.warning("insert_batch llamado con lista vacía, nada que insertar")
            return 0

        try:
            cursor = self.conn.cursor()
            cursor.execute("BEGIN")

            rows = [
                (
                    rec.get("house_id", "aruba"),
                    str(rec.get("timestamp")),
                    rec.get("sensor_id"),
                    rec.get("event_type"),
                    rec.get("value"),
                )
                for rec in list_of_dicts
            ]

            cursor.executemany(
                """
                INSERT OR IGNORE INTO sensor_events (house_id, timestamp, sensor_id, event_type, value)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

            self.conn.commit()
            inserted = cursor.rowcount if cursor.rowcount != -1 else len(rows)
            skipped = len(rows) - inserted
            if skipped > 0:
                logger.info(
                    f"{inserted} eventos insertados, {skipped} ya existían (ignorados)"
                )
            else:
                logger.info(f"Insertados {inserted} eventos correctamente")
            return inserted

        except sqlite3.OperationalError as e:
            self.conn.rollback()
            logger.error(f"BD bloqueada u error operacional: {e}")
            raise
        except sqlite3.Error as e:
            self.conn.rollback()
            logger.error(f"Error al insertar eventos: {e}")
            raise

    def query_to_dataframe(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        if self.conn is None:
            raise RuntimeError("You must call connect() before query_to_dataframe()")

        try:
            df = pd.read_sql_query(sql, self.conn, params=list(params))
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
            return df
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            logger.error(f"Error al ejecutar consulta: {e}")
            raise

    def query_house(self, house_id: str) -> pd.DataFrame:
        return self.query_to_dataframe(
            "SELECT * FROM sensor_events WHERE house_id = ? ORDER BY timestamp",
            params=(house_id,),
        )

    def list_houses(self) -> list[str]:
        if self.conn is None:
            raise RuntimeError("You must call connect() before list_houses()")
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT house_id FROM sensor_events ORDER BY house_id")
        return [row[0] for row in cursor.fetchall()]

    def get_stats(self, house_id: str | None = None) -> dict:
        if self.conn is None:
            raise RuntimeError("You must call connect() before get_stats()")

        try:
            cursor = self.conn.cursor()
            where = "WHERE house_id = ?" if house_id else ""
            params = (house_id,) if house_id else ()

            # `where` is a fixed constant ("WHERE house_id = ?" or ""), never
            # user input; values are passed as bound parameters.
            cursor.execute(
                f"SELECT COUNT(*) FROM sensor_events {where}", params  # noqa: S608
            )
            count = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM sensor_events {where}",  # noqa: S608
                params,
            )
            min_ts, max_ts = cursor.fetchone()

            cursor.execute(
                f"SELECT COUNT(DISTINCT sensor_id) FROM sensor_events {where}", params  # noqa: S608
            )
            n_sensors = cursor.fetchone()[0]

            stats = {
                "house_id": house_id or "todas",
                "count": count,
                "date_range": (min_ts, max_ts),
                "sensors": n_sensors,
            }
            logger.info(f"Estadísticas obtenidas: {stats}")
            return stats
        except sqlite3.Error as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            raise

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None
            logger.info("Conexión a la base de datos cerrada")
