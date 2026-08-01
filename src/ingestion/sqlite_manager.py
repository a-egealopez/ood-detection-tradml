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
            raise RuntimeError("Debes llamar a connect() antes de create_tables()")

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
            raise RuntimeError("Debes llamar a connect() antes de insert_events()")

        if "house_id" not in df.columns:
            df = df.copy()
            df["house_id"] = house_id

        records = df.to_dict("records")
        return self.insert_batch(records)

    def insert_batch(self, list_of_dicts: list[dict]) -> int:
        if self.conn is None:
            raise RuntimeError("Debes llamar a connect() antes de insert_batch()")

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
            raise RuntimeError("Debes llamar a connect() antes de query_to_dataframe()")

        try:
            df = pd.read_sql_query(sql, self.conn, params=params)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except (sqlite3.Error, pd.errors.DatabaseError) as e:
            logger.error(f"Error al ejecutar consulta: {e}")
            raise

    def query_house(self, house_id: str) -> pd.DataFrame:
        return self.query_to_dataframe(
            "SELECT * FROM sensor_events WHERE house_id = ? ORDER BY timestamp",
            params=(house_id,),
        )

    def query_date_range(
        self, start_date: str, end_date: str, house_id: str | None = None
    ) -> pd.DataFrame:
        if house_id is None:
            sql = """
                SELECT * FROM sensor_events
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """
            return self.query_to_dataframe(sql, params=(start_date, end_date))

        sql = """
            SELECT * FROM sensor_events
            WHERE house_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        return self.query_to_dataframe(sql, params=(house_id, start_date, end_date))

    def list_houses(self) -> list[str]:
        if self.conn is None:
            raise RuntimeError("Debes llamar a connect() antes de list_houses()")
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT house_id FROM sensor_events ORDER BY house_id")
        return [row[0] for row in cursor.fetchall()]

    def get_stats(self, house_id: str | None = None) -> dict:
        if self.conn is None:
            raise RuntimeError("Debes llamar a connect() antes de get_stats()")

        try:
            cursor = self.conn.cursor()
            where = "WHERE house_id = ?" if house_id else ""
            params = (house_id,) if house_id else ()

            cursor.execute(f"SELECT COUNT(*) FROM sensor_events {where}", params)
            count = cursor.fetchone()[0]

            cursor.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM sensor_events {where}",
                params,
            )
            min_ts, max_ts = cursor.fetchone()

            cursor.execute(
                f"SELECT COUNT(DISTINCT sensor_id) FROM sensor_events {where}", params
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    db = SQLiteDataManager("data/sensor_data_test.db")
    db.connect()
    db.create_tables()

    batch = [
        {
            "house_id": "aruba",
            "timestamp": "2024-01-01 10:30:00",
            "sensor_id": "M001",
            "event_type": "motion",
            "value": 1.0,
        },
        {
            "house_id": "aruba",
            "timestamp": "2024-01-01 10:31:00",
            "sensor_id": "M001",
            "event_type": "motion",
            "value": 0.0,
        },
    ]
    count1 = db.insert_batch(batch)
    count2 = db.insert_batch(batch)

    print(f" Primera inserción: {count1} eventos nuevos")
    print(f"s Segunda inserción (idempotente): {count2} eventos nuevos")
    assert count2 == 0

    print(db.get_stats())
    db.close()
