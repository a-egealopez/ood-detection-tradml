"""
event_driven_extractors.py

Extracción de vectores de características para series temporales basadas en eventos
(sensores IoT ON/OFF, logs de actividad), agrupando por ventana (por defecto: día).

Tres enfoques con literatura propia dentro del análisis de event-driven time series.
Cada uno se anota con el tipo de anomalía que permite detectar, según la taxonomía
estándar de Chandola, Banerjee & Kumar, "Anomaly Detection: A Survey" (ACM Computing
Surveys, 2009): puntual (point), contextual (conditional) y colectiva (collective).
Para las anomalías de secuencia se usa además Chandola, Banerjee & Kumar, "Anomaly
Detection for Discrete Sequences: A Survey" (IEEE TKDE, 2012).

1. WindowAggregationExtractor -> CONTEXTUAL + COLECTIVA (a nivel de ventana/día)
   Resume todo un día en estadísticas agregadas (conteo, entropía, % nocturno). El
   contexto (hora/franja) está codificado explícitamente en features como
   `night_activity_ratio`, por lo que el mismo volumen de eventos puede ser normal
   de día y anómalo de noche (anomalía CONTEXTUAL, formalizada por Song et al.,
   "Conditional Anomaly Detection", 2007). Como cada vector representa el día
   completo, un día que se salga del patrón habitual es una anomalía COLECTIVA: no
   hay un evento individual culpable, es el conjunto del día.
   NO sirve para anomalías puntuales: al agregar, el detalle de un evento suelto se
   pierde (un único evento raro puede pasar desapercibido en el promedio del día).

2. IntervalStatisticsExtractor -> PUNTUAL (a nivel de intervalo crudo) o
   COLECTIVA (si se agrega por ventana, como hace `extract()`)
   Analiza el tiempo entre eventos consecutivos (point process / renewal process;
   Fano, 1947). Los valores crudos de `diagnostics()["intervals_seconds"]` son
   directamente aptos para detección PUNTUAL: un único hueco anómalamente largo o
   corto es un outlier puntual en sí mismo (p.ej., "el sensor dejó de disparar 6h").
   Al agregar por día (mean/CV/Fano factor, que es lo que hace `extract()`), el
   vector resultante deja de señalar el instante exacto y pasa a describir el
   "ritmo" de todo el día -> ahí la anomalía que se detecta es COLECTIVA (un día
   demasiado regular o demasiado "bursty" en su conjunto).

3. NGramTransitionExtractor -> COLECTIVA / DE SECUENCIA (pattern-based)
   Cadena de Markov de primer orden sobre la secuencia de sensores disparados. No
   analiza el timestamp ni la magnitud de nada individual, solo el ORDEN de los
   eventos. Por diseño solo puede detectar anomalías COLECTIVAS de tipo secuencia:
   un día donde el orden de activaciones rompe el patrón habitual (p.ej.
   "Cocina -> Baño -> Cocina -> Cocina" repetido, en vez de la rutina esperada),
   aunque cada sensor individual y cada intervalo sean perfectamente normales.
   NO sirve para anomalías puntuales ni contextuales puras (ignora tiempo y hora).

Recomendación práctica: para cubrir los tres tipos de anomalía en un mismo pipeline,
combina los tres extractores (o aplica el de intervalos también a nivel crudo, sin
agregar) en vez de usar solo uno.

Las tres clases comparten la misma interfaz: extract(df) -> (X, dates) y
diagnostics(group) -> dict con los valores intermedios necesarios para graficar
cómo se calculó el vector de features de una ventana concreta.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

EPSILON = 1e-10


def _entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs + EPSILON)))


class WindowAggregationExtractor:
    """Agregación estadística clásica por ventana (tumbling window).

    Tipo de anomalía: CONTEXTUAL (el contexto hora/franja está codificado en las
    features) + COLECTIVA a nivel de día (el vector resume el día completo).
    No apta para anomalías puntuales (evento individual raro dentro de un día normal).
    """

    ANOMALY_TYPES: List[str] = ["contextual", "colectiva"]

    FEATURE_NAMES: List[str] = [
        "n_events", "n_sensors", "activity_hours", "avg_gap_minutes",
        "night_activity_ratio", "entropy_hourly", "entropy_sensor",
    ]

    def extract(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        rows, dates = [], []
        for date, group in df.groupby("date"):
            rows.append(self._features_for_window(group))
            dates.append(date)
        return np.array(rows), np.array(dates)

    def _features_for_window(self, group: pd.DataFrame) -> List[float]:
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        group["hour"] = group["timestamp"].dt.hour

        n_events = len(group)
        if n_events == 0:
            return [0.0] * len(self.FEATURE_NAMES)

        n_sensors = group["sensor_id"].nunique()
        activity_hours = group["hour"].nunique()

        ts_sorted = group["timestamp"].sort_values()
        gaps = ts_sorted.diff().dropna().dt.total_seconds() / 60.0
        avg_gap = float(gaps.mean()) if len(gaps) > 0 else 0.0

        night_mask = (group["hour"] < 8) | (group["hour"] >= 22)
        night_ratio = float(night_mask.sum()) / n_events

        hourly_counts = group.groupby("hour").size().reindex(range(24), fill_value=0).values
        entropy_hourly = _entropy(hourly_counts)
        entropy_sensor = _entropy(group.groupby("sensor_id").size().values)

        return [n_events, n_sensors, activity_hours, avg_gap, night_ratio, entropy_hourly, entropy_sensor]

    def diagnostics(self, group: pd.DataFrame) -> Dict:
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        group["hour"] = group["timestamp"].dt.hour
        hourly_counts = group.groupby("hour").size().reindex(range(24), fill_value=0)
        return {
            "hourly_counts": hourly_counts,
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
        }


class IntervalStatisticsExtractor:
    """Estadística de intervalos entre eventos consecutivos (point-process / renewal process).

    Tipo de anomalía: PUNTUAL si se usan los intervalos crudos (`diagnostics()`),
    COLECTIVA si se usa el vector agregado por ventana que produce `extract()`
    (mean/CV/Fano factor describen el "ritmo" del día completo, no un instante).
    """

    ANOMALY_TYPES: List[str] = ["puntual (crudo)", "colectiva (agregado por ventana)"]

    FEATURE_NAMES: List[str] = ["n_events", "mean_iei_sec", "std_iei_sec", "cv_iei", "fano_factor"]

    def __init__(self, fano_bin_minutes: float = 30.0):
        self.fano_bin_minutes = fano_bin_minutes

    def extract(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        rows, dates = [], []
        for date, group in df.groupby("date"):
            rows.append(self._features_for_window(group))
            dates.append(date)
        return np.array(rows), np.array(dates)

    def _intervals_seconds(self, group: pd.DataFrame) -> np.ndarray:
        ts_sorted = pd.to_datetime(group["timestamp"]).sort_values()
        return ts_sorted.diff().dropna().dt.total_seconds().values

    def _fano_factor(self, group: pd.DataFrame) -> float:
        """Varianza / media del conteo de eventos en bins de tamaño fijo: mide 'burstiness'."""
        ts = pd.to_datetime(group["timestamp"]).sort_values()
        if len(ts) < 2:
            return 0.0
        bin_edges = pd.date_range(ts.min().floor("min"), ts.max().ceil("min") + pd.Timedelta(minutes=self.fano_bin_minutes), freq=f"{self.fano_bin_minutes}min")
        if len(bin_edges) < 2:
            return 0.0
        counts, _ = np.histogram(ts.astype("int64"), bins=bin_edges.astype("int64"))
        mean_c = counts.mean()
        if mean_c == 0:
            return 0.0
        return float(counts.var() / mean_c)

    def _features_for_window(self, group: pd.DataFrame) -> List[float]:
        n_events = len(group)
        intervals = self._intervals_seconds(group)
        if len(intervals) < 2:
            return [float(n_events), 0.0, 0.0, 0.0, 0.0]

        mean_iei = float(intervals.mean())
        std_iei = float(intervals.std())
        cv_iei = float(std_iei / (mean_iei + EPSILON))
        fano = self._fano_factor(group)
        return [float(n_events), mean_iei, std_iei, cv_iei, fano]

    def diagnostics(self, group: pd.DataFrame) -> Dict:
        return {
            "intervals_seconds": self._intervals_seconds(group),
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
        }


class NGramTransitionExtractor:
    """Cadena de Markov de primer orden sobre la secuencia temporal de sensores disparados.

    Tipo de anomalía: COLECTIVA de secuencia (pattern-based). Solo mira el orden de
    los eventos, no su instante ni su magnitud, así que no sirve para detectar
    anomalías puntuales ni contextuales puras.
    """

    ANOMALY_TYPES: List[str] = ["colectiva (secuencia)"]

    FEATURE_NAMES: List[str] = ["n_transitions", "transition_entropy", "top_transition_prob", "unique_bigrams_ratio"]

    def __init__(self, token_col: str = "sensor_id"):
        self.token_col = token_col
        self.vocabulary_: List[str] = []

    def fit_vocabulary(self, df: pd.DataFrame) -> "NGramTransitionExtractor":
        self.vocabulary_ = sorted(df[self.token_col].astype(str).unique().tolist())
        return self

    def _sequence(self, group: pd.DataFrame) -> List[str]:
        group = group.copy()
        group["timestamp"] = pd.to_datetime(group["timestamp"])
        return group.sort_values("timestamp")[self.token_col].astype(str).tolist()

    def extract(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.vocabulary_:
            self.fit_vocabulary(df)

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        rows, dates = [], []
        for date, group in df.groupby("date"):
            rows.append(self._features_for_window(group))
            dates.append(date)
        return np.array(rows), np.array(dates)

    def _transition_matrix(self, group: pd.DataFrame) -> pd.DataFrame:
        seq = self._sequence(group)
        vocab = self.vocabulary_ or sorted(set(seq))
        matrix = pd.DataFrame(0, index=vocab, columns=vocab, dtype=float)
        for a, b in zip(seq[:-1], seq[1:]):
            if a in matrix.index and b in matrix.columns:
                matrix.loc[a, b] += 1
        return matrix

    def _features_for_window(self, group: pd.DataFrame) -> List[float]:
        seq = self._sequence(group)
        n_transitions = max(len(seq) - 1, 0)
        if n_transitions == 0:
            return [0.0, 0.0, 0.0, 0.0]

        matrix = self._transition_matrix(group)
        counts = matrix.values.flatten()
        entropy = _entropy(counts)

        total = counts.sum()
        top_prob = float(counts.max() / total) if total > 0 else 0.0

        possible_bigrams = len(self.vocabulary_) ** 2 if self.vocabulary_ else 1
        unique_bigrams = int((counts > 0).sum())
        unique_ratio = float(unique_bigrams / possible_bigrams)

        return [float(n_transitions), entropy, top_prob, unique_ratio]

    def diagnostics(self, group: pd.DataFrame) -> Dict:
        return {
            "transition_matrix": self._transition_matrix(group),
            "sequence": self._sequence(group),
            "features": self._features_for_window(group),
            "feature_names": self.FEATURE_NAMES,
        }


def generate_synthetic_events(
    n_days: int = 5,
    pattern: str = "regular",
    n_sensors: int = 3,
    events_per_day: int = 80,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera un stream de eventos sintético con el mismo esquema que CASAS Aruba
    (timestamp, sensor_id, event_type, value), para ilustrar los extractores
    sin depender de la base de datos real.

    pattern:
        "regular"   -> eventos casi equiespaciados (proceso periódico con jitter pequeño)
        "bursty"    -> eventos concentrados en pocos clusters temporales (proceso "bursty")
        "day_night" -> mayoría de eventos en horario diurno, pocos de noche
    """
    rng = np.random.default_rng(seed)
    sensors = [f"Sensor_{i + 1}" for i in range(n_sensors)]
    base_day = pd.Timestamp("2024-01-01")
    records = []

    for day in range(n_days):
        day_start = base_day + pd.Timedelta(days=day)

        if pattern == "regular":
            offsets = np.linspace(0, 24 * 60, events_per_day, endpoint=False)
            offsets = offsets + rng.normal(0, 2, size=events_per_day)
        elif pattern == "bursty":
            n_clusters = max(3, events_per_day // 15)
            centers = rng.uniform(0, 24 * 60, size=n_clusters)
            per_cluster = max(events_per_day // n_clusters, 1)
            offsets = np.concatenate([rng.normal(c, 5, size=per_cluster) for c in centers])
        elif pattern == "day_night":
            day_events = int(events_per_day * 0.85)
            night_events = events_per_day - day_events
            offsets = np.concatenate([
                rng.uniform(8 * 60, 22 * 60, size=day_events),
                rng.uniform(0, 8 * 60, size=night_events // 2),
                rng.uniform(22 * 60, 24 * 60, size=night_events - night_events // 2),
            ])
        else:
            raise ValueError(f"Patrón desconocido: {pattern}")

        offsets = np.clip(offsets, 0, 24 * 60 - 0.01)
        timestamps = [day_start + pd.Timedelta(minutes=float(m)) for m in offsets]
        chosen_sensors = rng.choice(sensors, size=len(timestamps))
        event_types = rng.choice(["ON", "OFF"], size=len(timestamps))

        for ts, sensor, etype in zip(timestamps, chosen_sensors, event_types):
            records.append({
                "timestamp": ts, "sensor_id": sensor, "event_type": etype,
                "value": 1.0 if etype == "ON" else 0.0,
            })

    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
