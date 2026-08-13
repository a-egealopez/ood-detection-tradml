import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import SYNTHETIC_DATA_DIR

DATA_DIR = SYNTHETIC_DATA_DIR
SENSORS_MOTION = ["Bedroom", "Kitchen", "LivingRoom"]
SENSOR_DOOR = "OutsideDoor"

HOUSE_PROFILES = {
    "aruba": {"events_mean": 180, "night_ratio": 0.08, "n_days": 60, "seed": 1},
    "cairo": {"events_mean": 90, "night_ratio": 0.05, "n_days": 45, "seed": 2},
    "milan": {"events_mean": 260, "night_ratio": 0.15, "n_days": 50, "seed": 3},
    "tulum": {"events_mean": 130, "night_ratio": 0.10, "n_days": 40, "seed": 4},
}


def _events_for_day(rng, date, events_mean, night_ratio, anomalous=False):
    if anomalous:
        events_mean = events_mean * rng.uniform(2.5, 4.0)
        night_ratio = min(0.9, night_ratio * 4)

    n_events = max(1, int(rng.poisson(events_mean)))
    is_night = rng.random(n_events) < night_ratio

    hours = np.where(
        is_night,
        rng.choice(list(range(22, 24)) + list(range(8)), size=n_events),
        rng.integers(8, 22, size=n_events),
    )
    minutes = rng.integers(0, 60, size=n_events)
    seconds = rng.integers(0, 60, size=n_events)
    micros = rng.integers(0, 1_000_000, size=n_events)

    timestamps = [
        pd.Timestamp(date)
        + pd.Timedelta(
            hours=int(h), minutes=int(m), seconds=int(s), microseconds=int(u)
        )
        for h, m, s, u in zip(hours, minutes, seconds, micros, strict=True)
    ]
    timestamps.sort()

    rows = []
    for ts in timestamps:
        if rng.random() < 0.15:
            sensor = SENSOR_DOOR
            reading = rng.choice(["OPEN", "CLOSE"])
        else:
            sensor = rng.choice(SENSORS_MOTION)
            reading = rng.choice(["ON", "OFF"])
        rows.append(
            (ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M:%S.%f"), sensor, reading)
        )

    return rows


def generate_house_csv(
    house_id: str, events_mean: int, night_ratio: float, n_days: int, seed: int
) -> Path:
    rng = np.random.default_rng(seed)
    start_date = pd.Timestamp("2024-01-01")

    anomalous_days = set(rng.choice(n_days, size=max(1, n_days // 20), replace=False))

    all_rows = []
    for day_offset in range(n_days):
        date = start_date + pd.Timedelta(days=day_offset)
        all_rows.extend(
            _events_for_day(
                rng,
                date,
                events_mean,
                night_ratio,
                anomalous=day_offset in anomalous_days,
            )
        )

    df = pd.DataFrame(all_rows, columns=["date", "time", "sensor_id", "reading"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"casas_{house_id}_raw.csv"
    df.to_csv(out_path, header=False, index=False)
    return out_path


if __name__ == "__main__":
    for house_id, profile in HOUSE_PROFILES.items():
        path = generate_house_csv(house_id, **profile)
        print(
            f"✓ [{house_id}] {path} ({profile['n_days']} días, ~{profile['events_mean']} eventos/día)"
        )
