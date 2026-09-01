"""Generate synthetic CASAS-style CSVs using the Markov generator.

The events are produced by ``ingestion/markov_generator`` so the streams carry the
deliberate structure (asymmetric movement graph, latent day regime) that the
sequential/order detectors are meant to exploit. Output CSVs use the exact schema
the loader expects (``date, time, sensor_id, reading``, no header).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import SYNTHETIC_DATA_DIR
from ingestion.markov_generator import generate_house_stream

HOUSE_PROFILES = {
    "aruba": {"events_mean": 180, "night_ratio": 0.08, "n_days": 90, "seed": 1},
    "cairo": {"events_mean": 90, "night_ratio": 0.05, "n_days": 75, "seed": 2},
    "milan": {"events_mean": 260, "night_ratio": 0.15, "n_days": 85, "seed": 3},
    "tulum": {"events_mean": 130, "night_ratio": 0.10, "n_days": 70, "seed": 4},
}


def generate_house_csv(house_id: str, **profile) -> Path:
    df = generate_house_stream(house_id=house_id, **profile)
    SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SYNTHETIC_DATA_DIR / f"casas_{house_id}_raw.csv"
    df.to_csv(out_path, header=False, index=False)
    return out_path


if __name__ == "__main__":
    for house_id, profile in HOUSE_PROFILES.items():
        path = generate_house_csv(house_id, **profile)
        print(
            f"[{house_id}] {path} ({profile['n_days']} days, "
            f"~{profile['events_mean']} events/day)"
        )
