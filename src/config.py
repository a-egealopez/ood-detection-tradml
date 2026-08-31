import logging
from pathlib import Path

# Shared low-level constants. They live here (a package with no dependencies) so
# both ``features`` and ``detectors`` can import them without triggering a
# circular import (detectors/sequential/markov_sequence_detector imports from
# ``features``, so ``features`` must never import the ``detectors`` package).
EPSILON = 1e-8
DEFAULT_RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

HOUSES = ("aruba", "cairo", "milan", "tulum")

# Minimum days a house must have before it can be scored by the pipeline.
MIN_DAYS = 10

SOURCES = ("real", "synthetic")
REAL_DATA_DIR = DATA_DIR / "real"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
_DB_FILENAMES = {"real": "sensor_data.db", "synthetic": "sensor_data_synthetic.db"}


def _validate_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"Unknown source: '{source}'. Valid: {SOURCES}")


def raw_csv_path(house: str, source: str = "real") -> Path:
    if house not in HOUSES:
        raise ValueError(f"Unknown house: '{house}'. Valid: {HOUSES}")
    _validate_source(source)
    data_dir = REAL_DATA_DIR if source == "real" else SYNTHETIC_DATA_DIR
    return data_dir / f"casas_{house}_raw.csv"


def db_path(source: str = "real") -> Path:
    _validate_source(source)
    return DATA_DIR / _DB_FILENAMES[source]


DB_PATH = db_path("real")


def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / "app.log"),
        ],
    )
    return logging.getLogger()
