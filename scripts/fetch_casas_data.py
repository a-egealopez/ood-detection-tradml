"""Download and convert the real WSU CASAS datasets into the loader's CSV schema.

The CASAS Smart Home dataset (aruba, cairo, milan, tulum) is published on Zenodo
under a CC-BY-4.0 license (Cook, 2025, DOI: 10.5281/zenodo.17180309). Its files are
whitespace-separated text lines of the form::

    <date> <time> <sensor_id> <reading>[ <activity_label>...]

Each house ships one file (``tulum`` ships two: ``tulum1.txt`` + ``tulum2.txt``).
Sensors are PIR motion (``M###``), magnetic door (``D###``) and ambient temperature
(``T###``). The loader in ``src/ingestion/casas_loader.py`` expects comma-separated
CSVs with columns ``date,time,sensor_id,reading`` and only understands
``ON``/``OFF``/``OPEN``/``CLOSE`` readings, so this script:

1. Downloads ``new_labeled_data.zip`` from Zenodo (or reuses a local copy).
2. Parses every line, keeping only rows whose reading is an event
   (``ON``/``OFF``/``OPEN``/``CLOSE``) and whose sensor is motion or door.
   Activity labels, temperature values and garbled readings are dropped.
3. Writes one ``data/real/casas_<house>_raw.csv`` per house (``tulum`` merges
   its two files), matching the loader's schema.

Usage::

    python scripts/fetch_casas_data.py [--output-dir data/real] [--zip-path <local.zip>]
"""

import argparse
import logging
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import HOUSES, setup_logging

logger = logging.getLogger(__name__)

# Zenodo record that packages exactly the four houses of this project.
ZENODO_RECORD_URL = (
    "https://zenodo.org/api/records/17180309/files/new_labeled_data.zip/content"
)
DEFAULT_ZIP_PATH = Path(tempfile.gettempdir()) / "casas_new_labeled_data.zip"

# Which source files feed each house (``tulum`` is split across two files).
HOUSE_TO_FILES = {
    "aruba": ["aruba.txt"],
    "cairo": ["cairo.txt"],
    "milan": ["milan.txt"],
    "tulum": ["tulum1.txt", "tulum2.txt"],
}

# Readings the loader understands; everything else is dropped.
VALID_READINGS = {"ON", "OFF", "OPEN", "CLOSE"}
# Sensor prefixes that represent events (motion / door). Temperature rows (T###)
# carry numeric readings and are not events.
EVENT_SENSOR_PREFIXES = ("M", "D")


def _download_zip(zip_path: Path) -> Path:
    """Download the Zenodo archive to ``zip_path`` and return it."""
    if zip_path.exists():
        logger.info(f"Reusing local archive: {zip_path}")
        return zip_path

    logger.info(f"Downloading {ZENODO_RECORD_URL} -> {zip_path}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(ZENODO_RECORD_URL, zip_path)
    logger.info("Download complete")
    return zip_path


def _parse_lines(lines: list[str], source_name: str) -> list[str]:
    """Parse WSU text lines into loader-compatible ``date,time,sensor,reading`` rows.

    A line has the shape ``date time sensor reading [label...]``. Rows whose
    reading is not an event (temperature values, garbled tokens) or whose sensor
    is not motion/door are skipped, as are readings whose sensor column is missing.
    """
    rows: list[str] = []
    n_lines = len(lines)
    n_skipped = 0
    for line in lines:
        fields = line.split()
        if len(fields) < 4:
            n_skipped += 1
            continue
        date, time_, sensor_id, reading = fields[:4]
        if reading not in VALID_READINGS or not sensor_id.startswith(
            EVENT_SENSOR_PREFIXES
        ):
            n_skipped += 1
            continue
        rows.append(f"{date},{time_},{sensor_id},{reading}")

    logger.info(
        f"[{source_name}] {len(rows)}/{n_lines} event rows kept, "
        f"{n_skipped} dropped"
    )
    return rows


def convert_archive(zip_path: Path, output_dir: Path) -> dict[str, int]:
    """Extract and convert the Zenodo archive into per-house loader CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    with zipfile.ZipFile(zip_path) as archive:
        for house in HOUSES:
            house_members = HOUSE_TO_FILES[house]
            rows: list[str] = []
            for member in house_members:
                if member not in archive.namelist():
                    raise FileNotFoundError(
                        f"Expected {member!r} in {zip_path} but it is missing"
                    )
                with archive.open(member) as fh:
                    lines = fh.read().decode("utf-8").splitlines()
                rows.extend(_parse_lines(lines, member))
            out_csv = output_dir / f"casas_{house}_raw.csv"
            out_csv.write_text("\n".join(rows) + "\n")
            counts[house] = len(rows)
            logger.info(f"[{house}] Wrote {out_csv} ({len(rows)} rows)")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "real",
        help="Directory to write casas_<house>_raw.csv files (default: data/real)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=DEFAULT_ZIP_PATH,
        help="Local copy of new_labeled_data.zip to reuse instead of downloading",
    )
    args = parser.parse_args()

    setup_logging()

    zip_path = _download_zip(args.zip_path)
    counts = convert_archive(zip_path, args.output_dir)

    logger.info("Conversion complete:")
    for house in HOUSES:
        logger.info(f"  {house}: {counts[house]} event rows")


if __name__ == "__main__":
    main()
