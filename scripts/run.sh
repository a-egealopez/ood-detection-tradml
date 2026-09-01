#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root (this script lives in scripts/).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_SETUP=false
SOURCE=auto

for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --synthetic) SOURCE=synthetic ;;
        --real) SOURCE=real ;;
    esac
done

REAL_CSV_PATTERN="data/real/casas_*_raw.csv"
has_real_csvs() { compgen -G "$REAL_CSV_PATTERN" > /dev/null; }

# Resolve the data source. Default is auto-detect: real CASAS CSVs if present
# (users can drop their own WSU CASAS downloads into data/real/), synthetic
# otherwise so a fresh clone works without any data.
if [ "$SOURCE" = "auto" ]; then
    if has_real_csvs; then
        SOURCE=real
        echo "[1/4] Real CASAS CSVs found in data/real/ - using real data (use --synthetic to force)."
    else
        SOURCE=synthetic
        echo "[1/4] No real CASAS CSVs in data/real/ - using synthetic data (--real to require data)."
    fi
fi

if [ "$SOURCE" = "real" ] && ! has_real_csvs; then
    echo "[ERROR] --real requested but no data/real/casas_*_raw.csv found." >&2
    echo "        Download the WSU CASAS datasets into data/real/ and retry," >&2
    echo "        or run without --real to use the synthetic fixtures." >&2
    exit 1
fi

# [1/4] Virtualenv
if [ ! -d "venv" ]; then
    if [ "$SKIP_SETUP" = true ]; then
        echo "[ERROR] --skip-setup requires an existing ./venv. Run without it." >&2
        exit 1
    fi
    echo "[1/4] Creating virtualenv in ./venv ..."
    python3 -m venv venv
else
    echo "[1/4] Reusing existing ./venv"
fi

# Always use the venv, also with --skip-setup.
# shellcheck disable=SC1091
source venv/bin/activate

# [2/4] Dependencies
if [ "$SKIP_SETUP" = false ]; then
    echo "[2/4] Installing dependencies..."
    pip install -q -r requirements.txt
else
    echo "[1/4][2/4] --skip-setup: venv and dependencies reused."
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# [3/4] Data
if [ "$SOURCE" = "synthetic" ]; then
    echo "[3/4] Generating synthetic test data into data/synthetic/ ..."
    python scripts/generate_test_fixtures.py
fi

echo "[3/4] Loading $SOURCE data into SQLite (idempotent)..."
python src/ingestion/casas_loader.py --source "$SOURCE"

# [4/4] App
STREAMLIT_EXTRA=()
if grep -qi microsoft /proc/version 2>/dev/null; then
    # WSL has no native browser handler; auto-open fails with 'gio'. Launch
    # headless and let the user open the URL manually.
    STREAMLIT_EXTRA+=(--server.headless true)
fi

echo "[4/4] Launching the app at http://localhost:8501 (source: $SOURCE)..."
if [ -n "${STREAMLIT_EXTRA[*]}" ]; then
    echo "      If the browser does not open, visit http://localhost:8501"
fi
streamlit run app/streamlit_app.py "${STREAMLIT_EXTRA[@]}"
