#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository root (this script lives in scripts/).
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SKIP_SETUP=false
SOURCE=real
for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --synthetic) SOURCE=synthetic ;;
    esac
done

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
    if ! pip install -q -r requirements.txt; then
        # 'tick' is a fragile native (C++) build; install everything else first,
        # then attempt tick alone and continue with a warning if it fails.
        echo "[WARN] Full install failed (usually 'tick'). Retrying without it, then tick last..."
        tmpfile="$(mktemp)"
        grep -vi '^tick' requirements.txt > "$tmpfile" || true
        pip install -q -r "$tmpfile" || true
        rm -f "$tmpfile"
        pip install -q tick || \
            echo "[WARN] 'tick' could not be built; the Hawkes detector will be unavailable."
    fi
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
