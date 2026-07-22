#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SKIP_SETUP=false
SOURCE=real
for arg in "$@"; do
    case "$arg" in
        --skip-setup) SKIP_SETUP=true ;;
        --synthetic) SOURCE=synthetic ;;
    esac
done

if [ "$SKIP_SETUP" = false ]; then
    if [ ! -d "venv" ]; then
        echo "[1/4] Creando entorno virtual en ./venv ..."
        python3 -m venv venv
    else
        echo "[1/4] Entorno virtual ya existe, se reutiliza."
    fi

    source venv/bin/activate

    echo "[2/4] Instalando dependencias..."
    pip install -q -r requirements.txt
else
    echo "[1/4][2/4] --skip-setup: se omite creación de venv e instalación."
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

if [ "$SOURCE" = "synthetic" ]; then
    echo "[3/4] Generando datos sintéticos de prueba en data/synthetic/ ..."
    python scripts/generate_test_fixtures.py
fi

echo "[3/4] Cargando datos (fuente: $SOURCE) a SQLite (idempotente, seguro correr siempre)..."
python src/ingestion/casas_loader.py --source "$SOURCE"

echo "[4/4] Lanzando la app en http://localhost:8501 (elige la fuente '$SOURCE' en la barra lateral)..."
streamlit run app/streamlit_app.py