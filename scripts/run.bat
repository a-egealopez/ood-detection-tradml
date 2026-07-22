@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set SKIP_SETUP=0
set SOURCE=real

for %%A in (%*) do (
    if "%%A"=="--skip-setup" set SKIP_SETUP=1
    if "%%A"=="--synthetic" set SOURCE=synthetic
)

if %SKIP_SETUP%==0 (
    if not exist "venv\" (
        echo [1/4] Creando entorno virtual en .\venv ...
        python -m venv venv
    ) else (
        echo [1/4] Entorno virtual ya existe, se reutiliza.
    )

    call venv\Scripts\activate.bat

    echo [2/4] Instalando dependencias...
    pip install -q -r requirements.txt
) else (
    echo [1/4][2/4] --skip-setup: se omite creacion de venv e instalacion.
)

set PYTHONPATH=%cd%\src;%PYTHONPATH%

if "%SOURCE%"=="synthetic" (
    echo [3/4] Generando datos sinteticos de prueba en data\synthetic\ ...
    python scripts\generate_test_fixtures.py
)

echo [3/4] Cargando datos ^(fuente: %SOURCE%^) a SQLite ^(idempotente^)...
python src\ingestion\casas_loader.py --source %SOURCE%

echo [4/4] Lanzando la app en http://localhost:8501 ^(elige la fuente '%SOURCE%' en la barra lateral^)...
streamlit run app\streamlit_app.py

endlocal