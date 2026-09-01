@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set SKIP_SETUP=0
set SOURCE=auto

for %%A in (%*) do (
    if "%%A"=="--skip-setup" set SKIP_SETUP=1
    if "%%A"=="--synthetic" set SOURCE=synthetic
    if "%%A"=="--real" set SOURCE=real
)

if "%SOURCE%"=="auto" (
    if exist "data\real\casas_*_raw.csv" (
        set SOURCE=real
        echo [1/4] Real CASAS CSVs found in data\real\ - using real data (use --synthetic to force^).
    ) else (
        set SOURCE=synthetic
        echo [1/4] No real CASAS CSVs in data\real\ - using synthetic data (--real to require data^).
    )
)

if "%SOURCE%"=="real" if not exist "data\real\casas_*_raw.csv" (
    echo [ERROR] --real requested but no data\real\casas_*_raw.csv found. 1>&2
    echo         Download the WSU CASAS datasets into data\real\ and retry, 1>&2
    echo         or run without --real to use the synthetic fixtures. 1>&2
    exit /b 1
)

if %SKIP_SETUP%==0 (
    if not exist "venv\" (
        echo [1/4] Creating virtualenv in .\venv ...
        python -m venv venv
    ) else (
        echo [1/4] Reusing existing .\venv
    )
) else (
    echo [1/4][2/4] --skip-setup: venv and dependencies reused.
)

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] --skip-setup requires an existing .\venv. Run without it.
    exit /b 1
)
call venv\Scripts\activate.bat

if %SKIP_SETUP%==0 (
    echo [2/4] Installing dependencies...
    pip install -q -r requirements.txt
)

set PYTHONPATH=%cd%\src;%PYTHONPATH%

if "%SOURCE%"=="synthetic" (
    echo [3/4] Generating synthetic test data into data\synthetic\ ...
    python scripts\generate_test_fixtures.py
)

echo [3/4] Loading %SOURCE% data into SQLite ^(idempotent^)...
python src\ingestion\casas_loader.py --source %SOURCE%

echo [4/4] Launching the app at http://localhost:8501 ^(source: %SOURCE%^)...
streamlit run app\streamlit_app.py

endlocal
