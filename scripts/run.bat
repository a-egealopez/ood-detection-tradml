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
    if errorlevel 1 (
        echo [WARN] Full install failed ^(usually 'tick'^). Retrying without it, then tick last...
        findstr /v /i /r "^tick" requirements.txt > "%TEMP%\req_without_tick.txt"
        pip install -q -r "%TEMP%\req_without_tick.txt"
        del "%TEMP%\req_without_tick.txt"
        pip install -q tick
        if errorlevel 1 echo [WARN] 'tick' could not be built; the Hawkes detector will be unavailable.
    )
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