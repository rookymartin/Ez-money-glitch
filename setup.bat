@echo off
echo Ez-Money-Glitch Setup
echo =====================

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

if exist venv (
    echo Virtual environment already exists, skipping creation.
) else (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo.
echo Setup complete! To get started:
echo   venv\Scripts\activate
echo   python run.py signals AAPL --no-nn
echo   python run.py backtest
echo   python run.py dashboard
pause
