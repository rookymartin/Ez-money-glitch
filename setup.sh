#!/usr/bin/env bash
set -e

echo "Ez-Money-Glitch Setup"
echo "====================="

if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 not found. Install Python 3.10+ from python.org"
    exit 1
fi

if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation."
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt

echo ""
echo "Setup complete! To get started:"
echo "  source venv/bin/activate"
echo "  python run.py signals AAPL --no-nn"
echo "  python run.py backtest"
echo "  python run.py dashboard"
