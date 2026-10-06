#!/usr/bin/env bash
set -e

# Navigate to project root
cd "$(dirname "$0")"

echo "==================================================="
echo "  Stellar-Forge Environment Setup & Launcher (Linux)"
echo "==================================================="

# Check for Python 3
if ! command -v python3 &>/dev/null; then
    echo "[Stellar-Forge] ERROR: python3 not found. Please install Python 3.10+."
    exit 1
fi

# Verify / create virtual environment
if [ -f "venv/bin/python" ]; then
    if ! venv/bin/python --version &>/dev/null; then
        echo "[Stellar-Forge] Virtual environment is broken. Recreating..."
        rm -rf venv
        python3 -m venv venv
    fi
else
    echo "[Stellar-Forge] Virtual environment not found. Creating..."
    rm -rf venv
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check and install dependencies
echo "[Stellar-Forge] Checking and installing dependencies..."
pip install -r requirements.txt --quiet

python3 scripts/build_terrain.py --if-needed

# Configure PYTHONPATH and run
export PYTHONPATH="$(pwd)/engine:${PYTHONPATH}"
echo "[Stellar-Forge] Starting Stellar-Forge simulation..."
python3 engine/main.py
