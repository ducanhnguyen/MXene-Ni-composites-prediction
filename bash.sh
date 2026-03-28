#!/usr/bin/env bash

set -e

echo "========================================"
echo "Setting up Python environment"
echo "========================================"

# Upgrade pip
python -m pip install --upgrade pip

# Check requirements.txt
if [ ! -f requirements.txt ]; then
  echo "❌ requirements.txt not found!"
  exit 1
fi

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "========================================"
echo "✅ All dependencies installed successfully"
echo "========================================"
