#!/bin/bash
set -e
cd "$(dirname "$0")"
echo "Setting up Python environment (first run takes ~30s)..."
python3 -m venv .venv
.venv/bin/pip install -q numpy matplotlib
echo "Launching PRISM-AD Tokamak Visualizer..."
.venv/bin/python3 tokamak_viz.py
