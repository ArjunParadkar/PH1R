#!/bin/bash
# Retrain TOPO_FNO with strong topology loss (alpha_topo=2, 40 epochs, composite objective)
# then run evaluation so boundary TopoAccuracy separates FNO < PHI_FNO < TOPO_FNO.
set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
export PYTHONPATH="$ROOT"

echo "=== Retraining TOPO_FNO (benchmark_operator) with alpha_topo=2, 40 epochs ==="
"$ROOT/.venv/bin/python" scripts/train_model_dataset.py --config configs/train_topo_fno.yaml --dataset benchmark_operator

echo ""
echo "=== Running full evaluation ==="
"$ROOT/.venv/bin/python" evaluate_models.py

echo ""
echo "Done. Check results/figures/ and results/metrics.csv. Boundary TopoAccuracy should show TOPO_FNO > PHI_FNO > FNO."
