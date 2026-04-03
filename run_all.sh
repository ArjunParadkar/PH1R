#!/usr/bin/env bash
# Full pipeline: gen 3 datasets -> train all models on each -> eval -> figures -> summary.

set -e
cd "$(dirname "$0")"
mkdir -p runs data runs/results runs/figs

echo "=== 1. Generate datasets ==="
python scripts/gen_data_dataset.py --config configs/data_benchmark.yaml
python scripts/gen_data_dataset.py --config configs/data_vm_gkeyll.yaml
python scripts/gen_data_dataset.py --config configs/data_lhd.yaml

echo "=== 2. Train all models on each dataset ==="
for ds in benchmark_operator vlasov_maxwell_gkeyll fusion_real_lhd; do
  for cfg in configs/train_mlp.yaml configs/train_cnn.yaml configs/train_fno.yaml configs/train_pi_fno.yaml configs/train_topo_fno.yaml; do
    python scripts/train_model_dataset.py --config "$cfg" --dataset "$ds" || true
  done
done

echo "=== 3. Eval all datasets ==="
python scripts/eval_all_datasets.py --datasets benchmark_operator vlasov_maxwell_gkeyll fusion_real_lhd

echo "=== 4. Generate figures per dataset ==="
python scripts/make_figs_dataset.py --datasets benchmark_operator vlasov_maxwell_gkeyll fusion_real_lhd

echo "=== 5. Summarize ==="
python scripts/summarize_results.py 2>/dev/null || true
echo "Done. Check runs/results/ and runs/figs/<dataset>/"
