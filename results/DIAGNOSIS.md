# Evaluation diagnosis (sanity checks)

## What was done

1. **Sanity check A (checkpoint identity)**  
   - For each model we print: checkpoint path, parameter count, fingerprint (sum of all weights).  
   - **Result:** All five models now have **different fingerprints** (PHI_FNO and TOPO_FNO were sharing the same checkpoint before; TOPO_FNO was retrained so they differ).  
   - Fingerprint uses the sum of all parameters so different checkpoints get different values.

2. **Sanity check B (topology loss in training)**  
   - For TOPO_FNO we log per epoch: `spectral_loss`, `topo_loss`, `total_loss`, `grad_norm`.  
   - **Result:** During training, `topo_loss` is non-constant (~9e-7), `grad_norm` is large and changes. So the topology term is in the loss and gets backpropagated.  
   - To run the **alpha stress test**:  
     `PYTHONPATH=. .venv/bin/python scripts/alpha_stress_test.py`  
     (trains 5 epochs with alpha_topo in {0, 10, 100} and compares boundary topo_error.)

3. **Sanity check C (PDs from model output)**  
   - For one boundary sample we compute:  
     TopoError(truth, pred_FNO), TopoError(truth, pred_TOPO), TopoError(pred_FNO, pred_TOPO).  
   - **Result:** TopoError(pred_FNO, pred_TOPO) ≈ 0.01 (nonzero), so FNO and TOPO_FNO outputs differ. Persistence diagrams are computed from the **model output grids**, not from a shared source.  
   - Saved: `results/debug/boundary_sample_*.npy` and `boundary_sample_clouds.png`.

## Why boundary TopoAccuracy is still similar across models

- **Training** uses a **differentiable surrogate** (multi-scale top-k L2 on density) in `prism_ad/topology_loss.py`.  
- **Evaluation** uses **cubical persistence + Wasserstein** (GUDHI + persim).  
- Minimizing the surrogate does not necessarily minimize the evaluation Wasserstein, so boundary TopoAccuracy can stay similar even when topology loss is active.  
- To align training and evaluation you would need either:  
  - a **differentiable persistence** layer (e.g. topologylayer) so the same topological quantity is minimized and evaluated, or  
  - a **surrogate** that is a better proxy for Wasserstein (e.g. learned or hand-designed).

## Data splits

- **Train / validation / boundary / OOD** are all defined in `splits.json` and used in evaluation.  
- `evaluate_models.py` uses: train (first 300), validation, boundary, OOD (first 200).

## Figures

- Only these 6 figures are written (old ones are removed each run):  
  `figure1_runtime_comparison.png`, `figure2_spectral_generalization.png`, `figure3_topo_error.png`, `figure4_boundary_topo_accuracy.png`, `figure5_correlation.png`, `figure6_qualitative.png`
