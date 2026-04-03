# PRISM-AD: Full Implementation Summary

**PRISM-AD** stands for *Topology-Aware Physics-Constrained Operator Learning of Eigenmode Instability Spectra in Vlasov–Maxwell plasma systems*. The goal is to learn a map from plasma parameters to eigenvalue spectra (as 2D density images) with emphasis on **structure-preserving approximation** near stability transitions (where γ_max ≈ 0) and **topology** (components and holes in the spectrum), not only pointwise MSE. This document describes everything implemented: data, libraries, methods, models, training, evaluation, and outputs.

---

## 1. Libraries and Dependencies

| Library | Role |
|--------|------|
| **NumPy** | Arrays, random sampling, linear algebra, solver matrix construction. |
| **SciPy** | Sparse linear algebra: `scipy.sparse.linalg.eigs` (ARPACK) for rightmost eigenvalues; `scipy.sparse.diags`, `LinearOperator` for matrix-free operator. |
| **PyTorch** | All neural models, training loop, autograd, mixed precision (`torch.amp`), DataLoader, checkpointing. |
| **PyYAML** | Reading config files (`configs/*.yaml`) for data, training, and runtime. |
| **Matplotlib** | All figures: training/val curves, density maps, model comparison grids, bar charts, persistence diagrams, speedup plots. |
| **GUDHI** (optional) | Cubical persistence on 2D images: build cubical complex from density grid, compute H0/H1 persistence diagrams for evaluation. |
| **Persim** (optional) | Wasserstein distance between persistence diagrams (H0 and H1) for the reported topology metric. |
| **TopologyLayer** (optional) | Differentiable persistent homology: `LevelSetLayer2D` (superlevel on 2D grid) + `TopKBarcodeLengths` for H0/H1; used as topology regularizer in training when available. |

If GUDHI/Persim are not installed, **structure_proxy_distance** is reported (L2 between L2-normalized flattened images); it is **not** a topology metric—only GUDHI+Persim provide TopoDist (W0, W1). If TopologyLayer is not installed, a **differentiable fallback** is used in training: top-k pixel values (and for H1, top-k of negated image) as a simple featurization, with L2 loss between pred and truth features.

---

## 2. Data Pipeline

### 2.1 Parameter Space and Configuration

- **Source of truth:** `configs/data.yaml`.
- **Parameter space:** Each sample is a vector **p** ∈ ℝ^P with P = 8 (configurable). Parameters are drawn uniformly in [param_low, param_high] = [-1, 1].
- **Dataset size:** N_total = 600 (configurable). Train fraction 0.8 → 480 train, 120 val.
- **Boundary subset:** Indices i where |γ_max_i| < ε with ε = 0.05 (boundary_epsilon). Used for boundary-weighted loss and boundary MSE reporting.
- **OOD subset:** Same number of samples with parameters scaled by **ood_shift_factor** = 1.5 (i.e. params × 1.5). Used only for OOD MSE evaluation.

### 2.2 Eigenvalue Solver (Surrogate for Runtime Benchmark)

- **Purpose:** Surrogate eigenvalue operator used to benchmark iterative eigensolver runtime vs neural inference. Used for (1) generating training/val/OOD targets and (2) speedup comparison.
- **Implementation:** `prism_ad/solver.py`.
  - A **parameter-conditioned linear operator** A(p): tridiagonal matrix of size n_dim = 80. Diagonal and off-diagonals depend on **p**.
  - Matrix-free: `LinearOperator` with `matvec = A(p) @ x`.
  - **Eigenvalues:** `scipy.sparse.linalg.eigs(L, k=min(20, n_dim-1), which='LI', tol=1e-8)` (ARPACK, **largest imaginary part**). Instability is γ > 0 (λ = ω + iγ); γ_max = max(Im(λ)) is the growth rate, so we target largest imaginary part. On failure, a single default eigenvalue is returned.
- **Output per sample:** Complex array of length n_eigs. γ_max = max(Im(λ)).

### 2.3 From Eigenvalues to 2D Density Grid (Output Representation)

- **Fixed representation:** 128×128 image on a fixed (ω, γ) domain.
- **Domain:** ω ∈ [omega_min, omega_max] = [-0.5, 0.5], γ ∈ [gamma_min, gamma_max] = [-0.3, 0.3] (from config).
- **σ units:** density_sigma = 0.02 is in **physical (ω, γ) coordinate units**, not pixels. Gaussian is computed using the actual meshgrid values (omega_g, gamma_g) in physical space.
- **Method:** `prism_ad/grid_utils.py` — **eigs_to_density**:
  - **Out-of-domain:** Eigenvalues outside [omega_min, omega_max] × [gamma_min, gamma_max] are **discarded**. Then the grid is built and **renormalized** so the grid sums to 1.
  - For each in-domain eigenvalue λ = ω + iγ, add a Gaussian blob exp(-((ω'-ω)² + (γ'-γ)²) / (2σ²)) on the (ω', γ') meshgrid (physical units).
  - Sum over in-domain eigenvalues, then **normalize so the grid sums to 1**. Stored as float32.
- **Constants** (grid size, domain, σ, density_sigma_units, normalization) are in **data/processed/meta.json**.

### 2.4 Dataset Artifacts

- **data/processed/dataset.pt** (PyTorch):  
  `params` [N, P], `eigs` (list of complex arrays), `grid_truth` [N, 1, 128, 128], `gamma_max` [N], `ood_params`, `grid_ood`, `gamma_max_ood`.
- **data/processed/meta.json:** grid_size, omega/gamma bounds, density_sigma, normalization type, param_dim.
- **data/processed/splits.json:** train_idx, val_idx, boundary_idx, boundary_epsilon, ood_shift_factor.

**Script:** `scripts/gen_data.py` — reads config, samples params, runs solver for in-domain and OOD params, builds grids, saves the above.

---

## 3. Models (All Output [B, 1, 128, 128])

Every model maps **p** ∈ ℝ^P → density grid. Output is passed through **softplus** (nonnegative) then **sum-to-one normalization** (same as ground truth). Implemented in `prism_ad/models.py`.

### 3.1 MLP Baseline

- Input: **p** [B, P].
- Layers: P → 512 → 512 → 16384 (GELU).
- Reshape to [B, 1, 128, 128]; softplus; normalize.

### 3.2 CNN Baseline

- Embed **p** → 256 (FC), then 256 → 64×8×8; reshape to [B, 64, 8, 8].
- Decoder: ConvTranspose2d stack 8→16→32→64→128 (channels 64→64→32→16→8→1), GELU. Bilinear interpolation to 128×128 if needed.
- Softplus; normalize.

### 3.3 FNO Baseline (Minimal 2D FNO)

- **No input field:** only parameters. A **param-conditioned field** is built: **p** → embed → [B, C, 1, 1] with C = 16, broadcast to [B, C, 128, 128]. Two coordinate channels (x, y) in [-1, 1] are appended → input [B, C+2, 128, 128].
- **FNO blocks (×4):** Fourier convolution in 2D (modes 16×16), then 1×1 conv, GELU. Only the first C channels are updated; the 2 coordinate channels are carried through.
- Final 1×1 conv: (C+2) → 1. Softplus; normalize.
- Fourier block: real FFT2D, learnable weights in Fourier space (modes × modes), inverse FFT.

### 3.4 PI-FNO (Physics-Informed FNO)

- Same architecture as FNO plus an **auxiliary head**: from the same embedding vector **e** used to build the field, a linear layer predicts **3-class logits** (stable / boundary / unstable).
- **Stability labels:** From ground-truth γ_max and a threshold δ (boundary_delta = 0.05):  
  stable = γ_max < -δ, boundary = |γ_max| ≤ δ, unstable = γ_max > δ → classes 1, 0, 2.
- **Loss terms (in training):**
  - **L_mse:** Mean (pred - truth)² over pixels.
  - **L_boundary:** Boundary-weighted MSE: weight per sample b_i = exp(-|γ_max_i| / ε), then mean of b_i · (pred - truth)² (emphasizes samples near γ_max ≈ 0).
  - **L_aux:** Cross-entropy on the 3-class stability head.
  - **L_PI = L_mse + λ_boundary · L_boundary + λ_aux · L_aux** (λ’s from config, no tuning).

### 3.5 Topo-FNO

- Same as PI-FNO (same model class, same boundary + aux loss).
- **Additional loss:** **α · L_topo**, where L_topo is a **differentiable topology loss** (see below). Total loss = L_PI + α · L_topo (α from config).

---

## 4. Topology: Training Loss and Evaluation Metric

### 4.1 Differentiable Topology Loss (Training)

- **Purpose:** Encourage the predicted density image to have similar topological structure (components and holes) to the ground truth.
- **Implementation:** `prism_ad/topology_loss.py`.
- **Input:** Pred and truth [B, 1, 128, 128]. Internally **downsampled to 32×32** (config: topo_grid) for speed and, when using TopologyLayer, for stability.
- **Primary path (if TopologyLayer available):**
  - **LevelSetLayer2D(size=(32,32), maxdim=1, sublevel=False):** Superlevel-set persistence on the 2D grid (high density = early in filtration).
  - **TopKBarcodeLengths(dim=0, k=16)** and **TopKBarcodeLengths(dim=1, k=16):** Top 16 birth-death lengths for H0 and H1 as a differentiable vector.
  - **L_topo = ‖feat0(pred) - feat0(truth)‖² + β ‖feat1(pred) - feat1(truth)‖²** (β from config). Batched by running the layer per sample and stacking.
- **Fallback (no TopologyLayer):** Featurize each image by the **top-k pixel values** (and for “H1” surrogate, top-k of negated image); same L2 loss between pred and truth features. Fully differentiable.

### 4.2 Topology Evaluation Metric (Reporting and Figures)

- **Purpose:** Report a single “topology distance” between predicted and true density images (and W0, W1 separately when available).
- **Implementation:** `prism_ad/persistence_eval.py`.
- **With GUDHI + Persim:**
  - **Cubical complex** on the 2D image (superlevel: cells = -density so high density appears first). Compute persistence; extract H0 and H1 diagrams (birth, death) pairs.
  - **W0 = Wasserstein(D0(pred), D0(truth)), W1 = Wasserstein(D1(pred), D1(truth))** via persim.
  - **TopoDist = W0 + β·W1** (β = 1 in config).
- **Without GUDHI/Persim:** **structure_proxy_distance** is reported (L2 between L2-normalized flattened images); it is **not** a topology metric. Only GUDHI+Persim provide TopoDist (W0, W1). Tables and figures label “TopoDist” only when real; otherwise “Structure proxy dist”.

---

## 5. Training Pipeline

- **Entry point:** `scripts/train_model.py` — loads a train config (e.g. `configs/train_mlp.yaml`), builds the corresponding model, loads data via `prism_ad/dataset.get_dataloaders`, and calls **train(model, dataset, config)**.
- **Unified training:** `prism_ad/training.py` — **train(model, dataset, config)**:
  - Device: from config (e.g. cuda); mixed precision (autocast + GradScaler) only when device is CUDA.
  - Optimizer: Adam, lr from config.
  - For each epoch: **train_one_epoch** (MSE + boundary weight + aux if applicable + α·L_topo if Topo-FNO), then **evaluate** on val and boundary loaders (MSE and, when topo_fn is given, topology loss value).
  - **Loss equation:** L = MSE + λ_boundary·L_boundary + λ_aux·L_aux + α·L_topo (logged in metrics.jsonl and summary.json).
  - Best checkpoint by val MSE saved as **run_dir/checkpoint.pt** (state_dict + epoch).
  - Logs: **run_dir/metrics.jsonl** (first line: loss_equation; per-epoch train_mse, val_mse, boundary_mse, val_topodist, runtime); **run_dir/train_curve.png**, **run_dir/val_curve.png**; **run_dir/summary.json** (best_val_mse, final_train_mse, epochs, loss_equation, torch_version, cudnn_deterministic).

---

## 6. Evaluation and Figures

### 6.1 Evaluation Script (`scripts/eval_all.py`)

- **Input:** Glob of run directories (e.g. `runs/*`). Only directories that contain **checkpoint.pt** are evaluated.
- **Per run:** Load model (MLP/CNN/FNO/PI-FNO/Topo-FNO) from checkpoint; run on **val** indices (MSE, collect preds/truths for topology); on **boundary** indices (boundary MSE); on **OOD** samples (OOD MSE). **Topology distance** (and W0, W1 when available) is computed on validation predictions via **topology_distance_batch**.
- **Outputs:** **run_dir/eval_metrics.json** (val_mse, boundary_mse, ood_mse, topo_w0, topo_w1, topo_dist); **run_dir/eval_preds.pt** (val_preds, val_truths). **runs/eval_all_metrics.json** aggregates all runs.

### 6.2 Figures Script (`scripts/make_figs.py`)

- **Input:** Same runs glob; uses **data/processed**, **runs/eval_all_metrics.json**, and optionally **runs/runtime.json**.
- **Figures produced:**
  - **runs/figs/density_mapping.png:** One example eigenvalue density (128×128) with (ω, γ) axes.
  - **runs/figs/model_comparison_grid.png:** One val sample: truth vs each model’s prediction and |pred - truth|.
  - **runs/figs/ph_distance_barchart.png:** Bar chart of val MSE and TopoDist per model.
  - **runs/figs/improvement_bars.png:** Δ MSE and Δ TopoDist for PI-FNO vs FNO, Topo-FNO vs PI-FNO, Topo-FNO vs FNO.
  - **runs/figs/boundary_stress.png:** Scatter of MSE vs |γ_max| for a subset of val samples (all models).
  - **runs/figs/ood_plot.png:** OOD MSE per model (bar).
  - **runs/figs/mse_lies_panel.png:** Three samples where FNO has relatively low MSE but high TopoDist (or high ratio); show truth vs FNO pred.
  - **runs/figs/runtime_speedup.png:** Log-scale speedup (solver time / model inference time) per model (from runtime.json).
  - **runs/<name>/figs/persistence_diagrams.png:** For one run, H0 and H1 persistence diagrams (truth vs pred) from GUDHI when available.

---

## 7. Runtime Benchmark and Summary

### 7.1 Runtime (`scripts/bench_runtime.py`)

- **Solver:** For n_samples (e.g. 50) parameter vectors, time **solve_one** per sample; report mean (and std) solver time per sample in ms.
- **Models:** For each trained model (with checkpoint), measure inference time (batch size 1 and 32), with warmup; store mean time per sample (ms).
- **Output:** **runs/runtime.json** (solver_ms_per_sample, solver_std_ms, and per-model *_ms, *_std_ms). **runs/figs/runtime_speedup.png** = solver_time / model_time (log scale).

### 7.2 Summarize (`scripts/summarize_results.py`)

- Reads **runs/eval_all_metrics.json** and **runs/runtime.json**.
- **Prints:** Speedup (solver vs each model), Val/Boundary/OOD MSE table, Topology (W0, W1, TopoDist) table, improvement deltas (PI-FNO vs FNO, Topo-FNO vs PI-FNO, Topo-FNO vs FNO) with absolute and % change, and paths to all figure PNGs.
- **Writes:** **runs/results_summary.json** (speedup, val_mse, boundary_mse, ood_mse, topo_dist, deltas, figure_paths).

---

## 8. Config Files

- **configs/data.yaml:** n_total, train_frac, boundary_epsilon, ood_shift_factor, param_dim, param bounds, grid_size, (ω, γ) bounds, density_sigma, n_eigs, solver_tol, seed.
- **configs/train_mlp.yaml, train_cnn.yaml, train_fno.yaml:** model name, data config path, epochs (10), batch_size (32), lr (1e-3), device (cuda), mixed_precision (true), run_dir (e.g. runs/mlp). FNO configs add fno_modes, fno_layers, fno_channels.
- **configs/train_pi_fno.yaml, train_topo_fno.yaml:** Same as FNO plus lambda_boundary, lambda_aux, boundary_delta; Topo-FNO adds alpha_topo, beta_topo, topo_grid (32).
- **configs/runtime.yaml:** data config, runs dir, n_warmup, n_samples, batch_sizes for inference timing.

---

## 9. Scripts and One-Shot Run

| Script | Purpose |
|--------|---------|
| **scripts/gen_data.py** | Sample params, run solver, build grids, write dataset.pt, meta.json, splits.json. |
| **scripts/train_model.py** | Train one model (MLP/CNN/FNO/PI-FNO/Topo-FNO) from its config. |
| **scripts/eval_all.py** | Evaluate all runs with checkpoints; write eval_metrics.json per run and eval_all_metrics.json. |
| **scripts/make_figs.py** | Generate all figures under runs/figs/ and runs/<name>/figs/. |
| **scripts/bench_runtime.py** | Measure solver and model inference times; write runtime.json and speedup figure. |
| **scripts/summarize_results.py** | Print verification checklist and write results_summary.json. |

**One-shot:** `run_all.sh` runs in order: gen_data → train (mlp, cnn, fno, pi_fno, topo_fno) → eval_all → make_figs → bench_runtime → summarize_results. Results and figures are in **runs/** and **runs/figs/**.

---

## 10. Summary of Methods Used

- **Eigenvalue computation:** ARPACK (SciPy) for rightmost eigenvalues of a parameter-dependent tridiagonal operator (surrogate for expensive plasma solvers).
- **Output representation:** Fixed 128×128 (ω, γ) grid; Gaussian mixture on eigenvalues with fixed σ; sum-to-one normalization.
- **Baselines:** MLP (fully connected), CNN (param embedding + transposed conv decoder), FNO (param-conditioned 2D Fourier neural operator with coordinate channels).
- **Physics-informed terms:** Boundary-weighted MSE (samples near γ_max ≈ 0) and auxiliary 3-class stability head (stable/boundary/unstable from γ_max).
- **Topology in training:** Differentiable loss on persistence-derived features (TopologyLayer: LevelSet2D + TopKBarcodeLengths) or fallback (top-k pixel L2).
- **Topology in evaluation:** Cubical persistence (GUDHI) + Wasserstein on diagrams (Persim), or L2 structure proxy.
- **Training:** Single pipeline with Adam, optional mixed precision; combined loss = MSE + λ_boundary·L_boundary + λ_aux·L_aux + α·L_topo (for Topo-FNO).
- **Evaluation:** Val/boundary/OOD MSE; topology distance (W0, W1, TopoDist); improvement deltas between PI-FNO, Topo-FNO, and FNO.
- **Speedup:** Solver time per sample vs model inference time per sample (log-scale plot and numbers in results_summary.json).

This is the full implementation from data generation through to reported results and figures.
