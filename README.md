# Progressive Trust Propagation for Urban Region Classification

Code for reproducing the experiments in *"When Labels Are Far Away: Progressive Trust Propagation for Urban Region Classification under Spatial Label Scarcity."*

## Repository Structure

```
├── config.py               # Global constants (cities, training hyperparameters, device)
├── data_utils.py           # Data loading and preprocessing (normalization, mask creation)
├── base_models.py          # All base classifiers: RF, MLP, GCN, GAT, GraphSAGE, SGC, APPNP, GPRGNN, H2GCN
├── trainer.py              # Training loop (early stopping) and Random Forest wrapper
├── evaluation.py           # Metric computation (AUC, Recall, Macro-F1, Micro-F1)
├── post_processing.py      # Post-processing methods: LP, C&S, and PTP (our method)
├── run_experiment.py        # Main experiment runner (two-phase: cache predictions → evaluate)
└── Graph_Dataset_Block_Balanced/   # Data directory (expected location for .pt graph files)
```

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- PyTorch Geometric
- scikit-learn
- pandas, numpy

## Data

Each city benchmark is stored as a single PyG `Data` object (e.g., `London_05pct.pt`) containing node features, edge indices, labels, train/test masks, and the Borderland (bridge) mask. Place these files under `Graph_Dataset_Block_Balanced/`.

> **Note:** The dataset files are too large to host on GitHub. They will be shared via Google Drive.

## Quick Start

**Run a single city with PTP only (1 run):**

```bash
python run_experiment.py --city London --ratio 05pct --n_runs 1 --imlp_only
```

**Full reproduction (all cities, 10 runs, all post-processing methods):**

```bash
python run_experiment.py --city all --ratio 05pct --n_runs 10
```

## Two-Phase Execution

The experiment runner separates training from evaluation. This avoids retraining base models when sweeping post-processing configurations.

**Phase 1 — Train base models and cache predictions:**

```bash
python run_experiment.py --city all --phase cache
```

Predictions are saved as `.pt` files in the `cache/` directory.

**Phase 2 — Apply post-processing and evaluate:**

```bash
python run_experiment.py --city all --phase eval
```

By default, `--phase both` runs both phases sequentially.

## Command-Line Options

| Argument | Default | Description |
|---|---|---|
| `--city` | `all` | City to run: `Harbin`, `London`, `Paris`, or `all` |
| `--ratio` | `05pct` | Training ratio: `05pct` through `25pct`, or `all` |
| `--n_runs` | `10` | Number of runs (seeds 42–51) |
| `--base_models` | all 9 | Subset of base models, e.g., `--base_models MLP GCN` |
| `--phase` | `both` | `cache` (train only), `eval` (post-process only), or `both` |
| `--base_only` | off | Skip all post-processing |
| `--imlp_only` | off | Run only base + PTP (skips LP and C&S) |
| `--cache_dir` | `cache` | Directory for cached base predictions |
| `--output_dir` | `results` | Directory for result CSVs |

## Outputs

Results are written to `results/`:

- `results_{City}_{Ratio}.csv` — Per-run metrics for every (base model, post-processing, region) combination.
- `all_results.csv` — Combined results across all cities and ratios.
- `summary.csv` — Aggregated mean ± std for all metrics.

## PTP Hyperparameters

PTP uses fixed hyperparameters across all experiments (set in `run_experiment.py`):

| Symbol | Parameter | Value |
|---|---|---|
| *K* | `num_stages` | 3 |
| *α* | `alpha_calibrate` / `alpha_aggregate` | 0.999 |
| *γ* | `gamma` | 0.5 |
| *τ* | `conf_threshold` | 0.8 |
| *λ* | `dense_weight` | 0.8 |
| *β* | `sharpness` | 10.0 |
| *τ_m* | `margin_thresh` | 0.3 |
| *T* | `num_iter_calibrate` / `num_iter_aggregate` | 200 |

To modify these, edit the `IMLPConfig(...)` instantiation in `run_experiment.py`.
