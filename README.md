# Explainable Detection of Byzantine Attacks in Drone Swarms

A research-grade implementation of a spatio-temporal Graph Attention Network (GAT+GRU) for detecting Byzantine faults in UAV swarms, with full explainability via GNNExplainer and permutation-based feature importance.

---

## Overview

No public dataset exists specifically for Byzantine swarm attacks. This project bridges the gap by:

1. Using the [UAVs-Dataset-Under-Normal-and-Cyberattacks](https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks) — ~55K rows, 53 features (16 physical + 37 cyber), covering Benign / DoS / Replay / Evil Twin / FDI attacks.
2. **Simulating swarm topology** on top of the real per-drone telemetry to create graph-structured data.
3. **Injecting Byzantine-specific attacks** (false-state, intermittent, colluding, delay) at the swarm level.

### Proposed Model — GAT+GRU

```
Input (B, T, N, F)
  └─ GRU  (2 layers, per-node temporal encoding)
       └─ GATv2Conv  (2 layers, multi-head spatial aggregation)
            └─ MLP classifier  →  node-level binary prediction
```

---

## Project Structure

```
iod/
├── data/                      # Raw & processed data
│   └── raw/                   # Cloned dataset repo
├── src/                       # Core source modules
│   ├── config.py              # All hyperparameters & paths
│   ├── data_loader.py         # Load & merge CSV
│   ├── preprocessing.py       # Clean, normalise, split
│   └── utils.py               # Logging, Timer, JSON helpers
├── models/                    # Model architectures
│   ├── gat_temporal.py        # Proposed GAT+GRU model
│   ├── graphsage_temporal.py  # GraphSAGE+GRU baseline
│   ├── mlp.py                 # Baseline MLP
│   ├── lstm.py                # Baseline LSTM
│   ├── cnn.py                 # Baseline 1D-CNN
│   ├── gcn.py                 # Baseline GCN
│   └── model_utils.py         # FocalLoss, EarlyStopping
├── attacks/                   # Byzantine attack injection
│   ├── base_attack.py         # Abstract interface
│   ├── false_state.py         # False-state / FDI
│   ├── intermittent.py        # Intermittent attack
│   ├── colluding.py           # Colluding drones
│   ├── delay.py               # Stale-data delay
│   └── attack_manager.py      # Multi-attack orchestrator
├── graphs/                    # Swarm graph construction
│   ├── swarm_simulator.py     # Tabular → temporal swarm graphs
│   ├── knn_graph.py           # KNN topology
│   ├── distance_graph.py      # Distance-threshold topology
│   ├── hexagonal_graph.py     # Hexagonal lattice topology
│   └── temporal_graph.py      # Temporal sequence builder
├── evaluation/                # Metrics & evaluation
│   ├── metrics.py             # Accuracy, F1, ROC-AUC, etc.
│   ├── swarm_metrics.py       # Localisation accuracy, latency, per-node FPR
│   ├── evaluator.py           # Full pipeline + plot generation
│   ├── ablation.py            # Ablation study utilities
│   └── explainability.py      # GNNExplainer + SHAP + temporal importance
├── scripts/                   # Runnable entry points
│   ├── download_data.py       # Clone dataset repo
│   ├── preprocess.py          # Run preprocessing pipeline
│   ├── build_graphs.py        # Build PyG graph dataset
│   ├── train_baselines.py     # Train MLP / LSTM / CNN / GCN
│   ├── train_gnn_temporal.py  # Train proposed GAT+GRU model
│   ├── evaluate.py            # Full evaluation & plots
│   ├── explain.py             # Explainability outputs
│   ├── ablation.py            # Ablation study
│   └── run_all.py             # End-to-end pipeline
├── outputs/                   # Generated results (git-ignored)
│   ├── figures/
│   ├── tables/
│   └── models/
└── requirements.txt
```

---

## Installation

```bash
# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust for your CUDA version)
# See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```

---

## Quick Start

### Run the complete pipeline

```bash
python scripts/run_all.py
```

### Run individual stages

```bash
# Stage 1: Download dataset
python scripts/download_data.py

# Stage 2: Preprocess
python scripts/preprocess.py

# Stage 3: Build graph dataset
python scripts/build_graphs.py --graph_type knn --attack_type false_state

# Stage 6a: Train baselines
python scripts/train_baselines.py --model all --epochs 50

# Stage 6b: Train proposed model
python scripts/train_gnn_temporal.py --model gat --epochs 100

# Stage 7: Evaluate all models
python scripts/evaluate.py --attack_type false_state

# Stage 8: Explainability
python scripts/explain.py --attack_type false_state

# Stage 9: Ablation study
python scripts/ablation.py --study all --epochs 30
```

---

## Configuration

All hyperparameters and paths are in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_DRONES` | 20 | Drones per swarm snapshot |
| `ATTACKER_RATIO` | 0.20 | Fraction of Byzantine drones |
| `NUM_TIMESTEPS` | 50 | Temporal window length |
| `NUM_SWARM_SNAPSHOTS` | 1000 | Total graph sequences generated |
| `HIDDEN_DIM` | 128 | Model hidden layer width |
| `NUM_HEADS` | 4 | GAT attention heads |
| `GRU_LAYERS` | 2 | GRU depth |
| `EPOCHS` | 100 | Max training epochs |
| `PATIENCE` | 15 | Early stopping patience |

---

## Byzantine Attack Types

| Attack | Description |
|--------|-------------|
| `false_state` | Add Gaussian noise to attacker node features (FDI spoofing) |
| `intermittent` | Activate false-state with probability `p_active` per timestep |
| `colluding` | Group of drones report the same coordinated false value |
| `delay` | Replace current features with features from `d` timesteps ago |

---

## Outputs

All results are saved to `outputs/`:

- `outputs/models/` — Trained model checkpoints (`.pt` files)
- `outputs/figures/` — Confusion matrices, ROC/PR curves, SHAP plots, temporal heatmaps
- `outputs/tables/` — LaTeX tables, JSON metric files, ablation results

---

## Citation

If you use this code in your research, please cite the underlying dataset:

```bibtex
@dataset{uav_dataset,
  author  = {Mughal, Umair Ahmed},
  title   = {UAVs Dataset Under Normal and Cyberattacks},
  year    = {2023},
  url     = {https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks}
}
```
