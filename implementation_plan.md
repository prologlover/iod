# Explainable Detection of Byzantine Attacks in Drone Swarms

Full research-grade implementation for graph-based, temporal, and explainable detection of Byzantine faults in UAV swarms, using real UAV cyber-physical datasets augmented with simulated swarm behavior.

## Background

No public dataset exists specifically for Byzantine swarm attacks. We bridge this gap by:
1. Using the **UAVs-Dataset-Under-Normal-and-Cyberattacks** (primary) — a single CSV with ~55K rows, 53 features (16 physical + 37 cyber), covering Benign/DoS/Replay/Evil Twin/FDI attacks
2. Optionally enriching with the **UAV-NIDD** dataset from Zenodo (secondary) — covering 10+ attack types
3. **Simulating swarm topology** on top of the real per-drone telemetry to create graph-structured data
4. **Injecting Byzantine-specific attacks** (false-state, intermittent, colluding, delay) at the swarm level

## User Review Required

> [!IMPORTANT]
> **GPU Requirement**: The GAT-LSTM model and GNNExplainer benefit significantly from CUDA. The code will auto-detect and fall back to CPU, but training may be slow (~10x). Do you have a CUDA-capable GPU available?

> [!IMPORTANT]
> **Secondary Dataset**: The UAV-NIDD dataset is ~1.6 GB. Downloading it is optional and adds robustness testing. Should I include it or skip it to save time/bandwidth?

> [!WARNING]
> **Estimated Build Time**: This is a 30+ file project. Full implementation will take significant generation time. The plan is structured to build incrementally so each stage is independently testable.

---

## Proposed Changes

### Project Structure

```
f:\My papers\iod\
├── data/                          # Raw & processed data
│   └── raw/                       # Cloned dataset repo + downloads
├── src/                           # Core source modules
│   ├── __init__.py
│   ├── config.py                  # Global configuration
│   ├── data_loader.py             # Load & merge CSV files
│   ├── preprocessing.py           # Clean, normalize, encode, split
│   └── utils.py                   # Shared utilities
├── models/                        # All model architectures
│   ├── __init__.py
│   ├── mlp.py                     # Baseline MLP
│   ├── lstm.py                    # Baseline LSTM
│   ├── cnn.py                     # Baseline 1D-CNN
│   ├── gcn.py                     # Baseline GCN
│   ├── gat_temporal.py            # Proposed GAT + GRU model
│   └── model_utils.py             # Focal loss, early stopping, etc.
├── attacks/                       # Byzantine attack injection
│   ├── __init__.py
│   ├── base_attack.py             # Abstract attack interface
│   ├── false_state.py             # False-state attack
│   ├── intermittent.py            # Intermittent attack
│   ├── colluding.py               # Colluding attack
│   ├── delay_attack.py            # Delay/stale-data attack
│   └── attack_manager.py          # Orchestrate multi-attack scenarios
├── graphs/                        # Graph construction
│   ├── __init__.py
│   ├── knn_graph.py               # KNN-based topology
│   ├── distance_graph.py          # Distance-based topology
│   ├── hexagonal_graph.py         # Fixed hexagonal topology
│   ├── swarm_simulator.py         # Simulate swarm from tabular data
│   └── temporal_graph.py          # Temporal graph snapshots
├── evaluation/                    # Metrics & evaluation
│   ├── __init__.py
│   ├── metrics.py                 # All metric computations
│   ├── swarm_metrics.py           # Swarm-specific metrics
│   └── evaluator.py               # Full evaluation pipeline
├── outputs/                       # Generated results (gitignored)
│   ├── figures/
│   ├── tables/
│   └── models/
├── scripts/                       # Runnable entry points
│   ├── download_data.py           # Clone repo + optional download
│   ├── preprocess.py              # Run preprocessing pipeline
│   ├── build_graphs.py            # Build graph dataset
│   ├── train_baselines.py         # Train MLP/LSTM/CNN
│   ├── train_gnn_temporal.py      # Train proposed GAT+GRU model
│   ├── evaluate.py                # Run full evaluation
│   ├── explain.py                 # Run SHAP + GNNExplainer
│   ├── ablation.py                # Run ablation studies
│   └── run_all.py                 # Full pipeline end-to-end
├── requirements.txt
└── README.md
```

---

### Stage 1: Setup & Data Acquisition

#### [NEW] [requirements.txt](file:///f:/My papers/iod/requirements.txt)
- All dependencies: torch, torch-geometric, pandas, numpy, scikit-learn, matplotlib, seaborn, shap, networkx, scipy

#### [NEW] [download_data.py](file:///f:/My papers/iod/scripts/download_data.py)
- Clone `uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks` into `data/raw/`
- Optionally download UAV-NIDD from Zenodo
- Verify file integrity

---

### Stage 2: Data Loading & Preprocessing

#### [NEW] [config.py](file:///f:/My papers/iod/src/config.py)
- Central configuration: paths, hyperparameters, random seeds, device selection
- Swarm parameters: num_drones (default 20), communication_range, time_steps

#### [NEW] [data_loader.py](file:///f:/My papers/iod/src/data_loader.py)
- Load `Dataset_T-ITS.csv`
- Parse the cyber/physical feature ranges from the dataset structure
- Merge cyber + physical features per sample
- Assign attack-type labels based on row indices (as documented in the dataset)
- Return unified DataFrame with columns: features + `attack_type` + `is_attack`

#### [NEW] [preprocessing.py](file:///f:/My papers/iod/src/preprocessing.py)
- Handle missing values (median imputation for numeric, mode for categorical)
- Normalize features (StandardScaler with saved scaler for inference)
- Encode labels (LabelEncoder for multi-class, binary for attack/benign)
- Stratified train/val/test split (70/15/15)
- Save processed data to `data/processed/`

---

### Stage 3: Swarm Graph Construction

#### [NEW] [swarm_simulator.py](file:///f:/My papers/iod/graphs/swarm_simulator.py)
The core innovation: convert tabular per-drone data into a swarm graph.

**Approach**:
- Each time step: sample `N` rows from the dataset (one per drone in the swarm)
- Assign positions to drones (use physical features if available, else simulate 2D grid)
- Compute inter-drone distances for edge construction
- Create node features: original features + derived consistency features (deviation from neighbor mean)
- Label: if the sampled row was from an attack class, mark node as Byzantine

This creates temporal graph snapshots: `G_t = (V, E, X_t, Y_t)` for `t = 1..T`

#### [NEW] [knn_graph.py](file:///f:/My papers/iod/graphs/knn_graph.py)
- Build edges using K-nearest neighbors (K=5 default) on drone positions
- Returns PyG `Data` object with edge_index, node features, labels

#### [NEW] [distance_graph.py](file:///f:/My papers/iod/graphs/distance_graph.py)
- Build edges where distance < communication_range threshold
- Weighted edges (inverse distance)

#### [NEW] [hexagonal_graph.py](file:///f:/My papers/iod/graphs/hexagonal_graph.py)
- Fixed hexagonal lattice topology (common in swarm formations)
- Each drone connected to up to 6 neighbors

#### [NEW] [temporal_graph.py](file:///f:/My papers/iod/graphs/temporal_graph.py)
- Create sequences of graph snapshots for temporal modeling
- Each sequence: `[G_{t-W+1}, ..., G_t]` with window size W
- Returns list of PyG `Data` objects with temporal dimension

---

### Stage 4: Byzantine Attack Injection

#### [NEW] [base_attack.py](file:///f:/My papers/iod/attacks/base_attack.py)
```python
class ByzantineAttack(ABC):
    def apply(self, graph_data, target_nodes, timestep) -> Data
    def get_config(self) -> dict
```

#### [NEW] [false_state.py](file:///f:/My papers/iod/attacks/false_state.py)
- Modify position/velocity features of target nodes
- Configurable: noise_scale, feature_indices, perturbation_type (gaussian, adversarial, fixed_offset)

#### [NEW] [intermittent.py](file:///f:/My papers/iod/attacks/intermittent.py)
- Activate attack randomly with probability `p_active` per timestep
- When active, apply false-state or random perturbation
- Models realistic intermittent Byzantine behavior

#### [NEW] [colluding.py](file:///f:/My papers/iod/attacks/colluding.py)
- Multiple target drones coordinate to report the same wrong value
- Configurable: collusion_group_size, shared_false_value_strategy

#### [NEW] [delay_attack.py](file:///f:/My papers/iod/attacks/delay_attack.py)
- Replace current features with features from `d` timesteps ago
- Configurable delay: 1–10 timesteps

#### [NEW] [attack_manager.py](file:///f:/My papers/iod/attacks/attack_manager.py)
- Orchestrate mixed attack scenarios
- Configure: attack_types, num_attackers (default 20% of swarm), intensity
- Generate labeled datasets with attack metadata

---

### Stage 5: Model Architectures

#### [NEW] [mlp.py](file:///f:/My papers/iod/models/mlp.py)
- 3-layer MLP with BatchNorm, Dropout, ReLU
- Input: flattened node features (no graph structure)

#### [NEW] [lstm.py](file:///f:/My papers/iod/models/lstm.py)
- 2-layer LSTM with attention over time steps
- Input: temporal feature sequences per node

#### [NEW] [cnn.py](file:///f:/My papers/iod/models/cnn.py)
- 1D-CNN over temporal feature sequences
- Conv1D → BatchNorm → ReLU → MaxPool → FC

#### [NEW] [gcn.py](file:///f:/My papers/iod/models/gcn.py)
- 3-layer GCN (GCNConv from PyG)
- No temporal modeling — single snapshot classification

#### [NEW] [gat_temporal.py](file:///f:/My papers/iod/models/gat_temporal.py) — **Proposed Model**
Architecture:
```
Input → GATConv (multi-head attention, 2 layers)
      → Per-node temporal embedding via GRU (2 layers)
      → Classification head (MLP)
      → Node-level output (binary or multi-class)
```
- GAT captures spatial/neighbor relationships with learnable attention
- GRU captures temporal dynamics across graph snapshots
- Attention weights are extractable for explainability

#### [NEW] [model_utils.py](file:///f:/My papers/iod/models/model_utils.py)
- `FocalLoss`: handles class imbalance (alpha, gamma configurable)
- `EarlyStopping`: patience-based with best model checkpoint
- `get_class_weights`: compute inverse-frequency weights
- Training loop utilities

---

### Stage 6: Training Pipeline

#### [NEW] [train_baselines.py](file:///f:/My papers/iod/scripts/train_baselines.py)
- Train MLP, LSTM, CNN on tabular/sequential data
- Hyperparameters from config
- Save best models to `outputs/models/`

#### [NEW] [train_gnn_temporal.py](file:///f:/My papers/iod/scripts/train_gnn_temporal.py)
- Main training script for GAT+GRU model
- Uses PyG DataLoader for batched graph training
- Focal loss + early stopping
- Logs training/validation metrics per epoch
- Saves best model + training curves

---

### Stage 7: Evaluation

#### [NEW] [metrics.py](file:///f:/My papers/iod/evaluation/metrics.py)
- Standard: Accuracy, Precision, Recall, F1, ROC-AUC, MCC
- Returns dict of all metrics

#### [NEW] [swarm_metrics.py](file:///f:/My papers/iod/evaluation/swarm_metrics.py)
- **Attack localization accuracy**: % of Byzantine nodes correctly identified
- **Detection latency**: number of timesteps until first correct detection
- **Per-node FPR**: false positive rate computed per drone across time

#### [NEW] [evaluator.py](file:///f:/My papers/iod/evaluation/evaluator.py)
- Run all models on test set
- Generate comparison tables (LaTeX-ready)
- Generate all plots:
  - Confusion matrices (heatmap)
  - ROC curves (per model, overlaid)
  - Precision-Recall curves
  - Training loss/accuracy curves

---

### Stage 8: Explainability

#### [NEW] [explain.py](file:///f:/My papers/iod/scripts/explain.py)
Two explainability approaches:

**SHAP Analysis**:
- Use `shap.DeepExplainer` or `KernelExplainer` on the GAT model
- Generate feature importance bar charts
- Generate SHAP summary plots (beeswarm)
- Identify top-K features that drive Byzantine detection

**GNNExplainer / Attention Weights**:
- Extract GAT attention weights per edge
- Visualize which neighbor drones influenced classification
- Use PyG's `GNNExplainer` to generate node-level explanations
- Temporal analysis: show when attention shifts (attack onset detection)

**Outputs**:
- Feature importance plots (global + per-attack-type)
- Node-level explanation graphs (networkx visualization)
- Temporal attention heatmaps

---

### Stage 9: Ablation Study

#### [NEW] [ablation.py](file:///f:/My papers/iod/scripts/ablation.py)
Compare:
1. **GAT+GRU** (full model) vs **GAT-only** (no temporal) vs **GRU-only** (no graph)
2. Performance across attack types (false-state, intermittent, colluding, delay)
3. Impact of attacker ratio (10%, 20%, 30%, 40%)
4. Graph topology comparison (KNN vs distance vs hexagonal)

Generate comparison tables and bar charts.

---

### Stage 10: Documentation

#### [NEW] [README.md](file:///f:/My papers/iod/README.md)
- Project overview, architecture diagram
- Installation instructions
- Quick-start commands
- Full pipeline walkthrough
- Citation information

---

## Verification Plan

### Automated Tests
1. **Data Pipeline**: Run `python scripts/download_data.py` → verify CSV loaded, features counted
2. **Graph Construction**: Run `python scripts/build_graphs.py` → verify PyG Data objects have correct shapes
3. **Attack Injection**: Verify labels are correctly assigned, feature perturbations are within expected ranges
4. **Training**: Run `python scripts/train_gnn_temporal.py --epochs 5` → verify loss decreases
5. **Full Pipeline**: Run `python scripts/run_all.py` → verify all outputs generated in `outputs/`

### Manual Verification
- Inspect generated figures for visual correctness
- Verify LaTeX tables are well-formatted
- Check that SHAP explanations are meaningful (position/velocity features ranked high for false-state attacks)
