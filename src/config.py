"""
Global configuration for Byzantine Attack Detection in Drone Swarms.

All hyperparameters, paths, and experimental settings are centralised here
so that every module imports a single source of truth.
"""

import os
import torch
import numpy as np
import random
from pathlib import Path

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
MODEL_DIR = OUTPUT_DIR / "models"

# Create directories
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURE_DIR, TABLE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset
# ============================================================
PRIMARY_DATASET_REPO = "https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks.git"
PRIMARY_CSV_NAME = "Dataset_T-ITS.csv"
SECONDARY_DATASET_URL = "https://zenodo.org/records/15125851/files/UAV-NDD%20CSV.zip?download=1"

# Row-index ranges for the primary dataset (documented in the repo README).
# Format: (label, cyber_start, cyber_end, phys_start, phys_end)  — 0-indexed
DATASET_RANGES = {
    "Benign":       {"cyber": (0, 9425),      "physical": (9426, 13716)},
    "DoS":          {"cyber": (13717, 25388),  "physical": (25389, 26362)},
    "Replay":       {"cyber": (26363, 38369),  "physical": (38370, 39343)},
    "EvilTwin":     {"cyber": (39344, 45027),  "physical": (45028, 50501)},
    "FDI":          {"cyber": (50502, 53975),  "physical": (53976, 54783)},
}

# ============================================================
# Swarm Simulation
# ============================================================
NUM_DRONES = 20                    # Drones per swarm snapshot
COMMUNICATION_RANGE = 50.0         # Metres – used for distance-based graph
KNN_K = 5                         # Neighbours for KNN graph
NUM_TIMESTEPS = 50                 # Temporal window length
NUM_SWARM_SNAPSHOTS = 1000         # Total swarm graphs to generate
ATTACKER_RATIO = 0.2               # Fraction of drones that are Byzantine
SWARM_AREA = (200.0, 200.0)        # 2-D area for swarm positions (metres)

# ============================================================
# Byzantine Attack Parameters
# ============================================================
FALSE_STATE_NOISE_SCALE = 3.0      # σ for Gaussian perturbation
INTERMITTENT_P_ACTIVE = 0.4        # Probability of attack being active
COLLUDING_GROUP_SIZE = 4           # Number of drones colluding
DELAY_STEPS = 3                    # Timesteps of delay for stale data

# ============================================================
# Model Hyperparameters
# ============================================================
HIDDEN_DIM = 128
NUM_HEADS = 4                      # GAT attention heads
GAT_LAYERS = 2
GRU_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 15                      # Early stopping patience
FOCAL_ALPHA = 0.75                 # Focal loss alpha
FOCAL_GAMMA = 2.0                  # Focal loss gamma

# ============================================================
# Data Splits
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Reproducibility
# ============================================================
SEED = 42


def set_seed(seed: int = SEED):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Auto-set seed on import
set_seed()
