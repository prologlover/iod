"""
Swarm Simulator — convert tabular per-drone data into temporal swarm graphs.

Core idea:
  • Each time-step, sample N rows from the dataset (one per drone).
  • Assign 2-D positions (simulated or derived from physical features).
  • Build a graph snapshot with node features, edges, and labels.
  • Stack T consecutive snapshots to form a temporal sequence.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ATTACKER_RATIO,
    NUM_DRONES,
    NUM_SWARM_SNAPSHOTS,
    NUM_TIMESTEPS,
    SEED,
    SWARM_AREA,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Position assignment
# ------------------------------------------------------------------

def _generate_positions(n: int, area: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """
    Generate random 2-D positions for *n* drones within the given area.

    Returns shape (n, 2).
    """
    x = rng.uniform(0, area[0], size=n)
    y = rng.uniform(0, area[1], size=n)
    return np.stack([x, y], axis=1)


# ------------------------------------------------------------------
# Single snapshot
# ------------------------------------------------------------------

def build_single_snapshot(
    df: pd.DataFrame,
    feature_cols: List[str],
    rng: np.random.Generator,
    num_drones: int = NUM_DRONES,
    attacker_ratio: float = ATTACKER_RATIO,
) -> Dict:
    """
    Sample *num_drones* rows from the dataset, assign positions,
    and create a single-timestep swarm snapshot.

    Returns a dict with:
      - node_features : np.ndarray (num_drones, num_features)
      - positions     : np.ndarray (num_drones, 2)
      - labels        : np.ndarray (num_drones,)  — binary (is_attack)
      - attack_types  : list[str]
    """
    # Decide how many attackers
    n_attackers = max(1, int(num_drones * attacker_ratio))
    n_benign = num_drones - n_attackers

    # Sample benign and attack rows
    benign_pool = df[df["is_attack"] == 0]
    attack_pool = df[df["is_attack"] == 1]

    if len(benign_pool) < n_benign:
        benign_sample = benign_pool.sample(n=n_benign, replace=True, random_state=rng.integers(1e9))
    else:
        benign_sample = benign_pool.sample(n=n_benign, replace=False, random_state=rng.integers(1e9))

    if len(attack_pool) < n_attackers:
        attack_sample = attack_pool.sample(n=n_attackers, replace=True, random_state=rng.integers(1e9))
    else:
        attack_sample = attack_pool.sample(n=n_attackers, replace=False, random_state=rng.integers(1e9))

    combined = pd.concat([benign_sample, attack_sample], ignore_index=True)
    # Shuffle so attackers aren't always at the end
    perm = rng.permutation(len(combined))
    combined = combined.iloc[perm].reset_index(drop=True)

    node_features = combined[feature_cols].values.astype(np.float32)
    labels = combined["is_attack"].values.astype(np.int64)
    attack_types = combined["attack_type"].tolist()
    positions = _generate_positions(num_drones, SWARM_AREA, rng)

    return {
        "node_features": node_features,
        "positions": positions,
        "labels": labels,
        "attack_types": attack_types,
    }


# ------------------------------------------------------------------
# Temporal sequence
# ------------------------------------------------------------------

def build_temporal_sequence(
    df: pd.DataFrame,
    feature_cols: List[str],
    rng: np.random.Generator,
    num_drones: int = NUM_DRONES,
    num_timesteps: int = NUM_TIMESTEPS,
    attacker_ratio: float = ATTACKER_RATIO,
) -> List[Dict]:
    """
    Build a sequence of T snapshots (one temporal window).

    The same drone "identities" persist across the window:
      - Positions evolve smoothly (random walk).
      - Attacker assignments are fixed for the window.

    Returns list of T dicts (same format as build_single_snapshot).
    """
    # Decide which drones are attackers for this window
    n_attackers = max(1, int(num_drones * attacker_ratio))
    attacker_indices = set(rng.choice(num_drones, size=n_attackers, replace=False).tolist())

    benign_pool = df[df["is_attack"] == 0]
    attack_pool = df[df["is_attack"] == 1]

    # Initial positions
    positions = _generate_positions(num_drones, SWARM_AREA, rng)

    snapshots = []
    for t in range(num_timesteps):
        # Sample features for each drone
        rows = []
        for i in range(num_drones):
            if i in attacker_indices:
                row = attack_pool.sample(n=1, random_state=rng.integers(1e9))
            else:
                row = benign_pool.sample(n=1, random_state=rng.integers(1e9))
            rows.append(row)

        combined = pd.concat(rows, ignore_index=True)
        node_features = combined[feature_cols].values.astype(np.float32)
        labels = combined["is_attack"].values.astype(np.int64)
        attack_types = combined["attack_type"].tolist()

        snapshots.append({
            "node_features": node_features,
            "positions": positions.copy(),
            "labels": labels,
            "attack_types": attack_types,
        })

        # Evolve positions (random walk with small steps)
        positions += rng.normal(0, 2.0, size=positions.shape)
        positions = np.clip(positions, 0, [SWARM_AREA[0], SWARM_AREA[1]])

    return snapshots


# ------------------------------------------------------------------
# Full dataset generation
# ------------------------------------------------------------------

def generate_swarm_dataset(
    df: pd.DataFrame,
    feature_cols: List[str],
    num_snapshots: int = NUM_SWARM_SNAPSHOTS,
    num_drones: int = NUM_DRONES,
    num_timesteps: int = NUM_TIMESTEPS,
    attacker_ratio: float = ATTACKER_RATIO,
    seed: int = SEED,
) -> List[List[Dict]]:
    """
    Generate the full set of temporal swarm sequences.

    Returns a list of *num_snapshots* temporal windows,
    each containing *num_timesteps* snapshot dicts.
    """
    rng = np.random.default_rng(seed)
    dataset = []

    for idx in range(num_snapshots):
        seq = build_temporal_sequence(
            df, feature_cols, rng,
            num_drones=num_drones,
            num_timesteps=num_timesteps,
            attacker_ratio=attacker_ratio,
        )
        dataset.append(seq)

        if (idx + 1) % 100 == 0:
            logger.info(f"  Generated {idx + 1}/{num_snapshots} swarm sequences")

    logger.info(f"Swarm dataset complete: {len(dataset)} sequences × {num_timesteps} timesteps × {num_drones} drones")
    return dataset
