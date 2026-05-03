"""
Build Graph Dataset — Stage 3 entry point.

Loads preprocessed data, simulates swarm topology, converts to PyG graph
sequences, and saves them to disk.

Usage
-----
    python scripts/build_graphs.py [--graph_type knn|distance|hexagonal]
                                   [--snapshots 1000]
                                   [--attack_type false_state]
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    NUM_SWARM_SNAPSHOTS, PROCESSED_DATA_DIR, SEED,
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger, Timer
from graphs.swarm_simulator import generate_swarm_dataset
from graphs.temporal_graph import build_temporal_graphs
from attacks import inject_attacks

logger = get_logger(__name__)

GRAPH_DIR = PROCESSED_DATA_DIR / "graphs"


def main(
    graph_type: str = "knn",
    num_snapshots: int = NUM_SWARM_SNAPSHOTS,
    attack_type: str = "false_state",
):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    logger.info("=" * 60)
    logger.info("STAGE 3: Build Graph Dataset")
    logger.info(f"  graph_type   = {graph_type}")
    logger.info(f"  num_snapshots = {num_snapshots}")
    logger.info(f"  attack_type  = {attack_type}")
    logger.info("=" * 60)

    # Load & preprocess
    logger.info("Loading raw data...")
    raw_df = load_and_merge()
    prep = run_preprocessing(raw_df)

    train_df = prep["train_df"]
    val_df = prep["val_df"]
    test_df = prep["test_df"]
    feature_cols = prep["feature_cols"]
    logger.info(f"Features: {len(feature_cols)}")

    splits = {
        "train": (train_df, int(num_snapshots * 0.70)),
        "val":   (val_df,   int(num_snapshots * 0.15)),
        "test":  (test_df,  max(1, int(num_snapshots * 0.15))),
    }

    for split_name, (df_split, n_snap) in splits.items():
        logger.info(f"\n--- {split_name.upper()} ({n_snap} snapshots) ---")

        with Timer(f"Simulate swarm ({split_name})"):
            tabular_seq = generate_swarm_dataset(
                df_split, feature_cols, num_snapshots=n_snap,
                seed=rng.integers(10_000),
            )

        with Timer(f"Build {graph_type} graphs ({split_name})"):
            pyg_seq = build_temporal_graphs(tabular_seq, graph_type=graph_type)

        with Timer(f"Inject {attack_type} attack ({split_name})"):
            attacked_seq = inject_attacks(pyg_seq, attack_type, rng)

        save_path = GRAPH_DIR / f"{split_name}_{graph_type}_{attack_type}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(attacked_seq, f)
        logger.info(f"Saved {len(attacked_seq)} sequences -> {save_path}")

    logger.info("\nGraph dataset build complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build temporal swarm graph dataset.")
    parser.add_argument("--graph_type", default="knn",
                        choices=["knn", "distance", "hexagonal"])
    parser.add_argument("--snapshots", type=int, default=NUM_SWARM_SNAPSHOTS)
    parser.add_argument("--attack_type", default="false_state")
    args = parser.parse_args()

    main(
        graph_type=args.graph_type,
        num_snapshots=args.snapshots,
        attack_type=args.attack_type,
    )
