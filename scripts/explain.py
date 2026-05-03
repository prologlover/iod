"""
Explainability script — Stage 8.

Generates SHAP-style feature importance, GNNExplainer sub-graphs,
and temporal importance heatmaps for the trained GAT+GRU model.

Usage
-----
    python scripts/explain.py [--attack_type false_state]
                              [--graph_type knn]
                              [--seq_idx 0]
                              [--n_sequences 20]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEVICE, DROPOUT, GRU_LAYERS, HIDDEN_DIM, MODEL_DIR, NUM_HEADS,
    NUM_SWARM_SNAPSHOTS, SEED,
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger, save_json
from graphs.swarm_simulator import generate_swarm_dataset
from graphs.temporal_graph import build_temporal_graphs, pack_temporal_batch
from attacks import inject_attacks
from models import get_model
from evaluation.explainability import generate_full_explanation

logger = get_logger(__name__)


def main(
    attack_type: str = "false_state",
    graph_type: str = "knn",
    seq_idx: int = 0,
    n_sequences: int = 20,
):
    logger.info("=" * 60)
    logger.info("STAGE 8: Explainability")
    logger.info(f"  attack_type  = {attack_type}")
    logger.info(f"  graph_type   = {graph_type}")
    logger.info(f"  n_sequences  = {n_sequences}")
    logger.info("=" * 60)

    rng = np.random.default_rng(SEED)

    # Load data
    raw_df = load_and_merge()
    prep = run_preprocessing(raw_df)
    test_df = prep["test_df"]
    feature_cols = prep["feature_cols"]
    num_features = len(feature_cols)

    # Build a small test batch
    test_tab = generate_swarm_dataset(
        test_df, feature_cols, n_sequences, seed=rng.integers(10_000)
    )
    test_pyg = build_temporal_graphs(test_tab, graph_type=graph_type)
    test_seqs = inject_attacks(test_pyg, attack_type, rng)

    # Pack into batch tensor
    features, labels, final_graphs = pack_temporal_batch(test_seqs)
    features = features.to(DEVICE)

    # Load model
    checkpoint = MODEL_DIR / "best_gat_temporal.pt"
    model = get_model(
        "gat",
        in_channels=num_features,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        gru_layers=GRU_LAYERS,
    ).to(DEVICE)

    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        logger.info(f"Loaded model from {checkpoint}")
    else:
        logger.warning("No trained model found; using random weights (demonstration only).")

    model.eval()

    # Run full explanation
    logger.info(f"Generating explanations for sequence index {seq_idx}...")
    report = generate_full_explanation(
        model=model,
        features=features,
        final_graphs=final_graphs,
        feature_names=feature_cols,
        seq_idx=seq_idx,
        save_prefix=f"{attack_type}_{graph_type}",
    )

    logger.info("\nExplanation report summary:")
    for k, v in report.items():
        if k != "explanations":
            logger.info(f"  {k}: {v if not isinstance(v, dict) else list(v.keys())}")

    logger.info("Explainability stage complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", default="false_state")
    parser.add_argument("--graph_type", default="knn",
                        choices=["knn", "distance", "hexagonal"])
    parser.add_argument("--seq_idx", type=int, default=0)
    parser.add_argument("--n_sequences", type=int, default=20)
    args = parser.parse_args()

    main(
        attack_type=args.attack_type,
        graph_type=args.graph_type,
        seq_idx=args.seq_idx,
        n_sequences=args.n_sequences,
    )
