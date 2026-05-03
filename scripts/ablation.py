"""
Ablation study script — Stage 9.

Compares:
  1. GAT+GRU (full) vs GAT-only (no temporal) vs GRU-only (no graph)
  2. Performance across attack types
  3. Impact of attacker ratio (10%–40%)
  4. Graph topology comparison (KNN vs distance vs hexagonal)

Usage
-----
    python scripts/ablation.py [--study all|components|attacks|ratios|topology]
                               [--epochs 30]
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ATTACKER_RATIO, BATCH_SIZE, DEVICE, DROPOUT, EPOCHS,
    FOCAL_ALPHA, FOCAL_GAMMA, GRU_LAYERS, HIDDEN_DIM, LEARNING_RATE,
    MODEL_DIR, NUM_HEADS, NUM_SWARM_SNAPSHOTS, PATIENCE, SEED,
    TABLE_DIR, WEIGHT_DECAY,
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger, save_json
from graphs.swarm_simulator import generate_swarm_dataset
from graphs.temporal_graph import build_temporal_graphs, pack_temporal_batch
from attacks import inject_attacks
from models import get_model
from models.model_utils import FocalLoss, EarlyStopping
from evaluation.metrics import calculate_metrics
from evaluation.ablation import ablation_results_to_latex

logger = get_logger(__name__)

ATTACK_TYPES = ["false_state", "intermittent", "colluding", "delay"]
RATIOS = [0.1, 0.2, 0.3, 0.4]
TOPOLOGIES = ["knn", "distance", "hexagonal"]


# ------------------------------------------------------------------ #
#  Quick training loop for ablation                                   #
# ------------------------------------------------------------------ #

def quick_train_eval(model, train_seqs, val_seqs, test_seqs, epochs: int):
    """Mini training loop that returns test metrics."""
    import random
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    early_stop = EarlyStopping(patience=min(PATIENCE, epochs // 3))

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_seqs)
        for i in range(0, len(train_seqs), BATCH_SIZE):
            batch = train_seqs[i : i + BATCH_SIZE]
            features, labels, final_graphs = pack_temporal_batch(batch)
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(features, final_graphs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_seqs), BATCH_SIZE):
                batch = val_seqs[i : i + BATCH_SIZE]
                features, labels, final_graphs = pack_temporal_batch(batch)
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                logits = model(features, final_graphs)
                val_loss += criterion(logits, labels).item()
        if early_stop.step(val_loss, model):
            break

    # Evaluate on test
    all_true, all_pred, all_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(test_seqs), BATCH_SIZE):
            batch = test_seqs[i : i + BATCH_SIZE]
            features, labels, final_graphs = pack_temporal_batch(batch)
            features = features.to(DEVICE)
            logits = model(features, final_graphs)
            probs = F.softmax(logits, dim=-1)[:, :, 1].cpu().numpy().ravel()
            preds = (probs >= 0.5).astype(int)
            all_true.extend(labels.numpy().ravel())
            all_pred.extend(preds)
            all_prob.extend(probs)

    return calculate_metrics(
        np.array(all_true), np.array(all_pred), np.array(all_prob)
    )


def make_gat(num_features):
    return get_model(
        "gat", in_channels=num_features, hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS, dropout=DROPOUT, gru_layers=GRU_LAYERS,
    ).to(DEVICE)


def generate_seqs(df, feature_cols, n, attack_type, graph_type, seed, attacker_ratio=ATTACKER_RATIO):
    from src.config import NUM_DRONES, NUM_TIMESTEPS, SWARM_AREA
    rng = np.random.default_rng(seed)
    tab = generate_swarm_dataset(
        df, feature_cols, n, attacker_ratio=attacker_ratio, seed=seed,
    )
    pyg = build_temporal_graphs(tab, graph_type=graph_type)
    return inject_attacks(pyg, attack_type, rng)


# ------------------------------------------------------------------ #
#  Study 1: Component ablation (GAT+GRU vs partial)                   #
# ------------------------------------------------------------------ #

def study_components(train_df, val_df, test_df, feature_cols, epochs, n_snapshots=None):
    logger.info("\n=== Study 1: Component Ablation ===")
    n_feat = len(feature_cols)
    total = n_snapshots or NUM_SWARM_SNAPSHOTS
    n_train = max(4, int(total * 0.35))
    n_val = max(2, int(total * 0.10))
    n_test = max(2, int(total * 0.10))
    rng = np.random.default_rng(SEED)

    train_s = generate_seqs(train_df, feature_cols, n_train, "false_state", "knn", SEED)
    val_s = generate_seqs(val_df, feature_cols, n_val, "false_state", "knn", SEED + 1)
    test_s = generate_seqs(test_df, feature_cols, n_test, "false_state", "knn", SEED + 2)

    configs = {
        "GAT+GRU (full)":      {"gru_layers": GRU_LAYERS},
        "GAT+GRU (GRU-1)":     {"gru_layers": 1},
        "GraphSAGE+GRU":       None,
    }

    results = {}
    for name, override in configs.items():
        if override is None:
            model = get_model(
                "graphsage", in_channels=n_feat, hidden_dim=HIDDEN_DIM,
                dropout=DROPOUT, gru_layers=GRU_LAYERS,
            ).to(DEVICE)
        else:
            model = get_model(
                "gat", in_channels=n_feat, hidden_dim=HIDDEN_DIM,
                num_heads=NUM_HEADS, dropout=DROPOUT, **override,
            ).to(DEVICE)

        logger.info(f"  Training: {name}")
        metrics = quick_train_eval(model, train_s, val_s, test_s, epochs)
        results[name] = metrics
        logger.info(f"  F1={metrics.get('f1', 0):.4f}")

    return results


# ------------------------------------------------------------------ #
#  Study 2: Attack-type robustness                                    #
# ------------------------------------------------------------------ #

def study_attacks(train_df, val_df, test_df, feature_cols, epochs, n_snapshots=None):
    logger.info("\n=== Study 2: Attack Type Robustness ===")
    n_feat = len(feature_cols)
    total = n_snapshots or NUM_SWARM_SNAPSHOTS
    n_train = max(4, int(total * 0.35))
    n_val = max(2, int(total * 0.08))
    n_test = max(2, int(total * 0.08))

    results = {}
    for attack in ATTACK_TYPES:
        logger.info(f"  Attack: {attack}")
        train_s = generate_seqs(train_df, feature_cols, n_train, attack, "knn", SEED)
        val_s = generate_seqs(val_df, feature_cols, n_val, attack, "knn", SEED + 1)
        test_s = generate_seqs(test_df, feature_cols, n_test, attack, "knn", SEED + 2)
        model = make_gat(n_feat)
        metrics = quick_train_eval(model, train_s, val_s, test_s, epochs)
        results[attack] = metrics
        logger.info(f"  F1={metrics.get('f1', 0):.4f}")

    return results


# ------------------------------------------------------------------ #
#  Study 3: Attacker ratio                                            #
# ------------------------------------------------------------------ #

def study_ratios(train_df, val_df, test_df, feature_cols, epochs, n_snapshots=None):
    logger.info("\n=== Study 3: Attacker Ratio ===")
    n_feat = len(feature_cols)
    total = n_snapshots or NUM_SWARM_SNAPSHOTS
    n_train = max(4, int(total * 0.30))
    n_val = max(2, int(total * 0.08))
    n_test = max(2, int(total * 0.08))

    results = {}
    for ratio in RATIOS:
        logger.info(f"  Ratio: {ratio:.0%}")
        train_s = generate_seqs(train_df, feature_cols, n_train, "false_state", "knn",
                                SEED, attacker_ratio=ratio)
        val_s = generate_seqs(val_df, feature_cols, n_val, "false_state", "knn",
                              SEED + 1, attacker_ratio=ratio)
        test_s = generate_seqs(test_df, feature_cols, n_test, "false_state", "knn",
                               SEED + 2, attacker_ratio=ratio)
        model = make_gat(n_feat)
        metrics = quick_train_eval(model, train_s, val_s, test_s, epochs)
        results[f"{ratio:.0%}"] = metrics
        logger.info(f"  F1={metrics.get('f1', 0):.4f}")

    return results


# ------------------------------------------------------------------ #
#  Study 4: Graph topology                                            #
# ------------------------------------------------------------------ #

def study_topology(train_df, val_df, test_df, feature_cols, epochs, n_snapshots=None):
    logger.info("\n=== Study 4: Graph Topology ===")
    n_feat = len(feature_cols)
    total = n_snapshots or NUM_SWARM_SNAPSHOTS
    n_train = max(4, int(total * 0.30))
    n_val = max(2, int(total * 0.08))
    n_test = max(2, int(total * 0.08))

    results = {}
    for topo in TOPOLOGIES:
        logger.info(f"  Topology: {topo}")
        train_s = generate_seqs(train_df, feature_cols, n_train, "false_state", topo, SEED)
        val_s = generate_seqs(val_df, feature_cols, n_val, "false_state", topo, SEED + 1)
        test_s = generate_seqs(test_df, feature_cols, n_test, "false_state", topo, SEED + 2)
        model = make_gat(n_feat)
        metrics = quick_train_eval(model, train_s, val_s, test_s, epochs)
        results[topo] = metrics
        logger.info(f"  F1={metrics.get('f1', 0):.4f}")

    return results


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main(study: str = "all", epochs: int = 30, n_snapshots: int = None):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    raw_df = load_and_merge()
    prep = run_preprocessing(raw_df)
    train_df = prep["train_df"]
    val_df = prep["val_df"]
    test_df = prep["test_df"]
    feature_cols = prep["feature_cols"]

    all_results = {}

    if study in ("all", "components"):
        all_results["components"] = study_components(train_df, val_df, test_df, feature_cols, epochs, n_snapshots)
    if study in ("all", "attacks"):
        all_results["attacks"] = study_attacks(train_df, val_df, test_df, feature_cols, epochs, n_snapshots)
    if study in ("all", "ratios"):
        all_results["ratios"] = study_ratios(train_df, val_df, test_df, feature_cols, epochs, n_snapshots)
    if study in ("all", "topology"):
        all_results["topology"] = study_topology(train_df, val_df, test_df, feature_cols, epochs, n_snapshots)

    save_json(all_results, TABLE_DIR / "ablation_full_results.json")
    logger.info(f"\nAblation results saved to {TABLE_DIR / 'ablation_full_results.json'}")

    # LaTeX for component study if available
    if "components" in all_results:
        latex = ablation_results_to_latex(all_results["components"])
        latex_path = TABLE_DIR / "ablation_components.tex"
        latex_path.write_text(latex)
        logger.info(f"LaTeX table saved to {latex_path}")

    logger.info("Ablation study complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", default="all",
                        choices=["all", "components", "attacks", "ratios", "topology"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--snapshots", type=int, default=None,
                        help="Total swarm snapshots per study. Use small value for quick testing.")
    args = parser.parse_args()

    main(study=args.study, epochs=args.epochs, n_snapshots=args.snapshots)
