"""
Full evaluation script — Stage 7.

Loads trained models, runs inference on the test set, and generates
comparison tables, confusion matrices, ROC/PR curves.

Usage
-----
    python scripts/evaluate.py [--attack_type false_state]
                               [--graph_type knn]
                               [--threshold 0.5]
"""
import argparse
import sys
from pathlib import Path

import json

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DEVICE, DROPOUT, GRU_LAYERS, HIDDEN_DIM, MODEL_DIR, NUM_HEADS,
    NUM_SWARM_SNAPSHOTS, SEED, BATCH_SIZE,
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger
from graphs.swarm_simulator import generate_swarm_dataset
from graphs.temporal_graph import build_temporal_graphs, pack_temporal_batch
from attacks import inject_attacks
from models import get_model
from models.mlp import MLP
from models.lstm import LSTMModel
from models.cnn import CNNModel
from models.gcn import GCNModel
from evaluation.evaluator import run_evaluation, predict_gnn, predict_baseline

logger = get_logger(__name__)


def load_model_if_exists(path: Path, model):
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        logger.info(f"Loaded weights from {path}")
    else:
        logger.warning(f"No checkpoint found at {path} — using random weights (for testing).")
    return model.to(DEVICE)


def load_threshold(threshold_path: Path, default: float = 0.5) -> float:
    """Load the saved optimal threshold, falling back to *default* if missing."""
    if threshold_path.exists():
        data = json.loads(threshold_path.read_text())
        t = float(data.get("threshold", default))
        logger.info(f"Loaded threshold {t:.3f} from {threshold_path}")
        return t
    logger.info(f"No threshold file at {threshold_path} — using default {default}")
    return default


def main(attack_type: str = "false_state", graph_type: str = "knn",
         threshold: float = 0.5, n_snapshots: int = None):
    logger.info("=" * 60)
    logger.info("STAGE 7: Evaluation")
    logger.info(f"  attack_type = {attack_type}")
    logger.info(f"  graph_type  = {graph_type}")
    logger.info("=" * 60)

    rng = np.random.default_rng(SEED)

    logger.info("Loading and preprocessing data...")
    raw_df = load_and_merge()
    prep = run_preprocessing(raw_df)
    test_df = prep["test_df"]
    feature_cols = prep["feature_cols"]
    num_features = len(feature_cols)

    total = n_snapshots if n_snapshots else NUM_SWARM_SNAPSHOTS
    n_test = max(50, int(total * 0.15))
    logger.info(f"Generating {n_test} test sequences...")
    test_tab = generate_swarm_dataset(test_df, feature_cols, n_test, seed=rng.integers(10_000))
    test_pyg = build_temporal_graphs(test_tab, graph_type=graph_type)
    test_seqs = inject_attacks(test_pyg, attack_type, rng)

    # Instantiate models
    gat_model = load_model_if_exists(
        MODEL_DIR / "best_gat_temporal.pt",
        get_model("gat", in_channels=num_features, hidden_dim=HIDDEN_DIM,
                  num_heads=NUM_HEADS, dropout=DROPOUT, gru_layers=GRU_LAYERS),
    )
    graphsage_model = load_model_if_exists(
        MODEL_DIR / "best_graphsage_temporal.pt",
        get_model("graphsage", in_channels=num_features, hidden_dim=HIDDEN_DIM,
                  dropout=DROPOUT, gru_layers=GRU_LAYERS),
    )
    mlp_model = load_model_if_exists(
        MODEL_DIR / "best_mlp.pt",
        MLP(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
    )
    lstm_model = load_model_if_exists(
        MODEL_DIR / "best_lstm.pt",
        LSTMModel(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
    )
    cnn_model = load_model_if_exists(
        MODEL_DIR / "best_cnn.pt",
        CNNModel(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
    )
    gcn_model = load_model_if_exists(
        MODEL_DIR / "best_gcn.pt",
        GCNModel(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
    )

    # Load per-model optimal thresholds (saved during training); fall back to CLI value
    thresholds = {
        "GAT+GRU":       load_threshold(MODEL_DIR / "threshold_gat_temporal.json",      threshold),
        "GraphSAGE+GRU": load_threshold(MODEL_DIR / "threshold_graphsage_temporal.json", threshold),
        "MLP":           load_threshold(MODEL_DIR / "threshold_mlp.json",                threshold),
        "LSTM":          load_threshold(MODEL_DIR / "threshold_lstm.json",               threshold),
        "1D-CNN":        load_threshold(MODEL_DIR / "threshold_cnn.json",                threshold),
        "GCN":           load_threshold(MODEL_DIR / "threshold_gcn.json",                threshold),
    }

    logger.info("Running inference...")
    raw_predictions = {
        "GAT+GRU":       predict_gnn(gat_model, test_seqs),
        "GraphSAGE+GRU": predict_gnn(graphsage_model, test_seqs),
        "MLP":           predict_baseline(mlp_model, test_seqs, "mlp"),
        "LSTM":          predict_baseline(lstm_model, test_seqs, "lstm"),
        "1D-CNN":        predict_baseline(cnn_model, test_seqs, "cnn"),
        "GCN":           predict_baseline(gcn_model, test_seqs, "gcn"),
    }

    logger.info("Computing metrics and generating plots...")
    # run_evaluation accepts a single global threshold; we compute per-model metrics ourselves
    from evaluation.metrics import calculate_metrics
    from evaluation.evaluator import plot_confusion_matrix, plot_roc_curves, plot_pr_curves
    from src.config import FIGURE_DIR, TABLE_DIR
    from src.utils import save_json
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for model_name, (y_true, y_prob) in raw_predictions.items():
        t = thresholds[model_name]
        y_pred = (y_prob >= t).astype(int)
        all_metrics[model_name] = calculate_metrics(y_true, y_pred, y_prob)
        all_metrics[model_name]["threshold"] = t
        plot_confusion_matrix(y_true, y_pred, model_name, save_prefix=attack_type)

    plot_roc_curves(raw_predictions, save_prefix=attack_type)
    plot_pr_curves(raw_predictions, save_prefix=attack_type)
    save_json(all_metrics, TABLE_DIR / f"{attack_type}_metrics.json")

    # Print summary table
    header = (
        f"\n{'Model':<20}  {'Threshold':>9}  {'Accuracy':>9}  "
        f"{'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'ROC-AUC':>9}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in all_metrics.items():
        logger.info(
            f"{name:<20}  {m.get('threshold', 0.5):>9.3f}  "
            f"{m.get('accuracy', 0):>9.4f}  "
            f"{m.get('precision', 0):>9.4f}  "
            f"{m.get('recall', 0):>9.4f}  "
            f"{m.get('f1', 0):>9.4f}  "
            f"{m.get('roc_auc', 0):>9.4f}"
        )

    logger.info("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", default="false_state")
    parser.add_argument("--graph_type", default="knn",
                        choices=["knn", "distance", "hexagonal"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--snapshots", type=int, default=None,
                        help="Total swarm snapshots (derives test split from this).")
    args = parser.parse_args()

    main(
        attack_type=args.attack_type,
        graph_type=args.graph_type,
        threshold=args.threshold,
        n_snapshots=args.snapshots,
    )
