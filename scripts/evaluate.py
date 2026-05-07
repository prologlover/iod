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
from typing import Dict, List

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


def _evaluate_once(
    attack_type: str,
    graph_type: str,
    threshold: float,
    n_snapshots: int,
    eval_seed: int,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(eval_seed)

    logger.info(f"Running single evaluation with seed={eval_seed}")

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
        plot_confusion_matrix(y_true, y_pred, model_name, save_prefix=f"{attack_type}_seed{eval_seed}")

    plot_roc_curves(raw_predictions, save_prefix=f"{attack_type}_seed{eval_seed}")
    plot_pr_curves(raw_predictions, save_prefix=f"{attack_type}_seed{eval_seed}")
    save_json(all_metrics, TABLE_DIR / f"{attack_type}_metrics_seed{eval_seed}.json")
    return all_metrics


def _aggregate_metrics(seed_metrics: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    model_names = seed_metrics[0].keys()
    aggregated = {}
    for model_name in model_names:
        metric_names = seed_metrics[0][model_name].keys()
        aggregated[model_name] = {}
        for metric_name in metric_names:
            vals = [m[model_name][metric_name] for m in seed_metrics]
            aggregated[model_name][metric_name] = float(np.mean(vals))
            aggregated[model_name][f"{metric_name}_std"] = float(np.std(vals))
    return aggregated


def main(attack_type: str = "false_state", graph_type: str = "knn",
         threshold: float = 0.5, n_snapshots: int = None, eval_seeds: int = 1):
    logger.info("=" * 60)
    logger.info("STAGE 7: Evaluation")
    logger.info(f"  attack_type = {attack_type}")
    logger.info(f"  graph_type  = {graph_type}")
    logger.info("=" * 60)

    eval_seed_values = [SEED + i for i in range(eval_seeds)]
    all_runs = [
        _evaluate_once(
            attack_type=attack_type,
            graph_type=graph_type,
            threshold=threshold,
            n_snapshots=n_snapshots,
            eval_seed=s,
        )
        for s in eval_seed_values
    ]

    from src.config import TABLE_DIR
    from src.utils import save_json

    all_metrics = _aggregate_metrics(all_runs) if eval_seeds > 1 else all_runs[0]
    out_name = f"{attack_type}_metrics_seedavg.json" if eval_seeds > 1 else f"{attack_type}_metrics.json"
    save_json(all_metrics, TABLE_DIR / out_name)

    # Print summary table
    header = (
        f"\n{'Model':<20}  {'Threshold':>9}  {'Accuracy':>9}  "
        f"{'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'ROC-AUC':>9}  {'FPR':>9}"
    )
    logger.info(header)
    logger.info("-" * len(header))
    for name, m in all_metrics.items():
        if eval_seeds > 1:
            acc_txt = f"{m.get('accuracy', 0):.4f}±{m.get('accuracy_std', 0):.4f}"
            prec_txt = f"{m.get('precision', 0):.4f}±{m.get('precision_std', 0):.4f}"
            rec_txt = f"{m.get('recall', 0):.4f}±{m.get('recall_std', 0):.4f}"
            f1_txt = f"{m.get('f1', 0):.4f}±{m.get('f1_std', 0):.4f}"
            auc_txt = f"{m.get('roc_auc', 0):.4f}±{m.get('roc_auc_std', 0):.4f}"
            fpr_txt = f"{m.get('fpr', 0):.4f}±{m.get('fpr_std', 0):.4f}"
        else:
            acc_txt = f"{m.get('accuracy', 0):.4f}"
            prec_txt = f"{m.get('precision', 0):.4f}"
            rec_txt = f"{m.get('recall', 0):.4f}"
            f1_txt = f"{m.get('f1', 0):.4f}"
            auc_txt = f"{m.get('roc_auc', 0):.4f}"
            fpr_txt = f"{m.get('fpr', 0):.4f}"
        logger.info(
            f"{name:<20}  {m.get('threshold', 0.5):>9.3f}  "
            f"{acc_txt:>9}  "
            f"{prec_txt:>9}  "
            f"{rec_txt:>9}  "
            f"{f1_txt:>9}  "
            f"{auc_txt:>9}  "
            f"{fpr_txt:>9}"
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
    parser.add_argument("--eval_seeds", type=int, default=1,
                        help="Number of evaluation seeds. >1 reports mean±std.")
    args = parser.parse_args()

    main(
        attack_type=args.attack_type,
        graph_type=args.graph_type,
        threshold=args.threshold,
        n_snapshots=args.snapshots,
        eval_seeds=args.eval_seeds,
    )
