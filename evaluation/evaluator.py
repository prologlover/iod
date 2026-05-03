"""
Full evaluation pipeline.

Runs all trained models on the test set and generates:
  - Comparison metrics table (console + LaTeX)
  - Confusion matrices (PNG)
  - ROC curves (overlaid, PNG)
  - Precision-Recall curves (PNG)
  - Training curve plots (PNG) from saved logs
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURE_DIR, TABLE_DIR, DEVICE
from src.utils import get_logger, save_json
from evaluation.metrics import calculate_metrics, metrics_to_latex_row
from evaluation.swarm_metrics import swarm_metrics_summary

logger = get_logger(__name__)


# ================================================================== #
#  Prediction helpers                                                 #
# ================================================================== #

def predict_gnn(model, graph_sequences, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the spatio-temporal GNN model on graph sequences and collect
    flat (node-level) predictions.

    Returns
    -------
    y_true : (M,) int array
    y_prob : (M,) float array  — P(attack)
    """
    from graphs.temporal_graph import pack_temporal_batch

    model.eval()
    all_true, all_prob = [], []

    for i in range(0, len(graph_sequences), batch_size):
        batch = graph_sequences[i : i + batch_size]
        features, labels, final_graphs = pack_temporal_batch(batch)
        features = features.to(DEVICE)

        with torch.no_grad():
            logits = model(features, final_graphs)  # (B, N, 2)
            probs = F.softmax(logits, dim=-1)[:, :, 1]  # (B, N)

        all_true.append(labels.cpu().numpy().ravel())
        all_prob.append(probs.cpu().numpy().ravel())

    return np.concatenate(all_true), np.concatenate(all_prob)


def predict_baseline(model, graph_sequences, model_type: str,
                     batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run MLP / LSTM / CNN / GCN on graph sequences.

    model_type : 'mlp', 'lstm', 'cnn', 'gcn'
    """
    from graphs.temporal_graph import pack_temporal_batch

    model.eval()
    all_true, all_prob = [], []

    for i in range(0, len(graph_sequences), batch_size):
        batch = graph_sequences[i : i + batch_size]
        features, labels, final_graphs = pack_temporal_batch(batch)
        features = features.to(DEVICE)

        with torch.no_grad():
            if model_type == "mlp":
                # Use only the last timestep features
                x = features[:, -1, :, :]  # (B, N, F)
                logits = model(x)
            elif model_type in ("lstm", "cnn"):
                logits = model(features)       # (B, N, C)
            elif model_type == "gcn":
                # Use last-timestep graph
                x_last = features[:, -1, :, :]  # (B, N, F)
                B_g, N_g, _ = x_last.shape
                logits_list = []
                for b in range(B_g):
                    g = final_graphs[b].clone()
                    g.x = x_last[b].to(DEVICE)
                    lo = model(g.x, g.edge_index.to(DEVICE))  # (N, 2)
                    logits_list.append(lo)
                logits = torch.stack(logits_list, dim=0)  # (B, N, 2)
            else:
                raise ValueError(f"Unknown model_type '{model_type}'")

            probs = F.softmax(logits, dim=-1)[:, :, 1]

        all_true.append(labels.cpu().numpy().ravel())
        all_prob.append(probs.cpu().numpy().ravel())

    return np.concatenate(all_true), np.concatenate(all_prob)


# ================================================================== #
#  Plot helpers                                                       #
# ================================================================== #

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray,
    model_name: str, save_prefix: str = "",
) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"], fontsize=11)
    ax.set_yticklabels(["Benign", "Attack"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)

    plt.tight_layout()
    pref = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{pref}cm_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_prefix: str = "",
) -> Path:
    """
    Overlay ROC curves for multiple models.

    Parameters
    ----------
    results : dict mapping model_name -> (y_true, y_prob)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, (y_true, y_prob)), color in zip(results.items(), colors):
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pref = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{pref}roc_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_pr_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_prefix: str = "",
) -> Path:
    """Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    for (name, (y_true, y_prob)), color in zip(results.items(), colors):
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = auc(rec, prec)
        ax.plot(rec, prec, lw=2, color=color, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pref = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{pref}pr_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    save_prefix: str = "",
) -> Path:
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict with keys 'train_loss', 'val_loss', and optionally
              'train_acc', 'val_acc'
    """
    has_acc = "train_acc" in history

    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14 if has_acc else 7, 5))
    if not has_acc:
        axes = [axes]

    # Loss
    ax = axes[0]
    ax.plot(history["train_loss"], label="Train", color="#1976D2", lw=2)
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", color="#E53935", lw=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"{model_name} — Loss", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if has_acc:
        ax = axes[1]
        ax.plot(history["train_acc"], label="Train", color="#43A047", lw=2)
        if "val_acc" in history:
            ax.plot(history["val_acc"], label="Val", color="#FB8C00", lw=2)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Accuracy", fontsize=11)
        ax.set_title(f"{model_name} — Accuracy", fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pref = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{pref}training_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path


# ================================================================== #
#  Main evaluation pipeline                                           #
# ================================================================== #

def run_evaluation(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    threshold: float = 0.5,
    save_prefix: str = "eval",
) -> Dict[str, Dict]:
    """
    Full evaluation pipeline.

    Parameters
    ----------
    model_predictions : dict mapping model_name -> (y_true, y_prob)
        All models share the same y_true.
    threshold : float
        Classification threshold.
    save_prefix : str
        Prefix for all output files.

    Returns
    -------
    dict mapping model_name -> metric dict
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    latex_rows = []

    for model_name, (y_true, y_prob) in model_predictions.items():
        y_pred = (y_prob >= threshold).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        all_metrics[model_name] = metrics

        logger.info(f"\n{model_name}:")
        for k, v in metrics.items():
            logger.info(f"  {k:20s}: {v:.4f}")

        # Confusion matrix
        plot_confusion_matrix(y_true, y_pred, model_name, save_prefix=save_prefix)

        # LaTeX row
        latex_rows.append(metrics_to_latex_row(model_name, "all", metrics))

    # Overlay ROC curves
    plot_roc_curves(model_predictions, save_prefix=save_prefix)
    plot_pr_curves(model_predictions, save_prefix=save_prefix)

    # Save LaTeX table
    header = (
        r"\begin{table}[h]\centering"
        r"\caption{Model Comparison on Byzantine Detection Test Set}"
        r"\label{tab:results}"
        r"\begin{tabular}{llccccc}"
        r"\toprule"
        r"Model & Attack & Accuracy & Precision & Recall & F1 & ROC-AUC \\"
        r"\midrule"
    )
    footer = r"\bottomrule\end{tabular}\end{table}"
    latex_table = header + "\n" + "\n".join(latex_rows) + "\n" + footer

    latex_path = TABLE_DIR / f"{save_prefix}_results.tex"
    latex_path.write_text(latex_table)
    logger.info(f"LaTeX table saved to {latex_path}")

    # Save JSON
    save_json(all_metrics, TABLE_DIR / f"{save_prefix}_metrics.json")

    return all_metrics
