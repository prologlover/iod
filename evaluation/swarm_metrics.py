"""
Swarm-specific evaluation metrics.

These metrics go beyond standard classification scores to characterise
how well the detector performs in a dynamic swarm context.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple


def attack_localization_accuracy(
    y_true_seq: List[np.ndarray],
    y_pred_seq: List[np.ndarray],
) -> float:
    """
    Fraction of Byzantine nodes correctly identified across all timesteps.

    Parameters
    ----------
    y_true_seq : list of (N,) arrays — true binary labels per timestep
    y_pred_seq : list of (N,) arrays — predicted binary labels per timestep

    Returns
    -------
    float in [0, 1]
    """
    correct, total = 0, 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        attacker_mask = y_true == 1
        if attacker_mask.sum() == 0:
            continue
        correct += (y_pred[attacker_mask] == 1).sum()
        total += attacker_mask.sum()
    return float(correct / total) if total > 0 else 0.0


def detection_latency(
    y_true_seq: List[np.ndarray],
    y_pred_seq: List[np.ndarray],
) -> int:
    """
    Number of timesteps until the first correctly detected Byzantine node.

    Returns -1 if no correct detection occurs.
    """
    for t, (y_true, y_pred) in enumerate(zip(y_true_seq, y_pred_seq)):
        attacker_mask = y_true == 1
        if attacker_mask.any() and (y_pred[attacker_mask] == 1).any():
            return t
    return -1


def per_node_false_positive_rate(
    y_true_seq: List[np.ndarray],
    y_pred_seq: List[np.ndarray],
) -> np.ndarray:
    """
    False positive rate computed per drone across the temporal sequence.

    Parameters
    ----------
    y_true_seq : list of (N,) arrays
    y_pred_seq : list of (N,) arrays

    Returns
    -------
    np.ndarray of shape (N,) with per-drone FPR
    """
    y_true_mat = np.stack(y_true_seq, axis=0)  # (T, N)
    y_pred_mat = np.stack(y_pred_seq, axis=0)  # (T, N)

    benign_mask = y_true_mat == 0
    fp = ((y_pred_mat == 1) & benign_mask).sum(axis=0).astype(float)
    tn_fp = benign_mask.sum(axis=0).astype(float)
    tn_fp = np.where(tn_fp == 0, 1.0, tn_fp)

    return fp / tn_fp


def swarm_metrics_summary(
    y_true_seq: List[np.ndarray],
    y_pred_seq: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute all swarm-specific metrics in one call.

    Returns
    -------
    dict with keys: localization_accuracy, detection_latency,
                    mean_per_node_fpr, max_per_node_fpr
    """
    fpr_per_node = per_node_false_positive_rate(y_true_seq, y_pred_seq)
    return {
        "localization_accuracy": attack_localization_accuracy(y_true_seq, y_pred_seq),
        "detection_latency": detection_latency(y_true_seq, y_pred_seq),
        "mean_per_node_fpr": float(fpr_per_node.mean()),
        "max_per_node_fpr": float(fpr_per_node.max()),
    }


def attacker_ratio_sweep_metrics(
    results_by_ratio: Dict[float, Dict[str, float]],
) -> str:
    """
    Format a dict of {ratio: metrics} as a readable summary string.
    """
    lines = [
        f"{'Ratio':>8}  {'F1':>8}  {'Localization':>14}  {'Latency':>9}",
        "-" * 50,
    ]
    for ratio in sorted(results_by_ratio.keys()):
        m = results_by_ratio[ratio]
        lines.append(
            f"{ratio:>8.0%}  "
            f"{m.get('f1', 0):>8.4f}  "
            f"{m.get('localization_accuracy', 0):>14.4f}  "
            f"{m.get('detection_latency', -1):>9}"
        )
    return "\n".join(lines)
