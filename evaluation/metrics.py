"""
Evaluation metrics.

Calculates binary classification metrics for attack detection,
including per-attack-type breakdowns and ROC-AUC.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from typing import Dict, List, Optional


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calculate primary performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (0 for benign, 1 for attack)
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray, optional
        Predicted probabilities for the positive class (for AUC)

    Returns
    -------
    dict
        Dictionary of metric scores.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["fpr"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    metrics["fnr"] = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["avg_precision"] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["avg_precision"] = 0.0

    return metrics


def calculate_per_attack_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_types: List[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate metrics broken down by attack type.

    Parameters
    ----------
    y_true : binary labels (N,)
    y_pred : predicted binary labels (N,)
    attack_types : list of attack type strings per node
    y_prob : predicted probabilities (N,) optional

    Returns
    -------
    dict mapping attack_type -> metric dict
    """
    results = {}
    unique_types = sorted(set(attack_types))

    for atype in unique_types:
        mask = np.array([t == atype for t in attack_types])
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        yprob = y_prob[mask] if y_prob is not None else None
        results[atype] = calculate_metrics(yt, yp, yprob)

    return results


def formatted_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> str:
    """Returns a string representation of the confusion matrix and metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0)

    m = calculate_metrics(y_true, y_pred, y_prob)

    lines = [
        "═" * 40,
        "       Classification Report",
        "═" * 40,
        f"  Accuracy:          {m['accuracy']:.4f}",
        f"  Precision:         {m['precision']:.4f}",
        f"  Recall:            {m['recall']:.4f}",
        f"  F1 Score:          {m['f1']:.4f}",
    ]
    if "roc_auc" in m:
        lines.append(f"  ROC-AUC:           {m['roc_auc']:.4f}")
        lines.append(f"  Avg Precision:     {m['avg_precision']:.4f}")
    lines += [
        "─" * 40,
        "       Confusion Matrix",
        "─" * 40,
        f"  TN: {tn:<8}  FP: {fp}",
        f"  FN: {fn:<8}  TP: {tp}",
        "═" * 40,
    ]
    return "\n".join(lines)


def metrics_to_latex_row(
    model_name: str, attack_type: str, metrics: Dict[str, float]
) -> str:
    """Format metrics as a LaTeX table row for the paper."""
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    vals = [f"{metrics.get(c, 0.0):.4f}" for c in cols]
    return f"{model_name} & {attack_type} & " + " & ".join(vals) + r" \\"
