"""
Model utilities: FocalLoss, EarlyStopping, class-weight helpers.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha is applied per-class: alpha for the positive class (attack),
    (1 - alpha) for the negative class (benign).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : (N, C) logits or (B, N, C) for node-level tasks
        targets : (N,) or (B, N) integer labels
        """
        orig_shape = inputs.shape
        inputs_flat = inputs.view(-1, orig_shape[-1])
        targets_flat = targets.view(-1)

        ce_loss = nn.CrossEntropyLoss(reduction="none")(inputs_flat, targets_flat)
        pt = torch.exp(-ce_loss)

        # Per-sample alpha: self.alpha for attack (class 1), (1-alpha) for benign (class 0)
        alpha_t = torch.where(
            targets_flat == 1,
            torch.full_like(ce_loss, self.alpha),
            torch.full_like(ce_loss, 1.0 - self.alpha),
        )
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs to wait after last improvement.
    min_delta : float
        Minimum change to qualify as an improvement.
    checkpoint_path : Path
        Where to save the best model checkpoint.
    mode : str
        'min' to stop when metric stops decreasing (e.g. loss),
        'max' to stop when metric stops increasing (e.g. F1).
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        checkpoint_path: Optional[Path] = None,
        mode: str = "min",
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.best_score: float = float("inf") if mode == "min" else float("-inf")
        self.counter: int = 0
        self.best_epoch: int = 0

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float, model: nn.Module) -> bool:
        """
        Call after each epoch.

        Parameters
        ----------
        score : float
            The monitored metric value (loss or F1, etc.).

        Returns
        -------
        bool : True if training should stop.
        """
        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.checkpoint_path is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1

        return self.counter >= self.patience

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best checkpoint into the model."""
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            model.load_state_dict(torch.load(self.checkpoint_path, map_location="cpu"))
        return model


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_thresholds: int = 200,
) -> tuple:
    """
    Sweep thresholds and return the one that maximises the validation F1 score.

    Parameters
    ----------
    y_true : (N,) binary ground-truth labels
    y_prob : (N,) predicted probability of the positive class
    n_thresholds : int
        Number of candidate thresholds to evaluate.

    Returns
    -------
    best_threshold : float
    best_f1 : float
    """
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_thresh, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh, best_f1


def get_class_weights(labels: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for loss balancing.

    Parameters
    ----------
    labels : 1-D integer array
    num_classes : int

    Returns
    -------
    torch.Tensor of shape (num_classes,)
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
