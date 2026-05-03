"""
Baseline MLP model for tabular Byzantine detection.

Operates on flattened per-node feature vectors with no graph structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    3-layer MLP with BatchNorm, Dropout, and ReLU activations.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_dim : int
        Width of hidden layers.
    num_classes : int
        Number of output classes (default 2: benign/attack).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, F) or (B, N, F) — node feature tensor

        Returns
        -------
        logits : same leading dims + (num_classes,)
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        logits = self.net(x_flat)
        return logits.reshape(*orig_shape[:-1], -1)
