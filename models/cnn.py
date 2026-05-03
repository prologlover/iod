"""
Baseline 1D-CNN model for temporal Byzantine detection.

Applies 1D convolutions over the time dimension of per-node sequences.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    1D-CNN over temporal feature sequences.

    Architecture: Conv1D → BatchNorm → ReLU → MaxPool (×2) → Flatten → FC

    Parameters
    ----------
    in_channels : int
        Number of input features per time step.
    hidden_dim : int
        Number of convolution filters.
    num_classes : int
        Output classes (default 2).
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

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, N, F) or (M, T, F) temporal node features

        Returns
        -------
        logits : (B, N, num_classes) or (M, num_classes)
        """
        if x.dim() == 4:
            B, T, N, F_dim = x.shape
            # Reshape to (B*N, F, T) — Conv1d expects (batch, channels, length)
            x_seq = x.permute(0, 2, 3, 1).reshape(B * N, F_dim, T)
        else:
            # (M, T, F) → (M, F, T)
            x_seq = x.permute(0, 2, 1)
            B, N = None, None

        h = self.conv_block1(x_seq)
        h = self.conv_block2(h)
        h = self.global_pool(h).squeeze(-1)  # (M, hidden_dim)

        logits = self.classifier(h)

        if B is not None:
            logits = logits.view(B, N, -1)

        return logits
