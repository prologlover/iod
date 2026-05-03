"""
Baseline LSTM model with temporal attention for Byzantine detection.

Processes per-node feature sequences without any graph topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    """
    2-layer LSTM with a learned attention mechanism over time steps.

    Parameters
    ----------
    in_channels : int
        Number of input features per time step.
    hidden_dim : int
        LSTM hidden size.
    num_classes : int
        Output classes (default 2).
    num_layers : int
        LSTM depth (default 2).
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        # Attention: score each timestep output
        self.attention = nn.Linear(hidden_dim, 1)

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
        x : (B, T, N, F) temporal node features
            or (M, T, F) for a flat batch of sequences

        Returns
        -------
        logits : (B, N, num_classes) or (M, num_classes)
        """
        if x.dim() == 4:
            B, T, N, F_dim = x.shape
            # Reshape to (B*N, T, F)
            x_seq = x.permute(0, 2, 1, 3).reshape(B * N, T, F_dim)
        else:
            x_seq = x  # (M, T, F)
            B, N = None, None

        lstm_out, _ = self.lstm(x_seq)  # (M, T, H)

        # Attention weights
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (M, T)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (M, T, 1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (M, H)

        logits = self.classifier(context)  # (M, num_classes)

        if B is not None:
            logits = logits.view(B, N, -1)

        return logits
