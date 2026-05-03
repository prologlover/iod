"""
Baseline GCN model for single-snapshot Byzantine detection.

Uses three GCNConv layers from PyTorch Geometric without temporal modeling.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from typing import List


class GCNModel(nn.Module):
    """
    3-layer GCN for node-level binary classification on a single graph snapshot.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_dim : int
        Hidden layer width.
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
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_classes),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Single-graph forward pass.

        Parameters
        ----------
        x : (N, F) node feature matrix
        edge_index : (2, E) edge index

        Returns
        -------
        logits : (N, num_classes)
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        return self.classifier(h)

    def forward_graphs(self, graphs: List[Data]) -> torch.Tensor:
        """
        Batch forward pass over a list of graph snapshots.

        Parameters
        ----------
        graphs : list of PyG Data objects

        Returns
        -------
        logits : (B, N, num_classes) — assumes fixed N across graphs
        """
        device = graphs[0].x.device
        B = len(graphs)
        N = graphs[0].x.shape[0]

        batch = Batch.from_data_list(graphs).to(device)
        logits_flat = self.forward(batch.x, batch.edge_index)  # (B*N, C)
        return logits_flat.view(B, N, -1)
