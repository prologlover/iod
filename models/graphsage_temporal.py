"""
Spatio-Temporal GraphSAGE Model.

Combines a GRU layer to process temporal feature sequences
with a GraphSAGE layer to process final spatial topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from typing import List
from torch_geometric.data import Data, Batch

class GraphSAGETemporalModel(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int, 
        dropout: float,
        gru_layers: int = 2
    ):
        super().__init__()
        
        # Temporal component
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0
        )
        
        # Spatial component
        self.sage1 = SAGEConv(hidden_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.dropout = dropout
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
        
    def forward(self, features: torch.Tensor, final_graphs: List[Data]):
        B, T, N, F_dim = features.shape
        
        # Temporal
        x_seq = features.view(B * N, T, F_dim)
        gru_out, _ = self.gru(x_seq)
        temporal_emb = gru_out[:, -1, :]
        temporal_emb = temporal_emb.view(B, N, -1)
        
        # Spatial
        device = features.device
        
        graph_list = []
        for b in range(B):
            g = final_graphs[b].clone()
            g.x = temporal_emb[b]
            graph_list.append(g)
            
        batch = Batch.from_data_list(graph_list).to(device)
        
        x, edge_index = batch.x, batch.edge_index
        
        x = self.sage1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.sage2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        logits = self.classifier(x)
        logits = logits.view(B, N, 2)
        
        return logits
