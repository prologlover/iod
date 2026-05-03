"""
Spatio-Temporal GAT Model.

Combines a GRU layer to process temporal feature sequences
with a GATv2 layer to process final spatial topology.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import List
from torch_geometric.data import Data, Batch

class GATTemporalModel(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int, 
        num_heads: int, 
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
        # GATv2 is more robust for dynamic graphs
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
        self.gat2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, concat=True, dropout=dropout)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification: benign vs attack
        )
        
    def forward(self, features: torch.Tensor, final_graphs: List[Data]):
        """
        Parameters
        ----------
        features : (B, T, N, F) tensor
            Temporal node features.
        final_graphs : List[Data]
            The final timestep graphs containing edge connectivity.
            
        Returns
        -------
        logits : (B, N, 2)
        """
        B, T, N, F_dim = features.shape
        
        # 1. Temporal Processing
        # Flatten B and N to process sequences through GRU
        # x_seq shape: (B*N, T, F_dim)
        x_seq = features.view(B * N, T, F_dim)
        
        gru_out, _ = self.gru(x_seq)
        
        # Take the output from the last timestep -> (B*N, hidden_dim)
        temporal_emb = gru_out[:, -1, :]
        
        # Reshape to (B, N, hidden_dim)
        temporal_emb = temporal_emb.view(B, N, -1)
        
        # 2. Spatial Processing
        # We need to apply GAT using the edge_index of final_graphs
        # Easiest way in PyG is to create a dynamic batch
        device = features.device
        
        graph_list = []
        for b in range(B):
            # Create a shallow copy with replaced node features
            g = final_graphs[b].clone()
            g.x = temporal_emb[b]
            graph_list.append(g)
            
        batch = Batch.from_data_list(graph_list).to(device)
        
        # Apply GAT
        x, edge_index = batch.x, batch.edge_index
        
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x) # Output is (B*N, hidden_dim)
        
        # 3. Classification
        logits = self.classifier(x)
        
        # Reshape back to (B, N, 2)
        logits = logits.view(B, N, 2)
        
        return logits
