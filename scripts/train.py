"""
Training script for Spatio-Temporal Byzantine Drone Detection.
"""
import sys
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MODEL_DIR,
    FOCAL_ALPHA, FOCAL_GAMMA, HIDDEN_DIM, NUM_HEADS, GAT_LAYERS, GRU_LAYERS, DROPOUT, DEVICE,
    NUM_SWARM_SNAPSHOTS
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from graphs.swarm_simulator import generate_swarm_dataset
from attacks import inject_attacks
from graphs.temporal_graph import build_temporal_graphs, pack_temporal_batch
from models import get_model
from src.utils import get_logger

logger = get_logger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # inputs: (B, N, 2)
        # targets: (B, N)
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs.view(-1, 2), targets.view(-1))
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss)
        return focal_loss.mean()

def create_dataset_pipeline(df, feature_cols, num_snapshots, attack_type, rng):
    """Pipeline from Raw DF -> Swarm Tabular -> PyG graphs."""
    tabular_seq = generate_swarm_dataset(
        df, feature_cols, num_snapshots=num_snapshots, 
        seed=rng.integers(10000)
    )
    pyg_seq = build_temporal_graphs(tabular_seq, graph_type="knn")
    attacked_seq = inject_attacks(pyg_seq, attack_type, rng)
    return attacked_seq

def get_batches(seqs, batch_size):
    for i in range(0, len(seqs), batch_size):
        yield seqs[i:i + batch_size]

def train():
    rng = np.random.default_rng(42)
    
    logger.info("Loading and Preprocessing Data...")
    raw_df = load_and_merge()
    prep = run_preprocessing(raw_df)
    
    train_df = prep['train_df']
    val_df = prep['val_df']
    feature_cols = prep['feature_cols']
    num_features = len(feature_cols)
    
    logger.info("Generating Training Graphs (FDI attack)...")
    # Using smaller size for memory efficiency if needed
    train_seqs = create_dataset_pipeline(train_df, feature_cols, int(NUM_SWARM_SNAPSHOTS * 0.7), "fdi", rng)
    val_seqs = create_dataset_pipeline(val_df, feature_cols, int(NUM_SWARM_SNAPSHOTS * 0.15), "fdi", rng)
    
    # Init Model
    logger.info("Initializing Model...")
    model = get_model(
        "gat",
        in_channels=num_features,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        gru_layers=GRU_LAYERS
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    
    best_val_loss = float('inf')
    
    logger.info("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # Shuffle train sequences
        random.shuffle(train_seqs)
        
        for batch_seqs in get_batches(train_seqs, BATCH_SIZE):
            features, labels, final_graphs = pack_temporal_batch(batch_seqs)
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(features, final_graphs)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_seqs) / BATCH_SIZE
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_seqs in get_batches(val_seqs, BATCH_SIZE):
                features, labels, final_graphs = pack_temporal_batch(batch_seqs)
                features, labels = features.to(DEVICE), labels.to(DEVICE)
                
                logits = model(features, final_graphs)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
        val_loss /= len(val_seqs) / BATCH_SIZE
        
        logger.info(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_DIR, "best_gat_temporal.pt")
            torch.save(model.state_dict(), save_path)
            logger.info(f"--> Saved better model to {save_path}")

    logger.info("Training Complete.")

if __name__ == "__main__":
    train()
