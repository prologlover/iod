"""
Train the proposed spatio-temporal GAT+GRU model.

Usage
-----
    python scripts/train_gnn_temporal.py [--model gat|graphsage]
                                         [--epochs 100]
                                         [--attack_type false_state]
                                         [--graph_type knn|distance|hexagonal]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    BATCH_SIZE, DEVICE, DROPOUT, EPOCHS, FOCAL_ALPHA, FOCAL_GAMMA,
    GAT_LAYERS, GRU_LAYERS, HIDDEN_DIM, LEARNING_RATE, MODEL_DIR,
    NUM_HEADS, NUM_SWARM_SNAPSHOTS, PATIENCE, SEED, WEIGHT_DECAY,
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger, save_json, count_parameters
from graphs.swarm_simulator import generate_swarm_dataset
from graphs.temporal_graph import build_temporal_graphs, pack_temporal_batch
from attacks import inject_attacks
from models import get_model
from models.model_utils import FocalLoss, EarlyStopping, find_best_threshold

logger = get_logger(__name__)


def get_batches(seqs, batch_size):
    for i in range(0, len(seqs), batch_size):
        yield seqs[i : i + batch_size]


def train_epoch(model, seqs, optimizer, criterion):
    model.train()
    total_loss, n_batches = 0.0, 0
    random.shuffle(seqs)

    for batch in get_batches(seqs, BATCH_SIZE):
        features, labels, final_graphs = pack_temporal_batch(batch)
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(features, final_graphs)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def eval_epoch(model, seqs, criterion):
    """Returns (avg_loss, val_f1)."""
    from sklearn.metrics import f1_score

    model.eval()
    total_loss, n_batches = 0.0, 0
    all_labels, all_probs = [], []

    for batch in get_batches(seqs, BATCH_SIZE):
        features, labels, final_graphs = pack_temporal_batch(batch)
        features, labels = features.to(DEVICE), labels.to(DEVICE)
        logits = model(features, final_graphs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

        probs = F.softmax(logits, dim=-1)[..., 1].cpu().numpy().ravel()
        all_labels.extend(labels.cpu().numpy().ravel().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(1, n_batches)
    y_pred = (np.array(all_probs) >= 0.5).astype(int)
    val_f1 = float(f1_score(np.array(all_labels), y_pred, zero_division=0))
    return avg_loss, val_f1


def main(
    model_name: str = "gat",
    epochs: int = EPOCHS,
    attack_type: str = "false_state",
    graph_type: str = "knn",
    n_snapshots: int = None,
):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    logger.info("=" * 60)
    logger.info("STAGE 6: Train Spatio-Temporal GNN")
    logger.info(f"  model      = {model_name}")
    logger.info(f"  epochs     = {epochs}")
    logger.info(f"  attack     = {attack_type}")
    logger.info(f"  graph_type = {graph_type}")
    logger.info(f"  device     = {DEVICE}")
    logger.info("=" * 60)

    # Data
    logger.info("Loading and preprocessing data...")
    raw_df = load_and_merge()
    prep = run_preprocessing(raw_df)
    train_df, val_df = prep["train_df"], prep["val_df"]
    feature_cols = prep["feature_cols"]
    num_features = len(feature_cols)
    logger.info(f"Features: {num_features}")

    total = n_snapshots if n_snapshots else NUM_SWARM_SNAPSHOTS
    n_train = max(4, int(total * 0.70))
    n_val = max(2, int(total * 0.15))

    logger.info("Generating swarm sequences...")
    train_tab = generate_swarm_dataset(train_df, feature_cols, n_train, seed=rng.integers(10_000))
    val_tab = generate_swarm_dataset(val_df, feature_cols, n_val, seed=rng.integers(10_000))

    train_pyg = build_temporal_graphs(train_tab, graph_type=graph_type)
    val_pyg = build_temporal_graphs(val_tab, graph_type=graph_type)

    train_seqs = inject_attacks(train_pyg, attack_type, rng)
    val_seqs = inject_attacks(val_pyg, attack_type, rng)

    # Model
    model_kwargs = dict(
        in_channels=num_features,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        gru_layers=GRU_LAYERS,
    )
    if model_name == "gat":
        model_kwargs["num_heads"] = NUM_HEADS

    model = get_model(model_name, **model_kwargs).to(DEVICE)
    logger.info(f"Model parameters: {count_parameters(model):,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    checkpoint = MODEL_DIR / f"best_{model_name}_temporal.pt"
    # Use F1 (higher = better) to drive early stopping instead of loss
    early_stop = EarlyStopping(patience=PATIENCE, checkpoint_path=checkpoint, mode="max")

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    logger.info("Starting training...")
    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_seqs, optimizer, criterion)
        v_loss, v_f1 = eval_epoch(model, val_seqs, criterion)
        scheduler.step()

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)

        logger.info(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
            f"Val F1: {v_f1:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}"
        )

        if early_stop.step(v_f1, model):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    model = early_stop.load_best(model)

    # Find the optimal classification threshold on the validation set
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in get_batches(val_seqs, BATCH_SIZE):
            features, labels, final_graphs = pack_temporal_batch(batch)
            features = features.to(DEVICE)
            logits = model(features, final_graphs)
            probs = F.softmax(logits, dim=-1)[..., 1].cpu().numpy().ravel()
            all_labels.extend(labels.numpy().ravel().tolist())
            all_probs.extend(probs.tolist())

    best_thresh, best_val_f1 = find_best_threshold(
        np.array(all_labels), np.array(all_probs)
    )
    save_json(
        {"threshold": best_thresh, "val_f1": best_val_f1},
        MODEL_DIR / f"threshold_{model_name}_temporal.json",
    )
    logger.info(
        f"Optimal threshold: {best_thresh:.3f}  (val F1 @ threshold = {best_val_f1:.4f})"
    )

    save_json(history, MODEL_DIR / f"history_{model_name}_temporal.json")
    logger.info(f"Training complete. Best model: {checkpoint}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gat", choices=["gat", "graphsage"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--attack_type", default="false_state")
    parser.add_argument("--graph_type", default="knn",
                        choices=["knn", "distance", "hexagonal"])
    parser.add_argument("--snapshots", type=int, default=None,
                        help="Total swarm snapshots. Use small value for quick testing.")
    args = parser.parse_args()

    main(
        model_name=args.model,
        epochs=args.epochs,
        attack_type=args.attack_type,
        graph_type=args.graph_type,
        n_snapshots=args.snapshots,
    )
