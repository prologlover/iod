"""
Train baseline models: MLP, LSTM, 1D-CNN, GCN.

Usage
-----
    python scripts/train_baselines.py [--model mlp|lstm|cnn|gcn|all]
                                      [--epochs 50]
                                      [--attack_type false_state]
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
    HIDDEN_DIM, LEARNING_RATE, MODEL_DIR, NUM_SWARM_SNAPSHOTS,
    PATIENCE, SEED, WEIGHT_DECAY,
)
from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger, save_json
from graphs.swarm_simulator import generate_swarm_dataset
from graphs.temporal_graph import build_temporal_graphs, pack_temporal_batch
from attacks import inject_attacks
from models.mlp import MLP
from models.lstm import LSTMModel
from models.cnn import CNNModel
from models.gcn import GCNModel
from models.model_utils import FocalLoss, EarlyStopping, find_best_threshold

logger = get_logger(__name__)

BASELINE_MODELS = ["mlp", "lstm", "cnn", "gcn"]


def get_batches(seqs, batch_size):
    for i in range(0, len(seqs), batch_size):
        yield seqs[i : i + batch_size]


def train_epoch(model, seqs, optimizer, criterion, model_type):
    model.train()
    total_loss = 0.0
    random.shuffle(seqs)

    for batch in get_batches(seqs, BATCH_SIZE):
        features, labels, final_graphs = pack_temporal_batch(batch)
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        if model_type == "mlp":
            x = features[:, -1, :, :]
            logits = model(x)
        elif model_type in ("lstm", "cnn"):
            logits = model(features)
        elif model_type == "gcn":
            x_last = features[:, -1, :, :]
            B_g, N_g, _ = x_last.shape
            logits_list = []
            for b in range(B_g):
                g = final_graphs[b].clone()
                g.x = x_last[b].to(DEVICE)
                logits_list.append(model(g.x, g.edge_index.to(DEVICE)))
            logits = torch.stack(logits_list, dim=0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(seqs) // BATCH_SIZE)


@torch.no_grad()
def eval_epoch(model, seqs, criterion, model_type):
    """Returns (avg_loss, val_f1_best, val_best_threshold) over validation sequences."""
    from models.model_utils import find_best_threshold

    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch in get_batches(seqs, BATCH_SIZE):
        features, labels, final_graphs = pack_temporal_batch(batch)
        features, labels = features.to(DEVICE), labels.to(DEVICE)

        if model_type == "mlp":
            x = features[:, -1, :, :]
            logits = model(x)
        elif model_type in ("lstm", "cnn"):
            logits = model(features)
        elif model_type == "gcn":
            x_last = features[:, -1, :, :]
            B_g, N_g, _ = x_last.shape
            logits_list = []
            for b in range(B_g):
                g = final_graphs[b].clone()
                g.x = x_last[b].to(DEVICE)
                logits_list.append(model(g.x, g.edge_index.to(DEVICE)))
            logits = torch.stack(logits_list, dim=0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = F.softmax(logits, dim=-1)[..., 1].cpu().numpy().ravel()
        all_labels.extend(labels.cpu().numpy().ravel().tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(1, len(seqs) // BATCH_SIZE)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    best_t, val_f1_best = find_best_threshold(y_true, y_prob)
    return avg_loss, val_f1_best, best_t


def build_model(model_type: str, num_features: int):
    if model_type == "mlp":
        return MLP(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    if model_type == "lstm":
        return LSTMModel(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    if model_type == "cnn":
        return CNNModel(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    if model_type == "gcn":
        return GCNModel(in_channels=num_features, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    raise ValueError(f"Unknown model: {model_type}")


def _collect_val_probs(model, seqs, model_type):
    """Collect all validation (y_true, y_prob) pairs for threshold tuning."""
    all_labels, all_probs = [], []
    model.eval()
    with torch.no_grad():
        for batch in get_batches(seqs, BATCH_SIZE):
            features, labels, final_graphs = pack_temporal_batch(batch)
            features = features.to(DEVICE)
            if model_type == "mlp":
                logits = model(features[:, -1, :, :])
            elif model_type in ("lstm", "cnn"):
                logits = model(features)
            elif model_type == "gcn":
                x_last = features[:, -1, :, :]
                logits_list = []
                for b in range(x_last.shape[0]):
                    g = final_graphs[b].clone()
                    g.x = x_last[b].to(DEVICE)
                    logits_list.append(model(g.x, g.edge_index.to(DEVICE)))
                logits = torch.stack(logits_list, dim=0)
            probs = F.softmax(logits, dim=-1)[..., 1].cpu().numpy().ravel()
            all_labels.extend(labels.numpy().ravel().tolist())
            all_probs.extend(probs.tolist())
    return np.array(all_labels), np.array(all_probs)


def train_model(model_type: str, train_seqs, val_seqs, num_features: int, epochs: int):
    model = build_model(model_type, num_features).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    checkpoint = MODEL_DIR / f"best_{model_type}.pt"
    # Use F1 (higher = better) to drive early stopping instead of loss
    early_stop = EarlyStopping(patience=PATIENCE, checkpoint_path=checkpoint, mode="max")

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_best_threshold": []}

    logger.info(f"\n{'='*60}")
    logger.info(f"Training: {model_type.upper()}")
    logger.info(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t_loss = train_epoch(model, train_seqs, optimizer, criterion, model_type)
        v_loss, v_f1, v_t = eval_epoch(model, val_seqs, criterion, model_type)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["val_f1"].append(v_f1)
        history["val_best_threshold"].append(v_t)

        logger.info(
            f"Epoch {epoch:03d} | Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | Val F1(best-th): {v_f1:.4f} | Best-th: {v_t:.3f}"
        )

        if early_stop.step(v_f1, model):
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    model = early_stop.load_best(model)

    # Find the optimal classification threshold on the validation set
    y_true_val, y_prob_val = _collect_val_probs(model, val_seqs, model_type)
    best_thresh, best_val_f1 = find_best_threshold(y_true_val, y_prob_val)
    save_json(
        {"threshold": best_thresh, "val_f1": best_val_f1},
        MODEL_DIR / f"threshold_{model_type}.json",
    )
    logger.info(
        f"Optimal threshold: {best_thresh:.3f}  (val F1 @ threshold = {best_val_f1:.4f})"
    )

    save_json(history, MODEL_DIR / f"history_{model_type}.json")
    logger.info(f"Best model saved to {checkpoint}")
    return model, history


def main(model_types=None, epochs=EPOCHS, attack_type="false_state", n_snapshots=None):
    if model_types is None:
        model_types = BASELINE_MODELS

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

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

    logger.info("Generating swarm graph sequences...")
    train_tabular = generate_swarm_dataset(train_df, feature_cols, n_train, seed=rng.integers(10_000))
    val_tabular = generate_swarm_dataset(val_df, feature_cols, n_val, seed=rng.integers(10_000))

    train_pyg = build_temporal_graphs(train_tabular, graph_type="knn")
    val_pyg = build_temporal_graphs(val_tabular, graph_type="knn")

    train_seqs = inject_attacks(train_pyg, attack_type, rng)
    val_seqs = inject_attacks(val_pyg, attack_type, rng)

    for model_type in model_types:
        train_model(model_type, train_seqs, val_seqs, num_features, epochs)

    logger.info("\nAll baseline models trained.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        help="Model name or 'all'. E.g. mlp, lstm, cnn, gcn, all")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--attack_type", default="false_state")
    parser.add_argument("--snapshots", type=int, default=None,
                        help="Total swarm snapshots (overrides config). Use small value for quick testing.")
    args = parser.parse_args()

    model_list = BASELINE_MODELS if args.model == "all" else [args.model]
    main(model_types=model_list, epochs=args.epochs, attack_type=args.attack_type,
         n_snapshots=args.snapshots)
