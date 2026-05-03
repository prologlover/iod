"""
Explainability Module.

Provides three levels of interpretability:
  1. GNNExplainer   — sub-graph and feature masks per node
  2. SHAP           — Shapley-value feature importance (global + local)
  3. Temporal       — per-timestep contribution analysis

All visualisations are saved to the configured figures directory.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import to_networkx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import FIGURE_DIR, DEVICE
from src.utils import get_logger, save_json

logger = get_logger(__name__)


# ================================================================== #
#  Spatial Wrapper (reusable by GNNExplainer & SHAP)                  #
# ================================================================== #

class SpatialWrapper(nn.Module):
    """
    Wraps the spatial GAT/GraphSAGE portion of the model so that PyG
    Explainer can interpret predictions based on GRU temporal embeddings.
    """

    def __init__(self, full_model):
        super().__init__()
        self.model = full_model

    def forward(self, x, edge_index):
        if hasattr(self.model, "gat1"):
            h = self.model.gat1(x, edge_index)
            h = F.elu(h)
            h = self.model.gat2(h, edge_index)
            h = F.elu(h)
        elif hasattr(self.model, "sage1"):
            h = self.model.sage1(x, edge_index)
            h = F.relu(h)
            h = self.model.sage2(h, edge_index)
            h = F.relu(h)
        else:
            raise ValueError("Model architecture not recognised for SpatialWrapper.")
        return self.model.classifier(h)


# ================================================================== #
#  Temporal embedding extractor                                       #
# ================================================================== #

def extract_temporal_embeddings(
    model: nn.Module,
    features: torch.Tensor,
) -> torch.Tensor:
    """
    Extract temporal embeddings from the GRU for a batch.

    Parameters
    ----------
    model : the full spatio-temporal model
    features : (B, T, N, F) tensor

    Returns
    -------
    temporal_emb : (B, N, H)
    """
    model.eval()
    B, T, N, F_dim = features.shape
    x_seq = features.view(B * N, T, F_dim)
    with torch.no_grad():
        gru_out, _ = model.gru(x_seq)
        temporal_emb = gru_out[:, -1, :].view(B, N, -1)
    return temporal_emb


# ================================================================== #
#  1. GNNExplainer                                                    #
# ================================================================== #

def explain_node(
    full_model: nn.Module,
    features: torch.Tensor,
    final_graphs: List[Data],
    target_node: int,
    seq_idx: int = 0,
    save_prefix: str = "",
) -> Dict:
    """
    Run GNNExplainer on a specific node to identify which neighbours
    and features contributed to its classification.

    Returns
    -------
    dict with 'node_mask' and 'edge_mask' arrays (numpy), plus saved figure path.
    """
    temporal_emb = extract_temporal_embeddings(full_model, features)
    device = features.device

    x = temporal_emb[seq_idx].to(device)
    edge_index = final_graphs[seq_idx].edge_index.to(device)
    y_true = final_graphs[seq_idx].y[target_node].item()

    spatial_model = SpatialWrapper(full_model).to(device)
    spatial_model.eval()

    explainer = Explainer(
        model=spatial_model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    explanation = explainer(x, edge_index, index=target_node)

    # --- Visualise ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # (a) Sub-graph importance
    ax = axes[0]
    G = to_networkx(final_graphs[seq_idx], to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    # Node colours from labels
    labels = final_graphs[seq_idx].y.cpu().numpy()
    node_colors = ["#ef5350" if l == 1 else "#66bb6a" for l in labels]
    node_sizes = [500 if i == target_node else 200 for i in range(len(labels))]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           edgecolors="white", linewidths=1.5, ax=ax)

    # Edge importance from mask
    if explanation.edge_mask is not None:
        edge_mask = explanation.edge_mask.detach().cpu().numpy()
        edge_mask_norm = edge_mask / (edge_mask.max() + 1e-8)
        edge_colors = [plt.cm.YlOrRd(w) for w in edge_mask_norm]
        edge_widths = 1 + 3 * edge_mask_norm
    else:
        edge_colors = ["#cccccc"] * G.number_of_edges()
        edge_widths = [0.5] * G.number_of_edges()

    nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color=edge_colors,
                           width=edge_widths, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    label_str = "Attacker" if y_true == 1 else "Benign"
    ax.set_title(f"GNNExplainer — Node {target_node} ({label_str})",
                 fontsize=13, fontweight="bold")
    ax.axis("off")

    # (b) Feature importance bar chart
    ax = axes[1]
    if explanation.node_mask is not None:
        feat_importance = explanation.node_mask[target_node].detach().cpu().numpy()
        top_k = min(20, len(feat_importance))
        top_indices = np.argsort(feat_importance)[-top_k:][::-1]
        top_vals = feat_importance[top_indices]

        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
        ax.barh(range(top_k), top_vals[::-1], color=colors[::-1])
        ax.set_yticks(range(top_k))
        ax.set_yticklabels([f"Feat {i}" for i in top_indices[::-1]], fontsize=8)
        ax.set_xlabel("Importance Score", fontsize=11)
        ax.set_title("Top Feature Importances", fontsize=13, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No feature mask available", ha="center", va="center")

    plt.tight_layout()
    prefix = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{prefix}gnn_explainer_node_{target_node}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"GNNExplainer figure saved to {save_path}")

    result = {"target_node": target_node, "true_label": y_true, "figure": str(save_path)}
    if explanation.node_mask is not None:
        result["feature_importance"] = feat_importance.tolist()
    if explanation.edge_mask is not None:
        result["edge_mask"] = edge_mask.tolist()

    return result


# ================================================================== #
#  2. Temporal Contribution Analysis                                  #
# ================================================================== #

def temporal_importance(
    model: nn.Module,
    features: torch.Tensor,
    final_graphs: List[Data],
    seq_idx: int = 0,
    save_prefix: str = "",
) -> Dict:
    """
    Measure how much each timestep contributes to the final prediction
    by masking out individual timesteps and measuring prediction change.

    Returns per-timestep importance scores for all nodes.
    """
    model.eval()
    device = features.device
    B, T, N, F_dim = features.shape

    # Baseline predictions
    with torch.no_grad():
        baseline_logits = model(features, final_graphs)
        baseline_probs = F.softmax(baseline_logits, dim=-1)  # (B, N, 2)
        baseline_attack_prob = baseline_probs[seq_idx, :, 1].cpu().numpy()  # (N,)

    # Leave-one-out per timestep
    importance = np.zeros((T, N))
    for t in range(T):
        masked_features = features.clone()
        masked_features[:, t, :, :] = 0.0  # zero-out timestep t

        with torch.no_grad():
            masked_logits = model(masked_features, final_graphs)
            masked_probs = F.softmax(masked_logits, dim=-1)
            masked_attack_prob = masked_probs[seq_idx, :, 1].cpu().numpy()

        # Importance = drop in attack probability when timestep is removed
        importance[t] = np.abs(baseline_attack_prob - masked_attack_prob)

    # --- Visualise ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) Heatmap
    ax = axes[0]
    labels = final_graphs[seq_idx].y.cpu().numpy()
    sorted_idx = np.argsort(labels)  # benign first, attackers last
    im = ax.imshow(importance[:, sorted_idx].T, aspect="auto", cmap="magma",
                   interpolation="nearest")
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Node (sorted: benign → attacker)", fontsize=11)
    ax.set_title("Temporal Importance Heatmap", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Δ P(attack)")

    # (b) Aggregated per-timestep importance
    ax = axes[1]
    mean_importance = importance.mean(axis=1)
    attacker_mask = labels == 1
    benign_mask = labels == 0
    att_imp = importance[:, attacker_mask].mean(axis=1) if attacker_mask.any() else np.zeros(T)
    ben_imp = importance[:, benign_mask].mean(axis=1) if benign_mask.any() else np.zeros(T)

    ax.plot(range(T), att_imp, "o-", color="#ef5350", label="Attackers", linewidth=2)
    ax.plot(range(T), ben_imp, "s-", color="#66bb6a", label="Benign", linewidth=2)
    ax.fill_between(range(T), att_imp, alpha=0.15, color="#ef5350")
    ax.fill_between(range(T), ben_imp, alpha=0.15, color="#66bb6a")
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("Mean Importance", fontsize=11)
    ax.set_title("Temporal Importance by Drone Type", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    prefix = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{prefix}temporal_importance.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Temporal importance figure saved to {save_path}")

    return {
        "importance_matrix": importance.tolist(),
        "mean_per_timestep": mean_importance.tolist(),
        "figure": str(save_path),
    }


# ================================================================== #
#  3. SHAP-based Feature Importance                                   #
# ================================================================== #

def shap_feature_importance(
    model: nn.Module,
    features: torch.Tensor,
    final_graphs: List[Data],
    feature_names: List[str] = None,
    n_background: int = 50,
    save_prefix: str = "",
) -> Dict:
    """
    Compute approximate SHAP values for the spatial GNN features.

    Uses a permutation-based approach on the GRU temporal embeddings
    to estimate feature contributions (avoids full SHAP dependency
    for environments where shap is not installed).

    Parameters
    ----------
    model : trained spatio-temporal model
    features : (B, T, N, F) tensor
    final_graphs : list of final-timestep graphs
    feature_names : optional list of feature name strings
    n_background : number of background samples for reference
    save_prefix : prefix for saved figures

    Returns
    -------
    dict with SHAP-like importance values and figure path
    """
    model.eval()
    device = features.device
    B, T, N, F_dim = features.shape

    # Use permutation importance on the input features
    # Baseline predictions
    with torch.no_grad():
        baseline_logits = model(features, final_graphs)
        baseline_probs = F.softmax(baseline_logits, dim=-1)  # (B, N, 2)
        baseline_pred = baseline_probs[:, :, 1].cpu().numpy()  # (B, N)

    # Permutation importance per feature
    importance = np.zeros(F_dim)
    n_repeats = 3

    for f_idx in range(F_dim):
        score_drops = []
        for _ in range(n_repeats):
            perm_features = features.clone()
            # Shuffle this feature across the node dimension
            perm_idx = torch.randperm(N)
            perm_features[:, :, :, f_idx] = perm_features[:, :, perm_idx, f_idx]

            with torch.no_grad():
                perm_logits = model(perm_features, final_graphs)
                perm_probs = F.softmax(perm_logits, dim=-1)
                perm_pred = perm_probs[:, :, 1].cpu().numpy()

            score_drops.append(np.abs(baseline_pred - perm_pred).mean())

        importance[f_idx] = np.mean(score_drops)

    # --- Visualise ---
    fig, ax = plt.subplots(figsize=(10, 8))

    top_k = min(25, F_dim)
    top_indices = np.argsort(importance)[-top_k:][::-1]
    top_vals = importance[top_indices]

    if feature_names is not None:
        top_names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_indices]
    else:
        top_names = [f"Feature {i}" for i in top_indices]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_k))
    ax.barh(range(top_k), top_vals[::-1], color=colors[::-1], edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Permutation Importance (Δ P(attack))", fontsize=11)
    ax.set_title("Feature Importance (Permutation-based)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    prefix = f"{save_prefix}_" if save_prefix else ""
    save_path = FIGURE_DIR / f"{prefix}feature_importance_shap.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance figure saved to {save_path}")

    return {
        "importance": importance.tolist(),
        "top_features": list(zip([int(i) for i in top_indices], top_vals.tolist())),
        "figure": str(save_path),
    }


# ================================================================== #
#  4. Combined Explanation Report                                     #
# ================================================================== #

def generate_full_explanation(
    model: nn.Module,
    features: torch.Tensor,
    final_graphs: List[Data],
    feature_names: List[str] = None,
    target_nodes: List[int] = None,
    seq_idx: int = 0,
    save_prefix: str = "",
) -> Dict:
    """
    Generate a comprehensive explainability report combining all methods.

    Parameters
    ----------
    model : trained model
    features : (B, T, N, F)
    final_graphs : list of final graphs
    feature_names : list of feature name strings
    target_nodes : list of node indices to explain (default: 1 attacker + 1 benign)
    seq_idx : batch index
    save_prefix : prefix for figures

    Returns
    -------
    dict with all explanation results
    """
    labels = final_graphs[seq_idx].y.cpu().numpy()

    if target_nodes is None:
        # Pick 1 attacker and 1 benign node
        attacker_nodes = np.where(labels == 1)[0]
        benign_nodes = np.where(labels == 0)[0]
        target_nodes = []
        if len(attacker_nodes) > 0:
            target_nodes.append(int(attacker_nodes[0]))
        if len(benign_nodes) > 0:
            target_nodes.append(int(benign_nodes[0]))

    report = {"seq_idx": seq_idx, "explanations": {}}

    # 1. GNNExplainer per target node
    for node_id in target_nodes:
        logger.info(f"Explaining node {node_id} ...")
        try:
            result = explain_node(
                model, features, final_graphs,
                target_node=node_id, seq_idx=seq_idx,
                save_prefix=save_prefix,
            )
            report["explanations"][f"node_{node_id}"] = result
        except Exception as e:
            logger.warning(f"GNNExplainer failed for node {node_id}: {e}")

    # 2. Temporal importance
    logger.info("Computing temporal importance...")
    try:
        temporal_result = temporal_importance(
            model, features, final_graphs,
            seq_idx=seq_idx, save_prefix=save_prefix,
        )
        report["temporal"] = temporal_result
    except Exception as e:
        logger.warning(f"Temporal importance failed: {e}")

    # 3. Feature importance
    logger.info("Computing feature importance...")
    try:
        shap_result = shap_feature_importance(
            model, features, final_graphs,
            feature_names=feature_names,
            save_prefix=save_prefix,
        )
        report["feature_importance"] = shap_result
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")

    # Save combined report
    report_path = FIGURE_DIR / f"{save_prefix}_explanation_report.json"
    save_json(report, report_path)
    logger.info(f"Full explanation report saved to {report_path}")

    return report
