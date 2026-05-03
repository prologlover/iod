"""
Temporal graph builder.

Creates sequences of PyG Data objects (graph snapshots) from simulated
swarm data, suitable for temporal GNN models (GAT + GRU).
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.knn_graph import build_knn_graph
from graphs.distance_graph import build_distance_graph
from graphs.hexagonal_graph import build_hexagonal_graph
from src.config import KNN_K, COMMUNICATION_RANGE
from src.utils import get_logger

logger = get_logger(__name__)

GRAPH_BUILDERS = {
    "knn": lambda nf, pos, lab: build_knn_graph(nf, pos, lab, k=KNN_K),
    "distance": lambda nf, pos, lab: build_distance_graph(nf, pos, lab, comm_range=COMMUNICATION_RANGE),
    "hexagonal": lambda nf, pos, lab: build_hexagonal_graph(nf, lab),
}


def snapshots_to_pyg(
    snapshots: list,
    graph_type: str = "knn",
) -> List[Data]:
    """
    Convert a list of snapshot dicts (from swarm_simulator) into PyG Data objects.

    Parameters
    ----------
    snapshots : list of dicts with keys node_features, positions, labels
    graph_type : one of "knn", "distance", "hexagonal"

    Returns
    -------
    list of PyG Data objects
    """
    builder = GRAPH_BUILDERS[graph_type]
    graphs = []
    for snap in snapshots:
        g = builder(snap["node_features"], snap["positions"], snap["labels"])
        graphs.append(g)
    return graphs


def build_temporal_graphs(
    sequences: list,
    graph_type: str = "knn",
) -> List[List[Data]]:
    """
    Convert all temporal sequences into lists of PyG Data graph sequences.

    Parameters
    ----------
    sequences : list of temporal windows (each a list of snapshot dicts)
    graph_type : one of "knn", "distance", "hexagonal"

    Returns
    -------
    list of lists of PyG Data objects — outer list = sequences, inner = timesteps
    """
    all_graph_sequences = []
    for idx, seq in enumerate(sequences):
        graph_seq = snapshots_to_pyg(seq, graph_type=graph_type)
        all_graph_sequences.append(graph_seq)

        if (idx + 1) % 100 == 0:
            logger.info(f"  Converted {idx + 1}/{len(sequences)} sequences to graphs")

    logger.info(f"Temporal graph dataset built: {len(all_graph_sequences)} sequences, type={graph_type}")
    return all_graph_sequences


def pack_temporal_batch(
    graph_sequences: List[List[Data]],
) -> Tuple[torch.Tensor, torch.Tensor, List[Data]]:
    """
    Pack temporal graph sequences into tensors suitable for GRU input.

    For each sequence, stack node features across time → (T, N, F),
    and take the labels from the last timestep.

    Returns
    -------
    features : (B, T, N, F) tensor
    labels   : (B, N) tensor  — labels from final timestep
    graphs   : list of final-timestep Data objects (for edge info)
    """
    B = len(graph_sequences)
    T = len(graph_sequences[0])
    N = graph_sequences[0][0].x.shape[0]
    F = graph_sequences[0][0].x.shape[1]

    features = torch.zeros(B, T, N, F)
    labels = torch.zeros(B, N, dtype=torch.long)
    final_graphs = []

    for b in range(B):
        for t in range(T):
            features[b, t] = graph_sequences[b][t].x
        labels[b] = graph_sequences[b][-1].y
        final_graphs.append(graph_sequences[b][-1])

    return features, labels, final_graphs
