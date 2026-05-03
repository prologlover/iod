"""
KNN-based graph topology builder.

Constructs edges between drones using K-nearest neighbours
on their 2-D positions.
"""

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data

from src.config import KNN_K


def build_knn_graph(
    node_features: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    k: int = KNN_K,
) -> Data:
    """
    Build a PyG Data object with KNN-based edge topology.

    Parameters
    ----------
    node_features : (N, F)
    positions     : (N, 2)
    labels        : (N,)
    k             : number of nearest neighbours

    Returns
    -------
    torch_geometric.data.Data
    """
    n = positions.shape[0]
    k_actual = min(k, n - 1)  # can't have more neighbours than N-1

    tree = cKDTree(positions)
    distances, indices = tree.query(positions, k=k_actual + 1)  # +1 because self is included

    src_list = []
    dst_list = []
    edge_weights = []

    for i in range(n):
        for j_idx in range(1, k_actual + 1):  # skip self (index 0)
            neighbour = indices[i, j_idx]
            dist = distances[i, j_idx]
            src_list.append(i)
            dst_list.append(neighbour)
            edge_weights.append(1.0 / (dist + 1e-6))  # inverse distance weight

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(labels, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float32),
    )
    return data
