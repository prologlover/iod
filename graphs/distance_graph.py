"""
Distance-based graph topology builder.

Connects drones within a communication range threshold,
with inverse-distance edge weights.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch_geometric.data import Data

from src.config import COMMUNICATION_RANGE


def build_distance_graph(
    node_features: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray,
    comm_range: float = COMMUNICATION_RANGE,
) -> Data:
    """
    Build a PyG Data object with distance-based edge topology.

    An edge exists between drones i and j iff dist(i, j) < comm_range.

    Parameters
    ----------
    node_features : (N, F)
    positions     : (N, 2)
    labels        : (N,)
    comm_range    : communication range in metres

    Returns
    -------
    torch_geometric.data.Data
    """
    dist_matrix = cdist(positions, positions, metric="euclidean")

    src_list = []
    dst_list = []
    edge_weights = []

    n = positions.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix[i, j] < comm_range:
                src_list.append(i)
                dst_list.append(j)
                edge_weights.append(1.0 / (dist_matrix[i, j] + 1e-6))

    # Fall-back: if graph is disconnected, add at least 1-NN per node
    if len(src_list) == 0:
        for i in range(n):
            dists_i = dist_matrix[i].copy()
            dists_i[i] = np.inf
            j = int(np.argmin(dists_i))
            src_list.extend([i, j])
            dst_list.extend([j, i])
            w = 1.0 / (dist_matrix[i, j] + 1e-6)
            edge_weights.extend([w, w])

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
