"""
Hexagonal lattice graph topology builder.

Arranges drones on a hexagonal grid and connects each to up to 6 neighbours.
Common in real swarm formations.
"""

import math

import numpy as np
import torch
from torch_geometric.data import Data


def _hex_grid_positions(n: int, spacing: float = 15.0) -> np.ndarray:
    """
    Generate a hexagonal-grid layout for *n* nodes.

    Returns positions (n, 2).
    """
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    positions = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= n:
                break
            x = c * spacing
            if r % 2 == 1:
                x += spacing / 2  # offset odd rows
            y = r * spacing * math.sqrt(3) / 2
            positions.append([x, y])

    return np.array(positions[:n], dtype=np.float32)


def _hex_neighbours(n: int, cols: int) -> list:
    """
    Compute hexagonal neighbour lists for an n-node grid
    with the given number of columns.

    Returns list of (src, dst) tuples.
    """
    rows = int(math.ceil(n / cols))
    edges = set()

    for idx in range(n):
        r = idx // cols
        c = idx % cols

        # Same-row neighbours
        if c + 1 < cols and idx + 1 < n:
            edges.add((idx, idx + 1))
            edges.add((idx + 1, idx))

        # Upper-row neighbours
        if r + 1 < rows:
            # Directly below
            below = (r + 1) * cols + c
            if below < n:
                edges.add((idx, below))
                edges.add((below, idx))

            # Diagonal below
            if r % 2 == 0:
                diag = (r + 1) * cols + c - 1
            else:
                diag = (r + 1) * cols + c + 1

            if 0 <= diag % cols < cols and 0 <= diag < n:
                edges.add((idx, diag))
                edges.add((diag, idx))

    return list(edges)


def build_hexagonal_graph(
    node_features: np.ndarray,
    labels: np.ndarray,
    spacing: float = 15.0,
) -> Data:
    """
    Build a PyG Data object with hexagonal lattice topology.

    Parameters
    ----------
    node_features : (N, F)
    labels        : (N,)
    spacing       : distance between adjacent nodes in metres

    Returns
    -------
    torch_geometric.data.Data
    """
    n = node_features.shape[0]
    cols = int(math.ceil(math.sqrt(n)))

    positions = _hex_grid_positions(n, spacing)
    edges = _hex_neighbours(n, cols)

    if edges:
        src_list, dst_list = zip(*edges)
    else:
        src_list, dst_list = [], []

    edge_index = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
        pos=torch.tensor(positions, dtype=torch.float32),
    )
    return data
