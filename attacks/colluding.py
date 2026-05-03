"""
Colluding Attack.

A group of attacking drones perfectly synchronize their features 
to create a strong, consistent false signal.
"""
import torch
import numpy as np
from typing import List
from torch_geometric.data import Data

def inject_colluding(
    sequence: List[Data], 
    group_size: int,
    rng: np.random.Generator
) -> List[Data]:
    """
    Apply Colluding attack to a temporal graph sequence.
    
    Randomly selects chunks of attackers and forces them to broadcast
    the exact same features at each timestep.
    
    Parameters
    ----------
    sequence : List[Data]
        A sequence of graph snapshots (T timesteps).
    group_size : int
        Number of attackers that collude and share the same feature vector.
    rng : np.random.Generator
        Random number generator.
        
    Returns
    -------
    List[Data]
        The sequence with colluding node features.
    """
    new_seq = []
    
    num_nodes = sequence[0].x.shape[0]
    attackers = (sequence[0].y == 1).nonzero(as_tuple=True)[0]
    
    if len(attackers) == 0:
        return [g.clone() for g in sequence]
        
    # Group attackers into colluding clusters
    shuffled_attackers = attackers[rng.permutation(len(attackers))]
    clusters = [
        shuffled_attackers[i:i + group_size] 
        for i in range(0, len(shuffled_attackers), group_size)
    ]
    
    for t, data in enumerate(sequence):
        new_data = data.clone()
        
        for cluster in clusters:
            if len(cluster) == 0:
                continue
            
            # Select the leader's features and broadcast to the rest
            leader_idx = cluster[0]
            leader_features = new_data.x[leader_idx]
            
            for follower_idx in cluster[1:]:
                new_data.x[follower_idx] = leader_features
                
        new_seq.append(new_data)
        
    return new_seq
