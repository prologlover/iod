"""
False State (FDI) Attack.

Simulates sensor spoofing by adding Gaussian noise or offsetting
the features of attacking drones.
"""
import torch
from typing import List
from torch_geometric.data import Data

def inject_false_state(sequence: List[Data], noise_scale: float) -> List[Data]:
    """
    Apply False State attack to a temporal graph sequence.
    
    Parameters
    ----------
    sequence : List[Data]
        A sequence of graph snapshots (T timesteps).
    noise_scale : float
        Standard deviation of the Gaussian noise applied to features.
        
    Returns
    -------
    List[Data]
        The sequence with perturbed node features for attackers.
    """
    new_seq = []
    for data in sequence:
        new_data = data.clone()
        attackers = (new_data.y == 1)
        if attackers.any():
            # Add Gaussian noise
            noise = torch.randn_like(new_data.x[attackers]) * noise_scale
            new_data.x[attackers] += noise
        new_seq.append(new_data)
        
    return new_seq
