"""
Delay Attack.

Simulates a delayed/stale transmission where attackers broadcast
their features from earlier timesteps (T - DELAY_STEPS) instead
of their true current state.
"""
import torch
from typing import List
from torch_geometric.data import Data

def inject_delay(
    sequence: List[Data], 
    delay_steps: int
) -> List[Data]:
    """
    Apply Delay attack to a temporal graph sequence.
    
    Parameters
    ----------
    sequence : List[Data]
        A sequence of graph snapshots (T timesteps).
    delay_steps : int
        Number of timesteps to lag the feature reporting.
        
    Returns
    -------
    List[Data]
        The sequence with delayed node features.
    """
    new_seq = []
    
    attackers = (sequence[0].y == 1)
    
    for t, data in enumerate(sequence):
        new_data = data.clone()
        
        if attackers.any():
            # Find the historical feature index
            hist_t = max(0, t - delay_steps)
            
            # The historical features
            hist_features = sequence[hist_t].x[attackers]
            
            # Attacker nodes send old features
            new_data.x[attackers] = hist_features
            
        new_seq.append(new_data)
        
    return new_seq
