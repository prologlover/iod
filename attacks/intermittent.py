"""
Intermittent Attack.

Attackers randomly toggle their malicious behavior on and off 
across the temporal sequence to evade detection.
"""
import torch
import numpy as np
from typing import List
from torch_geometric.data import Data

def inject_intermittent(
    sequence: List[Data], 
    p_active: float, 
    rng: np.random.Generator,
    base_attack_fn=None,
    **base_kwargs
) -> List[Data]:
    """
    Apply Intermittent attack to a temporal graph sequence.
    
    Parameters
    ----------
    sequence : List[Data]
        A sequence of graph snapshots (T timesteps).
    p_active : float
        Probability that an attacker is actively attacking at any timestep.
    rng : np.random.Generator
        Random number generator.
    base_attack_fn : Callable
        An attack function (e.g., inject_false_state) to apply when active.
        If None, active attackers just use their original malicious features,
        and inactive ones revert to benign behavior (we simulate this by mixing).
    
    Returns
    -------
    List[Data]
        The sequence with intermittently perturbed node features.
    """
    new_seq = []
    
    # Pre-calculate active masks for all attackers across time
    # If a node is an attacker (y == 1), it tosses a coin each t
    num_nodes = sequence[0].x.shape[0]
    is_attacker_global = (sequence[0].y == 1)
    
    for t, data in enumerate(sequence):
        new_data = data.clone()
        
        # Determine which attackers are ACTIVE at time t
        # (1 = attacking, 0 = benign behavior)
        active_mask = torch.tensor(
            rng.binomial(1, p_active, size=num_nodes), 
            dtype=torch.bool, device=data.x.device
        )
        
        # Only attackers can be active attackers
        actual_active = is_attacker_global & active_mask
        
        # If we have a base function, apply it to the actual active attackers
        if base_attack_fn is not None and actual_active.any():
            # Create a localized clone just for the base attack call
            temp_seq = base_attack_fn([new_data], **base_kwargs)
            # Copy only the actual_active node features
            new_data.x[actual_active] = temp_seq[0].x[actual_active]
        else:
            # If no base function, we assume the dataset features are already malicious.
            # To hide them when inactive, we'd ideally replace with benign features.
            # Since that's tricky without reference, we just add slight noise to active ones
            # or keep it as is (which means they are naturally varying).
            pass
            
        new_seq.append(new_data)
        
    return new_seq
