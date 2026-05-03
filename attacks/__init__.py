"""
Byzantine attack injection package.
"""
import logging
import numpy as np
from typing import List
from torch_geometric.data import Data

from src.config import (
    FALSE_STATE_NOISE_SCALE, 
    INTERMITTENT_P_ACTIVE, 
    COLLUDING_GROUP_SIZE, 
    DELAY_STEPS
)

from .false_state import inject_false_state
from .intermittent import inject_intermittent
from .colluding import inject_colluding
from .delay import inject_delay

logger = logging.getLogger(__name__)

def inject_attacks(
    graph_sequences: List[List[Data]], 
    attack_type: str, 
    rng: np.random.Generator
) -> List[List[Data]]:
    """
    Apply a specific Byzantine attack to all temporal graph sequences.
    
    Parameters
    ----------
    graph_sequences : List[List[Data]]
        List of temporal windows, each a list of PyG Data snapshots.
    attack_type : str
        Type of attack ('none', 'false_state', 'intermittent', 'colluding', 'delay')
    rng : np.random.Generator
        Random generator for deterministic reproducibility.
        
    Returns
    -------
    List[List[Data]]
        The attacked dataset.
    """
    attack_type = attack_type.lower()
    
    if attack_type == "none" or attack_type == "benign":
        return [ [g.clone() for g in seq] for seq in graph_sequences ]
        
    logger.info(f"Injecting '{attack_type}' attacks...")
    
    attacked_sequences = []
    for seq in graph_sequences:
        if attack_type == "false_state" or attack_type == "fdi":
            new_seq = inject_false_state(seq, FALSE_STATE_NOISE_SCALE)
            
        elif attack_type == "intermittent":
            # For intermittent, we randomly toggle false_state
            new_seq = inject_intermittent(
                seq, INTERMITTENT_P_ACTIVE, rng,
                base_attack_fn=inject_false_state, 
                noise_scale=FALSE_STATE_NOISE_SCALE
            )
            
        elif attack_type == "colluding":
            new_seq = inject_colluding(seq, COLLUDING_GROUP_SIZE, rng)
            
        elif attack_type == "delay" or attack_type == "replay":
            new_seq = inject_delay(seq, DELAY_STEPS)
            
        else:
            logger.warning(f"Unknown attack type {attack_type}, returning original.")
            new_seq = [g.clone() for g in seq]
            
        attacked_sequences.append(new_seq)
        
    return attacked_sequences
