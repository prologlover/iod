"""
Attack Manager — orchestrate multi-attack scenarios across swarm sequences.

Supports single-attack, mixed-attack, and sweep configurations.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    ATTACKER_RATIO,
    COLLUDING_GROUP_SIZE,
    DELAY_STEPS,
    FALSE_STATE_NOISE_SCALE,
    INTERMITTENT_P_ACTIVE,
)
from src.utils import get_logger

from .false_state import inject_false_state
from .intermittent import inject_intermittent
from .colluding import inject_colluding
from .delay import inject_delay

logger = get_logger(__name__)

SUPPORTED_ATTACKS = ("none", "false_state", "fdi", "intermittent", "colluding", "delay", "replay")


class AttackManager:
    """
    High-level orchestrator for Byzantine attack injection.

    Parameters
    ----------
    attack_types : list[str]
        One or more attack types to apply per sequence.
        When multiple types are given each sequence is assigned one
        type at random (mixed scenario).
    attacker_ratio : float
        Fraction of drones that are Byzantine.
    noise_scale : float
        Noise magnitude for false-state / intermittent attacks.
    p_active : float
        Activation probability for intermittent attacks.
    collusion_group_size : int
        Number of colluding nodes per group.
    delay_steps : int
        Staleness delay for delay attacks.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        attack_types: List[str] = None,
        attacker_ratio: float = ATTACKER_RATIO,
        noise_scale: float = FALSE_STATE_NOISE_SCALE,
        p_active: float = INTERMITTENT_P_ACTIVE,
        collusion_group_size: int = COLLUDING_GROUP_SIZE,
        delay_steps: int = DELAY_STEPS,
        seed: int = 42,
    ):
        if attack_types is None:
            attack_types = ["false_state"]
        self.attack_types = [a.lower() for a in attack_types]
        self.attacker_ratio = attacker_ratio
        self.noise_scale = noise_scale
        self.p_active = p_active
        self.collusion_group_size = collusion_group_size
        self.delay_steps = delay_steps
        self.rng = np.random.default_rng(seed)

        for a in self.attack_types:
            if a not in SUPPORTED_ATTACKS:
                raise ValueError(f"Unknown attack type '{a}'. Supported: {SUPPORTED_ATTACKS}")

    # ------------------------------------------------------------------ #

    def inject(
        self,
        graph_sequences: List[List[Data]],
        attack_type: Optional[str] = None,
    ) -> List[List[Data]]:
        """
        Apply attack(s) to all temporal graph sequences.

        Parameters
        ----------
        graph_sequences : list of temporal windows
        attack_type : override the instance-level attack type (optional)

        Returns
        -------
        list of attacked temporal windows
        """
        attacked = []
        for seq in graph_sequences:
            a_type = attack_type if attack_type is not None else self._sample_attack()
            attacked.append(self._apply_single(seq, a_type))
        return attacked

    def sweep(
        self,
        graph_sequences: List[List[Data]],
    ) -> Dict[str, List[List[Data]]]:
        """
        Run every configured attack type separately on the same sequences.

        Returns
        -------
        dict mapping attack_type -> attacked sequences
        """
        results = {}
        for a_type in self.attack_types:
            logger.info(f"  Sweeping attack: {a_type}")
            results[a_type] = self.inject(graph_sequences, attack_type=a_type)
        return results

    def get_config(self) -> Dict:
        return {
            "attack_types": self.attack_types,
            "attacker_ratio": self.attacker_ratio,
            "noise_scale": self.noise_scale,
            "p_active": self.p_active,
            "collusion_group_size": self.collusion_group_size,
            "delay_steps": self.delay_steps,
        }

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _sample_attack(self) -> str:
        return self.attack_types[self.rng.integers(len(self.attack_types))]

    def _apply_single(self, seq: List[Data], attack_type: str) -> List[Data]:
        if attack_type in ("none", "benign"):
            return [g.clone() for g in seq]
        if attack_type in ("false_state", "fdi"):
            return inject_false_state(seq, self.noise_scale)
        if attack_type == "intermittent":
            return inject_intermittent(
                seq, self.p_active, self.rng,
                base_attack_fn=inject_false_state,
                noise_scale=self.noise_scale,
            )
        if attack_type == "colluding":
            return inject_colluding(seq, self.collusion_group_size, self.rng)
        if attack_type in ("delay", "replay"):
            return inject_delay(seq, self.delay_steps)
        logger.warning(f"Unknown attack '{attack_type}'; returning original sequence.")
        return [g.clone() for g in seq]
