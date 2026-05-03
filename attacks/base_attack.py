"""
Abstract base class for Byzantine attack injection.
"""
from abc import ABC, abstractmethod
from typing import Dict, List

from torch_geometric.data import Data


class ByzantineAttack(ABC):
    """
    Abstract interface that all Byzantine attack classes must implement.

    Subclasses must override :meth:`apply` to perturb graph data and
    :meth:`get_config` to expose their configuration as a dict.
    """

    @abstractmethod
    def apply(
        self,
        graph_data: Data,
        target_nodes: List[int],
        timestep: int,
    ) -> Data:
        """
        Apply the attack to a single graph snapshot.

        Parameters
        ----------
        graph_data : PyG Data
            The original (unperturbed) graph snapshot.
        target_nodes : list[int]
            Node indices that are Byzantine attackers.
        timestep : int
            Current timestep index within the temporal sequence.

        Returns
        -------
        Data
            A new (perturbed) graph snapshot.
        """

    @abstractmethod
    def get_config(self) -> Dict:
        """Return a serialisable dict describing the attack configuration."""

    def __repr__(self) -> str:
        cfg = self.get_config()
        params = ", ".join(f"{k}={v}" for k, v in cfg.items())
        return f"{self.__class__.__name__}({params})"
