"""
Model architectures package.

Spatio-temporal graph models (proposed):
  - gat        : GAT + GRU (proposed main model)
  - graphsage  : GraphSAGE + GRU (comparison)

Baseline models:
  - mlp        : Multi-layer Perceptron (tabular, no graph/temporal)
  - lstm       : LSTM with attention (temporal, no graph)
  - cnn        : 1D-CNN (temporal, no graph)
  - gcn        : GCN (graph, no temporal)
"""
from typing import Dict, Type
import torch.nn as nn

from .gat_temporal import GATTemporalModel
from .graphsage_temporal import GraphSAGETemporalModel
from .mlp import MLP
from .lstm import LSTMModel
from .cnn import CNNModel
from .gcn import GCNModel

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "gat": GATTemporalModel,
    "graphsage": GraphSAGETemporalModel,
    "mlp": MLP,
    "lstm": LSTMModel,
    "cnn": CNNModel,
    "gcn": GCNModel,
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Factory function to instantiate a model by name.

    Parameters
    ----------
    model_name : str
        One of: 'gat', 'graphsage', 'mlp', 'lstm', 'cnn', 'gcn'
    **kwargs
        Hyperparameters forwarded to the model constructor.

    Returns
    -------
    nn.Module
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name](**kwargs)
