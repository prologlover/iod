"""
Ablation Study Module.

Systematically evaluates the contribution of each model component:
  - Temporal (GRU) depth
  - Spatial (GNN) architecture
  - Graph topology type
  - Attack type robustness
"""

import sys
import os
import copy
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    HIDDEN_DIM, NUM_HEADS, DROPOUT, GRU_LAYERS, DEVICE,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    FOCAL_ALPHA, FOCAL_GAMMA, TABLE_DIR,
)
from src.utils import get_logger, save_json
from models import get_model
from evaluation.metrics import calculate_metrics

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
#  Ablation configurations                                            #
# ------------------------------------------------------------------ #

ABLATION_CONFIGS = {
    # -- Temporal ablations --
    "full_model": {
        "model": "gat",
        "kwargs": {"hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS,
                   "dropout": DROPOUT, "gru_layers": GRU_LAYERS},
    },
    "no_temporal": {
        "model": "gat",
        "kwargs": {"hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS,
                   "dropout": DROPOUT, "gru_layers": 1},
        "description": "Single GRU layer – minimal temporal context.",
    },
    "shallow_gru": {
        "model": "gat",
        "kwargs": {"hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS,
                   "dropout": DROPOUT, "gru_layers": 1},
        "description": "GRU depth = 1 vs default 2.",
    },
    # -- Spatial ablations --
    "graphsage_baseline": {
        "model": "graphsage",
        "kwargs": {"hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
                   "gru_layers": GRU_LAYERS},
        "description": "Replace GAT with GraphSAGE.",
    },
    # -- Capacity ablations --
    "small_hidden": {
        "model": "gat",
        "kwargs": {"hidden_dim": 64, "num_heads": NUM_HEADS,
                   "dropout": DROPOUT, "gru_layers": GRU_LAYERS},
        "description": "Hidden dim 64 instead of 128.",
    },
    "large_hidden": {
        "model": "gat",
        "kwargs": {"hidden_dim": 256, "num_heads": NUM_HEADS,
                   "dropout": DROPOUT, "gru_layers": GRU_LAYERS},
        "description": "Hidden dim 256 instead of 128.",
    },
    "no_dropout": {
        "model": "gat",
        "kwargs": {"hidden_dim": HIDDEN_DIM, "num_heads": NUM_HEADS,
                   "dropout": 0.0, "gru_layers": GRU_LAYERS},
        "description": "Dropout disabled.",
    },
}


def configure_ablation(
    model_kwargs: Dict[str, Any], ablation_type: str
) -> Dict[str, Any]:
    """
    Modify model keyword arguments to test ablation configurations.

    Parameters
    ----------
    model_kwargs : dict
        Original kwargs (hidden_dim, gru_layers, etc)
    ablation_type : str
        Key in ABLATION_CONFIGS

    Returns
    -------
    dict
        Modified kwargs.
    """
    if ablation_type not in ABLATION_CONFIGS:
        logger.warning(f"Unknown ablation '{ablation_type}', returning original.")
        return model_kwargs

    cfg = ABLATION_CONFIGS[ablation_type]
    new_kwargs = cfg["kwargs"].copy()
    logger.info(f"Ablation '{ablation_type}': {cfg.get('description', '')}")
    return new_kwargs


def build_ablation_model(
    ablation_type: str, in_channels: int
) -> nn.Module:
    """Instantiate a model for the given ablation configuration."""
    cfg = ABLATION_CONFIGS[ablation_type]
    model_name = cfg["model"]
    kwargs = cfg["kwargs"].copy()
    kwargs["in_channels"] = in_channels
    return get_model(model_name, **kwargs)


def run_ablation_study(
    train_fn,
    eval_fn,
    in_channels: int,
    ablation_types: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study across all (or specified) configurations.

    Parameters
    ----------
    train_fn : callable(model) -> trained_model
        A function that trains the given model and returns it.
    eval_fn : callable(model) -> dict
        A function that evaluates the model and returns metric dict.
    in_channels : int
        Number of input features.
    ablation_types : list[str], optional
        Subset of ABLATION_CONFIGS keys to run. None = all.

    Returns
    -------
    dict mapping ablation_type -> metric dict
    """
    if ablation_types is None:
        ablation_types = list(ABLATION_CONFIGS.keys())

    results = {}
    for ablation in ablation_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Ablation: {ablation}")
        logger.info(f"{'='*60}")

        model = build_ablation_model(ablation, in_channels)
        model = model.to(DEVICE)

        trained_model = train_fn(model)
        metrics = eval_fn(trained_model)
        results[ablation] = metrics

        logger.info(f"  Results: {metrics}")

    # Save results
    save_path = TABLE_DIR / "ablation_results.json"
    save_json(results, save_path)
    logger.info(f"Ablation results saved to {save_path}")

    return results


def ablation_results_to_latex(results: Dict[str, Dict[str, float]]) -> str:
    """Format ablation results as a LaTeX table."""
    header = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{Ablation Study Results}" + "\n"
        r"\label{tab:ablation}" + "\n"
        r"\begin{tabular}{lcccc}" + "\n"
        r"\toprule" + "\n"
        r"Configuration & Accuracy & Precision & Recall & F1 \\" + "\n"
        r"\midrule" + "\n"
    )

    rows = []
    for name, metrics in results.items():
        display_name = name.replace("_", " ").title()
        row = (
            f"{display_name} & "
            f"{metrics.get('accuracy', 0):.4f} & "
            f"{metrics.get('precision', 0):.4f} & "
            f"{metrics.get('recall', 0):.4f} & "
            f"{metrics.get('f1', 0):.4f} "
            r"\\"
        )
        rows.append(row)

    footer = (
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )

    return header + "\n".join(rows) + "\n" + footer
