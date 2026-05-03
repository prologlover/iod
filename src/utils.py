"""
Shared utility functions used across the project.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class Timer:
    """Simple context-manager timer."""

    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        print(f"  [timer]  {self.description}: {self.elapsed:.2f}s")


def save_json(data: Dict[str, Any], path: Path):
    """Save a dictionary as a formatted JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serialiser)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def _json_serialiser(obj):
    """Handle numpy/torch types for JSON serialisation."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(data, device: torch.device):
    """Move a PyG Data object or tensor to the target device."""
    if hasattr(data, "to"):
        return data.to(device)
    return data
