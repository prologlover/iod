"""
Preprocessing entry-point script.

Usage:
    python scripts/preprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_and_merge
from src.preprocessing import run_preprocessing
from src.utils import get_logger

logger = get_logger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("STAGE: Data Loading & Preprocessing")
    logger.info("=" * 60)

    df = load_and_merge(use_secondary=False)
    result = run_preprocessing(df)

    logger.info(f"Number of features: {len(result['feature_cols'])}")
    logger.info(f"Classes: {list(result['label_encoder'].classes_)}")
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
