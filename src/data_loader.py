"""
Data loader for UAV cyber-physical datasets.

Loads the primary dataset (UAVs-Dataset-Under-Normal-and-Cyberattacks)
and optionally the secondary UAV-NIDD dataset.  Merges cyber and physical
features and assigns attack-type labels based on documented row ranges.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    DATASET_RANGES,
    PRIMARY_CSV_NAME,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from src.utils import get_logger, Timer

logger = get_logger(__name__)


# ------------------------------------------------------------------
# Primary dataset
# ------------------------------------------------------------------

def load_primary_dataset(csv_path: Path = None) -> pd.DataFrame:
    """
    Load the primary UAV cyber-physical CSV.

    The single CSV contains interleaved cyber and physical rows for
    multiple attack categories.  We label every row with its attack
    type and a binary is_attack flag.

    Returns
    -------
    pd.DataFrame
        Columns = original features + ['attack_type', 'is_attack', 'domain']
    """
    if csv_path is None:
        csv_path = RAW_DATA_DIR / "UAVs-Dataset-Under-Normal-and-Cyberattacks" / PRIMARY_CSV_NAME

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Primary CSV not found at {csv_path}.  "
            "Run `python scripts/download_data.py` first."
        )

    with Timer("Loading primary CSV"):
        df = pd.read_csv(csv_path, low_memory=False)

    logger.info(f"Raw shape: {df.shape}")

    # --- Assign labels based on documented row ranges ---
    df["attack_type"] = "Unknown"
    df["is_attack"] = 0
    df["domain"] = "unknown"

    for label, ranges in DATASET_RANGES.items():
        cs, ce = ranges["cyber"]
        ps, pe = ranges["physical"]

        # Cyber rows
        mask_cyber = df.index.to_series().between(cs, ce)
        df.loc[mask_cyber, "attack_type"] = label
        df.loc[mask_cyber, "domain"] = "cyber"

        # Physical rows
        mask_phys = df.index.to_series().between(ps, pe)
        df.loc[mask_phys, "attack_type"] = label
        df.loc[mask_phys, "domain"] = "physical"

    # Binary attack flag
    df.loc[df["attack_type"] != "Benign", "is_attack"] = 1

    label_counts = df["attack_type"].value_counts()
    logger.info(f"Label distribution:\n{label_counts}")

    return df


def merge_cyber_physical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cyber and physical feature sets into a unified representation.

    The dataset stores cyber features in the first 37 columns and physical
    features in the remaining 16 columns.  Many rows have only one domain
    populated (NaN for the other).  We perform a forward-fill strategy
    within each attack segment to create complete feature vectors.

    Returns
    -------
    pd.DataFrame  with no fully-NaN feature columns.
    """
    meta_cols = ["attack_type", "is_attack", "domain"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    logger.info(f"Feature columns: {len(feature_cols)}")

    # Convert all feature columns to numeric (force errors -> NaN)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward-fill within each attack-type group to bridge cyber/physical gaps
    df[feature_cols] = df.groupby("attack_type")[feature_cols].transform(
        lambda g: g.ffill().bfill()
    )

    # Drop columns that are still entirely NaN
    null_cols = [c for c in feature_cols if df[c].isna().all()]
    if null_cols:
        logger.info(f"Dropping {len(null_cols)} fully-null columns: {null_cols[:5]}...")
        df.drop(columns=null_cols, inplace=True)

    return df


# ------------------------------------------------------------------
# Secondary dataset (optional)
# ------------------------------------------------------------------

def load_secondary_dataset(zip_dir: Path = None) -> pd.DataFrame:
    """
    Load the UAV-NIDD dataset CSVs from the extracted ZIP.

    Returns a single concatenated DataFrame or None if not available.
    """
    if zip_dir is None:
        zip_dir = RAW_DATA_DIR / "UAV-NIDD"

    if not zip_dir.exists():
        logger.warning("Secondary dataset not found. Skipping.")
        return None

    csv_files = list(zip_dir.rglob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in UAV-NIDD directory.")
        return None

    dfs = []
    for f in csv_files:
        try:
            part = pd.read_csv(f, low_memory=False)
            part["source_file"] = f.stem
            dfs.append(part)
            logger.info(f"  Loaded {f.name}: {part.shape}")
        except Exception as e:
            logger.warning(f"  Could not load {f.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    logger.info(f"Secondary dataset combined shape: {combined.shape}")
    return combined


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def load_and_merge(use_secondary: bool = False) -> pd.DataFrame:
    """Full load -> merge pipeline for the primary dataset."""
    df = load_primary_dataset()
    df = merge_cyber_physical(df)

    if use_secondary:
        sec = load_secondary_dataset()
        if sec is not None:
            # We only use secondary data for additional attack diversity
            logger.info("Secondary dataset loaded; kept separately for robustness tests.")

    return df


if __name__ == "__main__":
    df = load_and_merge()
    print(f"\nFinal dataset shape: {df.shape}")
    print(df["attack_type"].value_counts())
    print(df.head())
