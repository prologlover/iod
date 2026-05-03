"""
Preprocessing pipeline for the UAV dataset.

Steps:
  1. Handle missing values (median imputation)
  2. Normalise features (StandardScaler)
  3. Encode labels
  4. Stratified train / val / test split
  5. Save artefacts for downstream use
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.utils import get_logger, Timer

logger = get_logger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numerics with column median."""
    meta_cols = ["attack_type", "is_attack", "domain"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    missing_before = df[feature_cols].isna().sum().sum()
    logger.info(f"Missing values before imputation: {missing_before}")

    for col in feature_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    missing_after = df[feature_cols].isna().sum().sum()
    logger.info(f"Missing values after imputation: {missing_after}")
    return df


def normalise_features(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    Apply StandardScaler to all numeric feature columns.

    Parameters
    ----------
    df : DataFrame
    fit : bool
        If True, fit a new scaler. If False, use the provided scaler.
    scaler : StandardScaler or None

    Returns
    -------
    df, scaler
    """
    meta_cols = ["attack_type", "is_attack", "domain"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].values)
        # Save the scaler
        scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
    else:
        assert scaler is not None, "Must provide a scaler when fit=False"
        df[feature_cols] = scaler.transform(df[feature_cols].values)

    return df, scaler


def encode_labels(df: pd.DataFrame):
    """
    Encode the multi-class attack_type labels.

    Returns
    -------
    df : DataFrame with 'label' column (int-encoded)
    le : LabelEncoder
    """
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["attack_type"])

    le_path = PROCESSED_DATA_DIR / "label_encoder.pkl"
    joblib.dump(le, le_path)
    logger.info(f"Label encoder saved. Classes: {list(le.classes_)}")

    return df, le


def split_data(df: pd.DataFrame):
    """
    Stratified train / val / test split.

    Returns
    -------
    train_df, val_df, test_df
    """
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=SEED,
        stratify=df["label"],
    )

    # Second split: val vs test
    relative_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=SEED,
        stratify=temp_df["label"],
    )

    logger.info(
        f"Split sizes - Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}"
    )
    return train_df, val_df, test_df


def run_preprocessing(df: pd.DataFrame):
    """
    Run the full preprocessing pipeline.

    Returns
    -------
    dict with keys: train_df, val_df, test_df, scaler, label_encoder, feature_cols
    """
    with Timer("Full preprocessing"):
        # 1. Missing values
        df = handle_missing_values(df)

        # 2. Encode labels (before normalisation so we have the label column)
        df, le = encode_labels(df)

        # 3. Normalise
        df, scaler = normalise_features(df)

        # 4. Split
        train_df, val_df, test_df = split_data(df)

        # 5. Save to disk
        train_df.to_parquet(PROCESSED_DATA_DIR / "train.parquet", index=False)
        val_df.to_parquet(PROCESSED_DATA_DIR / "val.parquet", index=False)
        test_df.to_parquet(PROCESSED_DATA_DIR / "test.parquet", index=False)
        logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")

        meta_cols = ["attack_type", "is_attack", "domain", "label"]
        feature_cols = [c for c in df.columns if c not in meta_cols]

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "scaler": scaler,
        "label_encoder": le,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    from src.data_loader import load_and_merge

    df = load_and_merge()
    result = run_preprocessing(df)
    print(f"Feature columns ({len(result['feature_cols'])}): {result['feature_cols'][:10]}...")
