"""
Download / clone the required datasets.

Usage:
    python scripts/download_data.py [--secondary]
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import PRIMARY_DATASET_REPO, SECONDARY_DATASET_URL, RAW_DATA_DIR
from src.utils import get_logger

logger = get_logger(__name__)


def clone_primary():
    """Clone the primary UAV dataset repository."""
    dest = RAW_DATA_DIR / "UAVs-Dataset-Under-Normal-and-Cyberattacks"
    if dest.exists():
        logger.info(f"Primary dataset already exists at {dest}")
        return dest

    logger.info("Cloning primary dataset repository...")
    subprocess.run(
        ["git", "clone", PRIMARY_DATASET_REPO, str(dest)],
        check=True,
    )
    logger.info(f"Primary dataset cloned to {dest}")
    return dest


def download_secondary():
    """Download and extract the UAV-NIDD dataset from Zenodo."""
    dest_dir = RAW_DATA_DIR / "UAV-NIDD"
    zip_path = RAW_DATA_DIR / "UAV-NDD_CSV.zip"

    if dest_dir.exists():
        logger.info(f"Secondary dataset already exists at {dest_dir}")
        return dest_dir

    logger.info("Downloading secondary dataset (UAV-NIDD) from Zenodo...")
    try:
        import urllib.request
        urllib.request.urlretrieve(SECONDARY_DATASET_URL, str(zip_path))
        logger.info(f"Downloaded to {zip_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.info("You can manually download from: https://zenodo.org/records/15125851")
        return None

    # Extract
    logger.info("Extracting ZIP...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(dest_dir))
    zip_path.unlink()
    logger.info(f"Extracted to {dest_dir}")
    return dest_dir


def main():
    parser = argparse.ArgumentParser(description="Download UAV datasets")
    parser.add_argument(
        "--secondary",
        action="store_true",
        help="Also download the optional UAV-NIDD dataset (~1.6 GB)",
    )
    args = parser.parse_args()

    clone_primary()

    if args.secondary:
        download_secondary()

    logger.info("Data download complete.")


if __name__ == "__main__":
    main()
