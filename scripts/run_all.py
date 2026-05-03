"""
Full pipeline end-to-end — runs all stages sequentially.

Usage
-----
    python scripts/run_all.py [--skip_download]
                              [--epochs 100]
                              [--attack_type false_state]
                              [--graph_type knn]
                              [--ablation_epochs 30]
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)

SCRIPTS = Path(__file__).resolve().parent


def run_stage(name: str, cmd: list, check: bool = True):
    logger.info(f"\n{'='*60}")
    logger.info(f"  STAGE: {name}")
    logger.info(f"{'='*60}")
    result = subprocess.run(
        [sys.executable] + cmd,
        cwd=str(SCRIPTS.parent),
    )
    if check and result.returncode != 0:
        logger.error(f"Stage '{name}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)
    return result


def main(
    skip_download: bool = False,
    epochs: int = 100,
    attack_type: str = "false_state",
    graph_type: str = "knn",
    ablation_epochs: int = 30,
):
    logger.info("=" * 60)
    logger.info("  FULL PIPELINE: Byzantine Attack Detection in Drone Swarms")
    logger.info("=" * 60)

    # Stage 1: Download data
    if not skip_download:
        run_stage("Download Data", ["scripts/download_data.py"])

    # Stage 2: Preprocessing
    run_stage("Preprocessing", ["scripts/preprocess.py"])

    # Stage 3: Build graphs
    run_stage(
        "Build Graphs",
        ["scripts/build_graphs.py",
         "--graph_type", graph_type,
         "--attack_type", attack_type],
    )

    # Stage 6a: Train baselines
    run_stage(
        "Train Baselines",
        ["scripts/train_baselines.py",
         "--model", "all",
         "--epochs", str(epochs),
         "--attack_type", attack_type],
    )

    # Stage 6b: Train proposed GAT+GRU model
    run_stage(
        "Train GAT+GRU",
        ["scripts/train_gnn_temporal.py",
         "--model", "gat",
         "--epochs", str(epochs),
         "--attack_type", attack_type,
         "--graph_type", graph_type],
    )

    # Stage 7: Evaluation
    run_stage(
        "Evaluation",
        ["scripts/evaluate.py",
         "--attack_type", attack_type,
         "--graph_type", graph_type],
    )

    # Stage 8: Explainability
    run_stage(
        "Explainability",
        ["scripts/explain.py",
         "--attack_type", attack_type,
         "--graph_type", graph_type],
        check=False,  # Non-fatal if optional deps missing
    )

    # Stage 9: Ablation study
    run_stage(
        "Ablation Study",
        ["scripts/ablation.py",
         "--study", "all",
         "--epochs", str(ablation_epochs)],
        check=False,
    )

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("  Results in: outputs/figures/  outputs/tables/  outputs/models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full Byzantine detection pipeline.")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip the data download step.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--attack_type", default="false_state")
    parser.add_argument("--graph_type", default="knn",
                        choices=["knn", "distance", "hexagonal"])
    parser.add_argument("--ablation_epochs", type=int, default=30)
    args = parser.parse_args()

    main(
        skip_download=args.skip_download,
        epochs=args.epochs,
        attack_type=args.attack_type,
        graph_type=args.graph_type,
        ablation_epochs=args.ablation_epochs,
    )
