#!/usr/bin/env python3
"""A leaner, more dynamic smoke test for the end-to-end pipeline."""

import math
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import mlflow


def run_command(cmd: Sequence[str]) -> None:
    """Execute a command and echo it."""
    printable = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"> {printable}")
    subprocess.check_call(list(cmd))


def verify_paths_exist(*paths: Path) -> None:
    """Assert that all provided paths exist."""
    for path in paths:
        assert path.exists(), f"Artifact not found at: {path}"
    print(f"✓ Verified existence of {len(paths)} artifact(s).")


def clean_path(path: Path) -> None:
    """Remove a directory tree if it exists."""
    if path.exists():
        shutil.rmtree(path)


def verify_mlflow_run(experiment_name: str, min_auc: float) -> tuple[str, str]:
    """Verify the latest MLflow run for the experiment, checking the best model's AUC score."""
    print(f"\n--- Verifying MLflow Run for experiment '{experiment_name}' ---")
    
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
        max_results=10,  # Fetch a few recent runs to find the parent
    )

    assert not runs.empty, f"No runs found in MLflow experiment '{experiment_name}'."

    parent_run = None
    for _, run in runs.iterrows():
        parent_id = run.get("tags.mlflow.parentRunId")
        run_name = run.get("tags.mlflow.runName")
        if (parent_id is None or (isinstance(parent_id, float) and math.isnan(parent_id))) and run_name == "Churn_Training_Pipeline":
            parent_run = run
            break

    assert parent_run is not None, "Could not find a parent run in the recent MLflow runs."

    latest_run_id = parent_run.run_id
    print(f"Verifying latest MLflow run: {latest_run_id}")

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(latest_run_id)
    run_data = run.data
    assert run_data is not None, "MLflow run data could not be fetched."

    auc_metric = run_data.metrics.get("best_model_roc_auc")
    assert auc_metric is not None, "'best_model_roc_auc' metric not found in MLflow run."
    assert auc_metric > min_auc, f"Best model AUC {auc_metric:.3f} is below the threshold of {min_auc}."

    best_model = run_data.tags.get("best_model")
    best_model_run_id = run_data.tags.get("best_model_run_id")
    assert best_model, "Best model tag not set on parent run."
    assert best_model_run_id, "Best model run id tag not set on parent run."
    
    print(f"  ✓ MLflow run verified.")
    print(f"  ✓ Best Model ROC AUC: {auc_metric:.3f} > {min_auc}")
    print(f"  ✓ Best Model: {best_model} (run: {best_model_run_id})")

    return latest_run_id, best_model_run_id


def main() -> None:
    """Main orchestrator for the smoke test."""
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    # Define paths
    sample_csv = project_root / "data" / "sample.csv"
    processed_dir = project_root / "data" / "processed_smoke"
    eval_dir = project_root / "evaluation" / "smoke"

    clean_path(processed_dir)
    clean_path(eval_dir)

    # --- 1. Data Preparation ---
    print("\n--- Running Data Preparation ---")
    run_command(["python", "src/data_prep.py", "--input", str(sample_csv), "--output", str(processed_dir)])
    verify_paths_exist(processed_dir / "X_train.csv")

    # --- 2. Model Training ---
    print("\n--- Running Model Training ---")
    run_command(["python", "src/train.py", "--data", str(processed_dir), "--models", "logreg", "rf"])

    # --- 3. MLflow Verification ---
    parent_run_id, best_model_run_id = verify_mlflow_run(experiment_name="churn-prediction", min_auc=0.5)

    # --- 4. Model Evaluation ---
    print("\n--- Running Model Evaluation ---")
    run_command(
        [
            "python",
            "src/evaluate.py",
            "--run-id",
            parent_run_id,
            "--data",
            str(processed_dir),
            "--output",
            str(eval_dir),
        ]
    )
    verify_paths_exist(
        eval_dir / "evaluation_report.json",
        eval_dir / "roc_curve.png",
        eval_dir / "precision_recall_curve.png",
    )

    print("\nSmoke test completed successfully!")


if __name__ == "__main__":
    main()
