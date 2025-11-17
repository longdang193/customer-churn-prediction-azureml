#!/usr/bin/env python3
"""Helper entry-point to invoke train.py with sweep-managed hyperparameters."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

# Mirror the hyperparameter keys defined in hpo_utils.build_parameter_space.
HYPERPARAM_KEYS: List[str] = [
    "rf_n_estimators",
    "rf_max_depth",
    "rf_min_samples_split",
    "rf_min_samples_leaf",
    "logreg_C",
    "logreg_solver",
    "xgboost_n_estimators",
    "xgboost_max_depth",
    "xgboost_learning_rate",
    "xgboost_subsample",
    "xgboost_colsample_bytree",
]


def _format_override_key(raw_key: str) -> str:
    """Convert CLI-friendly keys (e.g., rf_n_estimators) to train.py format (rf.n_estimators)."""
    for model_prefix in ("rf", "logreg", "xgboost"):
        expected_prefix = f"{model_prefix}_"
        if raw_key.startswith(expected_prefix):
            return f"{model_prefix}.{raw_key[len(expected_prefix):]}"
    return raw_key


def _add_hyperparam_arguments(parser: argparse.ArgumentParser) -> None:
    for key in HYPERPARAM_KEYS:
        parser.add_argument(f"--{key}", default=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Invoke train.py with sweep-managed hyperparameters.")
    parser.add_argument("--data", required=True, help="Processed data URI/folder.")
    parser.add_argument("--model-type", required=True, help="Model type to train (logreg|rf|xgboost).")
    parser.add_argument(
        "--model-artifact-dir",
        required=True,
        help="Directory/URI where the trained model artifacts should be stored.",
    )
    _add_hyperparam_arguments(parser)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    train_script = repo_root / "train.py"

    cli = [
        sys.executable,
        str(train_script),
        "--data",
        args.data,
        "--model-type",
        args.model_type,
        "--model-artifact-dir",
        args.model_artifact_dir,
    ]

    for key in HYPERPARAM_KEYS:
        value = getattr(args, key)
        if value is None or str(value).lower() == "none":
            continue
        cli.extend(["--set", f"{_format_override_key(key)}={value}"])

    subprocess.run(cli, check=True)


if __name__ == "__main__":
    main()

