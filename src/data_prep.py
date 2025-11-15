#!/usr/bin/env python3
"""Data preparation CLI for the churn dataset."""

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from data import encode_categoricals, load_data, remove_columns, save_artifacts, save_preprocessed_data, scale_features
from utils import DEFAULT_CONFIG, get_data_prep_config


def prepare_data(
    *,
    input_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    target_col: str,
    columns_to_remove: Iterable[str],
    categorical_cols: Iterable[str],
    stratify: bool,
) -> None:
    """Execute the end-to-end preprocessing pipeline.
    
    Args:
        input_path: Path to input CSV file or directory containing CSV file(s)
        output_dir: Directory to save preprocessed data and artifacts
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        target_col: Name of the target column
        columns_to_remove: Iterable of column names to remove
        categorical_cols: Iterable of categorical column names to encode
        stratify: Whether to stratify the train-test split
        
    Raises:
        ValueError: If target column is not present in data
    """
    print(f"{'=' * 70}\nDATA PREPARATION PIPELINE\n{'=' * 70}")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df, columns_removed = remove_columns(df, columns_to_remove)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if stratify else None
    )

    X_train_encoded, encoders = encode_categoricals(X_train, categorical_cols=categorical_cols)
    X_test_encoded, _ = encode_categoricals(X_test, categorical_cols=categorical_cols, encoders=encoders)

    encoded_categorical_cols = [col for col in categorical_cols if col in X_train_encoded.columns]

    X_train_scaled, scaler, scaled_numeric_cols = scale_features(
        X_train_encoded,
        exclude_cols=encoded_categorical_cols,
    )
    X_test_scaled, _, _ = scale_features(
        X_test_encoded,
        scaler=scaler,
        columns=scaled_numeric_cols,
    )

    save_preprocessed_data(
        output_dir,
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
    )

    metadata = {
        "feature_names": list(X_train_scaled.columns),
        "target_name": target_col,
        "n_train": len(X_train_scaled),
        "n_test": len(X_test_scaled),
        "categorical_encoded_columns": encoded_categorical_cols,
        "scaled_numeric_columns": list(scaled_numeric_cols),
        "dropped_columns": columns_removed,
    }
    save_artifacts(output_dir, encoders=encoders, scaler=scaler, metadata=metadata)

    print(
        f"\n{'=' * 70}\n✓ DATA PREPARATION COMPLETE\n{'=' * 70}\n"
        f"Features: {len(X_train_scaled.columns)} | Train: {len(X_train_scaled)} | Test: {len(X_test_scaled)}\n"
        f"Churn rate: {y_train.mean():.2%} (train) / {y_test.mean():.2%} (test)"
    )


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(
        description="Prepare churn data for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input", type=str, help="Input CSV file or directory containing CSV file(s)")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--test-size", type=float, help="Override test split proportion")
    parser.add_argument("--random-state", type=int, help="Override random seed")
    parser.add_argument("--target", type=str, help="Override target column name")
    args = parser.parse_args()

    config = get_data_prep_config(args)
    prepare_data(**config)


if __name__ == "__main__":
    main()
