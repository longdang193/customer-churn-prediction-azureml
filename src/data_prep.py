#!/usr/bin/env python3
"""Data preparation CLI for the churn dataset."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_value, load_config

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "data.yaml"
DEFAULT_COLUMNS_TO_REMOVE = ("RowNumber", "CustomerId", "Surname")
DEFAULT_CATEGORICAL = ("Geography", "Gender")


def parse_bool(value: Any, *, default: bool) -> bool:
    """Parse loose truthy/falsey values without relying on distutils."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    raise ValueError(f"Cannot interpret value '{value}' as boolean.")

def load_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV and report basic shape."""
    print(f"Loading data from: {path}")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def remove_columns(df: pd.DataFrame, columns: Iterable[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Drop any columns present in the frame and return the removed columns."""
    to_drop = [col for col in columns if col in df.columns]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Removed columns: {', '.join(to_drop)}")
    return df, to_drop


def encode_categoricals(
    df: pd.DataFrame,
    *,
    categorical_cols: Iterable[str],
    encoders: Optional[Dict[str, LabelEncoder]] = None,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Label-encode categorical columns, reusing encoders when provided."""
    encoders = encoders or {}
    df_encoded = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            continue
        if col not in encoders:
            encoders[col] = LabelEncoder().fit(df[col])
        df_encoded[col] = encoders[col].transform(df[col])
    return df_encoded, encoders


def scale_features(
    df: pd.DataFrame,
    *,
    scaler: Optional[StandardScaler] = None,
    columns: Optional[Iterable[str]] = None,
    exclude_cols: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, StandardScaler, List[str]]:
    """Standard-score the numeric columns."""
    df_scaled = df.copy()
    if columns is not None:
        numeric_cols = [col for col in columns if col in df.columns]
    else:
        numeric_cols = df.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
        if exclude_cols:
            exclude_set = {col for col in exclude_cols if col in df.columns}
            numeric_cols = [col for col in numeric_cols if col not in exclude_set]

    if not numeric_cols:
        scaler = scaler or StandardScaler()
        return df_scaled, scaler, []

    if scaler is None:
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])

    return df_scaled, scaler, list(numeric_cols)


def save_artifacts(
    output_dir: Path, *,
    encoders: Dict[str, LabelEncoder], 
    scaler: StandardScaler, 
    metadata: Dict[str, Any]
) -> None:
    """Save all preprocessing artifacts to disk."""
    with open(output_dir / "encoders.pkl", "wb") as fh:
        pickle.dump(encoders, fh)
    with open(output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open(output_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)


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
    """Execute the end-to-end preprocessing pipeline."""
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
    print(f"Split: {len(X_train)} train / {len(X_test)} test samples")

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

    X_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)

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


def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Load config from file and merge with CLI arguments."""
    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    cfg = get_config_value(config, "data", {}) or {}

    stratify_raw = get_config_value(cfg, "stratify", True)
    stratify = parse_bool(stratify_raw, default=True)

    return {
        "input_path": Path(args.input or get_config_value(cfg, "input_path", "data/churn.csv")),
        "output_dir": Path(args.output or get_config_value(cfg, "output_dir", "data/processed")),
        "test_size": float(args.test_size or get_config_value(cfg, "test_size", 0.2)),
        "random_state": int(args.random_state or get_config_value(cfg, "random_state", 42)),
        "target_col": args.target or get_config_value(cfg, "target_column", "Exited"),
        "columns_to_remove": get_config_value(cfg, "columns_to_remove", DEFAULT_COLUMNS_TO_REMOVE),
        "categorical_cols": get_config_value(cfg, "categorical_columns", DEFAULT_CATEGORICAL),
        "stratify": stratify,
    }


def main() -> None:
    """CLI entry-point."""
    parser = argparse.ArgumentParser(description="Prepare churn data for training.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, help="Input CSV file")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--config", type=str, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--test-size", type=float, help="Override test split proportion")
    parser.add_argument("--random-state", type=int, help="Override random seed")
    parser.add_argument("--target", type=str, help="Override target column name")
    args = parser.parse_args()

    config = get_config(args)
    prepare_data(**config)


if __name__ == "__main__":
    main()
