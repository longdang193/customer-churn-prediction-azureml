"""Data loading and artifact saving utilities."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path: Path) -> pd.DataFrame:
    """Read the raw CSV file(s).
    
    Args:
        path: Path to CSV file or directory containing CSV file(s)
        
    Returns:
        DataFrame containing the loaded data (concatenated if multiple files)
        
    Raises:
        FileNotFoundError: If no CSV file is found
    """
    path = Path(path)
    
    # If it's a directory, load all CSV files and concatenate them
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in directory: {path}")
        
        # Load and concatenate all CSV files
        dataframes = [pd.read_csv(csv_file) for csv_file in csv_files]
        df = pd.concat(dataframes, ignore_index=True)
        print(f"Loaded {len(csv_files)} CSV file(s) from {path}: {[f.name for f in csv_files]}")
        return df
    
    # Single file
    return pd.read_csv(path)


def save_artifacts(
    output_dir: Path,
    *,
    encoders: Dict[str, LabelEncoder],
    scaler: StandardScaler,
    metadata: Dict[str, Any]
) -> None:
    """Save all preprocessing artifacts to disk.
    
    Args:
        output_dir: Directory to save artifacts
        encoders: Dictionary of LabelEncoders
        scaler: StandardScaler instance
        metadata: Dictionary of metadata to save
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "encoders.pkl", "wb") as fh:
        pickle.dump(encoders, fh)
    with open(output_dir / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    with open(output_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2)


def save_preprocessed_data(
    output_dir: Path,
    *,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save preprocessed training and test data to CSV files.
    
    Args:
        output_dir: Directory to save CSV files
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False, header=True)
    y_test.to_csv(output_dir / "y_test.csv", index=False, header=True)

