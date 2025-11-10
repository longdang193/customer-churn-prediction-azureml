#!/usr/bin/env python3
"""Data Preparation Script for Bank Customer Churn Prediction."""

import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_config, get_config_value


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def remove_uninformative_columns(df: pd.DataFrame, cols_to_remove: list = None) -> pd.DataFrame:
    """Remove ID and high cardinality columns that don't help prediction."""
    if cols_to_remove is None:
        cols_to_remove = ['RowNumber', 'CustomerId', 'Surname']
    
    existing_cols = [col for col in cols_to_remove if col in df.columns]
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"Removed columns: {', '.join(existing_cols)}")
    
    return df


def encode_categorical_features(
    df: pd.DataFrame,
    encoders: dict = None,
    fit: bool = True,
    categorical_cols: list = None
) -> tuple[pd.DataFrame, dict]:
    """Encode categorical variables using Label Encoding."""
    if categorical_cols is None:
        categorical_cols = ['Geography', 'Gender']
    if encoders is None:
        encoders = {}
    
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            if fit:
                encoders[col] = LabelEncoder()
                df_encoded[col] = encoders[col].fit_transform(df[col])
            else:
                if col not in encoders:
                    raise ValueError(f"No encoder for column: {col}")
                df_encoded[col] = encoders[col].transform(df[col])
    
    return df_encoded, encoders


def scale_numerical_features(
    df: pd.DataFrame,
    target_col: str = 'Exited',
    scaler: StandardScaler = None,
    fit: bool = True
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numerical features using StandardScaler."""
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    df_scaled = df.copy()
    if fit:
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        if scaler is None:
            raise ValueError("Scaler required when fit=False")
        df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df_scaled, scaler


def prepare_data(
    input_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    target_col: str = 'Exited',
    cols_to_remove: list = None,
    categorical_cols: list = None,
    stratify: bool = True
) -> None:
    """Complete data preparation pipeline."""
    print(f"{'='*70}\nDATA PREPARATION PIPELINE\n{'='*70}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = load_data(input_path)
    df = remove_uninformative_columns(df, cols_to_remove)
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    stratify_param = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )
    print(f"Split: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    X_train, encoders = encode_categorical_features(X_train, fit=True, categorical_cols=categorical_cols)
    X_test, _ = encode_categorical_features(X_test, encoders=encoders, fit=False, categorical_cols=categorical_cols)
    
    X_train, scaler = scale_numerical_features(X_train, fit=True)
    X_test, _ = scale_numerical_features(X_test, scaler=scaler, fit=False)
    
    X_train.to_csv(output_path / 'X_train.csv', index=False)
    X_test.to_csv(output_path / 'X_test.csv', index=False)
    y_train.to_csv(output_path / 'y_train.csv', index=False, header=True)
    y_test.to_csv(output_path / 'y_test.csv', index=False, header=True)
    
    with open(output_path / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open(output_path / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'feature_names': list(X_train.columns),
        'target_name': target_col,
        'n_features': len(X_train.columns),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_size': test_size,
        'random_state': random_state
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*70}\n✓ DATA PREPARATION COMPLETE\n{'='*70}")
    print(f"Features: {len(X_train.columns)} | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Churn rate: {y_train.mean():.2%} (train), {y_test.mean():.2%} (test)")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Prepare bank churn data for training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Directory to save processed data'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: configs/data.yaml)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=None,
        help='Proportion of data for test set (overrides config)'
    )
    
    parser.add_argument(
        '--random-state',
        type=int,
        default=None,
        help='Random seed for reproducibility (overrides config)'
    )
    
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Name of target column (overrides config)'
    )
    
    args = parser.parse_args()
    
    # Determine config path (relative to project root)
    if args.config is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / 'configs' / 'data.yaml'
    else:
        config_path = Path(args.config)
    
    # Load configuration from YAML
    config = {}
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded configuration from: {config_path}")
    else:
        print(f"Config file not found: {config_path}, using defaults")
    
    # Get values from config or CLI args (CLI takes precedence)
    data_config = get_config_value(config, 'data', {})
    
    input_path = args.input or get_config_value(data_config, 'input_path', 'data/churn.csv')
    output_dir = args.output or get_config_value(data_config, 'output_dir', 'data/processed')
    test_size = args.test_size or get_config_value(data_config, 'test_size', 0.2)
    random_state = args.random_state or get_config_value(data_config, 'random_state', 42)
    target_col = args.target or get_config_value(data_config, 'target_column', 'Exited')
    cols_to_remove = get_config_value(data_config, 'columns_to_remove', ['RowNumber', 'CustomerId', 'Surname'])
    categorical_cols = get_config_value(data_config, 'categorical_columns', ['Geography', 'Gender'])
    stratify = get_config_value(data_config, 'stratify', True)
    
    # Run data preparation
    prepare_data(
        input_path=input_path,
        output_dir=output_dir,
        test_size=test_size,
        random_state=random_state,
        target_col=target_col,
        cols_to_remove=cols_to_remove,
        categorical_cols=categorical_cols,
        stratify=stratify
    )


if __name__ == '__main__':
    main()

