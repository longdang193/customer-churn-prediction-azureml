#!/usr/bin/env python3
"""Validate prepared data for correctness."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def validate_prepared_data(data_dir: str) -> bool:
    """
    Validate that prepared data is correct and complete.
    
    Args:
        data_dir: Directory containing processed data
        
    Returns:
        True if validation passes, False otherwise
    """
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("VALIDATING PREPARED DATA")
    print("=" * 70)
    print(f"Data directory: {data_dir}\n")
    
    success = True
    
    # Check required files exist
    print("[1/6] Checking required files...")
    required_files = [
        'X_train.csv', 'X_test.csv',
        'y_train.csv', 'y_test.csv',
        'encoders.pkl', 'scaler.pkl',
        'metadata.json'
    ]
    
    for file in required_files:
        if not (data_path / file).exists():
            print(f"  ✗ Missing file: {file}")
            success = False
        else:
            print(f"  ✓ {file}")
    
    if not success:
        return False
    
    # Load metadata
    print("\n[2/6] Loading metadata...")
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  Features: {metadata['n_features']}")
    print(f"  Train samples: {metadata['n_train']}")
    print(f"  Test samples: {metadata['n_test']}")
    print(f"  Test size: {metadata['test_size']}")
    
    # Load data
    print("\n[3/6] Loading datasets...")
    X_train = pd.read_csv(data_path / 'X_train.csv')
    X_test = pd.read_csv(data_path / 'X_test.csv')
    y_train = pd.read_csv(data_path / 'y_train.csv').squeeze()
    y_test = pd.read_csv(data_path / 'y_test.csv').squeeze()
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")
    
    # Validate shapes
    print("\n[4/6] Validating shapes...")
    if X_train.shape[0] != len(y_train):
        print(f"  ✗ Train features/target mismatch: {X_train.shape[0]} vs {len(y_train)}")
        success = False
    else:
        print(f"  ✓ Train features/target match: {X_train.shape[0]} samples")
    
    if X_test.shape[0] != len(y_test):
        print(f"  ✗ Test features/target mismatch: {X_test.shape[0]} vs {len(y_test)}")
        success = False
    else:
        print(f"  ✓ Test features/target match: {X_test.shape[0]} samples")
    
    if X_train.shape[1] != X_test.shape[1]:
        print(f"  ✗ Train/test feature mismatch: {X_train.shape[1]} vs {X_test.shape[1]}")
        success = False
    else:
        print(f"  ✓ Train/test features match: {X_train.shape[1]} features")
    
    # Check for missing values
    print("\n[5/6] Checking for missing values...")
    train_missing = X_train.isnull().sum().sum()
    test_missing = X_test.isnull().sum().sum()
    
    if train_missing > 0:
        print(f"  ✗ Train set has {train_missing} missing values")
        success = False
    else:
        print("  ✓ No missing values in train set")
    
    if test_missing > 0:
        print(f"  ✗ Test set has {test_missing} missing values")
        success = False
    else:
        print("  ✓ No missing values in test set")
    
    # Check target distribution
    print("\n[6/6] Checking target distribution...")
    train_churn = y_train.mean()
    test_churn = y_test.mean()
    
    print(f"  Train churn rate: {train_churn:.2%}")
    print(f"  Test churn rate: {test_churn:.2%}")
    
    # Check if distributions are similar (within 5%)
    if abs(train_churn - test_churn) > 0.05:
        print(f"  ⚠ Train/test churn rate difference: {abs(train_churn - test_churn):.2%}")
        print("    (This might indicate stratification issues)")
    else:
        print("  ✓ Train/test churn rates are similar")
    
    # Load and verify artifacts
    print("\n[Extra] Verifying preprocessing artifacts...")
    try:
        with open(data_path / 'encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        print(f"  ✓ Encoders loaded: {list(encoders.keys())}")
        
        with open(data_path / 'scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print(f"  ✓ Scaler loaded: {type(scaler).__name__}")
    except Exception as e:
        print(f"  ✗ Error loading artifacts: {e}")
        success = False
    
    # Final verdict
    print("\n" + "=" * 70)
    if success:
        print("✓ VALIDATION PASSED - Data is ready for training!")
    else:
        print("✗ VALIDATION FAILED - Please check errors above")
    print("=" * 70)
    
    return success


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(description='Validate prepared data')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing processed data'
    )
    
    args = parser.parse_args()
    
    success = validate_prepared_data(args.data_dir)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()

