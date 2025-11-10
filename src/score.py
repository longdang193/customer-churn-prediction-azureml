#!/usr/bin/env python3
"""Model Scoring Script for Bank Customer Churn Prediction."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Union

import pandas as pd
import numpy as np


def load_model(model_path: str) -> Any:
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_preprocessing_artifacts(data_dir: str) -> Dict[str, Any]:
    """Load encoders, scaler, and metadata."""
    data_path = Path(data_dir)
    
    with open(data_path / 'encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open(data_path / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return {
        'encoders': encoders,
        'scaler': scaler,
        'metadata': metadata
    }


def preprocess_input(
    df: pd.DataFrame,
    artifacts: Dict[str, Any],
    remove_uninformative: bool = True
) -> pd.DataFrame:
    """
    Preprocess input data using saved artifacts.
    
    Args:
        df: Input DataFrame
        artifacts: Dictionary with encoders, scaler, metadata
        remove_uninformative: Whether to remove ID columns
        
    Returns:
        Preprocessed DataFrame ready for prediction
    """
    df_processed = df.copy()
    
    if remove_uninformative:
        cols_to_remove = ['RowNumber', 'CustomerId', 'Surname']
        existing_cols = [col for col in cols_to_remove if col in df_processed.columns]
        if existing_cols:
            df_processed = df_processed.drop(columns=existing_cols)
    
    encoders = artifacts['encoders']
    scaler = artifacts['scaler']
    feature_names = artifacts['metadata']['feature_names']
    
    # Encode categorical features
    for col, encoder in encoders.items():
        if col in df_processed.columns:
            df_processed[col] = encoder.transform(df_processed[col])
    
    # Scale numerical features
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target if present
    if 'Exited' in numerical_cols:
        numerical_cols.remove('Exited')
    
    if numerical_cols:
        df_processed[numerical_cols] = scaler.transform(df_processed[numerical_cols])
    
    # Ensure correct feature order
    df_processed = df_processed[feature_names]
    
    return df_processed


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Make predictions using the model."""
    return model.predict(X)


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get prediction probabilities."""
    return model.predict_proba(X)


def score_batch(
    model_path: str,
    data_dir: str,
    input_path: str,
    output_path: str,
    include_proba: bool = True,
    threshold: float = 0.5
) -> None:
    """
    Score a batch of examples from CSV file.
    
    Args:
        model_path: Path to trained model
        data_dir: Directory with preprocessing artifacts
        input_path: Path to input CSV file
        output_path: Path to save predictions CSV
        include_proba: Whether to include probability scores
        threshold: Classification threshold (default: 0.5)
    """
    print(f"{'='*70}\nBATCH SCORING\n{'='*70}")
    
    print(f"\n[1/4] Loading model and artifacts...")
    model = load_model(model_path)
    artifacts = load_preprocessing_artifacts(data_dir)
    print(f"  Model: {type(model).__name__}")
    print(f"  Features: {len(artifacts['metadata']['feature_names'])}")
    
    print(f"\n[2/4] Loading input data...")
    df_input = pd.read_csv(input_path)
    print(f"  Input samples: {len(df_input)}")
    
    print(f"\n[3/4] Preprocessing and predicting...")
    X_processed = preprocess_input(df_input, artifacts)
    predictions = predict(model, X_processed)
    probabilities = predict_proba(model, X_processed)[:, 1]
    
    print(f"\n[4/4] Saving predictions...")
    df_output = df_input.copy()
    df_output['predicted_churn'] = predictions
    df_output['churn_probability'] = probabilities
    
    if 'Exited' in df_output.columns:
        df_output['correct'] = (df_output['Exited'] == df_output['predicted_churn']).astype(int)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)
    print(f"  ✓ Saved: {output_path}")
    
    print(f"\n{'='*70}\n✓ SCORING COMPLETE\n{'='*70}")
    print(f"Predictions: {predictions.sum()} churn, {len(predictions) - predictions.sum()} retain")
    print(f"Avg probability: {probabilities.mean():.3f}")
    
    if 'Exited' in df_output.columns:
        accuracy = df_output['correct'].mean()
        print(f"Accuracy (if labels available): {accuracy:.3f}")


def score_single(
    model_path: str,
    data_dir: str,
    input_data: Dict[str, Any],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Score a single example (for API deployment).
    
    Args:
        model_path: Path to trained model
        data_dir: Directory with preprocessing artifacts
        input_data: Dictionary with feature values
        threshold: Classification threshold
        
    Returns:
        Dictionary with prediction and probability
    """
    model = load_model(model_path)
    artifacts = load_preprocessing_artifacts(data_dir)
    
    df_input = pd.DataFrame([input_data])
    X_processed = preprocess_input(df_input, artifacts)
    
    prediction = predict(model, X_processed)[0]
    probability = predict_proba(model, X_processed)[0, 1]
    
    return {
        'prediction': int(prediction),
        'churn_probability': float(probability),
        'predicted_class': 'churn' if prediction == 1 else 'retain'
    }


def score_from_json(
    model_path: str,
    data_dir: str,
    input_path: str,
    output_path: str
) -> None:
    """Score from JSON input (for API-like usage)."""
    print(f"{'='*70}\nJSON SCORING\n{'='*70}")
    
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    
    if isinstance(input_data, list):
        results = []
        for item in input_data:
            result = score_single(model_path, data_dir, item)
            results.append(result)
    else:
        results = score_single(model_path, data_dir, input_data)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved predictions to: {output_path}")


def main():
    """Main function with CLI."""
    parser = argparse.ArgumentParser(
        description='Score bank churn prediction model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pkl file)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory with preprocessing artifacts (encoders, scaler, metadata)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input CSV or JSON file'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output CSV or JSON file for predictions'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold'
    )
    
    parser.add_argument(
        '--no-proba',
        action='store_true',
        help='Do not include probability scores in output'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Use JSON format for input/output'
    )
    
    args = parser.parse_args()
    
    if args.json:
        score_from_json(args.model, args.data_dir, args.input, args.output)
    else:
        score_batch(
            model_path=args.model,
            data_dir=args.data_dir,
            input_path=args.input,
            output_path=args.output,
            include_proba=not args.no_proba,
            threshold=args.threshold
        )


if __name__ == '__main__':
    main()

