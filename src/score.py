#!/usr/bin/env python3
"""Scoring utilities for the churn prediction pipeline."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config_loader import get_config_value, load_config

DEFAULT_CONFIG = Path(__file__).parents[1] / "configs" / "score.yaml"


def load_artifacts(model_path: Path, data_dir: Path) -> Dict[str, Any]:
    """Load the model and preprocessing artifacts."""
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    with open(data_dir / "encoders.pkl", "rb") as fh:
        encoders = pickle.load(fh)
    with open(data_dir / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    with open(data_dir / "metadata.json", "r") as fh:
        metadata = json.load(fh)
    return {"model": model, "encoders": encoders, "scaler": scaler, "metadata": metadata}


def preprocess(
    df: pd.DataFrame,
    *,
    encoders: Dict[str, Any],
    scaler: Any,
    feature_names: List[str],
    metadata: Dict[str, Any],
    drop_uninformative: bool = True,
) -> pd.DataFrame:
    """Apply the full preprocessing pipeline to raw data."""
    processed = df.copy()
    if drop_uninformative:
        if "dropped_columns" not in metadata:
            raise KeyError("Preprocessing metadata is missing 'dropped_columns'.")
        dropped_cols = metadata["dropped_columns"]
        if not isinstance(dropped_cols, Iterable):
            raise TypeError("'dropped_columns' metadata must be an iterable of column names.")
        processed = processed.drop(columns=[c for c in dropped_cols if c in processed.columns], errors='ignore')
    if "Exited" in processed.columns:
        processed = processed.drop(columns=["Exited"])

    for col, encoder in encoders.items():
        if col not in processed.columns:
            continue

        column = processed[col]
        # Only attempt guard if original values are non-numeric strings
        if column.dtype.kind in {"O", "U"}:
            series = column.astype(str)
            known_classes = set(encoder.classes_)
            unknown_mask = ~series.isin(known_classes)

            if unknown_mask.any():
                unknown_label = "__unknown__"
                if unknown_label not in known_classes:
                    encoder.classes_ = np.sort(np.append(encoder.classes_, unknown_label))
                    known_classes.add(unknown_label)
                series.loc[unknown_mask] = unknown_label

            processed[col] = encoder.transform(series)
        else:
            processed[col] = encoder.transform(column)
    
    missing = [col for col in feature_names if col not in processed.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")

    processed = processed[feature_names].copy()
    scaled_cols = [col for col in metadata.get("scaled_numeric_columns", []) if col in processed.columns]
    if scaled_cols:
        subset = processed.loc[:, scaled_cols].astype(float)
        processed.loc[:, scaled_cols] = scaler.transform(subset)
    return processed


def predict_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return churn probabilities, falling back to decision function when needed."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba, dtype=float)
        if proba.ndim == 1:
            return proba
        if proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()

    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        decision = np.asarray(decision, dtype=float)
        if decision.ndim > 1:
            decision = decision[:, 0]
        min_val = np.min(decision)
        max_val = np.max(decision)
        if np.isclose(max_val, min_val):
            return np.full(decision.shape, 0.5, dtype=float)
        return (decision - min_val) / (max_val - min_val)

    return np.full(shape=(len(X),), fill_value=0.5, dtype=float)


def ensure_path(path: Union[str, Path]) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def score_batch(
    model: Any, 
    df_input: pd.DataFrame, 
    artifacts: Dict[str, Any], 
    output_path: Path, 
    include_proba: bool
) -> None:
    """Score a batch of data and save the results."""
    X = preprocess(df_input, **artifacts, drop_uninformative=True)
    predictions = model.predict(X)
    probabilities = predict_probabilities(model, X) if include_proba else None

    df_output = df_input.copy()
    df_output["predicted_churn"] = predictions
    if include_proba and probabilities is not None:
        df_output["churn_probability"] = probabilities

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_path, index=False)
    print(f"  ✓ Saved predictions to: {output_path}")


def score_single(
    model: Any,
    payload: Dict[str, Any],
    artifacts: Dict[str, Any],
    *,
    include_proba: bool,
) -> Dict[str, Any]:
    """Score a single data point."""
    df_input = pd.DataFrame([payload])
    X = preprocess(df_input, **artifacts, drop_uninformative=True)
    prediction = model.predict(X)[0]
    probability = predict_probabilities(model, X)[0] if include_proba else None
    result = {
        "prediction": int(prediction),
        "predicted_class": "churn" if prediction == 1 else "retain",
    }
    if include_proba and probability is not None:
        result["churn_probability"] = float(probability)
    return result


def main() -> None:
    """CLI entry-point for scoring."""
    parser = argparse.ArgumentParser(description="Score churn models.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pkl)")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with preprocessing artifacts")
    parser.add_argument("--input", type=str, required=True, help="Input CSV or JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output path for scored data")
    parser.add_argument("--config", type=str, default=None, help=f"Config file (default: {DEFAULT_CONFIG})")
    parser.add_argument("--json", action="store_true", help="Treat input/output as JSON")
    args = parser.parse_args()

    config_path = Path(args.config or DEFAULT_CONFIG)
    config = load_config(str(config_path)) if config_path.exists() else {}
    score_config = get_config_value(config, 'scoring', {})
    include_probability = bool(get_config_value(score_config, 'include_probability', True))

    model_path = ensure_path(args.model)
    data_dir = ensure_path(args.data_dir)
    input_path = ensure_path(args.input)
    output_path = ensure_path(args.output)

    artifacts = load_artifacts(model_path, data_dir)
    model = artifacts.pop("model")
    # Re-structure artifacts for preprocess function
    metadata = artifacts["metadata"]
    preprocess_artifacts = {
        "encoders": artifacts["encoders"],
        "scaler": artifacts["scaler"],
        "feature_names": metadata["feature_names"],
        "metadata": metadata,
    }

    if args.json:
        with open(input_path, "r") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            results = [
                score_single(model, item, preprocess_artifacts, include_proba=include_probability)
                for item in payload
            ]
        else:
            results = score_single(model, payload, preprocess_artifacts, include_proba=include_probability)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\n✓ Saved JSON predictions to: {output_path}")
    else:
        score_batch(
            model=model,
            df_input=pd.read_csv(input_path),
            artifacts=preprocess_artifacts,
            output_path=output_path,
            include_proba=include_probability,
        )

if __name__ == "__main__":
    main()
