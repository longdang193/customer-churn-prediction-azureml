"""Model creation, configuration, and hyperparameter management utilities."""

from typing import Any, Dict, Optional

import pandas as pd

from models import get_logistic_regression, get_random_forest, get_xgboost

JSONDict = Dict[str, Any]
MODEL_FACTORY = {
    "logreg": get_logistic_regression,
    "rf": get_random_forest,
    "xgboost": get_xgboost,
}


def get_model(model_name: str, class_weight: Optional[str] = "balanced", random_state: int = 42) -> Any:
    """Return a configured estimator instance.
    
    Args:
        model_name: Model identifier ('logreg', 'rf', or 'xgboost')
        class_weight: Class weight strategy
        random_state: Random seed
        
    Returns:
        Configured scikit-learn compatible estimator
        
    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_FACTORY.keys())}")
    return MODEL_FACTORY[model_name](class_weight=class_weight, random_state=random_state)


def apply_hyperparameters(model: Any, hyperparams: Optional[JSONDict]) -> tuple[Any, JSONDict]:
    """Apply hyperparameters via set_params where supported.
    
    Args:
        model: Model instance to configure
        hyperparams: Dictionary of hyperparameters to apply
        
    Returns:
        Tuple of (model, applied_params) where applied_params contains only
        successfully applied parameters
    """
    if not hyperparams or not hasattr(model, "get_params"):
        return model, {}
    
        valid_params = model.get_params(deep=True)
        applied = {k: v for k, v in hyperparams.items() if k in valid_params}
        
        # Validate RandomForest min_samples_split (sklearn requirement)
        if "min_samples_split" in applied and applied["min_samples_split"] < 2:
            applied["min_samples_split"] = 2
        
    if not applied:
        return model, {}

    try:
            model.set_params(**applied)
    except Exception:
        return model, {}
    
    return model, applied


def apply_class_weight_adjustments(
    model_name: str,
    model: Any,
    y_train: pd.Series,
    class_weight: Optional[str],
    tuned_params: JSONDict,
) -> JSONDict:
    """Adjust XGBoost scale_pos_weight when class_weight='balanced'.
    
    Args:
        model_name: Model identifier
        model: Model instance
        y_train: Training labels
        class_weight: Class weight strategy
        tuned_params: Already applied hyperparameters
        
    Returns:
        Dictionary of additional parameters applied (empty if none)
    """
    if model_name != "xgboost" or class_weight != "balanced":
        return {}
    if "scale_pos_weight" in (tuned_params or {}):
        return {}
    if not hasattr(model, "get_params"):
        return {}
    
    params = model.get_params()
    if "scale_pos_weight" not in params:
        return {}
    
    positives = (y_train == 1).sum()
    negatives = (y_train == 0).sum()
    if positives == 0 or negatives == 0:
        return {}
    
    current = params.get("scale_pos_weight")
    if current not in (None, 0, 1):
        return {}
    
    scale_pos_weight = float(negatives / positives)
    try:
        model.set_params(scale_pos_weight=scale_pos_weight)
        return {"scale_pos_weight": scale_pos_weight}
    except Exception:
        return {}

