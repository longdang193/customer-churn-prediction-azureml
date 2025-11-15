"""Data loading and preprocessing utilities."""

from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def load_prepared_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load preprocessed training and test data.
    
    Args:
        data_dir: Directory containing preprocessed CSV files
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    data_path = Path(data_dir)
    return (
        pd.read_csv(data_path / 'X_train.csv'),
        pd.read_csv(data_path / 'X_test.csv'),
        pd.read_csv(data_path / 'y_train.csv').squeeze(),
        pd.read_csv(data_path / 'y_test.csv').squeeze(),
    )


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to balance training labels.
    
    Args:
        X_train: Training features
        y_train: Training labels
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_balanced, y_balanced)
        
    Raises:
        ImportError: If imbalanced-learn is not installed
    """
    if not SMOTE_AVAILABLE:
        raise ImportError("SMOTE not available. Install with: pip install imbalanced-learn")
    
    smote = SMOTE(random_state=random_state)
    X_bal, y_bal = smote.fit_resample(X_train, y_train)
    return pd.DataFrame(X_bal, columns=X_train.columns), pd.Series(y_bal)

