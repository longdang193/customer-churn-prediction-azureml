import pytest
import pandas as pd
from pathlib import Path

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a sample DataFrame for testing."""
    data = {
        'RowNumber': range(1, 11),
        'CustomerId': [15634602, 15647311, 15619304, 15701354, 15737888, 15574012, 15592531, 15656148, 15792365, 15600882],
        'Surname': ['Hargrave', 'Hill', 'Onio', 'Boni', 'Mitchell', 'Chu', 'Hsieh', 'Maclean', 'He', 'H?'],
        'CreditScore': [619, 608, 502, 699, 850, 645, 822, 376, 501, 684],
        'Geography': ['France', 'Spain', 'France', 'France', 'Spain', 'Spain', 'France', 'Germany', 'France', 'France'],
        'Gender': ['Female', 'Female', 'Female', 'Female', 'Male', 'Male', 'Male', 'Female', 'Male', 'Male'],
        'Age': [42, 41, 42, 39, 43, 44, 50, 29, 44, 27],
        'Tenure': [2, 1, 8, 1, 2, 8, 7, 4, 4, 2],
        'Balance': [0.0, 83807.86, 159660.8, 0.0, 125510.82, 113755.78, 0.0, 115046.74, 142051.07, 134603.88],
        'NumOfProducts': [1, 1, 3, 2, 1, 2, 2, 4, 2, 1],
        'HasCrCard': [1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        'IsActiveMember': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
        'EstimatedSalary': [101348.88, 112542.58, 113931.57, 93826.63, 79084.1, 149756.7, 10062.8, 119346.88, 74940.5, 71725.73],
        'Exited': [1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

