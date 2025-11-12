import sys
from pathlib import Path
import pandas as pd
import pytest

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_prep import (
    load_data,
    remove_columns,
    encode_categoricals,
    scale_features,
    prepare_data,
)

def test_load_data(tmp_path):
    """Test that data is loaded correctly from a CSV file."""
    # Arrange
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    file_path = tmp_path / "test.csv"
    df.to_csv(file_path, index=False)

    # Act
    loaded_df = load_data(file_path)

    # Assert
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (2, 2)
    pd.testing.assert_frame_equal(loaded_df, df)

def test_remove_columns(sample_dataframe):
    """Test the removal of specified columns."""
    # Arrange
    cols_to_remove = ['RowNumber', 'CustomerId']
    initial_cols = set(sample_dataframe.columns)

    # Act
    df = remove_columns(sample_dataframe, cols_to_remove)

    # Assert
    final_cols = set(df.columns)
    assert 'RowNumber' not in final_cols
    assert 'CustomerId' not in final_cols
    assert len(final_cols) == len(initial_cols) - 2

def test_encode_categoricals(sample_dataframe):
    """Test categorical feature encoding."""
    # Arrange
    categorical_cols = ['Geography', 'Gender']
    
    # Act
    df_encoded, encoders = encode_categoricals(
        sample_dataframe, categorical_cols=categorical_cols
    )

    # Assert
    assert 'Geography' in encoders
    assert 'Gender' in encoders
    assert pd.api.types.is_numeric_dtype(df_encoded['Geography'])
    assert pd.api.types.is_numeric_dtype(df_encoded['Gender'])
    assert df_encoded['Geography'].nunique() == 3
    assert df_encoded['Gender'].nunique() == 2

def test_scale_features(sample_dataframe):
    """Test numerical feature scaling."""
    # Arrange
    df_encoded, _ = encode_categoricals(
        sample_dataframe, categorical_cols=['Geography', 'Gender']
    )
    
    # Act
    # Drop target column before scaling, as the scaler would be fitted on it otherwise
    df_to_scale = df_encoded.drop(columns=['Exited'])
    df_scaled, scaler = scale_features(df_to_scale)
    df_scaled['Exited'] = df_encoded['Exited'] # Add it back for the assertion

    # Assert
    assert scaler is not None
    # Check that mean is close to 0 and std is close to 1 for a scaled column
    assert abs(df_scaled['CreditScore'].mean()) < 1e-9
    # StandardScaler uses population std, pandas uses sample std by default
    assert abs(df_scaled['CreditScore'].std(ddof=0) - 1) < 1e-9
    # Target column should not be scaled
    assert 'Exited' in df_scaled.columns
    assert abs(df_scaled['Exited'].mean() - sample_dataframe['Exited'].mean()) < 1e-9

def test_prepare_data_pipeline(tmp_path, sample_dataframe):
    """Test the full data preparation pipeline."""
    # Arrange
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()
    output_dir.mkdir()
    
    input_csv = input_dir / "sample.csv"
    sample_dataframe.to_csv(input_csv, index=False)

    # Act
    prepare_data(
        input_path=input_csv,
        output_dir=output_dir,
        test_size=0.2,
        random_state=42,
        target_col='Exited',
        columns_to_remove=['RowNumber', 'CustomerId', 'Surname'],
        categorical_cols=['Geography', 'Gender'],
        stratify=True,
    )

    # Assert
    # Check if all output files were created
    assert (output_dir / 'X_train.csv').exists()
    assert (output_dir / 'X_test.csv').exists()
    assert (output_dir / 'y_train.csv').exists()
    assert (output_dir / 'y_test.csv').exists()
    assert (output_dir / 'encoders.pkl').exists()
    assert (output_dir / 'scaler.pkl').exists()
    assert (output_dir / 'metadata.json').exists()

    # Check shapes of the output data
    x_train = pd.read_csv(output_dir / 'X_train.csv')
    x_test = pd.read_csv(output_dir / 'X_test.csv')
    assert x_train.shape == (8, 10)
    assert x_test.shape == (2, 10)

