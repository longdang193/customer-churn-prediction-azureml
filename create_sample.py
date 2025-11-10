"""
Script to create a sample dataset from the full churn.csv file.
This creates a stratified sample that preserves the distribution of the target variable (Exited).
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split


def create_sample(input_file='data/churn.csv', output_file='data/sample.csv', sample_size=1000, random_state=42):
    """
    Create a stratified sample from the full dataset.

    Parameters:
    -----------
    input_file : str
        Path to the full dataset CSV file
    output_file : str
        Path where the sample will be saved
    sample_size : int
        Number of rows in the sample (default: 1000)
    random_state : int
        Random seed for reproducibility
    """
    # Read the full dataset
    print(f"Reading full dataset from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Full dataset shape: {df.shape}")
    print(f"Churn distribution in full dataset:")
    print(df['Exited'].value_counts(normalize=True))

    # Calculate the proportion for stratified sampling
    # We want to preserve the churn rate
    churn_rate = df['Exited'].mean()

    # Calculate sample size (ensure it's not larger than the dataset)
    actual_sample_size = min(sample_size, len(df))

    # Create stratified sample
    # Use train_test_split with stratify to preserve the target distribution
    sample_df, _ = train_test_split(
        df,
        test_size=1 - (actual_sample_size / len(df)),
        stratify=df['Exited'],
        random_state=random_state
    )

    print(f"\nSample dataset shape: {sample_df.shape}")
    print(f"Churn distribution in sample:")
    print(sample_df['Exited'].value_counts(normalize=True))

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the sample
    sample_df.to_csv(output_file, index=False)
    print(f"\nSample saved to {output_file}")

    return sample_df


if __name__ == "__main__":
    # Create a sample of 1000 rows (adjust as needed)
    create_sample(sample_size=1000, random_state=42)
