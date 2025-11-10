# Creating a Sample Dataset

This guide shows you different ways to create a sample dataset from `data/churn.csv` for local testing.

## Method 1: Using the Python Script (Recommended)

If you have Python and pandas installed, use the provided script:

```bash
python create_sample.py
```

This will:
- Create a stratified sample of 1000 rows (preserving the churn distribution)
- Save it to `data/sample.csv`
- Preserve the original target variable distribution

### Customizing the Sample Size

Edit `create_sample.py` and change the `sample_size` parameter:

```python
create_sample(sample_size=500, random_state=42)  # For a smaller sample
```

## Method 2: Using Python Interactively

If you prefer to run it interactively (e.g., in Jupyter or Python REPL):

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the full dataset
df = pd.read_csv('data/churn.csv')

# Create a stratified sample (preserves churn distribution)
sample_df, _ = train_test_split(
    df,
    test_size=0.9,  # Keep 10% of data (adjust as needed)
    stratify=df['Exited'],
    random_state=42
)

# Save the sample
sample_df.to_csv('data/sample.csv', index=False)
print(f"Sample shape: {sample_df.shape}")
```

## Method 3: Simple Random Sample (Without Stratification)

If you don't need to preserve the target distribution:

```python
import pandas as pd

# Read the full dataset
df = pd.read_csv('data/churn.csv')

# Create a simple random sample
sample_df = df.sample(n=1000, random_state=42)

# Save the sample
sample_df.to_csv('data/sample.csv', index=False)
```

## Method 4: Using pandas with head() (Quick but not recommended)

For a quick test with the first N rows (not representative):

```python
import pandas as pd

df = pd.read_csv('data/churn.csv')
sample_df = df.head(1000)  # First 1000 rows
sample_df.to_csv('data/sample.csv', index=False)
```

## Recommended Sample Size

- **Local testing**: 500-1000 rows
- **Development**: 1000-2000 rows
- **Full dataset**: Use the original `data/churn.csv` (10,000+ rows)

## Verify Your Sample

After creating the sample, verify it maintains the churn distribution:

```python
import pandas as pd

# Load full dataset
full_df = pd.read_csv('data/churn.csv')
print("Full dataset churn rate:", full_df['Exited'].mean())

# Load sample
sample_df = pd.read_csv('data/sample.csv')
print("Sample churn rate:", sample_df['Exited'].mean())

# They should be similar!
```

## Installation Requirements

If you need to install dependencies:

```bash
pip install pandas scikit-learn
```

Or if using conda:

```bash
conda install pandas scikit-learn
```

