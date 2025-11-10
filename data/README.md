# Data Directory

## Files

- `sample.csv` - Sample dataset (1,000 rows) for local development and testing
- `churn.csv` - Full dataset (10,000 rows) uploaded to Azure Blob Storage

## Dataset Source

[Churn for Bank Customers - Kaggle](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers)

## Dataset Description

Bank customer churn prediction dataset with 14 columns:

- **RowNumber** - Row number
- **CustomerId** - Unique customer identifier
- **Surname** - Customer surname
- **CreditScore** - Credit score
- **Geography** - Country (France, Germany, Spain)
- **Gender** - Male/Female
- **Age** - Customer age
- **Tenure** - Years with the bank
- **Balance** - Account balance
- **NumOfProducts** - Number of products
- **HasCrCard** - Has credit card (0/1)
- **IsActiveMember** - Active member (0/1)
- **EstimatedSalary** - Estimated salary
- **Exited** - Target variable (0=stayed, 1=churned)

## Usage

**Local development:** Use `sample.csv`  
**Azure ML pipelines:** Reference the data asset created from `churn.csv`

```python
raw_data = "azureml:churn-data:1"
```

