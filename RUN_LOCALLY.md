# Running EDA Notebook in Codespace

The EDA notebook is configured to run locally in Codespace without any Azure dependencies.

## Quick Start

### 1. Open the Notebook

In VS Code (Codespace):
- Navigate to `notebooks/eda.ipynb`
- Click to open it
- VS Code will automatically detect it's a Jupyter notebook

### 2. Select Python Kernel

- Click "Select Kernel" in the top-right
- Choose your Python environment (usually "Python 3.12.1" or similar)

### 3. Run the Notebook

**Option A: Run All Cells**
- Click "Run All" button at the top
- Or use: `Shift + Enter` to run cells one by one

**Option B: Run from Terminal**
```bash
cd notebooks
jupyter notebook eda.ipynb
```

## What Happens Automatically

✅ **Environment Detection:** Notebook detects it's running in Codespace (not Azure)  
✅ **Data Loading:** Loads from `../data/churn.csv` automatically  
✅ **No Azure Login:** No authentication needed  
✅ **All Libraries:** pandas, numpy, matplotlib, seaborn already available  

## Expected Output

```
✓ Data loaded locally: churn.csv
Dataset shape: (10000, 14)
Churn Rate: 20.37%
```

## Troubleshooting

**If libraries are missing:**
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
```

**If data file not found:**
- Check that `data/churn.csv` exists in the project root
- The notebook will automatically fall back to `sample.csv` if needed

**To view plots in terminal:**
```bash
cd notebooks
jupyter nbconvert --to html --execute eda.ipynb
# Opens eda.html with all visualizations
```

## Differences from Azure ML

| Aspect | Codespace (Local) | Azure ML Compute |
|--------|------------------|------------------|
| Data Source | `../data/churn.csv` | Azure ML data asset |
| Authentication | None needed | Azure credentials |
| Environment Detection | `AZUREML_RUN_ID` not set | `AZUREML_RUN_ID` set |
| Cost | Free (GitHub) | Azure compute costs |

The same notebook works in both environments automatically! 🎉

