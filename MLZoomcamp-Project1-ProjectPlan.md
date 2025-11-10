## Data

Data: [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers). USE THIS!

## 1) Set up the project (repo + README)

1. Create a new GitHub repo — e.g., `bank-churn-mlops-pipeline`.
2. Add a `README.md` with:

	* Problem statement: Predict bank customer churn
	* Why it matters: Helps reduce customer loss
	* Example Input → Output
	* Tools: “Azure Machine Learning, MLflow, Docker, Scikit-learn”

## 2) Prepare data (local + cloud)

1. Save a small **sample dataset** in `data/sample.csv` for local testing.
2. Store the **full dataset** in Azure Blob or Data Lake.
3. In Azure ML Studio (or SDK), create a **Data asset** from that file/folder.

## 3) Exploratory Data Analysis

1. Create `notebooks/eda.ipynb` on your Azure ML Compute Instance.
2. Do EDA: distributions, correlations, churn rates, missing values.
3. Document key findings in the notebook and summarize in README.

## 4) Move logic from notebook → scripts

Create a `src/` folder with modular scripts:

```
src/
├── data_prep.py        # Cleans data, feature engineering
├── train.py            # Trains models (baseline + advanced)
├── evaluate.py         # Evaluates and logs metrics to MLflow
└── score.py            # Inference logic for deployment
```

## 5) Define the environment

1. Create `environment/conda.yml`:

	 ```yaml
   name: aml-pipeline
   channels:
	- conda-forge
   dependencies:
	- python=3.9
	- pip
	- pip:
		- pandas
		- scikit-learn
		- xgboost
		- mlflow
		- azureml-mlflow
   ```

2. Register it in AML as an **Environment**.
3. *(Optional)* You can also keep a `uv.lock` or `requirements.txt` for **local setup**:

	 ```
   uv pip compile environment/requirements.in > environment/requirements.txt
   ```

## 6) Local testing before AML (new step!) 🧪

Create a **new file** called:

> `local_test.py`

Purpose: run the pipeline locally on `data/sample.csv` to catch bugs early.

Example flow:

```python
# local_test.py
!python src/data_prep.py --input data/sample.csv --output data/processed/
!python src/train.py --data data/processed/ --out models/local/
!python src/evaluate.py --model models/local/
```

## 7) Create AML components

Turn each script into a component:

1. **data_ingestion** → `data_prep.py`
2. **feature_prep**
3. **train_model**
4. **hyperparam_tuning**
5. **register_model**

Each should be parameterized and reusable.

## 8) Build the Azure ML pipeline

Create a file called:

> `azureml_pipeline.py`

It should:

1. Connect to your AML workspace
2. Wire the components together
3. Submit and monitor the pipeline job
4. Verify all steps complete successfully

## 9) Containerization

Add a `Dockerfile` that can:

* Copy `score.py` + model artifacts
* Expose a scoring endpoint (for `az ml online-endpoint`)

## 10) Deploy and test

1. Register your best model at the end of training.
2. Deploy via **Managed Online Endpoint**.
3. Test with:

	 ```bash
   az ml online-endpoint invoke --name churn-endpoint --request-file sample.json
   ```

4. Add example input/output in the README.

## 11) Document everything

In `README.md`, include:

1. Project summary + dataset
2. How to run locally (`python local_test.py`)
3. How to run pipeline on Azure (`python azureml_pipeline.py`)
4. How to build Docker + deploy
5. Example API call and output
