# Step 4d: Model Scoring/Inference - Complete ✅

Model scoring script has been created and tested successfully.

## What Was Created

### `src/score.py` (257 lines)
Complete inference/scoring pipeline with:
- ✅ Batch scoring from CSV files
- ✅ Single prediction from JSON
- ✅ Automatic preprocessing using saved artifacts
- ✅ Prediction probabilities
- ✅ Classification threshold support
- ✅ CLI interface
- ✅ Production-ready for deployment

## Quick Start

### Batch Scoring (CSV)

```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/new_customers.csv \
  --output predictions/predictions.csv
```

### Single Prediction (JSON)

```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input customer.json \
  --output result.json \
  --json
```

## Features

### 1. Batch Scoring
- Process multiple customers from CSV
- Automatic preprocessing (encoding, scaling)
- Adds prediction columns to output
- Includes probability scores
- Optional accuracy calculation if labels present

### 2. Single Prediction (JSON)
- API-ready format
- Returns prediction dictionary
- Includes probability and class label
- Perfect for web services

### 3. Preprocessing
- Uses saved encoders and scaler
- Handles categorical encoding
- Applies feature scaling
- Removes uninformative columns
- Ensures correct feature order

### 4. Output Formats

**CSV Output:**
```csv
CustomerId,Geography,Gender,...,predicted_churn,churn_probability
12345,France,Female,...,0,0.245
67890,Germany,Male,...,1,0.782
```

**JSON Output:**
```json
{
  "prediction": 0,
  "churn_probability": 0.245,
  "predicted_class": "retain"
}
```

## Usage Examples

### Example 1: Batch Scoring

```bash
# Score new customers
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/new_customers.csv \
  --output predictions/new_customers_predictions.csv
```

**Input CSV:**
```csv
CustomerId,Geography,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary
12345,France,Female,42,2,0,1,1,1,101348.88
```

**Output CSV:**
```csv
CustomerId,Geography,Gender,...,predicted_churn,churn_probability
12345,France,Female,...,1,0.625
```

### Example 2: JSON Scoring (API-like)

**Input JSON (`customer.json`):**
```json
{
  "CreditScore": 619,
  "Geography": "France",
  "Gender": "Female",
  "Age": 42,
  "Tenure": 2,
  "Balance": 0,
  "NumOfProducts": 1,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 101348.88
}
```

**Command:**
```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input customer.json \
  --output result.json \
  --json
```

**Output JSON (`result.json`):**
```json
{
  "prediction": 1,
  "churn_probability": 0.625,
  "predicted_class": "churn"
}
```

### Example 3: Custom Threshold

```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/test.csv \
  --output predictions.csv \
  --threshold 0.6
```

### Example 4: Without Probabilities

```bash
python src/score.py \
  --model models/local/rf_model.pkl \
  --data-dir data/processed \
  --input data/test.csv \
  --output predictions.csv \
  --no-proba
```

## Test Results

### Batch Scoring Test:
```
Input: 19 samples
Output: predictions/test_batch.csv
Results:
  - 6 predicted churn
  - 13 predicted retain
  - Average probability: 0.384
  - Accuracy (if labels available): 94.7%
```

### JSON Scoring Test:
```
Input: Single customer JSON
Output: predictions/test_output.json
Result:
  {
    "prediction": 1,
    "churn_probability": 0.625,
    "predicted_class": "churn"
  }
```

## Integration with Pipeline

Complete end-to-end workflow:

```bash
# 1. Prepare data
python src/data_prep.py --input data/churn.csv --output data/processed

# 2. Train models
python src/train.py --data data/processed --out models/local

# 3. Evaluate model
python src/evaluate.py --model models/local/rf_model.pkl \
  --data data/processed --output evaluation/rf

# 4. Score new customers
python src/score.py --model models/local/rf_model.pkl \
  --data-dir data/processed --input data/new_customers.csv \
  --output predictions/predictions.csv
```

## Deployment Use Cases

### 1. Batch Processing
- Process daily customer lists
- Generate churn risk reports
- Identify at-risk customers

### 2. Real-time API
- Integrate with web services
- Single customer predictions
- REST API endpoint

### 3. Azure ML Endpoint
- Deploy as scoring script
- Batch inference
- Real-time inference

## Key Features

✅ **Automatic Preprocessing**: Uses saved encoders and scaler  
✅ **Batch & Single**: Supports both CSV and JSON  
✅ **Probability Scores**: Includes prediction probabilities  
✅ **Threshold Control**: Adjustable classification threshold  
✅ **Error Handling**: Graceful handling of missing features  
✅ **Production Ready**: Clean output, proper logging  

## Output Columns

### CSV Output:
- All original input columns
- `predicted_churn`: Binary prediction (0 or 1)
- `churn_probability`: Probability of churn (0-1)
- `correct`: Accuracy indicator (if labels present)

### JSON Output:
- `prediction`: Binary prediction (0 or 1)
- `churn_probability`: Probability of churn (0-1)
- `predicted_class`: Human-readable class ("churn" or "retain")

## Requirements

The scoring script requires:
- Trained model (`.pkl` file)
- Preprocessing artifacts (`encoders.pkl`, `scaler.pkl`, `metadata.json`)
- Input data in correct format

## Next Steps

✅ Data preparation (data_prep.py) - DONE  
✅ Model training (train.py) - DONE  
✅ Model evaluation (evaluate.py) - DONE  
✅ Model scoring (score.py) - DONE  

## Status

✅ **COMPLETE** - Model scoring script working for both batch and single predictions

**Ready for deployment in:**
- Azure ML endpoints
- FastAPI services
- Batch processing pipelines
- Real-time inference systems

