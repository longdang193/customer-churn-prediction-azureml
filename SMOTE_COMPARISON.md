# SMOTE Impact Analysis

Comparison of model performance with and without SMOTE (Synthetic Minority Oversampling Technique).

## What is SMOTE?

SMOTE synthesizes new samples for the minority class to balance the training dataset. This helps the model learn better decision boundaries for the underrepresented class.

**Key Points:**
- ✅ Applied **only to training data**
- ✅ Test data kept in **original imbalanced distribution**
- ✅ Generates synthetic samples using k-nearest neighbors
- ✅ Prevents overfitting to majority class

## Training Data Distribution

### Without SMOTE:
```
Class 0 (Retained): 6,370 samples (79.6%)
Class 1 (Churned):  1,630 samples (20.4%)
Total:              8,000 samples
```

### With SMOTE:
```
Class 0 (Retained): 6,370 samples (50.0%)
Class 1 (Churned):  6,370 samples (50.0%)  ← Generated 4,740 synthetic samples
Total:             12,740 samples
```

## Test Data (Unchanged):
```
Class 0 (Retained): 1,593 samples (79.6%)
Class 1 (Churned):    407 samples (20.4%)
Total:              2,000 samples
```

## Model Performance Comparison

### Logistic Regression

| Metric | Without SMOTE | With SMOTE | Change |
|--------|---------------|------------|--------|
| Accuracy | 0.708 | 0.708 | ±0.000 |
| Precision | 0.383 | 0.384 | +0.001 |
| Recall | 0.717 | 0.720 | +0.003 |
| F1 Score | 0.500 | 0.501 | +0.001 |
| ROC-AUC | 0.774 | 0.773 | -0.001 |

**Analysis:** Minimal impact on Logistic Regression. This is expected as LogReg already handles class imbalance reasonably well with class weights.

### Random Forest

| Metric | Without SMOTE | With SMOTE | Change |
|--------|---------------|------------|--------|
| Accuracy | 0.831 | 0.829 | -0.002 |
| Precision | 0.571 | 0.565 | -0.006 |
| Recall | 0.676 | 0.681 | +0.005 |
| F1 Score | 0.619 | 0.618 | -0.001 |
| ROC-AUC | 0.858 | 0.852 | -0.006 |

**Analysis:** Similar performance. RF with balanced class weights already performs well. SMOTE provides a slight improvement in recall (better at catching churners) at the cost of slightly lower precision.

## Confusion Matrix Analysis

### Random Forest - Without SMOTE:
```
                 Predicted
                 Retained  Churned
Actual Retained    1386      207     (87% correct)
       Churned      132      275     (68% correct)
```

### Random Forest - With SMOTE:
```
                 Predicted
                 Retained  Churned
Actual Retained    1381      212     (87% correct)
       Churned      130      277     (68% correct)
```

**Key Finding:** Very similar confusion matrices. Both approaches identify ~68% of churners correctly.

## When to Use SMOTE

### ✅ Use SMOTE When:
- Severe class imbalance (>90%)
- Minority class has very few samples (<100)
- Model struggles to learn minority class patterns
- Using algorithms that don't have built-in class weight handling

### ⚠️ Consider Alternatives When:
- Moderate imbalance (<80/20) - class weights may suffice
- Already getting good recall on minority class
- Risk of overfitting to synthetic samples
- Large dataset where sampling is more practical

## Our Case: Bank Churn (20% minority class)

**Recommendation:** 
- **Class weights** (current approach) work well for this dataset
- **SMOTE** provides marginal improvements
- Both approaches yield F1 ~0.62 with RF

**Best Practice:**
1. Start with class weights (simpler, faster)
2. Try SMOTE if minority class performance is poor
3. Evaluate both on validation set
4. Choose based on business requirements (precision vs recall)

## Usage

### Without SMOTE (Default):
```bash
python src/train.py --data data/processed --out models/local
```

### With SMOTE:
```bash
python src/train.py --data data/processed --out models/with_smote --use-smote
```

### With SMOTE + MLflow:
```bash
python src/train.py --data data/processed --out models/smote_exp --use-smote --use-mlflow
```

## Technical Implementation

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Test data remains unchanged
# Evaluate on original imbalanced test set
```

## Key Takeaways

1. ✅ **SMOTE works correctly** - balanced training data from 8K to 12.7K samples
2. ✅ **Test data unchanged** - validates on original distribution (79%/21%)
3. ✅ **Similar performance** - Both approaches yield competitive results
4. ✅ **Proper implementation** - Applied only after train/test split
5. ✅ **No data leakage** - SMOTE fitted only on training data

## Conclusion

For this bank churn dataset (20% minority class):
- Both **class weights** and **SMOTE** achieve similar performance (~0.62 F1)
- SMOTE generates 4,740 synthetic samples but provides minimal gain
- **Recommendation:** Use class weights for simplicity, try SMOTE for experimentation
- Monitor precision/recall trade-offs based on business cost of false positives/negatives

The infrastructure is now in place to experiment with both approaches using the `--use-smote` flag.

