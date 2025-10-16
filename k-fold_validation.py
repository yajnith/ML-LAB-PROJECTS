import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- 1. Data Loading and Target Transformation ---

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y_continuous = housing.target # Continuous price target

# Convert to a binary classification target: 
# 1 = High Value (price > $2.0 million), 0 = Low Value
# The target is in units of $100,000s, so 2.0 is the threshold.
THRESHOLD = 2.0 
y_binary = (y_continuous > THRESHOLD).astype(int) 

# --- 2. Preprocessing and K-Fold Setup ---

# Standardize features (essential for linear models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the classification model
log_reg = LogisticRegression(solver='liblinear', random_state=42) 

# Define K-Fold Cross-Validator (K=5)
K = 5
k_folds = KFold(n_splits=K, shuffle=True, random_state=42)

# Define metrics to track during cross-validation
scoring = ['accuracy', 'precision', 'recall', 'f1']

# --- 3. K-Fold Cross-Validation Execution ---

# cross_validate returns a dictionary of scores for each fold
cv_results = cross_validate(
    log_reg, 
    X_scaled, 
    y_binary, 
    cv=k_folds, 
    scoring=scoring,
    return_train_score=False # We only need test scores
)

# --- 4. Evaluation and Reporting ---

print(f"--- Logistic Regression Classification with {K}-Fold Cross-Validation ---")
print(f"Classification Task: Predict if house value > ${THRESHOLD * 100}k")
print("------------------------------------------------------------------------\n")

# Report metrics for each fold and their average
metrics_df = pd.DataFrame({
    'Accuracy (Fold Avg)': [cv_results['test_accuracy'].mean()],
    'Precision (Fold Avg)': [cv_results['test_precision'].mean()],
    'Recall (Fold Avg)': [cv_results['test_recall'].mean()],
    'F1 Score (Fold Avg)': [cv_results['test_f1'].mean()]
})

print(metrics_df.to_string(index=False, float_format='{:.4f}'.format))
print("\n------------------------------------------------------------------------")

# Report standard deviation (measure of model stability across folds)
std_devs = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Std Dev across Folds': [
        cv_results['test_accuracy'].std(), 
        cv_results['test_precision'].std(), 
        cv_results['test_recall'].std(), 
        cv_results['test_f1'].std()
    ]
}

std_dev_df = pd.DataFrame(std_devs)
print("Standard Deviation (Stability Check):")
print(std_dev_df.to_string(index=False, float_format='{:.4f}'.format))