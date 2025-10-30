import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report 
import matplotlib.pyplot as plt
import seaborn as sns # Used for better visualizations


# --- 1. Data Loading and Target Transformation ---

# Load the dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y_continuous = housing.target # Median House Value (continuous)

# Convert the continuous target variable into a binary classification problem:
# Define a threshold (e.g., houses with value > $2.0 million are "High Value")
# The MedHouseVal target is in hundreds of thousands of dollars, so 2.0 = $200,000
THRESHOLD = 2.0 
y_binary = (y_continuous > THRESHOLD).astype(int) 

# Check class balance
print(f"Total samples: {len(y_binary)}")
print(f"Class 1 (High Value) count: {y_binary.sum()}")
print(f"Class 0 (Low Value) count: {len(y_binary) - y_binary.sum()}")
print("-------------------------------------------------")


# --- 2. Preprocessing and Splitting ---

# Standardize features (crucial for linear models like Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, 
    y_binary, 
    test_size=0.3, 
    random_state=42
)


# --- 3. Model Training (Logistic Regression) ---

# Logistic Regression is the correct model for linear classification
log_reg = LogisticRegression(solver='liblinear', random_state=42) 
log_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = log_reg.predict(X_test)


# --- 4. Evaluation ---

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model used: Logistic Regression (Linear Classification)")
print(f"Classification Task: Predict if house value > ${THRESHOLD * 100}k")
print("-------------------------------------------------")
print(f"Test Set Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
# The classification report shows precision, recall, and F1-score for each class
print(classification_report(y_test, y_pred, target_names=['Low Value (0)', 'High Value (1)']))


# --- 5. Interpretation of Coefficients ---

# Interpretation: Coefficients indicate the change in the *log-odds* of the house being high value.
feature_coeffs = pd.Series(log_reg.coef_[0], index=X.columns)
print("\nTop 3 Features Predicting 'High Value' (Positive Coefficients):")
print(feature_coeffs.sort_values(ascending=False).head(3).to_string())
# --- ADD THIS CODE TO PROGRAM 1 (Logistic Regression) ---

# 1. Feature Importance Graph (Coefficients)
plt.figure(figsize=(10, 6))
# Get the coefficients and feature names from the trained log_reg model
feature_coeffs = pd.Series(log_reg.coef_[0], index=X.columns)
feature_coeffs.abs().sort_values(ascending=False).head(5).plot(kind='barh', color='darkblue')

plt.title('Top 5 Feature Importance (Absolute Coefficients)')
plt.xlabel('Magnitude of Coefficient')
plt.ylabel('Feature')
plt.gca().invert_yaxis()
plt.show()

# 2. Performance Metrics Bar Chart
from sklearn.metrics import precision_score, recall_score, f1_score

# Recalculate metrics on test set
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred)
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), palette='viridis')
plt.title('Classification Performance Metrics')
plt.ylim(0, 1.0)
plt.show()