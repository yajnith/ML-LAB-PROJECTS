import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preprocessing ---

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data[['MedInc', 'AveRooms']] # Use only 2 features for visualization clarity
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- 2. Model Training (Demonstrates 2-Way Splitting) ---

# Decision Trees are built via a series of binary (2-way) splits.
# We limit the max_depth to 3 to clearly visualize the splitting process.
dt_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_reg.fit(X_train, y_train)

# --- 3. Evaluation ---

y_pred = dt_reg.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("--- Decision Tree Regressor (Max Depth 3) ---")
print(f"Test RMSE (Root Mean Squared Error): ${rmse:.4f} million")
print("\nModel trained using iterative binary (2-way) splits.")

# --- 4. Visualization of the Splitting Structure ---

# Visualize the resulting tree structure to see the binary splits
plt.figure(figsize=(18, 10))
plot_tree(
    dt_reg, 
    feature_names=X.columns.tolist(), 
    filled=True, 
    rounded=True, 
    fontsize=10,
    impurity=False,
    precision=2
)
plt.title("Decision Tree Structure: Binary (2-Way) Splitting")
plt.show()

# Interpretation: Examine the tree visualization. Each node shows a condition 
# (e.g., MedInc <= 4.96) which always results in exactly two branches (True/False),
# confirming the binary split nature.