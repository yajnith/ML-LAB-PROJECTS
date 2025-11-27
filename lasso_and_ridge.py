import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Preprocessing imports
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set plot style
sns.set(style="whitegrid")

# -------------------------------------------------
# 1. Load Data
# -------------------------------------------------
# We use a direct URL to the 'train.csv' file from the Kaggle competition
data_url = "https_url_to_ames_housing_train.csv"  # Placeholder
# For a runnable example, let's use a common public version of this dataset:
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv"
# Boston housing column names:
column_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

try:
    data = pd.read_csv(data_url, header=None, sep=',', names=column_names)
    print("Dataset loaded successfully.")

    # Define features (X) and target (y)
    # MEDV (Median value of owner-occupied homes) is our target
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']

except Exception as e:
    print(f"Error loading data: {e}")
    print("Please download the 'train.csv' from the Ames housing link and load it locally.")
    # As a fallback, you could create dummy data here
    # X = ...
    # y = ...

# --- RE-ADAPTATION FOR AMES DATASET (if you load it manually) ---
# If you load the Ames 'train.csv' locally:
# data = pd.read_csv('train.csv')
# X = data.drop(['Id', 'SalePrice'], axis=1)
# # Log-transform the target variable for better performance (it's skewed)
# y = np.log1p(data['SalePrice'])
# -----------------------------------------------------------------


# -------------------------------------------------
# 2. Preprocessing Pipeline
# -------------------------------------------------
# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

# Preprocessing for numerical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values
    ('scaler', StandardScaler())                    # Scale data
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))      # One-hot encode
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -------------------------------------------------
# 3. Create and Train Models
# -------------------------------------------------

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Pipelines for each model
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Lasso(alpha=0.1))  # you can tune this
])

# Train models
print("\nTraining models...")
lr_pipeline.fit(X_train, y_train)
ridge_pipeline.fit(X_train, y_train)
lasso_pipeline.fit(X_train, y_train)
print("Models trained.")

# -------------------------------------------------
# 4. Evaluate Performance
# -------------------------------------------------
models = {
    "Linear Regression": lr_pipeline,
    "Ridge Regression": ridge_pipeline,
    "Lasso Regression": lasso_pipeline
}

print("\n--- Model Performance ---")
results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Model: {name}")
    print(f"  R-squared (R^2): {r2:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}\n")

    results[name] = {'r2': r2, 'rmse': rmse, 'model_obj': model}

# -------------------------------------------------
# 5. Analyze Regularization Effect (Coefficients)
# -------------------------------------------------
print("\n--- Coefficient Analysis ---")

try:
    # Use the *fitted* preprocessor from one of the trained pipelines
    fitted_preprocessor = lr_pipeline.named_steps['preprocessor']

    # Easiest way (for sklearn >= 1.0): get all output feature names directly
    try:
        feature_names = fitted_preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        num_features_out = numerical_features
        cat_features_out = []
        if len(categorical_features) > 0:
            ohe = fitted_preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_features_out = ohe.get_feature_names_out(categorical_features)
        feature_names = list(num_features_out) + list(cat_features_out)

    # Get coefficients from the trained models
    lr_coefs = lr_pipeline.named_steps['model'].coef_
    ridge_coefs = ridge_pipeline.named_steps['model'].coef_
    lasso_coefs = lasso_pipeline.named_steps['model'].coef_

    # Create a DataFrame for comparison
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'LinearReg': lr_coefs,
        'Ridge': ridge_coefs,
        'Lasso': lasso_coefs
    })

    # Basic stats about sparsity
    print(f"Total features after preprocessing: {len(feature_names)}")
    print(f"Linear Regression non-zero coefficients: {np.sum(lr_coefs != 0)}")
    print(f"Ridge Regression non-zero coefficients: {np.sum(ridge_coefs != 0)}")
    print(f"Lasso Regression non-zero coefficients: {np.sum(lasso_coefs != 0)}")
    print(f"Lasso eliminated {np.sum(lasso_coefs == 0)} features.")

    # Plot coefficient magnitudes
    plt.figure(figsize=(15, 6))

    # Lasso coefficients
    plt.subplot(1, 2, 1)
    lasso_coefs_sorted = pd.Series(lasso_coefs, index=feature_names).sort_values()
    # Plot only non-zero for Lasso to see selected features
    lasso_coefs_sorted[lasso_coefs_sorted != 0].plot(
        kind='barh',
        title='Lasso Coefficients (L1)'
    )

    # Ridge coefficients
    plt.subplot(1, 2, 2)
    ridge_coefs_sorted = pd.Series(ridge_coefs, index=feature_names).sort_values()
    ridge_coefs_sorted.plot(
        kind='barh',
        title='Ridge Coefficients (L2)'
    )

    plt.suptitle("Effect of Regularization on Model Coefficients")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

except Exception as e:
    print(f"\nCould not generate coefficient plot: {e}")
    print("This step requires a fitted preprocessor and compatible sklearn version.")
