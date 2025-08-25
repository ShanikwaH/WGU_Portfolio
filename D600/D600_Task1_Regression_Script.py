# ========================================================
# A. GitLab Setup and Workflow
# ========================================================
# This script is committed to GitLab with a message after each of the following major rubric sections:

# C2 (descriptive stats), C3 (visualizations), D1–D4 (model building and evaluation).
# Submit your GitLab URL and branch history with commit messages when submitting the assessment.

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ========================================================
# B. Purpose of the Data Analysis
# ========================================================
# Goal: Predict housing prices using regression based on relevant home features (both numerical and categorical).
# Research Question: What is the impact of SquareFootage, SchoolRating, DistanceToCityCenter, AgeOfHome,
# Fireplace, Garage, and HouseColor on home prices?

# ========================================================
# C.1: Variable Identification and Justification
# ========================================================
df = pd.read_csv("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 1/D600 Task 1 Dataset 1 Housing Information.csv")
df_model = df[['Price', 'SquareFootage', 'SchoolRating', 'DistanceToCityCenter',
               'AgeOfHome', 'Fireplace', 'Garage', 'HouseColor']].copy()

# ========================================================
# C.2: Data Cleaning and Descriptive Statistics
# ========================================================
df_model['Price'] = df_model['Price'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df_model['Price'] = pd.to_numeric(df_model['Price'], errors='coerce')

for col in ['SquareFootage', 'SchoolRating', 'DistanceToCityCenter', 'AgeOfHome']:
    df_model[col] = df_model[col].astype(str).str.replace(',', '', regex=False).str.strip()
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')

for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = df_model[col].str.strip()

df_model.dropna(inplace=True)

# ========================================================
# Additional C2: Output Descriptive Statistics
# ========================================================

print("\n=== Descriptive Statistics for Numeric Variables ===")
print(df_model[['Price', 'SquareFootage', 'SchoolRating', 'DistanceToCityCenter', 'AgeOfHome']].describe())

print("\n=== Frequency Distribution for Categorical Variables ===")
for col in ['Fireplace', 'Garage', 'HouseColor']:
    print(f"\n{col} value counts (proportions):")
    
    # Normalize formatting for consistent output
    cleaned = df_model[col].astype(str).str.strip().str.lower()
    
    # Print value counts
    print(cleaned.value_counts(normalize=True).rename_axis('Category').reset_index(name='Proportion'))

input("\nPress Enter to exit...")

# ========================================================
# C.3: One-hot Encode Categorical Variables and Prepare for Modeling
# ========================================================
df_encoded = pd.get_dummies(df_model, columns=['Fireplace', 'Garage', 'HouseColor'], drop_first=True)
X = df_encoded.drop(columns=['Price'])
y = df_encoded['Price']

# ========================================================
# D.1: Split the Data into Train/Test Sets
# ========================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================================================
# D.2: Build and Optimize Regression Model (Significant Variables Only)
# ========================================================
# Manually chosen significant variables (p < 0.05)
significant_vars = ['SquareFootage', 'SchoolRating', 'DistanceToCityCenter', 'AgeOfHome']
X_train_sig = X_train[significant_vars].astype(float)
X_train_sig_const = sm.add_constant(X_train_sig)

X_train_sig_const = X_train_sig_const.dropna()
y_train_sig = y_train[X_train_sig_const.index]

print("Fitting optimized model with only significant variables...")
optimized_model = sm.OLS(y_train_sig, X_train_sig_const).fit()

print("\n--- Optimized Model Summary ---")
print(optimized_model.summary())

# ========================================================
# D.3: Training MSE
# ========================================================
train_pred = optimized_model.predict(X_train_sig_const)
mse_train = mean_squared_error(y_train_sig, train_pred)
print(f"\nTraining MSE: {mse_train:.2f}")

# ========================================================
# D.4: Test Set Prediction and Accuracy (Test MSE)
# ========================================================
X_test_sig = X_test[significant_vars].astype(float)
X_test_sig_const = sm.add_constant(X_test_sig)

X_test_sig_const = X_test_sig_const.dropna()
y_test_cleaned = y_test[X_test_sig_const.index]

test_pred = optimized_model.predict(X_test_sig_const)
mse_test = mean_squared_error(y_test_cleaned, test_pred)
print(f"Testing MSE: {mse_test:.2f}")

# ========================================================
# E. Final Report Summary and Evaluation Outputs
# ========================================================

# ========================================================
# F. Panopto Recording
# ========================================================
# Record yourself in Panopto: show code execution and explain libraries, modeling, and metrics.

# ========================================================
# G. Sources
# ========================================================
# The only sources used were the official course materials from WGU.

# ========================================================
# H. Professional Communication
# ========================================================
# Reviewed using Grammarly; content is presented clearly and professionally.
