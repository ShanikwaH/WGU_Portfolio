# D600 Task 3 - Linear Regression with PCA

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for file saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

# A: Load the dataset
df = pd.read_csv("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 3/D600 Task 3 Dataset 1 Housing Information.csv")

# B1-B2: Define research question and analysis goal (done in written report)

# C1-C2: PCA Use & PCA Assumption (done in written report)

# D1: Select and clean continuous explanatory variables(features) response_var(target)
features = [
    'SquareFootage', 'NumBedrooms', 'BackyardSpace', 'CrimeRate', 'SchoolRating',
    'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'PropertyTaxRate',
    'RenovationQuality', 'LocalAmenities', 'TransportAccess', 'PreviousSalePrice'
]
target = 'Price'

# Filter out invalid records
df = df[df['PreviousSalePrice'] >= 0]

# D2: Standardize explanatory variables 
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
standardized_df = pd.DataFrame(X_scaled, columns=features)
standardized_df.to_csv("standardized_data.csv", index=False)

# D3: Descriptive statistics (before standardization)
desc_stats = df[features + [target]].describe().T
desc_stats.to_csv("descriptive_stats.csv")

# E2: Determine number of principal components to retain (eigenvalue > 1)
pca = PCA(n_components=None)
X_pca_full = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_var_ratio) # Cumulative explained variance

# Print all eigenvalues and retained PCs
print('Eigenvalues:', eigenvalues)
retained_indices = [i for i, val in enumerate(eigenvalues) if val > 1]
retained_pcs = [f'PC{i+1}' for i in retained_indices]
print(f'Retained principal components (eigenvalue > 1): {retained_pcs}')
print(f'Number of retained principal components: {len(retained_pcs)}')

# Use only retained PCs for all downstream analysis
X_pca = X_pca_full[:, retained_indices]
pca_components_df = pd.DataFrame(
    X_pca,
    columns=retained_pcs
)

# Re-run PCA with correct component count
pca = PCA(n_components=len(retained_indices))
X_pca = pca.fit_transform(X_scaled)
explained_var_ratio_retained = pca.explained_variance_ratio_
cumulative_variance_retained = np.cumsum(explained_var_ratio_retained)

# E1. Save loading matrix
loading_matrix = pd.DataFrame(
    pca.components_.T,
    columns=retained_pcs,
    index=features
)
loading_matrix.to_csv("loading_matrix.csv")

# E3: Save explained variance of retained components
variance_df = pd.DataFrame({
    "Principal Component": retained_pcs,
    "Explained Variance": explained_var_ratio_retained,
    "Cumulative Explained Variance": cumulative_variance_retained
})
variance_df.to_csv("variance_explained.csv", index=False)

# F1: Split dataset and include response variable
X = X_pca
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_df = pd.DataFrame(X_train, columns=retained_pcs)
train_df[target] = y_train
train_df.to_csv("training_dataset.csv", index=False)

test_df = pd.DataFrame(X_test, columns=retained_pcs)
test_df[target] = y_test
test_df.to_csv("test_dataset.csv", index=False)

# F2. Model Optimization using backward stepwise elimination
import statsmodels.api as sm
import pandas as pd

# Backward Stepwise Elimination function
def backward_elimination(X, y, significance_level=0.05):
    X = X.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    features = list(X.columns)
    while True:
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()
        pvalues = model.pvalues.iloc[1:]  # exclude intercept
        if pvalues.empty:
            break
        max_pval = pvalues.max()
        if max_pval > significance_level:
            worst_feature = pvalues.idxmax()
            features.remove(worst_feature)
        else:
            break
    return model, features

X_train_df = pd.DataFrame(X_train, columns=retained_pcs)
y_train_aligned = pd.Series(y_train).reset_index(drop=True)
X_train_df_aligned = X_train_df.reset_index(drop=True)
X_test_df = pd.DataFrame(X_test, columns=retained_pcs)
optimized_model, selected_features = backward_elimination(X_train_df, y_train)

print(f"Selected features: {selected_features}")

# Final optimized model
X_df = pd.DataFrame(X_train, columns=retained_pcs)
X_train_opt = sm.add_constant(X_df[selected_features].reset_index(drop=True))
y_train_series = pd.Series(y_train).reset_index(drop=True)
y_train_aligned = pd.Series(y_train).reset_index(drop=True)
optimized_model = sm.OLS(y_train_aligned, X_train_opt).fit()

# Extract and print key metrics
print("Extracted Model Parameters:")
print(f"Adjusted R²: {optimized_model.rsquared_adj}")
print(f"R²: {optimized_model.rsquared}")
print(f"F-statistic: {optimized_model.fvalue}")
print(f"p-value (F-statistic): {optimized_model.f_pvalue}")
print("Coefficients:", optimized_model.params)
print("P-values:", optimized_model.pvalues)

# Save regression equation
coefficients = optimized_model.params
equation = f"Predicted Price = {coefficients['const']:.2f}"
for feature in selected_features:
    equation += f" + {coefficients[feature]:.2f} * {feature}"
print("Final Regression Equation:", equation)

# Save regression coefficients and p-values
coefficients_df = pd.DataFrame({
    "Principal Component": optimized_model.params.index.tolist(),
    "Coefficient": optimized_model.params.tolist(),
    "P-Value": optimized_model.pvalues.tolist()
})
coefficients_df.to_csv("regression_coefficients.csv", index=False)

# Print summary
print(optimized_model.summary())

# F3. Predict on training data
X_train_opt = sm.add_constant(X_df[selected_features])
y_pred = optimized_model.predict(X_train_opt)
mse_train = mean_squared_error(y_train, y_pred)
print("Training MSE:", mse_train)

# F4. Predict on test data
X_test_df = pd.DataFrame(X_test, columns=retained_pcs)
X_test_selected = sm.add_constant(X_test_df[selected_features])
y_pred = optimized_model.predict(X_test_selected)
mse_test = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse_test)

# G5. Save model metrics
metrics = {
    "Adjusted R2": optimized_model.rsquared_adj,
    "R2": optimized_model.rsquared,
    "Training MSE": mse_train,
    "Test MSE": mse_test,
    "F-Statistic": optimized_model.fvalue,
    "F-Statistic P-Value": optimized_model.f_pvalue
}
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
metrics_df.to_csv("model_metrics.csv")

# H. Panopto
# This code was demonstrated during the Panopto recording, showing environment setup, regression model results, and interpretation.

# I. Citations
# The only sources used were the official course materials from WGU.

# J. Professional Communication
# Report and code have been reviewed for spelling, grammar, clarity, and format using Grammarly and peer review.