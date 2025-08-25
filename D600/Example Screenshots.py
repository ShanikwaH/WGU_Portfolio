import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For non-interactive plot saving
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 3/D600 Task 3 Dataset 1 Housing Information.csv")

# Select relevant features
features = [
    'SquareFootage', 'NumBedrooms', 'BackyardSpace', 'CrimeRate', 'SchoolRating',
    'AgeOfHome', 'DistanceToCityCenter', 'EmploymentRate', 'PropertyTaxRate',
    'RenovationQuality', 'LocalAmenities', 'TransportAccess', 'PreviousSalePrice'
]
target = 'Price'

# Filter out invalid records
df = df[df['PreviousSalePrice'] >= 0]

# Standardize the data
X = df[features]
y = df[target]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
standardized_df = pd.DataFrame(X_scaled, columns=features)
standardized_df.to_csv("standardized_data.csv", index=False)

# Apply PCA
pca = PCA(n_components=None)
X_pca_full = pca.fit_transform(X_scaled)
eigenvalues = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_var_ratio)

# Print all eigenvalues and retained PCs
print('Eigenvalues:', eigenvalues)
retained_indices = [i for i, val in enumerate(eigenvalues) if val > 1]
retained_pcs = [f'PC{i+1}' for i in retained_indices]
print(f'Retained principal components (eigenvalue > 1): {retained_pcs}')

# Calculate cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
components = [f"PC{i+1}" for i in range(len(cumulative_variance))]

# Create DataFrame for display
cumulative_variance_df = pd.DataFrame({
    "Principal Component": components,
    "Cumulative Explained Variance": cumulative_variance
})

# Extract relevant model metrics from previously saved variables
metrics = {
    "Adjusted R2": 0.6014,
    "R2": 0.6017,
    "Training MSE": 9_035_310_248.21,
    "Test MSE": 8_484_051_951.54,
    "F-Statistic": 2104.46,
    "F-Statistic P-Value": 0.00
}

# Create DataFrame
metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])

# Only use retained PCs for plots
retained_indices = [i for i, val in enumerate(eigenvalues) if val > 1]
retained_pcs = [f'PC{i+1}' for i in retained_indices]
retained_eigenvalues = eigenvalues[retained_indices]
retained_explained_var = explained_var_ratio[retained_indices]
retained_cumulative_var = np.cumsum(retained_explained_var)

# Scree plot (only retained PCs)
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(retained_eigenvalues)+1), y=retained_eigenvalues, marker='o')
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion')
plt.title('Scree Plot (Eigenvalues) - Retained PCs')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.xticks(ticks=range(1, len(retained_pcs)+1), labels=retained_pcs)
plt.legend()
plt.tight_layout()
plt.savefig("scree_plot_eigenvalues.png")
plt.close()

# Explained Variance plot (only retained PCs)
plt.figure(figsize=(10, 6))
sns.barplot(x=retained_pcs, y=retained_explained_var)
plt.title('Explained Variance by Retained Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("explained_variance_barplot.png")
plt.close()

# Cumulative Explained Variance plot (only retained PCs)
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(1, len(retained_cumulative_var)+1), y=retained_cumulative_var, marker='o')
plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
plt.title('Cumulative Explained Variance by Retained Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.ylim(0, 1.05)
plt.xticks(ticks=range(1, len(retained_pcs)+1), labels=retained_pcs)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 3/E4_Cumulative_Explained_Variance.png")
plt.close()

# Plot bar chart of model metrics
plt.figure(figsize=(12, 6))
sns.barplot(x="Metric", y="Value", data=metrics_df)
plt.title("Model Metrics Overview")
plt.xticks(rotation=45)
plt.tight_layout()

# Save plot
plt.savefig("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 3/model_metrics_chart.png")
plt.close()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Add constant for statsmodels
X_train_const = sm.add_constant(X_train)

# Fit OLS model
optimized_model = sm.OLS(y_train, X_train_const).fit()
y_train_pred = optimized_model.predict(X_train_const)
residuals = y_train - y_train_pred

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Residuals vs Fitted
ax1.scatter(y_train_pred, residuals, alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_title('Residuals vs Fitted Values')
ax1.set_xlabel('Fitted Values')
ax1.set_ylabel('Residuals')

# Histogram of residuals
ax2.hist(residuals, bins=30, edgecolor='black')
ax2.set_title('Residual Distribution')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("residual_analysis.png")
plt.close(fig)  # Explicitly close figure