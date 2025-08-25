# A. GitLab Repository Setup
# This script is tracked in GitLab. Commit messages were created for each major task in C2 through D4. A link to the repository and commit history will be submitted with the assessment.

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# B. Research Question and Goal
# Research Question: To what extent do housing features such as SquareFootage, Garage, RenovationQuality, Price, SchoolRating, and Fireplace predict whether a home is classified as luxury (IsLuxury)?
# Goal: Use logistic regression to analyze which housing features increase the probability of a home being luxury.

# C1. Data Preparation – Variable Selection and Justification
# Dependent variable: IsLuxury
# Independent variables: A mix of quantitative and categorical: SquareFootage, Garage, RenovationQuality, Price, SchoolRating, Fireplace
df = pd.read_csv("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 2/D600 Task 2 Dataset 1 Housing Information.csv")

# C1. Encode categorical variables
df['Garage'] = df['Garage'].map({'Yes': 1, 'No': 0})
df['Fireplace'] = df['Fireplace'].map({'Yes': 1, 'No': 0})

# Define variables
X = df[['SquareFootage', 'Garage', 'RenovationQuality', 'Price', 'SchoolRating', 'Fireplace']]
y = df['IsLuxury']

# C2. Descriptive statistics summary
variables = ['Price', 'SquareFootage', 'SchoolRating', 'BackyardSpace', 'Fireplace', 'Garage', 'RenovationQuality']
print("Descriptive Statistics Summary:\n")
print(df[variables].describe(include='all'))

# Frequency distributions for categorical/binary variables
categorical_vars = ['Garage', 'Fireplace', 'IsLuxury']
for var in categorical_vars:
    print(f"\nFrequency distribution for {var}:\n{df[var].value_counts()}")

# Also include mode separately for categorical/binary variables
for col in ['Fireplace', 'Garage']:
    mode = df[col].mode()[0]
    print(f"\nMode for {col}: {mode}")

# C3. Visualizations (univariate and bivariate) are generated separately and included in the report as .png images.

# D1. Split data into training and test sets (75/25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# D2. Backward Stepwise Elimination
while True:
    model = sm.Logit(y_train, sm.add_constant(X_train)).fit()
    p_values = model.pvalues.drop("const")
    if p_values.max() > 0.05:
        worst_feature = p_values.idxmax()
        print(f"Dropping {worst_feature} with p-value {p_values.max()}")
        X_train = X_train.drop(columns=worst_feature)
        X_test = X_test.drop(columns=worst_feature)
    else:
        break

# Final model with only significant predictors
X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
model = sm.Logit(y_train, X_train_const).fit()
print(model.summary())

# D3. Confusion matrix and accuracy for training data
y_train_pred = model.predict(X_train_const) >= 0.5
print("\nTraining Confusion Matrix (Optimized):")
print(confusion_matrix(y_train, y_train_pred))
print("Training Accuracy (Optimized):", accuracy_score(y_train, y_train_pred))

# D4. Confusion matrix and accuracy for test data
y_test_pred = model.predict(X_test_const) >= 0.5
print("\nTest Confusion Matrix (Optimized):")
print(confusion_matrix(y_test, y_test_pred))
print("Test Accuracy (Optimized):", accuracy_score(y_test, y_test_pred))

# E5. Assumption Check: Multicollinearity via correlation heatmap
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Independent Variables")
plt.tight_layout()
plt.show()

# E1. Libraries used:
# pandas – data manipulation
# statsmodels – regression modeling with p-values and model summary
# sklearn – data splitting and accuracy metrics
# seaborn/matplotlib – data visualization

# E2–E3. Optimization used backward stepwise elimination by manually removing predictors with p > 0.05

# E4–E5. Logistic regression assumptions checked via data types, correlation, visualizations

# E6. Regression equation discussed in the report using the format: log(p/(1−p)) = β0 + β1x1 + ... + βnxn

# E7–E9. Model metrics, insights, and real-world implications are discussed in the final report

# F. Panopto
# This code was demonstrated during the Panopto recording, showing environment setup, regression model results, and interpretation.

# G. Citations
# The only sources used were the official course materials from WGU.

# H. Professional Communication
# Report and code have been reviewed for spelling, grammar, clarity, and format using Grammarly and peer review.
