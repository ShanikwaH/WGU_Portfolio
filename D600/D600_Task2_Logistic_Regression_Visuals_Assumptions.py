
# A. GitLab Repository Setup 
# This script is tracked in GitLab. Commit messages were created for each major task in C2 through D4. A link to the repository and commit history will be submitted with the assessment.

import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# B. Research Question and Goal
# Research Question: To what extent do housing features such as SquareFootage, Garage, RenovationQuality, Price, SchoolRating, and Fireplace predict whether a home is classified as luxury (IsLuxury)?
# Goal: Use logistic regression to analyze which housing features increase the probability of a home being luxury.

# C1. Data Preparation – Variable Selection and Justification
df = pd.read_csv("C:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D600/Task 2/D600 Task 2 Dataset 1 Housing Information.csv")
df['Garage'] = df['Garage'].map({'Yes': 1, 'No': 0})
df['Fireplace'] = df['Fireplace'].map({'Yes': 1, 'No': 0})

X = df[['SquareFootage', 'Garage', 'RenovationQuality', 'Price', 'SchoolRating', 'Fireplace']]
y = df['IsLuxury']

# C2. Descriptive statistics
variables = ['Price', 'SquareFootage', 'SchoolRating', 'BackyardSpace', 'Fireplace', 'Garage', 'RenovationQuality']
print("Descriptive Statistics Summary:\n")
print(df[variables].describe(include='all'))
for col in ['Fireplace', 'Garage']:
    print(f"Mode for {col}: {df[col].mode()[0]}")

# C3. Visualizations
for col in X.columns:
    plt.figure()
    sns.histplot(X[col], kde=True)
    plt.title(f'Univariate Distribution: {col}')
    plt.tight_layout()
    plt.show()

for col in X.columns:
    plt.figure()
    sns.boxplot(x=y, y=X[col])
    plt.title(f'{col} by IsLuxury')
    plt.tight_layout()
    plt.show()

# D1. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# D2. Backward Stepwise Elimination
while True:
    model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=False)
    pvals = model.pvalues.drop("const")
    if pvals.max() > 0.05:
        to_drop = pvals.idxmax()
        print(f"Dropping {to_drop} (p={pvals.max()})")
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    else:
        break

X_train_const = sm.add_constant(X_train)
X_test_const = sm.add_constant(X_test)
model = sm.Logit(y_train, X_train_const).fit()
print(model.summary())

# D3. Metrics
y_train_pred = model.predict(X_train_const) >= 0.5
y_test_pred = model.predict(X_test_const) >= 0.5

print("Training Confusion Matrix:")
print(confusion_matrix(y_train, y_train_pred))
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

print("Test Confusion Matrix:")
print(confusion_matrix(y_test, y_test_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# E5. Multicollinearity check
sns.heatmap(X.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Assumption Check: Linearity of the logit
df_filtered = df[X_train.columns]  # Only use columns from final model
pred_probs = model.predict(sm.add_constant(df_filtered))
logit_vals = np.log(pred_probs / (1 - pred_probs))
for col in df_filtered.columns:
    plt.figure()
    sns.scatterplot(x=df_filtered[col], y=logit_vals)
    plt.title(f'Logit vs {col}')
    plt.tight_layout()
    plt.show()
