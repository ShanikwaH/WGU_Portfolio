# D599 Task 2 - Statistical Analysis Script (Enhanced)

import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("Health Insurance Dataset 2025-05-04 00_21_56.xlsx", sheet_name="insurance")

# Clean for analysis
df_clean = df[['age', 'bmi', 'sex', 'smoker', 'Level']].dropna()

# -----------------------------
# UNIVARIATE ANALYSIS
# -----------------------------
print("Summary statistics for age and bmi:")
print(df_clean[['age', 'bmi']].describe())

print("Frequency for sex and smoker:")
print(df_clean['sex'].value_counts())
print(df_clean['smoker'].value_counts())

# -----------------------------
# PARAMETRIC TEST: t-test
# -----------------------------
print("\nRunning t-test for BMI vs Smoker...")

bmi_smoker_df = df[['bmi', 'smoker']].dropna()
smokers = bmi_smoker_df[bmi_smoker_df['smoker'] == 'yes']['bmi']
nonsmokers = bmi_smoker_df[bmi_smoker_df['smoker'] == 'no']['bmi']

t_stat, p_val = stats.ttest_ind(smokers, nonsmokers, equal_var=False)

print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
print(f"Mean BMI (Smokers): {smokers.mean():.2f}")
print(f"Mean BMI (Non-Smokers): {nonsmokers.mean():.2f}")

# -----------------------------
# PARAMETRIC ASSUMPTION TEST
# -----------------------------
print("\nRunning Shapiro-Wilk test for BMI normality...")
shapiro_stat, shapiro_p = stats.shapiro(df_clean['bmi'])
print(f"Shapiro-Wilk Test: Statistic = {shapiro_stat:.4f}, P-value = {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("BMI appears to be normally distributed (p > 0.05).")
else:
    print("BMI may not be normally distributed (p <= 0.05).")

# -----------------------------
# NONPARAMETRIC TEST: Chi-Square
# -----------------------------
print("\nRunning Chi-Square test for Level vs Smoker...")

level_smoker_df = df[['Level', 'smoker']].dropna()
contingency = pd.crosstab(level_smoker_df['Level'], level_smoker_df['smoker'])

chi2, chi2_p, chi2_dof, chi2_expected = stats.chi2_contingency(contingency)

print(f"Chi-square statistic: {chi2:.4f}, P-value: {chi2_p:.4f}, DOF: {chi2_dof}")

# -----------------------------
# CRAMER'S V for Level vs Smoker
# -----------------------------
def cramers_v(conf_matrix):
    chi2 = stats.chi2_contingency(conf_matrix)[0]
    n = conf_matrix.sum().sum()
    r, k = conf_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))

print("\nCalculating Cramer's V for Level vs Smoker...")
cramer_v_result = cramers_v(contingency)
print(f"Cramer's V: {cramer_v_result:.4f}")