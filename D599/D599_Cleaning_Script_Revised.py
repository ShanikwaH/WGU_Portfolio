
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("c:/Users/nikki/OneDrive/1 WGU Courses/MSDADS Courses/D599/Task 1/Employee Turnover Dataset.csv")

# Strip trailing/leading spaces in column names
df.columns = df.columns.str.strip()

# B1/B2 - Inspect Dataset Quality Issues

# Check for duplicate rows
duplicates_count = df.duplicated().sum()
print("Duplicate Rows Found:", duplicates_count)

# Screenshot 1 - Save duplicate count to file
with open("Screenshot_1_Duplicates.txt", "w") as f:
    f.write(f"Duplicate Rows Found: {duplicates_count}")

# Check for negative values in numerical columns
negative_values = (df.select_dtypes(include=["number"]) < 0).sum()
print("Negative Values by Column:")
print(negative_values)

# Screenshot 2 - Save negative values
negative_values.to_csv("Screenshot_2_Negative_Values.csv")

# Check for missing values
missing_values = df.isnull().sum()
print("Missing Values by Column:")
print(missing_values[missing_values > 0])

# Screenshot 3 - Save missing values
missing_values.to_csv("Screenshot_3_Missing_Values.csv")

# Outlier detection for AnnualSalary using IQR method
Q1 = df["AnnualSalary"].quantile(0.25)
Q3 = df["AnnualSalary"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df["AnnualSalary_Outlier"] = (df["AnnualSalary"] < lower_bound) | (df["AnnualSalary"] > upper_bound)
print("AnnualSalary Outliers Found:", df["AnnualSalary_Outlier"].sum())

# Screenshot 4 - Save boxplot of AnnualSalary
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["AnnualSalary"])
plt.title("Boxplot of AnnualSalary")
plt.tight_layout()
plt.savefig("Screenshot_4_Boxplot_AnnualSalary.png")
plt.close()

# View inconsistent values in PaycheckMethod
print("Before Cleaning PaycheckMethod:")
print(df["PaycheckMethod"].value_counts())

# Save screenshot before cleaning
df["PaycheckMethod"].value_counts().to_csv("Screenshot_5a_PaycheckMethod_Before.csv")

# Replace inconsistent category labels
df["PaycheckMethod"] = df["PaycheckMethod"].replace({"Mailed Check": "Mail Check", "DirectDeposit": "Direct Deposit"})

print("After Cleaning PaycheckMethod:")
print(df["PaycheckMethod"].value_counts())

# Save screenshot after cleaning
df["PaycheckMethod"].value_counts().to_csv("Screenshot_5b_PaycheckMethod_After.csv")

# C1 - Data Cleaning and Modifications

# Remove duplicate rows
df = df.drop_duplicates()

# Clean HourlyRate column
df["HourlyRate"] = df["HourlyRate"].replace(r'[$,]', '', regex=True).astype(float)

# Replace negative values with NaN
num_cols = df.select_dtypes(include=["number"]).columns
df[num_cols] = df[num_cols].where(df[num_cols] >= 0)

# Impute missing values
df["AnnualProfessionalDevHrs"] = df["AnnualProfessionalDevHrs"].fillna(df["AnnualProfessionalDevHrs"].median())
df["AnnualSalary"] = df["AnnualSalary"].fillna(df["AnnualSalary"].median())
df["DrivingCommuterDistance"] = df["DrivingCommuterDistance"].fillna(df["DrivingCommuterDistance"].median())
df["NumCompaniesPreviouslyWorked"] = df["NumCompaniesPreviouslyWorked"].fillna(df["NumCompaniesPreviouslyWorked"].median())
df["TextMessageOptIn"] = df["TextMessageOptIn"].fillna("Unknown")

# Optional: Cap outliers using IQR bounds (Winsorization)
df["AnnualSalary"] = np.where(df["AnnualSalary"] > upper_bound, upper_bound,
                              np.where(df["AnnualSalary"] < lower_bound, lower_bound, df["AnnualSalary"]))

# Remove temporary outlier flag
df.drop("AnnualSalary_Outlier", axis=1, inplace=True)

# Save cleaned dataset
df.to_csv("Cleaned_Employee_Turnover_Dataset.csv", index=False)
