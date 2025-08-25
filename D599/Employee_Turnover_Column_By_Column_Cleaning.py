# Column-by-Column Data Cleaning Script for Employee Turnover Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("Cleaned_Employee_Turnover_Dataset.csv")
print(df.head())

# Show info about all columns
print("\nDataset Info:")
print(df.info())

# Define columns to inspect
columns = [
    "EmployeeNumber", "Age", "Tenure", "Turnover", "HourlyRate", "HoursWeekly", "CompensationType",
    "AnnualSalary", "DrivingCommuterDistance", "JobRoleArea", "Gender", "MaritalStatus", "NumCompaniesPreviouslyWorked",
    "AnnualProfessionalDevHrs", "TextMessageOptIn", "PaycheckMethod"
]

# Column-by-column check
for col in columns:
    print(f"\n--- Column: {col} ---")
    print("Data type:", df[col].dtype)
    print("Missing values:", df[col].isnull().sum())

    if df[col].dtype in ['int64', 'float64']:
        print("Negative values:", (df[col] < 0).sum())
        print(df[col].describe())
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(f"Boxplot_{col}.png")
        plt.close()
    else:
        print(df[col].value_counts())

# --- Cleaning column: EmployeeNumber ---
print('\nEmployeeNumber unique values:')
print(df['EmployeeNumber'].unique())

# --- Cleaning column: Age ---
print('\nAge unique values:')
print(df['Age'].unique())

# --- Cleaning column: Tenure ---
print('\nTenure unique values:')
print(df['Tenure'].unique())

# --- Cleaning column: Turnover ---
print('\nTurnover unique values:')
print(df['Turnover'].unique())

# --- Cleaning column: HourlyRate ---
print('\nHourlyRate unique values:')
print(df['HourlyRate'].unique())

# --- Cleaning column: HoursWeekly ---
print('\nHoursWeekly unique values:')
print(df['HoursWeekly'].unique())

# --- Cleaning column: CompensationType ---
print('\nCompensationType unique values:')
print(df['CompensationType'].unique())

# --- Cleaning column: AnnualSalary ---
print('\nAnnualSalary unique values:')
print(df['AnnualSalary'].unique())

# --- Cleaning column: DrivingCommuterDistance ---
print('\nDrivingCommuterDistance unique values:')
print(df['DrivingCommuterDistance'].unique())

# --- Cleaning column: JobRoleArea ---
print('\nJobRoleArea unique values:')
print(df['JobRoleArea'].unique())

# --- Cleaning column: Gender ---
print('\nGender unique values:')
print(df['Gender'].unique())

# --- Cleaning column: MaritalStatus ---
print('\nMaritalStatus unique values:')
print(df['MaritalStatus'].unique())

# --- Cleaning column: NumCompaniesPreviouslyWorked ---
print('\nNumCompaniesPreviouslyWorked unique values:')
print(df['NumCompaniesPreviouslyWorked'].unique())

# --- Cleaning column: AnnualProfessionalDevHrs ---
print('\nAnnualProfessionalDevHrs unique values:')
print(df['AnnualProfessionalDevHrs'].unique())

# --- Cleaning column: PaycheckMethod ---
print('\nPaycheckMethod unique values:')
print(df['PaycheckMethod'].unique())

# --- Cleaning column: TextMessageOptIn ---
print('\nTextMessageOptIn unique values:')
print(df['TextMessageOptIn'].unique())