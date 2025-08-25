import pandas as pd

# Load the dataset
df = pd.read_excel("Health Insurance Dataset 2025-05-04 00_21_56.xlsx", sheet_name="insurance")

# Clean the dataset by selecting relevant columns and dropping rows with missing values
cleaned_df = df[['age', 'bmi', 'sex', 'smoker', 'Level']].dropna()

# Save cleaned dataset
cleaned_df.to_csv("D599_Task2_Cleaned_Dataset.csv", index=False)

print("Cleaned dataset saved as D599_Task2_Cleaned_Dataset.csv")