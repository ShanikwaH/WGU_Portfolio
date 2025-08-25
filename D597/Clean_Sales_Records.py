import pandas as pd

# Load CSV
df = pd.read_csv('Sales_Records.csv')

# Step 1: Remove leading/trailing spaces
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Step 2: Drop completely blank rows
df.dropna(how='all', inplace=True)

# Step 3: Check for missing values in required fields
required_columns = [
    'Region', 'Country', 'Item Type', 'Sales Channel', 'Order Priority',
    'Order Date', 'Order ID', 'Ship Date', 'Units Sold',
    'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit'
]
missing = df[required_columns].isnull().sum()
print("Missing values per column:\n", missing)

# Optional: Drop rows with missing required fields
df.dropna(subset=required_columns, inplace=True)

# Step 4: Convert date columns to date format
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')

# Step 5: Drop rows with invalid dates
df = df[df['Order Date'].notna() & df['Ship Date'].notna()]

# Step 6: Save cleaned file
df.to_csv('Cleaned_Sales_Records.csv', index=False)

print("Data cleaning complete. File saved as Cleaned_Sales_Records.csv")
