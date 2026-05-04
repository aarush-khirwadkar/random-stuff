import pandas as pd

file_path = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\Data for Datathon (Revised).xlsx"
df_daily = pd.read_excel(file_path, sheet_name="A - Daily")

# Check if Aug 2025 has non-null values
df_daily['Date_dt'] = pd.to_datetime(df_daily['Date'].str.slice(0, 8), format="%m/%d/%y")
aug_2025 = df_daily[(df_daily['Date_dt'].dt.year == 2025) & (df_daily['Date_dt'].dt.month == 8)]
print("Aug 2025 Daily Data (first 5 rows):")
print(aug_2025.head(5).to_string())

# Also check interval data year
df_interval = pd.read_excel(file_path, sheet_name="A - Interval")
print("Interval Data first row:")
print(df_interval.iloc[0])

