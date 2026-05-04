import pandas as pd

data_file = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\Data for Datathon (Revised).xlsx"

df_int = pd.read_excel(data_file, sheet_name="A - Interval")
print("df_int months:", df_int['Month'].unique())
print("df_int days (first 5):", df_int['Day'].head().tolist())

df_daily = pd.read_excel(data_file, sheet_name="A - Daily")
df_daily['Date_dt'] = pd.to_datetime(df_daily['Date'].str.slice(0, 8), format="%m/%d/%y")
print("df_daily months:", df_daily['Date_dt'].dt.month.unique())
print("df_daily days (first 5):", df_daily['Date_dt'].dt.day.head().tolist())

month_map = {'April': 4, 'May': 5, 'June': 6}
df_int['Month_num'] = df_int['Month'].map(month_map)
print("Mapped Month_num unique:", df_int['Month_num'].unique())

df_int['Year'] = 2024
df_daily['Month'] = df_daily['Date_dt'].dt.month
df_daily['Day'] = df_daily['Date_dt'].dt.day
df_daily['Year'] = df_daily['Date_dt'].dt.year

merged = pd.merge(df_int, df_daily, left_on=['Year', 'Month_num', 'Day'], right_on=['Year', 'Month', 'Day'], how='inner')
print("Merged 2024 rows:", len(merged))

df_int['Year'] = 2025
merged25 = pd.merge(df_int, df_daily, left_on=['Year', 'Month_num', 'Day'], right_on=['Year', 'Month', 'Day'], how='inner')
print("Merged 2025 rows:", len(merged25))
