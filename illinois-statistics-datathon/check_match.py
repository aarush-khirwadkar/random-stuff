import pandas as pd

data_file = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\Data for Datathon (Revised).xlsx"

df_int = pd.read_excel(data_file, sheet_name="A - Interval")
df_daily = pd.read_excel(data_file, sheet_name="A - Daily")
df_daily['Date_dt'] = pd.to_datetime(df_daily['Date'].str.slice(0, 8), format="%m/%d/%y")

int_sum = df_int.groupby(['Month', 'Day'])['Call Volume'].sum().reset_index()

apr1_sum = int_sum[(int_sum['Month'] == 'April') & (int_sum['Day'] == 1)]['Call Volume'].values[0]

daily_apr1_2024 = df_daily[(df_daily['Date_dt'] == '2024-04-01')]['Call Volume'].values[0]
daily_apr1_2025 = df_daily[(df_daily['Date_dt'] == '2025-04-01')]['Call Volume'].values[0]

print("Interval sum Apr 1:", apr1_sum)
print("Daily Apr 1 2024:", daily_apr1_2024)
print("Daily Apr 1 2025:", daily_apr1_2025)
