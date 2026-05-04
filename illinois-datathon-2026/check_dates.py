import pandas as pd

file_path = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\Data for Datathon (Revised).xlsx"

for cc in ['A', 'B', 'C', 'D']:
    df_daily = pd.read_excel(file_path, sheet_name=f"{cc} - Daily")
    df_interval = pd.read_excel(file_path, sheet_name=f"{cc} - Interval")
    
    print(f"--- Call Center {cc} ---")
    print("Daily Data:")
    print("First date:", df_daily['Date'].iloc[0])
    print("Last date:", df_daily['Date'].iloc[-1])
    print("Interval Data:")
    print("First month/day:", df_interval['Month'].iloc[0], df_interval['Day'].iloc[0])
    print("Last month/day:", df_interval['Month'].iloc[-1], df_interval['Day'].iloc[-1])

df_staffing = pd.read_excel(file_path, sheet_name="Daily Staffing")
print("--- Daily Staffing ---")
print("First date:", df_staffing['Unnamed: 0'].iloc[0])
print("Last date:", df_staffing['Unnamed: 0'].iloc[-1])
