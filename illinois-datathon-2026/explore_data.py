import pandas as pd
import json

file_path = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\Data for Datathon (Revised).xlsx"

# Load the excel file
excel_data = pd.ExcelFile(file_path)

# Print sheet names
print("Sheet Names:")
print(excel_data.sheet_names)

# For each sheet, print the first few rows
for sheet in excel_data.sheet_names:
    print(f"\n--- Sheet: {sheet} ---")
    df = pd.read_excel(file_path, sheet_name=sheet)
    print("Columns:")
    print(df.columns.tolist())
    print("Data sample:")
    print(df.head(2).to_json(orient="records"))

template_path = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\template_forecast_v00.csv"
template_df = pd.read_csv(template_path)
print("\n--- Template ---")
print("Columns:")
print(template_df.columns.tolist())
print("Data sample:")
print(template_df.head(2).to_json(orient="records"))
