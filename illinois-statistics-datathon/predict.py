import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

data_file = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\Data for Datathon (Revised).xlsx"
out_dir = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-statistics-datathon"
template_path = r"C:\Users\aarus\Desktop\)\random-stuff\illinois-datathon-2026\template_forecast_v00.csv"

# Helper for interval to index
def interval_to_idx(interval_str):
    try:
        if isinstance(interval_str, str):
            h, m, s = map(int, interval_str.split(':'))
            return h * 2 + (1 if m >= 30 else 0)
        else:
            return interval_str.hour * 2 + (1 if interval_str.minute >= 30 else 0)
    except:
        return 0

def idx_to_interval(idx):
    h = idx // 2
    m = (idx % 2) * 30
    return f"{h}:{m:02d}"

print("Loading Daily Data and Staffing...")
# Process Daily Data
daily_data = {}
for cc in ['A', 'B', 'C', 'D']:
    df = pd.read_excel(data_file, sheet_name=f"{cc} - Daily")
    df['Date_dt'] = pd.to_datetime(df['Date'].str.slice(0, 8), format="%m/%d/%y")
    df['Month'] = df['Date_dt'].dt.month
    df['Day'] = df['Date_dt'].dt.day
    df['DayOfWeek'] = df['Date_dt'].dt.dayofweek
    df['Year'] = df['Date_dt'].dt.year
    df = df.rename(columns={
        'Call Volume': 'Daily_CV',
        'CCT': 'Daily_CCT',
        'Service Level': 'Daily_SL',
        'Abandon Rate': 'Daily_ABR'
    })
    daily_data[cc] = df

# Process Staffing
df_staffing = pd.read_excel(data_file, sheet_name="Daily Staffing")
df_staffing = df_staffing.rename(columns={'Unnamed: 0': 'Date'})
df_staffing['Date_dt'] = pd.to_datetime(df_staffing['Date'])

for cc in ['A', 'B', 'C', 'D']:
    # Get global median for filling
    cc_staffing_median = df_staffing[cc].median()
    
    daily_data[cc] = pd.merge(daily_data[cc], df_staffing[['Date_dt', cc]].rename(columns={cc: 'Daily_Staffing'}), on='Date_dt', how='left')
    
    # Fill NaN with global median for that CC
    daily_data[cc]['Daily_Staffing'] = daily_data[cc]['Daily_Staffing'].fillna(cc_staffing_median)
    
    # Feature engineering
    daily_data[cc]['Workload'] = (daily_data[cc]['Daily_CV'] / daily_data[cc]['Daily_Staffing'].replace(0, 1)) * daily_data[cc]['Daily_CCT']

# Set up prediction DataFrame for August 2025
pred_rows = []
for day in range(1, 32):
    for interval in range(48):
        pred_rows.append({'Month_num': 8, 'Day': day, 'Interval_Idx': interval})

pred_df = pd.DataFrame(pred_rows)
pred_df['Year'] = 2025
pred_df['Date_dt'] = pd.to_datetime(pred_df['Year'].astype(str) + '-' + pred_df['Month_num'].astype(str).str.zfill(2) + '-' + pred_df['Day'].astype(str).str.zfill(2))
pred_df['DayOfWeek'] = pred_df['Date_dt'].dt.dayofweek

template_df = pd.read_csv(template_path)
final_res = pd.DataFrame()
final_res['Month'] = ['August'] * len(pred_df)
final_res['Day'] = pred_df['Day']
final_res['Interval'] = pred_df['Interval_Idx'].apply(idx_to_interval)

print("Training models and predicting...")
features = ['Month_num', 'Day', 'DayOfWeek', 'Interval_Idx', 'Daily_CV', 'Daily_CCT', 'Daily_SL', 'Daily_ABR', 'Daily_Staffing', 'Workload']

for cc in ['A', 'B', 'C', 'D']:
    print(f"--- Processing Call Center {cc} ---")
    df_int = pd.read_excel(data_file, sheet_name=f"{cc} - Interval")
    
    month_map = {'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8}
    df_int['Month_num'] = df_int['Month'].map(month_map)
    df_int['Interval_Idx'] = df_int['Interval'].apply(interval_to_idx)
    
    # Merge with 2024 daily data (assume interval data is 2024)
    df_int['Year'] = 2024
    train_df = pd.merge(df_int, daily_data[cc], left_on=['Year', 'Month_num', 'Day'], right_on=['Year', 'Month', 'Day'], how='inner')
    
    if len(train_df) == 0:
        df_int['Year'] = 2025
        train_df = pd.merge(df_int, daily_data[cc], left_on=['Year', 'Month_num', 'Day'], right_on=['Year', 'Month', 'Day'], how='inner')
    
    train_df = train_df.dropna(subset=features + ['Call Volume', 'CCT', 'Abandoned Rate'])
    
    X_train = train_df[features]
    
    rf_cv = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cv.fit(X_train, train_df['Call Volume'])
    
    rf_cct = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_cct.fit(X_train, train_df['CCT'])
    
    rf_abr = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_abr.fit(X_train, train_df['Abandoned Rate'])
    
    # Predict for Aug 2025
    cc_pred = pd.merge(pred_df, daily_data[cc], left_on=['Year', 'Date_dt', 'Day', 'DayOfWeek', 'Month_num'], right_on=['Year', 'Date_dt', 'Day', 'DayOfWeek', 'Month'], how='left')
    
    # Fill missing daily values with medians if any
    X_pred = cc_pred[features].fillna(cc_pred[features].median())
    
    cv_pred = rf_cv.predict(X_pred)
    cct_pred = rf_cct.predict(X_pred)
    # Add a 2% safety buffer to CCT to penalize underestimating handle time
    cct_pred = cct_pred * 1.02
    abr_pred = rf_abr.predict(X_pred)
    
    # Normalize predicted CV to a window around Daily_CV, biased upwards 
    # to avoid the heavy penalty of underestimating.
    cc_pred['pred_cv_raw'] = cv_pred
    daily_sums = cc_pred.groupby('Day')['pred_cv_raw'].transform('sum')
    # Replace 0 sum with 1 to avoid division by zero
    daily_sums = daily_sums.replace(0, 1)
    
    # Window: +0% to +10% of exact forecast. Biased to [1.00, 1.10] due to underestimation penalty.
    target_daily = daily_sums.clip(lower=cc_pred['Daily_CV'] * 1.00, upper=cc_pred['Daily_CV'] * 1.10)
    
    cc_pred['pred_cv_scaled'] = (cc_pred['pred_cv_raw'] / daily_sums) * target_daily
    
    # Save to final result
    final_res[f'Calls_Offered_{cc}'] = cc_pred['pred_cv_scaled'].fillna(0).round(2)
    final_res[f'Abandoned_Rate_{cc}'] = np.clip(abr_pred, 0, 1).round(4)
    final_res[f'Abandoned_Calls_{cc}'] = (final_res[f'Calls_Offered_{cc}'] * final_res[f'Abandoned_Rate_{cc}']).round(2)
    final_res[f'CCT_{cc}'] = cct_pred.round(2)

# Ensure columns match template
final_res = final_res[template_df.columns]
final_out_path = os.path.join(out_dir, "forecast_august_2025.csv")
final_res.to_csv(final_out_path, index=False)
print("Forecast generated successfully at:", final_out_path)
