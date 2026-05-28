# src/preprocess.py

import pandas as pd
import numpy as np

def preprocess(df: pd.DataFrame, resample_freq: str = 'h') -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Parse datetime
    - Handle missing values
    - Convert dtypes
    - Resample to hourly or daily
    """

    resample_freq = resample_freq.lower()

    # --- 3.1 Ensure the DataFrame has a datetime index ---
    if {'Date', 'Time'}.issubset(df.columns):
        df['Datetime'] = pd.to_datetime(
            df['Date'] + ' ' + df['Time'],
            format='%d/%m/%Y %H:%M:%S',
            dayfirst=True,
            errors='coerce'
        )
        df.drop(columns=['Date', 'Time'], inplace=True)
        df.set_index('Datetime', inplace=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index.dtype):
        df.index = pd.to_datetime(df.index, errors='coerce', dayfirst=True)

    # --- 3.2 Convert all columns to numeric (they may still be strings) ---
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 3.3 Report missing values before filling ---
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing)
    print(f"Total missing: {df.isnull().sum().sum()} rows ({df.isnull().mean().mean()*100:.2f}%)")

    # --- 3.4 Fill missing values using forward fill + backward fill ---
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # --- 3.5 Resample to hourly frequency (sum for power, mean for voltage) ---
    df_resampled = df.resample(resample_freq).agg({
        'Global_active_power':   'mean',
        'Global_reactive_power': 'mean',
        'Voltage':               'mean',
        'Global_intensity':      'mean',
        'Sub_metering_1':        'sum',
        'Sub_metering_2':        'sum',
        'Sub_metering_3':        'sum',
    })

    # --- 3.6 Drop any remaining NaNs after resample ---
    df_resampled.dropna(inplace=True)

    print(f"\nResampled shape ({resample_freq}): {df_resampled.shape}")
    return df_resampled


if __name__ == "__main__":
    from load_data import load_raw_data
    raw = load_raw_data("data/household_power_consumption.txt")
    df  = preprocess(raw, resample_freq='H')
    df.to_csv("data/processed_hourly.csv")
    print("Saved to data/processed_hourly.csv")