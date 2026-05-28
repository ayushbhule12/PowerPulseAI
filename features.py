# src/features.py

import pandas as pd
import numpy as np

def create_features(df: pd.DataFrame, target: str = 'Global_active_power') -> pd.DataFrame:
    """
    Add time-based, lag, rolling, and cyclical features.
    """
    df = df.copy()

    # ── 5.1 Calendar features ─────────────────────────────────────────────────
    df['hour']        = df.index.hour
    df['dayofweek']   = df.index.dayofweek       # 0 = Monday
    df['month']       = df.index.month
    df['quarter']     = df.index.quarter
    df['dayofyear']   = df.index.dayofyear
    df['weekofyear']  = df.index.isocalendar().week.astype(int)
    df['is_weekend']  = (df['dayofweek'] >= 5).astype(int)

    # ── 5.2 Cyclical encoding (sin/cos) for hour and month ────────────────────
    df['hour_sin']    = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)

    # ── 5.3 Lag features ──────────────────────────────────────────────────────
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f'lag_{lag}'] = df[target].shift(lag)

    # ── 5.4 Rolling statistics (window = 24 hours) ───────────────────────────
    for window in [6, 12, 24, 48]:
        df[f'rolling_mean_{window}'] = df[target].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}']  = df[target].shift(1).rolling(window).std()
        df[f'rolling_max_{window}']  = df[target].shift(1).rolling(window).max()
        df[f'rolling_min_{window}']  = df[target].shift(1).rolling(window).min()

    # ── 5.5 Drop NaN rows created by lags/rolling ────────────────────────────
    df.dropna(inplace=True)

    print(f"Feature matrix shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed_hourly.csv", index_col='Datetime', parse_dates=True)
    df_feat = create_features(df)
    df_feat.to_csv("data/featured_data.csv")
    print("Saved to data/featured_data.csv")