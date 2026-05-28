# src/split.py

import pandas as pd

def split_data(df: pd.DataFrame, target: str = 'Global_active_power', test_ratio: float = 0.2):
    """
    Chronological train/test split — NO shuffling for time-series.
    Returns X_train, X_test, y_train, y_test as DataFrames/Series.
    """
    split_idx = int(len(df) * (1 - test_ratio))

    train = df.iloc[:split_idx]
    test  = df.iloc[split_idx:]

    feature_cols = [c for c in df.columns if c != target]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test  = test[feature_cols]
    y_test  = test[target]

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train period: {train.index[0]} -> {train.index[-1]}")
    print(f"Test  period: {test.index[0]}  -> {test.index[-1]}")

    return X_train, X_test, y_train, y_test