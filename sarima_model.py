# src/sarima_model.py

import pandas as pd
import numpy as np
import warnings
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')


def check_stationarity(series: pd.Series) -> bool:
    """ADF test — returns True if already stationary."""
    result = adfuller(series.dropna())
    p_value = result[1]
    print(f"  ADF p-value: {p_value:.4f} → {'Stationary ✓' if p_value < 0.05 else 'Non-stationary'}")
    return p_value < 0.05


def train_sarima(
    full_series: pd.Series,
    train_size: int = 1500,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 0, 24)   # removed seasonal MA to avoid over-differencing
):
    """
    Correctly split the series internally so SARIMA trains on a
    recent chunk and forecasts the IMMEDIATELY following chunk.

    Parameters
    ----------
    full_series : the complete hourly series (train portion only)
    train_size  : how many hours to train on (default 1500 ≈ 62 days)
    """

    # ── Take the LAST (train_size + test_size) points of the series ───────────
    # We want: [......TRAIN_CHUNK | TEST_CHUNK] all contiguous in time
    use_tail   = min(len(full_series), train_size + 500)
    tail       = full_series.iloc[-use_tail:]

    train_chunk = tail.iloc[:train_size]
    test_chunk  = tail.iloc[train_size:]

    print(f"\nSARIMA internal split:")
    print(f"  Train chunk : {train_chunk.index[0]} → {train_chunk.index[-1]}  ({len(train_chunk)} pts)")
    print(f"  Test chunk  : {test_chunk.index[0]}  → {test_chunk.index[-1]}  ({len(test_chunk)} pts)")

    print("\nChecking stationarity...")
    check_stationarity(train_chunk)

    print(f"\nTraining SARIMA{order}x{seasonal_order} ...")
    model = SARIMAX(
        train_chunk,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
        simple_differencing=True    # speeds up training significantly
    )
    result = model.fit(disp=False, maxiter=200)
    print("  Done.")

    return result, test_chunk


def sarima_forecast(result, steps: int) -> np.ndarray:
    """Generate `steps` ahead one-shot forecast."""
    forecast = result.forecast(steps=steps)
    # Clip negatives (power can't be negative)
    return np.clip(forecast.values, 0, None)


def sarima_rolling_forecast(result, test_series: pd.Series,
                             train_series: pd.Series,
                             order=(1,1,1),
                             seasonal_order=(1,1,0,24),
                             max_steps: int = 200) -> np.ndarray:
    """
    Optional: walk-forward (rolling) forecast — refit every step.
    More accurate but very slow. Use max_steps to limit evaluation size.
    """
    history = list(train_series.values)
    preds   = []

    n = min(len(test_series), max_steps)
    print(f"Rolling forecast: {n} steps (this may take a few minutes)...")

    for i in range(n):
        m = SARIMAX(history, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                    simple_differencing=True)
        fit = m.fit(disp=False, maxiter=100)
        pred = fit.forecast(steps=1)[0]
        preds.append(max(pred, 0))
        history.append(test_series.values[i])

        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{n}")

    return np.array(preds)