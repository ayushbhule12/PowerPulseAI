# src/multistep_forecast.py
"""
Multi-step forecasting: predict next 24h, 48h, and 168h (7 days).
Uses a Direct Multi-Output strategy with XGBoost and a Seq2Seq LSTM.
"""

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, LSTM, Dense, RepeatVector,
                                     TimeDistributed, Dropout)
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')


# ── Horizons ──────────────────────────────────────────────────────────────────
HORIZONS = {
    '24h':  24,
    '48h':  48,
    '7day': 168
}


def make_multistep_dataset(series: np.ndarray,
                            look_back: int = 48,
                            horizon: int = 24):
    """
    Build (X, Y) where:
      X shape = (samples, look_back)
      Y shape = (samples, horizon)
    """
    X, Y = [], []
    for i in range(look_back, len(series) - horizon + 1):
        X.append(series[i - look_back:i])
        Y.append(series[i:i + horizon])
    return np.array(X), np.array(Y)


# ─────────────────────────────────────────────────────────────────────────────
#  Direct XGBoost multi-output forecaster
# ─────────────────────────────────────────────────────────────────────────────
def train_multistep_xgb(series: pd.Series,
                         look_back: int = 48,
                         horizon: int = 24,
                         label: str = '24h'):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    X, Y = make_multistep_dataset(scaled, look_back, horizon)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    base = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, random_state=42,
                        verbosity=0, n_jobs=-1)
    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, Y_train)

    preds_scaled = model.predict(X_test)

    # Inverse transform each step
    preds = scaler.inverse_transform(preds_scaled)
    truth = scaler.inverse_transform(Y_test)

    joblib.dump(model,  f"models/ms_xgb_{label}.pkl")
    joblib.dump(scaler, f"models/ms_xgb_scaler_{label}.pkl")
    print(f"  Multi-step XGB [{label}] saved.")
    return model, scaler, preds, truth, X_test


# ─────────────────────────────────────────────────────────────────────────────
#  Seq2Seq LSTM forecaster
# ─────────────────────────────────────────────────────────────────────────────
def build_seq2seq(look_back: int, horizon: int) -> tf.keras.Model:
    encoder_input = Input(shape=(look_back, 1))
    enc, state_h, state_c = tf.keras.layers.LSTM(
        128, return_state=True)(encoder_input)
    enc_out = RepeatVector(horizon)(enc)

    dec = LSTM(128, return_sequences=True)(enc_out)
    dec = Dropout(0.2)(dec)
    out = TimeDistributed(Dense(1))(dec)

    model = Model(encoder_input, out)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_seq2seq_lstm(series: pd.Series,
                        look_back: int = 48,
                        horizon: int = 24,
                        epochs: int = 30,
                        batch_size: int = 64,
                        label: str = '24h'):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()

    X, Y = make_multistep_dataset(scaled, look_back, horizon)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    Y_tr, Y_te = Y[:split], Y[split:]

    model = build_seq2seq(look_back, horizon)
    cb = EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X_tr, Y_tr, validation_data=(X_te, Y_te),
              epochs=epochs, batch_size=batch_size,
              callbacks=[cb], verbose=1)

    preds_scaled = model.predict(X_te).reshape(X_te.shape[0], horizon)
    truth_scaled = Y_te.reshape(X_te.shape[0], horizon)

    preds = scaler.inverse_transform(preds_scaled)
    truth = scaler.inverse_transform(truth_scaled)

    model.save(f"models/seq2seq_{label}.keras")
    joblib.dump(scaler, f"models/seq2seq_scaler_{label}.pkl")
    print(f"  Seq2Seq LSTM [{label}] saved.")
    return model, scaler, preds, truth


def future_forecast(series: pd.Series,
                     model, scaler,
                     look_back: int = 48,
                     horizon: int = 24,
                     model_type: str = 'xgb') -> pd.DataFrame:
    """
    Generate a real future forecast from the LAST look_back points.
    Returns a DataFrame with DatetimeIndex continuing from series.index[-1].
    """
    last = scaler.transform(
        series.values[-look_back:].reshape(-1, 1)
    ).flatten()

    if model_type == 'xgb':
        pred_scaled = model.predict(last.reshape(1, -1))
        pred = scaler.inverse_transform(pred_scaled).flatten()
    else:  # seq2seq
        inp = last.reshape(1, look_back, 1)
        pred_scaled = model.predict(inp).flatten()
        pred = scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).flatten()

    future_index = pd.date_range(
        start=series.index[-1] + pd.Timedelta(hours=1),
        periods=horizon, freq='h'
    )
    return pd.DataFrame({'forecast': np.clip(pred, 0, None)},
                         index=future_index)