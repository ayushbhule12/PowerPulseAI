# src/deep_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_sequences(series: np.ndarray, look_back: int = 48):
    """
    Convert 1D time series into (samples, timesteps, features) for LSTM.
    look_back = 48 means we use 48 hours of history to predict the next hour.
    """
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_lstm(look_back: int = 48):
    model = Sequential([
        Input(shape=(look_back, 1)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def build_gru(look_back: int = 48):
    model = Sequential([
        Input(shape=(look_back, 1)),
        GRU(128, return_sequences=True),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_deep_model(series: pd.Series, model_type: str = 'lstm',
                     look_back: int = 48, epochs: int = 30, batch_size: int = 64):
    # Scale
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = create_sequences(scaled, look_back)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split 80/20
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Build
    model = build_lstm(look_back) if model_type == 'lstm' else build_gru(look_back)
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save
    model.save(f"models/{model_type}_model.keras")
    joblib.dump(scaler, f"models/{model_type}_scaler.pkl")
    print(f"{model_type.upper()} saved.")

    return model, scaler, history, X_val, y_val