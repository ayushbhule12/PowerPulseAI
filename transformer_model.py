# src/transformer_model.py
"""
Vanilla Transformer encoder for time-series forecasting.
Uses multi-head self-attention over the look-back window.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib


# ── Transformer block ─────────────────────────────────────────────────────────
def transformer_encoder_block(inputs, head_size: int, num_heads: int,
                                ff_dim: int, dropout: float = 0.1):
    # Multi-head self-attention
    x = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # Feed-forward
    ff = Dense(ff_dim, activation='relu')(x)
    ff = Dropout(dropout)(ff)
    ff = Dense(inputs.shape[-1])(ff)
    return LayerNormalization(epsilon=1e-6)(ff + x)


def build_transformer(look_back: int,
                       num_blocks: int = 2,
                       head_size: int = 32,
                       num_heads: int = 4,
                       ff_dim: int = 64,
                       dropout: float = 0.1) -> tf.keras.Model:
    inp = Input(shape=(look_back, 1))
    x   = inp
    for _ in range(num_blocks):
        x = transformer_encoder_block(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse', metrics=['mae'])
    return model


def create_sequences(series: np.ndarray, look_back: int = 48):
    X, y = [], []
    for i in range(look_back, len(series)):
        X.append(series[i - look_back:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def train_transformer(series: pd.Series,
                       look_back: int = 48,
                       num_blocks: int = 2,
                       head_size: int = 32,
                       num_heads: int = 4,
                       ff_dim: int = 64,
                       epochs: int = 30,
                       batch_size: int = 64):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(
        series.values.reshape(-1, 1)
    ).flatten()

    X, y = create_sequences(scaled, look_back)
    X = X.reshape(X.shape[0], look_back, 1)

    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model = build_transformer(look_back, num_blocks,
                               head_size, num_heads, ff_dim)
    model.summary()

    callbacks = [
        EarlyStopping(patience=6, restore_best_weights=True,
                      monitor='val_loss'),
        ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    preds_scaled = model.predict(X_val).flatten()
    preds = scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).flatten()
    truth = scaler.inverse_transform(
        y_val.reshape(-1, 1)
    ).flatten()

    model.save("models/transformer_model.keras")
    joblib.dump(scaler, "models/transformer_scaler.pkl")
    print("Transformer model saved.")
    return model, scaler, history, preds, truth