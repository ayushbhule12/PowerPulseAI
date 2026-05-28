# main.py — Full advanced pipeline

import os
import numpy as np
import pandas as pd

os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

TARGET = 'Global_active_power'

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load & Preprocess
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Load & Preprocess")
print("="*60)
from src.load_data  import load_raw_data
from src.preprocess import preprocess

raw = load_raw_data("data/household_power_consumption.txt")
df  = preprocess(raw, resample_freq='H')
df.to_csv("data/processed_hourly.csv")
print("✅ data/processed_hourly.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: EDA")
print("="*60)
from src.eda import run_eda
run_eda(df.copy())

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — STL Decomposition
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: STL Decomposition")
print("="*60)
from src.stl_decompose import run_stl
stl_components = run_stl(df[TARGET], period=24)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: Feature Engineering")
print("="*60)
from src.features import create_features
df_feat = create_features(df)

# Add STL components as extra features
df_feat = df_feat.join(
    stl_components[['trend','seasonal','residual']], how='left'
).dropna()

df_feat.to_csv("data/featured_data.csv")
print("✅ data/featured_data.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train/Test Split
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: Train/Test Split")
print("="*60)
from src.split   import split_data
from src.evaluate import evaluate, compare_models

X_train, X_test, y_train, y_test = split_data(df_feat, target=TARGET)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6: Random Forest")
print("="*60)
from src.ml_models import train_random_forest, train_xgboost, predict

rf_model   = train_random_forest(X_train, y_train)
rf_preds   = predict(rf_model, X_test)
rf_results = evaluate(y_test.values, rf_preds, "Random Forest")

pd.DataFrame({'actual': y_test.values, 'predicted': rf_preds},
             index=y_test.index).to_csv('data/predictions_random_forest.csv')
print("✅ data/predictions_random_forest.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7: XGBoost")
print("="*60)
xgb_model   = train_xgboost(X_train, y_train)
xgb_preds   = predict(xgb_model, X_test)
xgb_results = evaluate(y_test.values, xgb_preds, "XGBoost")

pd.DataFrame({'actual': y_test.values, 'predicted': xgb_preds},
             index=y_test.index).to_csv('data/predictions_xgboost.csv')
print("✅ data/predictions_xgboost.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — SARIMA
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8: SARIMA")
print("="*60)
from src.sarima_model import train_sarima, sarima_forecast

train_series   = df[TARGET].iloc[:int(len(df) * 0.8)]
sarima_result, sarima_test_chunk = train_sarima(
    train_series, train_size=1500,
    order=(1,1,1), seasonal_order=(1,1,0,24)
)
sarima_preds   = sarima_forecast(sarima_result, steps=len(sarima_test_chunk))
sarima_true    = sarima_test_chunk.values[:len(sarima_preds)]
sarima_results = evaluate(sarima_true, sarima_preds, "SARIMA")

pd.DataFrame({'actual': sarima_true, 'predicted': sarima_preds},
             index=sarima_test_chunk.index[:len(sarima_preds)]
             ).to_csv('data/predictions_sarima.csv')
print("✅ data/predictions_sarima.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — LSTM
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9: LSTM")
print("="*60)
from src.deep_model import train_deep_model

lstm_model, lstm_scaler, _, X_val_l, y_val_l = train_deep_model(
    df[TARGET], model_type='lstm', look_back=48, epochs=30, batch_size=64)

lstm_preds = lstm_scaler.inverse_transform(
    lstm_model.predict(X_val_l)).flatten()
lstm_true  = lstm_scaler.inverse_transform(
    y_val_l.reshape(-1,1)).flatten()
lstm_results = evaluate(lstm_true, lstm_preds, "LSTM")

pd.DataFrame({'actual': lstm_true, 'predicted': lstm_preds}
             ).to_csv('data/predictions_lstm.csv', index=False)
print("✅ data/predictions_lstm.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — GRU
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10: GRU")
print("="*60)
gru_model, gru_scaler, _, X_val_g, y_val_g = train_deep_model(
    df[TARGET], model_type='gru', look_back=48, epochs=30, batch_size=64)

gru_preds = gru_scaler.inverse_transform(
    gru_model.predict(X_val_g)).flatten()
gru_true  = gru_scaler.inverse_transform(
    y_val_g.reshape(-1,1)).flatten()
gru_results = evaluate(gru_true, gru_preds, "GRU")

pd.DataFrame({'actual': gru_true, 'predicted': gru_preds}
             ).to_csv('data/predictions_gru.csv', index=False)
print("✅ data/predictions_gru.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — Transformer
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 11: Transformer")
print("="*60)
from src.transformer_model import train_transformer

trans_model, trans_scaler, _, trans_preds, trans_true = train_transformer(
    df[TARGET], look_back=48, num_blocks=2, epochs=30, batch_size=64)
trans_results = evaluate(trans_true, trans_preds, "Transformer")

pd.DataFrame({'actual': trans_true, 'predicted': trans_preds}
             ).to_csv('data/predictions_transformer.csv', index=False)
print("✅ data/predictions_transformer.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — Multi-step Forecasting (24h, 48h, 7day)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 12: Multi-step Forecasting")
print("="*60)
from src.multistep_forecast import (train_multistep_xgb, train_seq2seq_lstm,
                                     future_forecast, HORIZONS)

for label, horizon in HORIZONS.items():
    print(f"\n--- Horizon: {label} ({horizon}h) ---")

    # XGBoost multi-output
    ms_xgb, ms_scaler, ms_preds, ms_true, _ = train_multistep_xgb(
        df[TARGET], look_back=48, horizon=horizon, label=label)

    # Save mean forecast vs truth for dashboard
    pd.DataFrame({
        'actual_mean':    ms_true.mean(axis=1),
        'predicted_mean': ms_preds.mean(axis=1)
    }).to_csv(f'data/multistep_xgb_{label}.csv', index=False)

    # Real future forecast starting from last known point
    future = future_forecast(df[TARGET], ms_xgb, ms_scaler,
                              look_back=48, horizon=horizon, model_type='xgb')
    future.to_csv(f'data/future_forecast_{label}.csv')
    print(f"  ✅ Future forecast saved → data/future_forecast_{label}.csv")

# Seq2Seq LSTM for 24h
print("\n--- Seq2Seq LSTM [24h] ---")
from src.multistep_forecast import train_seq2seq_lstm
ss_model, ss_scaler, ss_preds, ss_true = train_seq2seq_lstm(
    df[TARGET], look_back=48, horizon=24, epochs=20, label='24h')

pd.DataFrame({
    'actual_mean':    ss_true.mean(axis=1),
    'predicted_mean': ss_preds.mean(axis=1)
}).to_csv('data/multistep_seq2seq_24h.csv', index=False)
print("✅ data/multistep_seq2seq_24h.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — SHAP Explainability
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 13: SHAP Explainability")
print("="*60)
from src.shap_explain import explain_model

shap_xgb, _ = explain_model(xgb_model, X_train, X_test,
                              model_name='XGBoost', sample_size=500)
shap_rf, _  = explain_model(rf_model,  X_train, X_test,
                              model_name='Random Forest', sample_size=300)
print("✅ SHAP plots saved to data/")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 14 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 14: Model Comparison")
print("="*60)
comparison = compare_models([
    rf_results, xgb_results, sarima_results,
    lstm_results, gru_results, trans_results
])
print("✅ data/model_comparison.csv")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — Advanced Anomaly Detection + Alert Report
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 15: Anomaly Detection + Alert Report")
print("="*60)
from src.anomaly import (detect_zscore, detect_rolling_threshold,
                          detect_isolation_forest, plot_anomalies)
from src.anomaly_alert import assign_severity, generate_alert_log, generate_html_report

# Basic anomaly flags
az = detect_zscore(df[TARGET], threshold=3.0)
ar = detect_rolling_threshold(df[TARGET], window=24, n_std=3.0)
ai = detect_isolation_forest(
        df, [TARGET,'Sub_metering_1','Sub_metering_2','Sub_metering_3'],
        contamination=0.01
     ).reindex(df.index, fill_value=False)

plot_anomalies(df[TARGET], az, "Z-Score Anomalies",       "data/anomaly_zscore.png")
plot_anomalies(df[TARGET], ar, "Rolling Threshold",        "data/anomaly_rolling.png")
plot_anomalies(df[TARGET], ai, "Isolation Forest",         "data/anomaly_iforest.png")

# Severity-based alert system
alert_df = assign_severity(df[TARGET], window=24)
alert_log = generate_alert_log(alert_df, save_path='data/alert_log.csv')
generate_html_report(alert_df, save_path='data/anomaly_report.html')

# Combined anomaly CSV for dashboard
pd.DataFrame({
    'value':   df[TARGET],
    'zscore':  az,
    'rolling': ar,
    'iforest': ai,
    'severity': alert_df['severity'],
    'z_score':  alert_df['z_score']
}).to_csv("data/anomalies.csv")
print("✅ data/anomalies.csv")
print("✅ data/alert_log.csv")
print("✅ data/anomaly_report.html")

# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("✅ Full advanced pipeline complete!")
print("   streamlit run dashboard/app.py")
print("="*60)