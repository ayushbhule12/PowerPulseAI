# src/ml_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

def train_random_forest(X_train, y_train, save_path="models/rf_model.pkl"):
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    joblib.dump(model, save_path)
    print(f"Random Forest saved -> {save_path}")
    return model


def train_xgboost(X_train, y_train, save_path="models/xgb_model.pkl"):
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    joblib.dump(model, save_path)
    print(f"XGBoost saved -> {save_path}")
    return model


def predict(model, X_test):
    return model.predict(X_test)