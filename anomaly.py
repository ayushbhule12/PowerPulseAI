# src/anomaly.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats

def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Flag points where |Z-score| > threshold."""
    z = np.abs(stats.zscore(series.dropna()))
    anomalies = pd.Series(False, index=series.index)
    anomalies.iloc[np.where(z > threshold)[0]] = True
    return anomalies


def detect_rolling_threshold(series: pd.Series,
                              window: int = 24,
                              n_std: float = 3.0) -> pd.Series:
    """Flag points outside rolling mean ± n_std * rolling_std."""
    roll_mean = series.rolling(window, center=True).mean()
    roll_std  = series.rolling(window, center=True).std()
    upper = roll_mean + n_std * roll_std
    lower = roll_mean - n_std * roll_std
    anomalies = (series > upper) | (series < lower)
    return anomalies.fillna(False)


def detect_isolation_forest(df: pd.DataFrame,
                             features: list,
                             contamination: float = 0.01) -> pd.Series:
    """
    Use Isolation Forest on multivariate features.
    Returns boolean Series (True = anomaly).
    """
    clf = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    preds = clf.fit_predict(df[features].dropna())
    # IsolationForest returns -1 for anomalies, 1 for normal
    anomalies = pd.Series(preds == -1, index=df.dropna().index)
    return anomalies


def plot_anomalies(series: pd.Series, anomalies: pd.Series, title: str, save_path: str):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(series.index, series.values, linewidth=0.7,
            color='steelblue', label='Consumption')
    ax.scatter(series[anomalies].index, series[anomalies].values,
               color='red', s=10, zorder=5, label='Anomaly')
    ax.set_title(title)
    ax.set_ylabel('Global Active Power (kW)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    n = anomalies.sum()
    print(f"Anomalies detected: {n} ({n/len(series)*100:.2f}%)")


if __name__ == "__main__":
    df = pd.read_csv("data/featured_data.csv", index_col='Datetime', parse_dates=True)
    target = 'Global_active_power'

    # Z-Score
    az = detect_zscore(df[target])
    plot_anomalies(df[target], az, "Z-Score Anomalies", "data/anomaly_zscore.png")

    # Rolling threshold
    ar = detect_rolling_threshold(df[target])
    plot_anomalies(df[target], ar, "Rolling Threshold Anomalies", "data/anomaly_rolling.png")

    # Isolation Forest (multivariate)
    feats = [target, 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    ai = detect_isolation_forest(df, feats)
    plot_anomalies(df[target], ai.reindex(df.index, fill_value=False),
                   "Isolation Forest Anomalies", "data/anomaly_iforest.png")