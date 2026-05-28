import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    sns = None
    _HAS_SEABORN = False

def run_eda(df: pd.DataFrame, show_plots: bool = False):
    """
    Runs Exploratory Data Analysis, saves plots to disk, and prints statistics.
    Safely keeps the input DataFrame immutable.
    """
    # Create a local copy to prevent side-effects on the source dataframe
    df_local = df.copy()
    target = 'Global_active_power'

    if target not in df_local.columns:
        raise ValueError(f"Target column '{target}' not found in the DataFrame.")

    # ── 4.1 Basic statistics ──────────────────────────────────────────────────
    print("=== Statistical Summary ===")
    print(df_local.describe().round(3))

    # ── 4.2 Full time-series plot ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(df_local.index, df_local[target], linewidth=0.5, color='steelblue')
    ax.set_title('Global Active Power – Full Period')
    ax.set_xlabel('Date')
    ax.set_ylabel('kW')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig('data/eda_full_series.png', dpi=150)
    if show_plots: plt.show()
    plt.close(fig)

    # ── 4.3 Monthly average consumption ──────────────────────────────────────
    monthly = df_local[target].resample('ME').mean()
    fig, ax = plt.subplots(figsize=(12, 4))
    monthly.plot(ax=ax, marker='o', color='darkorange')
    ax.set_title('Monthly Average Active Power')
    ax.set_ylabel('kW')
    plt.tight_layout()
    plt.savefig('data/eda_monthly.png', dpi=150)
    if show_plots: plt.show()
    plt.close(fig)

    # ── 4.4 Hour-of-day average (daily pattern) ───────────────────────────────
    # Safely extract hour without modifying original DataFrame structure
    hourly_avg = df_local.groupby(df_local.index.hour)[target].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    hourly_avg.plot(kind='bar', ax=ax, color='teal', edgecolor='white')
    ax.set_title('Average Consumption by Hour of Day')
    ax.set_xlabel('Hour')
    ax.set_ylabel('kW')
    plt.tight_layout()
    plt.savefig('data/eda_hourly.png', dpi=150)
    if show_plots: plt.show()
    plt.close(fig)

    # ── 4.5 Day-of-week average ───────────────────────────────────────────────
    dow_avg = df_local.groupby(df_local.index.dayofweek)[target].mean()
    dow_avg.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig, ax = plt.subplots(figsize=(8, 4))
    dow_avg.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='white')
    ax.set_title('Average Consumption by Day of Week')
    ax.set_ylabel('kW')
    plt.tight_layout()
    plt.savefig('data/eda_dow.png', dpi=150)
    if show_plots: plt.show()
    plt.close(fig)

    # ── 4.6 Correlation heatmap ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df_local.corr()
    if _HAS_SEABORN:
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, square=True)
    else:
        im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.index)
        for (i, j), val in np.ndenumerate(corr.values):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                    color='white' if abs(val) > 0.5 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('data/eda_corr.png', dpi=150)
    if show_plots: plt.show()
    plt.close(fig)

    # ── 4.7 Distribution of target ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    df_local[target].hist(bins=80, ax=ax, color='coral', edgecolor='white')
    ax.set_title('Distribution of Global Active Power')
    ax.set_xlabel('kW')
    plt.tight_layout()
    plt.savefig('data/eda_dist.png', dpi=150)
    if show_plots: plt.show()
    plt.close(fig)


if __name__ == "__main__":
    # Ensure data directory exists
    import os
    os.makedirs('data', exist_ok=True)
    
    df = pd.read_csv("data/processed_hourly.csv", index_col='Datetime', parse_dates=True)
    # Passed show_plots=True for local running
    run_eda(df, show_plots=True)