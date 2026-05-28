# src/stl_decompose.py
"""
STL (Seasonal-Trend decomposition using LOESS).
Saves component plots and CSVs.
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL


def run_stl(series: pd.Series,
             period: int = 24,
             save_prefix: str = 'data/stl') -> pd.DataFrame:
    """
    Decompose series into trend, seasonal, residual.
    period=24 for hourly data (daily seasonality).
    """
    print(f"Running STL decomposition (period={period})...")

    stl    = STL(series, period=period, robust=True)
    result = stl.fit()

    components = pd.DataFrame({
        'observed':  result.observed,
        'trend':     result.trend,
        'seasonal':  result.seasonal,
        'residual':  result.resid
    })
    components.to_csv(f'{save_prefix}_components.csv')

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    labels = ['Observed', 'Trend', 'Seasonal', 'Residual']
    colors = ['steelblue', 'darkorange', 'mediumseagreen', 'tomato']

    for ax, col, label, color in zip(axes, components.columns, labels, colors):
        ax.plot(components.index, components[col],
                linewidth=0.8, color=color)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('STL Decomposition — Global Active Power', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_plot.png', dpi=150)
    plt.close()
    print(f"STL plot saved → {save_prefix}_plot.png")

    return components