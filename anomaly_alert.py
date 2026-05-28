# src/anomaly_alert.py
"""
Anomaly alerting system with:
  - Severity levels: LOW / MEDIUM / HIGH / CRITICAL
  - HTML email-style report generation
  - Alert log CSV
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


# ── Severity thresholds (multiples of rolling std) ───────────────────────────
SEVERITY_THRESHOLDS = {
    'LOW':      (2.0, 2.5),
    'MEDIUM':   (2.5, 3.0),
    'HIGH':     (3.0, 4.0),
    'CRITICAL': (4.0, 9999)
}

SEVERITY_COLORS = {
    'LOW':      '#fff3cd',   # yellow
    'MEDIUM':   '#ffd699',   # orange-light
    'HIGH':     '#f8d7da',   # red-light
    'CRITICAL': '#dc3545'    # red
}

SEVERITY_BADGE = {
    'LOW':      '🟡',
    'MEDIUM':   '🟠',
    'HIGH':     '🔴',
    'CRITICAL': '🚨'
}


def assign_severity(series: pd.Series,
                     window: int = 24) -> pd.DataFrame:
    """
    For every point compute how many σ it deviates from the rolling mean.
    Assign a severity level based on SEVERITY_THRESHOLDS.
    Returns a DataFrame with columns: value, z_score, severity, is_anomaly
    """
    roll_mean = series.rolling(window, center=True, min_periods=1).mean()
    roll_std  = series.rolling(window, center=True, min_periods=1).std().fillna(1)

    z_score = ((series - roll_mean) / roll_std).abs()

    def _classify(z):
        for level, (lo, hi) in SEVERITY_THRESHOLDS.items():
            if lo <= z < hi:
                return level
        return None   # Normal

    severity   = z_score.apply(_classify)
    is_anomaly = severity.notna()

    return pd.DataFrame({
        'value':      series,
        'roll_mean':  roll_mean,
        'roll_std':   roll_std,
        'z_score':    z_score.round(3),
        'severity':   severity.fillna('NORMAL'),
        'is_anomaly': is_anomaly
    })


def generate_alert_log(alert_df: pd.DataFrame,
                        save_path: str = 'data/alert_log.csv') -> pd.DataFrame:
    """Save only anomalous rows as an alert log."""
    alerts = alert_df[alert_df['is_anomaly']].copy()
    alerts['detected_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    alerts.to_csv(save_path)
    print(f"Alert log saved → {save_path}  ({len(alerts)} alerts)")
    return alerts


def generate_html_report(alert_df: pd.DataFrame,
                          series_name: str = 'Global Active Power',
                          save_path: str = 'data/anomaly_report.html'):
    """
    Generate a self-contained HTML email-style anomaly report.
    """
    alerts    = alert_df[alert_df['is_anomaly']].copy()
    total     = len(alert_df)
    n_alerts  = len(alerts)
    rate      = n_alerts / total * 100

    # Severity counts
    sev_counts = alerts['severity'].value_counts().to_dict()
    critical   = sev_counts.get('CRITICAL', 0)
    high       = sev_counts.get('HIGH', 0)
    medium     = sev_counts.get('MEDIUM', 0)
    low        = sev_counts.get('LOW', 0)

    # Build alert rows HTML
    rows_html = ""
    for ts, row in alerts.sort_values('z_score', ascending=False).head(50).iterrows():
        bg    = SEVERITY_COLORS.get(row['severity'], '#ffffff')
        badge = SEVERITY_BADGE.get(row['severity'], '')
        rows_html += f"""
        <tr style="background:{bg}">
          <td>{ts}</td>
          <td>{row['value']:.4f} kW</td>
          <td>{row['roll_mean']:.4f} kW</td>
          <td>{row['z_score']:.2f}σ</td>
          <td><b>{badge} {row['severity']}</b></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  body      {{ font-family: Arial, sans-serif; background:#f4f6f9; margin:0; padding:20px; }}
  .card     {{ background:white; border-radius:10px; padding:24px; margin-bottom:20px;
               box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  h1        {{ color:#1a1a2e; }}
  h2        {{ color:#16213e; border-bottom:2px solid #e0e0e0; padding-bottom:8px; }}
  .kpi-row  {{ display:flex; gap:16px; flex-wrap:wrap; }}
  .kpi      {{ background:#f8f9fa; border-radius:8px; padding:16px 24px;
               text-align:center; flex:1; min-width:120px; }}
  .kpi .val {{ font-size:2em; font-weight:bold; }}
  .kpi .lbl {{ font-size:0.85em; color:#666; margin-top:4px; }}
  table     {{ width:100%; border-collapse:collapse; font-size:0.9em; }}
  th        {{ background:#1a1a2e; color:white; padding:10px; text-align:left; }}
  td        {{ padding:8px 10px; border-bottom:1px solid #eee; }}
  tr:hover  {{ opacity:0.88; }}
  .critical {{ color:#dc3545; font-weight:bold; }}
  .ts       {{ color:#666; font-size:0.82em; }}
</style>
</head>
<body>

<div class="card">
  <h1>⚡ Electricity Anomaly Detection Report</h1>
  <p class="ts">Generated: {datetime.now().strftime('%A, %d %B %Y at %H:%M:%S')}</p>
  <p>Series: <b>{series_name}</b> &nbsp;|&nbsp;
     Total records: <b>{total:,}</b> &nbsp;|&nbsp;
     Analysis window: rolling 24h</p>
</div>

<div class="card">
  <h2>📊 Summary Statistics</h2>
  <div class="kpi-row">
    <div class="kpi">
      <div class="val">{n_alerts:,}</div>
      <div class="lbl">Total Anomalies</div>
    </div>
    <div class="kpi">
      <div class="val">{rate:.2f}%</div>
      <div class="lbl">Anomaly Rate</div>
    </div>
    <div class="kpi" style="background:#fff3cd">
      <div class="val">🟡 {low}</div>
      <div class="lbl">Low Severity</div>
    </div>
    <div class="kpi" style="background:#ffd699">
      <div class="val">🟠 {medium}</div>
      <div class="lbl">Medium Severity</div>
    </div>
    <div class="kpi" style="background:#f8d7da">
      <div class="val">🔴 {high}</div>
      <div class="lbl">High Severity</div>
    </div>
    <div class="kpi" style="background:#dc3545;color:white">
      <div class="val">🚨 {critical}</div>
      <div class="lbl">Critical</div>
    </div>
  </div>
</div>

<div class="card">
  <h2>🚨 Top 50 Anomalies (sorted by severity)</h2>
  <table>
    <tr>
      <th>Timestamp</th>
      <th>Consumption</th>
      <th>Expected (Rolling Mean)</th>
      <th>Deviation</th>
      <th>Severity</th>
    </tr>
    {rows_html}
  </table>
</div>

<div class="card">
  <p style="color:#aaa;font-size:0.8em;text-align:center;">
    Auto-generated by Smart Electricity Forecasting System
  </p>
</div>

</body>
</html>"""

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML report saved → {save_path}")
    return save_path