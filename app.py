# dashboard/app.py — Full advanced dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, time, io
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ Smart Electricity Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡"
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0e1117; color:#fafafa; }
  [data-testid="stSidebar"]          { background:#161b22; }
  .metric-card {
    background:#1c2333; border-radius:10px; padding:18px 22px;
    text-align:center; border:1px solid #30363d;
  }
  .metric-card .val { font-size:1.9em; font-weight:700; color:#58a6ff; }
  .metric-card .lbl { font-size:0.82em; color:#8b949e; margin-top:4px; }
  .section-title    { font-size:1.25em; font-weight:600;
                      border-left:4px solid #58a6ff;
                      padding-left:10px; margin:18px 0 10px; }
  div[data-testid="stDataFrame"] { border-radius:8px; }
  .sev-critical { color:#ff4444; font-weight:bold; }
  .sev-high     { color:#ff8800; font-weight:bold; }
  .sev-medium   { color:#ffcc00; }
  .sev-low      { color:#88cc44; }
</style>
""", unsafe_allow_html=True)

TARGET = 'Global_active_power'

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_safe(path):
    d = pd.read_csv(path)
    dt_col = next(
        (c for c in d.columns if 'date' in c.lower()
         or 'time' in c.lower() or c.strip() == 'Unnamed: 0'),
        d.columns[0]
    )
    d = d.rename(columns={dt_col: 'Datetime'})
    d['Datetime'] = pd.to_datetime(d['Datetime'])
    return d.set_index('Datetime')


def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100
    return mae, rmse, mape, r2


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as w:
        df.to_excel(w, index=True)
    return buf.getvalue()


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    missing = [f for f in [
        "data/processed_hourly.csv", "data/featured_data.csv",
        "data/anomalies.csv",        "data/model_comparison.csv"
    ] if not os.path.exists(f)]
    if missing:
        return (None,)*6 + (missing,)

    df   = read_safe("data/processed_hourly.csv")
    feat = read_safe("data/featured_data.csv")
    anom = read_safe("data/anomalies.csv")
    comp = pd.read_csv("data/model_comparison.csv", index_col=0)

    stl = (read_safe("data/stl_components.csv")
           if os.path.exists("data/stl_components.csv") else None)

    alerts = (pd.read_csv("data/alert_log.csv", index_col=0, parse_dates=True)
              if os.path.exists("data/alert_log.csv") else pd.DataFrame())

    return df, feat, anom, comp, stl, alerts, []


df, feat, anom, comp, stl, alerts, missing = load_all()

if missing:
    st.error("Run `python main.py` first. Missing: " + ", ".join(missing))
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/48/electricity.png", width=40)
st.sidebar.title("⚡ Smart Electricity")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", [
    "📊 Overview",
    "🔍 EDA & Decomposition",
    "🤖 Forecasting",
    "🔮 Future Forecast",
    "📈 SHAP Explainability",
    "🚨 Anomaly Detection",
    "🔔 Alert Centre",
    "⚡ Live Simulation",
    "📋 Model Comparison",
])

st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date Filter")
min_d, max_d = df.index.min().date(), df.index.max().date()
date_range = st.sidebar.date_input("Range", [min_d, max_d],
                                    min_value=min_d, max_value=max_d)
if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
else:
    start, end = pd.Timestamp(min_d), pd.Timestamp(max_d)

df_v   = df.loc[start:end]
anom_v = anom.loc[start:end]

st.sidebar.markdown("---")
st.sidebar.caption(f"Period: {df.index.min().date()} → {df.index.max().date()}")
st.sidebar.caption(f"Records: {len(df):,}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("⚡ Smart Electricity Consumption Dashboard")
    st.caption("UCI Individual Household Electric Power Consumption — Hourly")
    st.markdown("---")

    # KPIs
    cols = st.columns(6)
    kpis = [
        ("Avg Power",   f"{df_v[TARGET].mean():.3f} kW"),
        ("Peak Power",  f"{df_v[TARGET].max():.3f} kW"),
        ("Min Power",   f"{df_v[TARGET].min():.3f} kW"),
        ("Std Dev",     f"{df_v[TARGET].std():.3f} kW"),
        ("Total kWh",   f"{df_v[TARGET].sum():.0f}"),
        ("Records",     f"{len(df_v):,}"),
    ]
    for col, (lbl, val) in zip(cols, kpis):
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main time series with rolling avg
    roll = df_v[TARGET].rolling(168).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_v.index, y=df_v[TARGET],
                             name='Hourly', opacity=0.5,
                             line=dict(color='#58a6ff', width=0.7)))
    fig.add_trace(go.Scatter(x=df_v.index, y=roll,
                             name='7-Day Rolling Avg',
                             line=dict(color='#f78166', width=2)))
    fig.update_layout(template='plotly_dark', height=380,
                      title='Global Active Power — Time Series',
                      hovermode='x unified',
                      margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Sub-metering
    sub_cols = [c for c in ['Sub_metering_1','Sub_metering_2','Sub_metering_3']
                if c in df_v.columns]
    if sub_cols:
        st.markdown('<div class="section-title">Sub-Metering Breakdown</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        labels = ['Kitchen','Laundry','HVAC/Water Heater']
        colors = ['#f78166','#56d364','#e3b341']
        for c, lbl, col in zip(sub_cols, labels, colors):
            fig2.add_trace(go.Scatter(x=df_v.index, y=df_v[c],
                                      name=lbl, opacity=0.8,
                                      line=dict(color=col, width=0.7)))
        fig2.update_layout(template='plotly_dark', height=320,
                            hovermode='x unified',
                            margin=dict(l=40,r=20,t=30,b=40))
        st.plotly_chart(fig2, use_container_width=True)

    # Download processed data
    st.markdown("---")
    st.download_button("⬇ Download Processed Data (CSV)",
                        df_v.to_csv().encode(),
                        file_name="processed_data.csv",
                        mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA & DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA & Decomposition":
    st.title("🔍 EDA & Seasonal Decomposition")
    st.markdown("---")

    tmp = df_v.copy()
    tmp['hour'] = tmp.index.hour
    tmp['dow']  = tmp.index.dayofweek
    tmp['month']= tmp.index.month

    c1, c2 = st.columns(2)
    with c1:
        ha = tmp.groupby('hour')[TARGET].mean()
        fig = px.bar(x=ha.index, y=ha.values, template='plotly_dark',
                     labels={'x':'Hour','y':'Avg kW'},
                     title='Avg Consumption by Hour',
                     color=ha.values, color_continuous_scale='teal')
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        da = tmp.groupby('dow')[TARGET].mean()
        da.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        fig = px.bar(x=da.index, y=da.values, template='plotly_dark',
                     labels={'x':'Day','y':'Avg kW'},
                     title='Avg Consumption by Weekday',
                     color=da.values, color_continuous_scale='purples')
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Monthly + Distribution
    c3, c4 = st.columns(2)
    with c3:
        monthly = df_v[TARGET].resample('ME').mean()
        fig = px.line(x=monthly.index, y=monthly.values, template='plotly_dark',
                      markers=True, title='Monthly Average',
                      labels={'x':'Month','y':'Avg kW'},
                      color_discrete_sequence=['#f78166'])
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(df_v, x=TARGET, nbins=80, template='plotly_dark',
                           title='Power Distribution',
                           color_discrete_sequence=['#58a6ff'])
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Box plot
    tmp2 = df_v[[TARGET]].copy()
    tmp2['Day'] = tmp2.index.dayofweek.map(
        {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
    fig = px.box(tmp2, x='Day', y=TARGET, template='plotly_dark',
                 color='Day', title='Consumption Distribution by Weekday',
                 category_orders={'Day':['Mon','Tue','Wed','Thu','Fri','Sat','Sun']})
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap: hour × weekday
    st.markdown('<div class="section-title">Hour × Weekday Heatmap</div>',
                unsafe_allow_html=True)
    pivot = tmp.groupby(['dow','hour'])[TARGET].mean().unstack()
    pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    fig = px.imshow(pivot, template='plotly_dark', aspect='auto',
                    color_continuous_scale='RdYlGn_r',
                    title='Avg kW by Hour and Weekday',
                    labels=dict(x='Hour', y='Day', color='kW'))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    # STL Decomposition
    if stl is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">STL Decomposition</div>',
                    unsafe_allow_html=True)
        stl_v = stl.loc[start:end]
        comps = ['observed','trend','seasonal','residual']
        colors_s = ['#58a6ff','#f78166','#56d364','#e3b341']
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            subplot_titles=comps, vertical_spacing=0.06)
        for i, (comp_col, color) in enumerate(zip(comps, colors_s), 1):
            if comp_col in stl_v.columns:
                fig.add_trace(go.Scatter(x=stl_v.index, y=stl_v[comp_col],
                                         name=comp_col,
                                         line=dict(color=color, width=0.8)), i, 1)
        fig.update_layout(template='plotly_dark', height=700,
                          showlegend=False,
                          margin=dict(l=40,r=20,t=60,b=40))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇ Download STL Components",
                            stl_v.to_csv().encode(),
                            "stl_components.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECASTING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Forecasting":
    st.title("🤖 Model Forecasting Results")
    st.markdown("---")

    all_models = ["Random Forest","XGBoost","SARIMA","LSTM","GRU","Transformer"]
    available  = [m for m in all_models
                  if os.path.exists(
                      f"data/predictions_{m.lower().replace(' ','_')}.csv")]

    if not available:
        st.warning("No prediction files found. Run main.py first.")
        st.stop()

    model_choice = st.selectbox("Select Model", available)
    fname = f"data/predictions_{model_choice.lower().replace(' ','_')}.csv"
    preds = pd.read_csv(fname)

    # Handle index
    if 'Datetime' in preds.columns:
        preds['Datetime'] = pd.to_datetime(preds['Datetime'])
        preds = preds.set_index('Datetime')
    elif preds.index.name == 'Datetime':
        preds.index = pd.to_datetime(preds.index)

    n_show = st.slider("Points to display", 100,
                        min(3000, len(preds)), min(500, len(preds)), 50)
    preds_show = preds.iloc[-n_show:]

    # Actual vs Predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=preds_show.index, y=preds_show['actual'],
                             name='Actual', line=dict(color='#58a6ff',width=1.5)))
    fig.add_trace(go.Scatter(x=preds_show.index, y=preds_show['predicted'],
                             name='Predicted',
                             line=dict(color='#f78166',width=1.5,dash='dash')))
    fig.update_layout(template='plotly_dark', height=420,
                      title=f'{model_choice} — Actual vs Predicted',
                      hovermode='x unified',
                      margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    resid = preds_show['actual'] - preds_show['predicted']
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=preds_show.index, y=resid,
                              mode='lines', name='Residual',
                              line=dict(color='#e3b341',width=0.8)))
    fig2.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.4)
    fig2.update_layout(template='plotly_dark', height=280,
                       title='Residuals', margin=dict(l=40,r=20,t=30,b=40))
    st.plotly_chart(fig2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig3 = px.scatter(x=preds_show['actual'], y=preds_show['predicted'],
                          template='plotly_dark', opacity=0.4,
                          labels={'x':'Actual','y':'Predicted'},
                          title='Actual vs Predicted Scatter',
                          color_discrete_sequence=['#58a6ff'])
        mn = min(preds_show[['actual','predicted']].min())
        mx = max(preds_show[['actual','predicted']].max())
        fig3.add_shape(type='line', x0=mn,y0=mn,x1=mx,y1=mx,
                       line=dict(color='#f78166',dash='dash'))
        fig3.update_layout(height=360)
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        fig4 = px.histogram(x=resid, nbins=60, template='plotly_dark',
                             title='Residual Distribution',
                             labels={'x':'Residual (kW)'},
                             color_discrete_sequence=['#e3b341'])
        fig4.update_layout(height=360)
        st.plotly_chart(fig4, use_container_width=True)

    # Metrics
    st.markdown("---")
    mae, rmse, mape, r2 = compute_metrics(preds['actual'].values,
                                           preds['predicted'].values)
    mc = st.columns(4)
    for col, lbl, val in zip(mc,
        ["MAE","RMSE","MAPE","R²"],
        [f"{mae:.4f} kW", f"{rmse:.4f} kW", f"{mape:.2f}%", f"{r2:.4f}"]):
        col.markdown(f"""
        <div class="metric-card">
          <div class="val">{val}</div><div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button("⬇ Download Predictions",
                        preds.to_csv().encode(),
                        f"predictions_{model_choice}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FUTURE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Future Forecast":
    st.title("🔮 Future Electricity Forecast")
    st.caption("Multi-step forecasts generated from the last known data point")
    st.markdown("---")

    horizon_choice = st.selectbox("Select Forecast Horizon",
                                   ["24h","48h","7day"])
    horizon_hrs    = {"24h":24, "48h":48, "7day":168}[horizon_choice]

    ff_path = f"data/future_forecast_{horizon_choice}.csv"
    if not os.path.exists(ff_path):
        st.warning(f"No future forecast file found for {horizon_choice}. Run main.py.")
        st.stop()

    ff = read_safe(ff_path)
    history_tail = df[TARGET].iloc[-168:]  # last 7 days for context

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_tail.index, y=history_tail.values,
                             name='Historical (last 7d)',
                             line=dict(color='#58a6ff',width=1.5)))
    fig.add_trace(go.Scatter(x=ff.index, y=ff['forecast'],
                             name=f'Forecast ({horizon_choice})',
                             line=dict(color='#56d364',width=2.5,dash='dot'),
                             fill='tozeroy', fillcolor='rgba(86,211,100,0.08)'))
    fig.add_vline(x=df.index[-1], line_dash='dash',
                  line_color='#e3b341', opacity=0.7,
                  annotation_text="Forecast Start",
                  annotation_position="top left")
    fig.update_layout(template='plotly_dark', height=440,
                      title=f'Future {horizon_choice} Forecast',
                      xaxis_title='Datetime', yaxis_title='kW',
                      hovermode='x unified',
                      margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Stats
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="metric-card">
      <div class="val">{ff['forecast'].mean():.3f} kW</div>
      <div class="lbl">Avg Forecast</div></div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card">
      <div class="val">{ff['forecast'].max():.3f} kW</div>
      <div class="lbl">Peak Forecast</div></div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card">
      <div class="val">{ff['forecast'].sum():.1f} kWh</div>
      <div class="lbl">Total Forecasted Energy</div></div>""",
      unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison across horizons
    st.markdown('<div class="section-title">All Horizon Comparison</div>',
                unsafe_allow_html=True)
    fig2 = go.Figure()
    colors_h = ['#56d364','#e3b341','#f78166']
    for (h, col) in zip(['24h','48h','7day'], colors_h):
        p = f"data/future_forecast_{h}.csv"
        if os.path.exists(p):
            d = read_safe(p)
            fig2.add_trace(go.Scatter(x=d.index, y=d['forecast'],
                                      name=h, line=dict(color=col)))
    fig2.update_layout(template='plotly_dark', height=360,
                       title='24h vs 48h vs 7-Day Forecasts',
                       hovermode='x unified',
                       margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig2, use_container_width=True)

    st.download_button("⬇ Download Forecast",
                        ff.to_csv().encode(),
                        f"future_forecast_{horizon_choice}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 SHAP Explainability":
    st.title("📈 SHAP Feature Explainability")
    st.markdown("---")

    shap_model = st.selectbox("Select Model", ["XGBoost","Random Forest"])
    slug = shap_model.lower().replace(" ","_")

    bar_img   = f"data/shap_bar_{slug}.png"
    bee_img   = f"data/shap_beeswarm_{slug}.png"
    shap_csv  = f"data/shap_importance_{slug}.csv"

    if not os.path.exists(bar_img):
        st.warning("SHAP files not found. Run main.py (Step 13).")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Mean |SHAP| Bar Chart</div>',
                    unsafe_allow_html=True)
        st.image(bar_img, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Beeswarm Summary</div>',
                    unsafe_allow_html=True)
        st.image(bee_img, use_container_width=True)

    # Interactive bar from CSV
    if os.path.exists(shap_csv):
        st.markdown('<div class="section-title">Top Features (Interactive)</div>',
                    unsafe_allow_html=True)
        shap_df = pd.read_csv(shap_csv)
        top_n   = st.slider("Top N features", 5, len(shap_df), 20)
        shap_top = shap_df.head(top_n)
        fig = px.bar(shap_top, x='mean_shap', y='feature',
                     orientation='h', template='plotly_dark',
                     title=f'Top {top_n} Features by Mean |SHAP|',
                     color='mean_shap', color_continuous_scale='blues',
                     labels={'mean_shap':'Mean |SHAP|','feature':'Feature'})
        fig.update_layout(height=max(400, top_n*22), yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("⬇ Download SHAP Values",
                            shap_df.to_csv(index=False).encode(),
                            f"shap_{slug}.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")
    st.markdown("---")

    method_map   = {"Z-Score":"zscore","Rolling Threshold":"rolling",
                    "Isolation Forest":"iforest"}
    method_label = st.selectbox("Method", list(method_map.keys()))
    method       = method_map[method_label]

    if method not in anom_v.columns:
        st.error(f"Column '{method}' missing. Re-run main.py.")
        st.stop()

    anom_flag = anom_v[method].astype(bool)
    abnormal  = anom_v[anom_flag]

    # KPIs
    kc = st.columns(4)
    for col, lbl, val in zip(kc,
        ["Total Points","Anomalies","Anomaly Rate","Avg Deviation"],
        [f"{len(anom_v):,}", f"{anom_flag.sum():,}",
         f"{anom_flag.mean()*100:.2f}%",
         f"{anom_v.loc[anom_flag,'z_score'].mean():.2f}σ"
         if 'z_score' in anom_v.columns else "N/A"]):
        col.markdown(f"""<div class="metric-card">
          <div class="val">{val}</div><div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot with severity coloring
    severity_colors = {'LOW':'#ffff00','MEDIUM':'#ffa500',
                       'HIGH':'#ff4444','CRITICAL':'#ff0000','NORMAL':None}

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=anom_v.index, y=anom_v['value'],
                             mode='lines', name='Power',
                             line=dict(color='#58a6ff',width=0.8), opacity=0.8))

    if 'severity' in anom_v.columns:
        for sev, color in severity_colors.items():
            if sev == 'NORMAL' or color is None:
                continue
            mask = anom_flag & (anom_v['severity'] == sev)
            if mask.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=anom_v[mask].index, y=anom_v[mask]['value'],
                    mode='markers', name=sev,
                    marker=dict(color=color, size=6, symbol='x')))
    else:
        fig.add_trace(go.Scatter(x=abnormal.index, y=abnormal['value'],
                                  mode='markers', name='Anomaly',
                                  marker=dict(color='red',size=5,symbol='x')))

    fig.update_layout(template='plotly_dark', height=440,
                      title=f'Anomalies — {method_label}',
                      hovermode='x unified',
                      margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Hour & Weekday breakdown
    if anom_flag.sum() > 0:
        c1, c2 = st.columns(2)
        with c1:
            ah = abnormal.index.hour.value_counts().sort_index()
            fig = px.bar(x=ah.index, y=ah.values, template='plotly_dark',
                         title='Anomalies by Hour',
                         labels={'x':'Hour','y':'Count'},
                         color=ah.values, color_continuous_scale='reds')
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            aw = abnormal.index.dayofweek.value_counts().sort_index()
            aw.index = [['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][i]
                        for i in aw.index]
            fig = px.bar(x=aw.index, y=aw.values, template='plotly_dark',
                         title='Anomalies by Weekday',
                         labels={'x':'Day','y':'Count'},
                         color=aw.values, color_continuous_scale='oranges')
            fig.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Method comparison
    st.markdown("---")
    methods = [m for m in ['zscore','rolling','iforest'] if m in anom_v.columns]
    rows = [{'Method':m, 'Anomalies':int(anom_v[m].astype(bool).sum()),
             'Rate %':round(anom_v[m].astype(bool).mean()*100,3)}
            for m in methods]
    st.dataframe(pd.DataFrame(rows).set_index('Method'), use_container_width=True)

    st.download_button("⬇ Download Anomaly Data",
                        anom_v.to_csv().encode(),
                        "anomalies.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ALERT CENTRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔔 Alert Centre":
    st.title("🔔 Anomaly Alert Centre")
    st.markdown("---")

    if alerts.empty:
        st.info("No alert log found. Run main.py to generate alerts.")
        st.stop()

    # Severity KPIs
    sev_counts = alerts['severity'].value_counts()
    sc = st.columns(4)
    for col, sev, emoji in zip(sc,
        ['CRITICAL','HIGH','MEDIUM','LOW'],
        ['🚨','🔴','🟠','🟡']):
        n = sev_counts.get(sev, 0)
        col.markdown(f"""<div class="metric-card">
          <div class="val">{emoji} {n}</div>
          <div class="lbl">{sev}</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    fc1, fc2 = st.columns(2)
    with fc1:
        sev_filter = st.multiselect("Filter by Severity",
                                     ['CRITICAL','HIGH','MEDIUM','LOW'],
                                     default=['CRITICAL','HIGH','MEDIUM','LOW'])
    with fc2:
        sort_by = st.selectbox("Sort by", ['z_score','value','severity'])

    filtered = alerts[alerts['severity'].isin(sev_filter)].sort_values(
        sort_by, ascending=False)

    # Timeline
    fig = go.Figure()
    colors_s = {'LOW':'#88cc44','MEDIUM':'#ffcc00',
                 'HIGH':'#ff8800','CRITICAL':'#ff4444'}
    for sev, color in colors_s.items():
        mask = filtered['severity'] == sev
        if mask.sum() > 0:
            fig.add_trace(go.Scatter(
                x=filtered[mask].index,
                y=filtered[mask]['value'],
                mode='markers', name=sev,
                marker=dict(color=color, size=7)
            ))
    fig.update_layout(template='plotly_dark', height=380,
                      title='Alert Timeline by Severity',
                      hovermode='closest',
                      margin=dict(l=40,r=20,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)

    # Severity distribution pie
    c1, c2 = st.columns(2)
    with c1:
        pie = alerts['severity'].value_counts()
        fig2 = px.pie(values=pie.values, names=pie.index,
                      template='plotly_dark',
                      title='Alert Severity Distribution',
                      color_discrete_map=colors_s)
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        hourly_alerts = alerts.index.hour.value_counts().sort_index()
        fig3 = px.bar(x=hourly_alerts.index, y=hourly_alerts.values,
                      template='plotly_dark',
                      title='Alerts by Hour of Day',
                      labels={'x':'Hour','y':'Count'},
                      color=hourly_alerts.values,
                      color_continuous_scale='reds')
        fig3.update_layout(height=360, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # Alert table
    st.markdown('<div class="section-title">Alert Log Table</div>',
                unsafe_allow_html=True)
    st.dataframe(filtered[['value','z_score','severity']].head(200),
                 use_container_width=True)

    # Downloads
    dc1, dc2 = st.columns(2)
    with dc1:
        st.download_button("⬇ Download Alert Log (CSV)",
                            filtered.to_csv().encode(),
                            "alert_log.csv","text/csv")
    with dc2:
        if os.path.exists("data/anomaly_report.html"):
            with open("data/anomaly_report.html","rb") as f:
                st.download_button("⬇ Download HTML Report",
                                    f.read(),
                                    "anomaly_report.html","text/html")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Live Simulation":
    st.title("⚡ Real-Time Consumption Simulation")
    st.caption("Simulates live meter data by replaying the test set")
    st.markdown("---")

    speed = st.slider("Simulation speed (points/refresh)", 1, 20, 5)
    auto  = st.toggle("▶ Start Live Simulation", value=False)

    # Session state for pointer
    if 'sim_idx' not in st.session_state:
        st.session_state.sim_idx = 0

    # Load test predictions (use RF as live source)
    live_path = "data/predictions_random_forest.csv"
    if not os.path.exists(live_path):
        st.warning("Run main.py first.")
        st.stop()

    live_df = pd.read_csv(live_path)
    if 'Datetime' in live_df.columns:
        live_df['Datetime'] = pd.to_datetime(live_df['Datetime'])
        live_df = live_df.set_index('Datetime')

    idx = st.session_state.sim_idx
    window = min(idx + 200, len(live_df))
    live_slice = live_df.iloc[max(0, window-200):window]

    # Current reading
    if len(live_slice) > 0:
        current_val = live_slice['actual'].iloc[-1]
        pred_val    = live_slice['predicted'].iloc[-1]
        err         = abs(current_val - pred_val)

        kc = st.columns(4)
        kc[0].markdown(f"""<div class="metric-card">
          <div class="val">{current_val:.3f} kW</div>
          <div class="lbl">Current Reading</div></div>""",
          unsafe_allow_html=True)
        kc[1].markdown(f"""<div class="metric-card">
          <div class="val">{pred_val:.3f} kW</div>
          <div class="lbl">Model Prediction</div></div>""",
          unsafe_allow_html=True)
        kc[2].markdown(f"""<div class="metric-card">
          <div class="val">{err:.3f} kW</div>
          <div class="lbl">Abs Error</div></div>""",
          unsafe_allow_html=True)

        # Alert if anomaly
        threshold = live_df['actual'].mean() + 2.5 * live_df['actual'].std()
        status = "🚨 HIGH ALERT" if current_val > threshold else "✅ Normal"
        color  = "#ff4444" if current_val > threshold else "#56d364"
        kc[3].markdown(f"""<div class="metric-card" style="border-color:{color}">
          <div class="val" style="color:{color}">{status}</div>
          <div class="lbl">Status</div></div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Live chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_slice.index, y=live_slice['actual'],
                                  name='Actual', mode='lines+markers',
                                  line=dict(color='#58a6ff',width=1.5),
                                  marker=dict(size=3)))
        fig.add_trace(go.Scatter(x=live_slice.index, y=live_slice['predicted'],
                                  name='Predicted',
                                  line=dict(color='#f78166',width=1.5,dash='dash')))
        fig.add_hline(y=threshold, line_dash='dot',
                      line_color='#ff4444', opacity=0.6,
                      annotation_text="Alert Threshold")
        fig.update_layout(template='plotly_dark', height=420,
                           title='Live Consumption Stream (last 200 points)',
                           hovermode='x unified',
                           margin=dict(l=40,r=20,t=40,b=40))
        st.plotly_chart(fig, use_container_width=True)

        # Rolling stats
        c1, c2 = st.columns(2)
        with c1:
            roll_mean = live_slice['actual'].rolling(24).mean()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=live_slice.index, y=live_slice['actual'],
                                      name='Actual', opacity=0.5,
                                      line=dict(color='#58a6ff',width=0.7)))
            fig2.add_trace(go.Scatter(x=live_slice.index, y=roll_mean,
                                      name='24h Rolling Mean',
                                      line=dict(color='#56d364',width=2)))
            fig2.update_layout(template='plotly_dark', height=300,
                                title='Rolling Mean', showlegend=True,
                                margin=dict(l=30,r=10,t=30,b=30))
            st.plotly_chart(fig2, use_container_width=True)

        with c2:
            fig3 = px.histogram(x=live_slice['actual'], nbins=30,
                                 template='plotly_dark',
                                 title='Live Distribution',
                                 labels={'x':'kW'},
                                 color_discrete_sequence=['#e3b341'])
            fig3.update_layout(height=300, margin=dict(l=30,r=10,t=30,b=30))
            st.plotly_chart(fig3, use_container_width=True)

    # Advance pointer and rerun
    if auto:
        st.session_state.sim_idx = min(
            st.session_state.sim_idx + speed, len(live_df) - 1)
        time.sleep(0.8)
        st.rerun()
    else:
        if st.button("⏭ Step Forward"):
            st.session_state.sim_idx = min(
                st.session_state.sim_idx + speed, len(live_df) - 1)
            st.rerun()
        if st.button("⏮ Reset"):
            st.session_state.sim_idx = 0
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Model Comparison":
    st.title("📋 Model Comparison")
    st.markdown("---")

    def highlight_best(s):
        if s.name in ['RMSE','MAE','MAPE']:
            return ['background-color:#1c4a2e;font-weight:bold'
                    if v == s.min() else '' for v in s]
        elif s.name == 'R2':
            return ['background-color:#1c4a2e;font-weight:bold'
                    if v == s.max() else '' for v in s]
        return ['' for _ in s]

    st.subheader("Performance Table")
    st.dataframe(comp.style.apply(highlight_best), use_container_width=True)
    st.markdown("---")

    # 4 metric bar charts
    metrics = [('RMSE','lower is better','#f78166'),
               ('MAE', 'lower is better','#58a6ff'),
               ('MAPE','lower is better','#e3b341'),
               ('R2',  'higher is better','#56d364')]
    for i in range(0, 4, 2):
        c1, c2 = st.columns(2)
        for col, (metric, note, color) in zip([c1,c2], metrics[i:i+2]):
            with col:
                fig = px.bar(comp.reset_index(), x='Model', y=metric,
                             template='plotly_dark',
                             title=f'{metric} ({note})',
                             text_auto='.4f',
                             color='Model')
                fig.update_layout(height=360, showlegend=False,
                                  margin=dict(l=30,r=10,t=40,b=60))
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("---")
    st.subheader("Radar Chart — Normalized Performance")
    comp_n = comp.copy()
    for c in ['RMSE','MAE','MAPE']:
        mn, mx = comp_n[c].min(), comp_n[c].max()
        comp_n[c] = 1 - (comp_n[c]-mn)/(mx-mn+1e-9)
    r2_mn, r2_mx = comp_n['R2'].min(), comp_n['R2'].max()
    comp_n['R2'] = (comp_n['R2']-r2_mn)/(r2_mx-r2_mn+1e-9)

    cats   = ['RMSE','MAE','MAPE','R2']
    colors_r = ['#58a6ff','#f78166','#56d364','#e3b341','#e879f9','#38bdf8']
    fig = go.Figure()
    for i, (name, row) in enumerate(comp_n.iterrows()):
        vals = [row[c] for c in cats] + [row[cats[0]]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats+[cats[0]], fill='toself',
            name=name, opacity=0.65,
            line=dict(color=colors_r[i % len(colors_r)])
        ))
    fig.update_layout(template='plotly_dark', height=500,
                      polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                      margin=dict(l=40,r=40,t=40,b=40))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("All metrics normalized 0–1. Higher = better on all axes.")

    # Download
    st.download_button("⬇ Download Comparison Table",
                        comp.to_csv().encode(),
                        "model_comparison.csv","text/csv")