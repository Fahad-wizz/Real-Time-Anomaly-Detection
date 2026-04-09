# dashboard_enhanced.py
# Enhanced Streamlit dashboard for Real-Time Anomaly Detection (minimal dark theme)
# Usage:
#    streamlit run dashboard_enhanced.py
#
# Requirements:
#    streamlit, pandas, plotly, requests
#
# The app reads:
#   - data/flow_features_with_preds.parquet    (preferred)
#   - fallback: data/flow_features.parquet
#
# Optional: to re-check a window with the model server enable the checkbox and ensure:
#   uvicorn model_server:app --reload --port 8000
#

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
import requests
from pathlib import Path
from datetime import datetime

# ---------- Config ----------
FEATURES_WITH_PRED_PATH = Path("data/flow_features_with_preds.parquet")
FEATURES_PATH = Path("data/flow_features.parquet")
POLL_INTERVAL = 5                 # seconds between auto-refresh (simulates realtime)
MAX_POINTS_TS = 1200              # how many windows to keep for time-series chart

# ---------- Page layout and minimal dark theme ----------
st.set_page_config(page_title="Anomaly Detection — Live Dashboard", layout="wide", initial_sidebar_state="expanded")
# Minimal dark theme CSS
st.markdown(
    """
    <style>
    /* background */
    .stApp { background-color: #0f1115; color: #e6eef8; }
    /* card backgrounds */
    .card { background-color: #111418; padding:12px; border-radius:8px; box-shadow:0 1px 0 rgba(255,255,255,0.02); }
    .kpi { font-size:28px; font-weight:700; color:#ffffff; }
    .kpi-sub { color:#9fb2d0; font-size:12px; }
    .small { font-size:12px; color:#9fb2d0; }
    /* table header */
    .st-table thead tr th { background-color: #0b0d10; color: #cfe3ff; }
    /* reduce chart bg grid */
    .plotly-graph-div .main-svg { background: transparent; }
    </style>
    """, unsafe_allow_html=True)

st.title("☁️ Real-Time Network Anomaly Dashboard")
st.markdown("Minimal dark theme • Streaming-style updates • IsolationForest + Autoencoder results")

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Controls")
    poll = st.number_input("Refresh interval (sec)", min_value=1, max_value=60, value=POLL_INTERVAL, step=1)
    last_n = st.slider("Show last N windows", min_value=50, max_value=2000, value=400, step=50)
    call_model_server = st.checkbox("Enable model-server re-check (http://localhost:8000/predict)", value=False)
    anomaly_threshold_ae = st.number_input("AE MSE threshold", min_value=0.0, value=1.0, step=0.1)
    st.markdown("---")
    st.markdown("⚙️ Tips:\n- Run `python stream_processor.py` then `python model_train.py`.\n- Start model server (`uvicorn model_server:app --port 8000`) if using re-check.")
    st.markdown("Dataset files: `data/flow_features_with_preds.parquet` (preferred)")

# ---------- Data loading helper ----------
@st.cache_data(ttl=10)
def load_features():
    if FEATURES_WITH_PRED_PATH.exists():
        df = pd.read_parquet(FEATURES_WITH_PRED_PATH)
    elif FEATURES_PATH.exists():
        df = pd.read_parquet(FEATURES_PATH)
        # ensure there's an 'iso_pred' column placeholder
        if 'iso_pred' not in df.columns:
            df['iso_pred'] = 0
    else:
        return pd.DataFrame()
    # normalize column names
    df = df.rename(columns=lambda c: c.strip())
    # ensure numeric types
    for c in ['proto_count','avg_len','std_len','pkt_count']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0
    # ensure window_start exists and convert to datetime for plotting
    if 'window_start' in df.columns:
        try:
            df['window_start_ts'] = pd.to_datetime(df['window_start'].astype(int) * 60, unit='s')  # window id -> epoch sec
        except Exception:
            # fallback if it's already an epoch value
            try:
                df['window_start_ts'] = pd.to_datetime(df['window_start'], unit='s')
            except Exception:
                df['window_start_ts'] = pd.Timestamp.now()
    else:
        df['window_start_ts'] = pd.Timestamp.now()
    # ensure iso_pred in 0/1
    if 'iso_pred' in df.columns:
        df['iso_pred'] = pd.to_numeric(df['iso_pred'], errors='coerce').fillna(0).astype(int)
    else:
        df['iso_pred'] = 0
    # add anomaly tag from AE (if recon mse stored)
    if 'recon_mse' in df.columns:
        df['ae_pred'] = (df['recon_mse'] > anomaly_threshold_ae).astype(int)
    else:
        df['ae_pred'] = 0
    return df

# ---------- Main refresh loop (single iteration per load; use st.experimental_rerun to refresh) ----------
st.markdown("### Live KPIs")
kpi1, kpi2, kpi3, kpi4 = st.columns([2,2,2,3])
df = load_features()
if df.empty:
    st.warning("No feature file found. Run `stream_processor.py` and `model_train.py` to generate `data/flow_features_with_preds.parquet`.")
    st.stop()

# compute KPIs
total_windows = len(df)
total_anomalies_iso = int(df['iso_pred'].sum())
total_anomalies_ae = int(df.get('ae_pred', pd.Series(0)).sum())
anomaly_rate = total_anomalies_iso / total_windows if total_windows>0 else 0.0
latest_window_time = df['window_start_ts'].max()

kpi1.markdown("<div class='card'><div class='kpi'>{}</div><div class='kpi-sub'>Total windows</div></div>".format(total_windows), unsafe_allow_html=True)
kpi2.markdown("<div class='card'><div class='kpi'>{}</div><div class='kpi-sub'>Anomalies (IsolationForest)</div></div>".format(total_anomalies_iso), unsafe_allow_html=True)
kpi3.markdown("<div class='card'><div class='kpi'>{:.2%}</div><div class='kpi-sub'>Anomaly rate</div></div>".format(anomaly_rate), unsafe_allow_html=True)
kpi4.markdown("<div class='card'><div class='kpi'>{}</div><div class='kpi-sub'>Latest window: {}</div></div>".format(latest_window_time.strftime("%Y-%m-%d %H:%M:%S"), ''), unsafe_allow_html=True)

# ---------- Time-series: anomalies over time ----------
st.markdown("### Anomaly count over time (per-window)")
ts_df = df.sort_values('window_start_ts').tail(MAX_POINTS_TS)
if ts_df.empty:
    st.info("No time-series points yet.")
else:
    ts_counts = ts_df.groupby('window_start_ts')['iso_pred'].sum().reset_index().rename(columns={'iso_pred':'anomaly_count'})
    fig_ts = px.line(ts_counts, x='window_start_ts', y='anomaly_count', markers=True, title="Anomalies per window (IsolationForest)")
    fig_ts.update_layout(template="plotly_dark", height=300, margin=dict(t=40,l=10,r=10,b=10))
    st.plotly_chart(fig_ts, use_container_width=True)

# ---------- Scatter: feature space with anomaly coloring ----------
st.markdown("### Feature-space (interactive)")
scatter_cols = st.multiselect("Choose axes", options=['avg_len','proto_count','std_len','pkt_count'], default=['avg_len','proto_count'])
x_col, y_col = scatter_cols[0], scatter_cols[1] if len(scatter_cols)>1 else scatter_cols[0]
fig_scatter = px.scatter(ts_df, x=x_col, y=y_col, color=ts_df['iso_pred'].map({0:'Normal',1:'Anomaly'}),
                         hover_data=['src_ip','window_start_ts','pkt_count'], title=f"{y_col} vs {x_col}")
fig_scatter.update_layout(template="plotly_dark", height=400)
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------- Top anomalous sources ----------
st.markdown("### Top anomalous source IPs (IsolationForest)")
top_src = df[df['iso_pred']==1].groupby('src_ip').size().reset_index(name='anomaly_windows').sort_values('anomaly_windows', ascending=False).head(10)
if top_src.empty:
    st.info("No anomalous sources detected yet.")
else:
    st.table(top_src)

# ---------- Recent anomalies table with optional model-server re-check ----------
st.markdown("### Recent flagged windows (IsolationForest)")
recent_anoms = df[df['iso_pred']==1].sort_values('window_start_ts', ascending=False).head(100)
if recent_anoms.empty:
    st.info("No recent anomalies to show.")
else:
    # small columns displayed
    disp = recent_anoms[['window_start_ts','src_ip','proto_count','avg_len','std_len','pkt_count','iso_pred']].copy()
    disp = disp.rename(columns={'window_start_ts':'time','src_ip':'src','proto_count':'proto_count','avg_len':'avg_len'})
    st.dataframe(disp.reset_index(drop=True).astype(str))

    if call_model_server:
        st.markdown("**Re-checking last 10 flagged windows with model server...**")
        recheck = recent_anoms.head(10)
        recheck_results = []
        for _, r in recheck.iterrows():
            payload = {
                'proto_count': float(r['proto_count']),
                'avg_len': float(r['avg_len']),
                'std_len': float(r['std_len']),
                'pkt_count': float(r['pkt_count'])
            }
            try:
                resp = requests.post('http://localhost:8000/predict', json=payload, timeout=3.0)
                j = resp.json()
                recheck_results.append({ 'window': str(r['window_start_ts']), 'src': r['src_ip'], 'server': j })
            except Exception as e:
                recheck_results.append({ 'window': str(r['window_start_ts']), 'src': r['src_ip'], 'server': f"ERROR: {e}" })
        st.write(recheck_results)

# ---------- Alerts panel (simple) ----------
st.markdown("### Alerts")
# define high severity as windows with high std_len or high avg_len and flagged
high_sev = df[(df['iso_pred']==1) & ((df['std_len'] > df['std_len'].quantile(0.9)) | (df['avg_len'] > df['avg_len'].quantile(0.95)))]
if high_sev.empty:
    st.write("No high-severity alerts.")
else:
    for _, r in high_sev.sort_values('window_start_ts', ascending=False).head(10).iterrows():
        st.markdown(f"**{r['window_start_ts']}** — src: `{r['src_ip']}` — avg_len: {r['avg_len']:.1f}, std_len: {r['std_len']:.1f} — flagged")

# ---------- Footer / auto-refresh ----------
st.markdown("---")
st.markdown(f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}. Poll interval: {poll} sec.")
# Auto-refresh button (uses Streamlit's experimental rerun) — we can't run background threads here so we simulate polling
refresh = st.button("Manual refresh")
if refresh:
    st.rerun()

# Use st_autorefresh style behavior: sleep then rerun (only if page is active)
if 'auto_refresh' not in st.session_state:
    st.session_state['auto_refresh'] = True

if st.session_state['auto_refresh']:
    time.sleep(poll)
    st.rerun()
