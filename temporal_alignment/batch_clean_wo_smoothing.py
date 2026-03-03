import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import re
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# =========================
# Configuration
# =========================

RAW_SEB_ROOT = Path("/Users/liuchenshu/Documents/Research/NUS/Project - Wang Chan - SEBLink/Human-study/TTSHDataProcessing/Organized Data")
DELTA_SEB_ROOT = Path("/Users/liuchenshu/Documents/Research/NUS/Project - Wang Chan - SEBLink/Human-study/TTSHDataProcessing/temporal_alignment/SEB_matched_to_CGM_window_avg")
CGM_ROOT = Path("/Users/liuchenshu/Documents/Research/NUS/Project - Wang Chan - SEBLink/Human-study/TTSHDataProcessing/cleaned_cgm_files")

# ---------------- HELPERS ----------------
def normalize_cols(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    return df


def get_subjects():
    return sorted([p.name for p in RAW_SEB_ROOT.iterdir() if p.is_dir()])


def get_days(subject):
    return sorted([p.name for p in (RAW_SEB_ROOT / subject).iterdir() if p.is_dir()])


def find_csv(folder, keyword=None):
    files = list(folder.glob("*.csv"))
    if keyword:
        files = [f for f in files if keyword in f.name.lower()]
    return files[0] if files else None


def load_seb(csv_path):
    df = pd.read_csv(csv_path)
    df = normalize_cols(df)

    if "Timestamp" not in df.columns:
        raise ValueError("Missing Timestamp")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    for c in df.columns:
        if c != "Timestamp":
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
    return df


def load_cgm(subject):
    m = re.match(r"P(\d+)", subject)
    if not m:
        return None

    sid = f"P{int(m.group(1)):03d}"
    f = CGM_ROOT / f"{sid}_cleaned.csv"
    if not f.exists():
        return None

    df = pd.read_csv(f)
    if "timestamp" not in df.columns or "glucose_mmol_L" not in df.columns:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["glucose_mmol_L"] = pd.to_numeric(df["glucose_mmol_L"], errors="coerce")
    df = df.dropna().sort_values("timestamp")
    return df


def normalize(series):
    vmin, vmax = series.min(), series.max()
    return (series - vmin) / (vmax - vmin if vmax != vmin else 1.0)

def resample_to_common_grid(t1, y1, t2, y2, freq="5min"):
    """
    Resample two time series onto a shared uniform grid
    """
    s1 = pd.Series(y1.values, index=t1)
    s2 = pd.Series(y2.values, index=t2)

    t_start = max(s1.index.min(), s2.index.min())
    t_end = min(s1.index.max(), s2.index.max())

    grid = pd.date_range(t_start, t_end, freq=freq)

    s1r = s1.reindex(grid).interpolate("time")
    s2r = s2.reindex(grid).interpolate("time")

    mask = s1r.notna() & s2r.notna()
    return grid[mask], s1r[mask].values, s2r[mask].values

# ---------------- Evaluation ----------------
def cross_correlation_with_lag(x, y):
    x = x - np.mean(x)
    y = y - np.mean(y)

    corr = np.correlate(x, y, mode="full")
    lags = np.arange(-len(x) + 1, len(x))

    best_idx = np.argmax(corr)
    best_lag = lags[best_idx]
    best_corr = corr[best_idx] / (np.std(x) * np.std(y) * len(x))

    return best_corr, best_lag

def compute_dtw(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("Input sequences must be 1D")
    
    dist, _ = fastdtw(x, y, dist=lambda a, b: abs(a - b))
    return dist

def derivative(x):
    return np.gradient(x)

def compute_ddtw(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    dx = derivative(x)
    dy = derivative(y)
    dist, _ = fastdtw(dx, dy, dist=lambda a, b: abs(a - b))
    return dist

# ---------------- UI ----------------
st.sidebar.header("Selection")

subjects = get_subjects()
subject = st.sidebar.selectbox("Subject", subjects)

days = get_days(subject)
day = st.sidebar.selectbox("Day", days)

show_raw = st.sidebar.checkbox("Show RAW SEB (Channel 1)", value=False)
show_cgm = st.sidebar.checkbox("Show CGM", value=True)
normalize_view = st.sidebar.checkbox("Normalize (shape only)", value=True)
compute_metrics = st.sidebar.checkbox("Compute Alignment Metrics", value=True)

st.title(f"{subject} — {day}")

# ---------------- LOAD DELTA SEB (PRIMARY) ----------------
delta_csv = find_csv(DELTA_SEB_ROOT / subject / day, keyword="delta")
if delta_csv is None:
    st.error("Delta SEB file not found.")
    st.stop()

df_delta = load_seb(delta_csv)
delta_col = [c for c in df_delta.columns if c != "Timestamp"][0]

# ---------------- LOAD RAW SEB (OPTIONAL) ----------------
df_raw = None
if show_raw:
    raw_csv = find_csv(RAW_SEB_ROOT / subject / day)
    if raw_csv is None:
        st.warning("RAW SEB file not found.")
    else:
        df_raw = load_seb(raw_csv)
        if "Channel1" not in df_raw.columns:
            st.warning("Channel1 not found in RAW SEB.")
            df_raw = None

# ---------------- LOAD CGM ----------------
df_cgm = load_cgm(subject)

# ---------------- TIME WINDOW ----------------
t0 = df_delta["Timestamp"].min()
t1 = df_delta["Timestamp"].max()

if df_raw is not None:
    t0 = min(t0, df_raw["Timestamp"].min())
    t1 = max(t1, df_raw["Timestamp"].max())

if df_cgm is not None:
    df_cgm = df_cgm[(df_cgm["timestamp"] >= t0) & (df_cgm["timestamp"] <= t1)]

# ---------------- NORMALIZATION ----------------
y_delta = df_delta[delta_col]
y_raw = df_raw["Channel1"] if df_raw is not None else None
y_cgm = df_cgm["glucose_mmol_L"] if df_cgm is not None else None

if normalize_view:
    y_delta = normalize(y_delta)
    if y_raw is not None:
        y_raw = normalize(y_raw)
    if y_cgm is not None:
        y_cgm = normalize(y_cgm)

# ---------------- ALIGNMENT METRICS ----------------
metrics = None

if compute_metrics and df_cgm is not None and not df_cgm.empty:
    grid, seb_vals, cgm_vals = resample_to_common_grid(
        df_delta["Timestamp"],
        y_delta,
        df_cgm["timestamp"],
        y_cgm,
        freq="5min"
    )
    print(f"shape of seb_vals: {seb_vals.shape}, shape of cgm_vals: {cgm_vals.shape}")

    if len(seb_vals) > 10:
        xcorr, lag = cross_correlation_with_lag(seb_vals, cgm_vals)
        dtw_dist = compute_dtw(seb_vals, cgm_vals)
        ddtw_dist = compute_ddtw(seb_vals, cgm_vals)

        metrics = {
            "Cross-correlation": round(float(xcorr), 3),
            "Best lag (samples)": int(lag),
            "Best lag (minutes)": int(lag * 5),
            "DTW distance": round(float(dtw_dist), 2),
            "Derivative DTW distance": round(float(ddtw_dist), 2),
            "Samples used": len(seb_vals)
        }

# ---------------- PLOT ----------------
fig = go.Figure()

# Delta SEB (PRIMARY)
fig.add_trace(go.Scatter(
    x=df_delta["Timestamp"],
    y=y_delta,
    mode="lines",
    name="SEB Delta (Ch2 − Ch1)"
))

# Raw SEB (OPTIONAL)
if df_raw is not None:
    fig.add_trace(go.Scatter(
        x=df_raw["Timestamp"],
        y=y_raw,
        mode="lines",
        name="SEB Channel 1 (Raw)",
        line=dict(dash="dot")
    ))

# CGM
if show_cgm and df_cgm is not None and not df_cgm.empty:
    fig.add_trace(go.Scatter(
        x=df_cgm["timestamp"],
        y=y_cgm,
        mode="lines+markers",
        name="CGM",
        yaxis="y2",
        line=dict(color="orange")
    ))

# alignment metrics
if metrics is not None:
    st.subheader("Alignment Quality Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Cross-correlation", metrics["Cross-correlation"])
    col2.metric("Lag (min)", metrics["Best lag (minutes)"])
    col3.metric("DTW distance", metrics["DTW distance"])

    with st.expander("Advanced metrics"):
        st.write({
            "Derivative DTW": metrics["Derivative DTW distance"],
            "Samples compared": metrics["Samples used"]
        })

fig.update_layout(
    xaxis_title="Time",
    yaxis_title="SEB",
    yaxis2=dict(
        title="CGM",
        overlaying="y",
        side="right"
    ),
    hovermode="x unified",
    height=650,
    legend=dict(orientation="h", y=1.05)
)

st.plotly_chart(fig, use_container_width=True)