from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

DATA_ROOT = Path("Organized Data")
CGM_ROOT = Path("cleaned_cgm_files")


# ---------------- helpers ----------------

def get_subjects():
    return sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir()])


def get_days(subject):
    subject_path = DATA_ROOT / subject
    return sorted([p.name for p in subject_path.iterdir() if p.is_dir()])


def load_subject_cgm(subject):
    """Load full CGM (all days combined) for one subject. Maps P01 → P001"""
    match = re.match(r"P(\d+)", subject)
    if not match:
        return None

    subject_num = int(match.group(1))
    cgm_subject = f"P{subject_num:03d}"
    cgm_file = CGM_ROOT / f"{cgm_subject}_cleaned.csv"
    if not cgm_file.exists():
        return None

    df = pd.read_csv(cgm_file)

    # timestamp -> datetime
    if "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # glucose -> numeric (fix str - str)
    if "glucose_mmol_L" not in df.columns:
        return None
    s = df["glucose_mmol_L"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["glucose_mmol_L"] = pd.to_numeric(s, errors="coerce")

    # clean
    df = df.dropna(subset=["timestamp", "glucose_mmol_L"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        return None
    return df


def compute_effective_duration(timestamps, threshold_sec=300):
    """
    Compute effective duration ignoring gaps.
    - timestamps: pd.Series of datetime
    - threshold_sec: gaps larger than this are ignored (default 5 min)
    """
    if len(timestamps) < 2:
        return pd.Timedelta(0)

    timestamps_sorted = timestamps.sort_values()
    deltas = timestamps_sorted.diff().iloc[1:]  # skip first NaT
    effective = deltas[deltas <= pd.Timedelta(seconds=threshold_sec)].sum()
    return effective


# ---------------- UI ----------------

st.sidebar.header("Data Selection")
subjects = get_subjects()
if not subjects:
    st.error("No subject folders found under 'Organized Data'.")
    st.stop()

subject = st.sidebar.selectbox("Select Subject", subjects)
days = get_days(subject)
if not days:
    st.error(f"No day folders found under Organized Data/{subject}.")
    st.stop()

day = st.sidebar.selectbox("Select Day", days)

# Checkbox to show CGM
show_cgm = st.sidebar.checkbox("Show CGM Time Series", value=True)

# Checkbox to normalize for trend comparison
normalize = st.sidebar.checkbox("Normalize Data for Trend Comparison", value=True)

st.title(f"Subject: {subject} — Day: {day}")

# ---------------- Load SEB ----------------

day_path = DATA_ROOT / subject / day
seb_files = list(day_path.glob("*_SEB.csv"))
if not seb_files:
    st.error("No SEB CSV file found.")
    st.stop()

df_ts = pd.read_csv(seb_files[0])

# timestamp -> datetime (supports ms)
if "Timestamp" not in df_ts.columns:
    st.error("SEB CSV missing 'Timestamp' column.")
    st.stop()

df_ts["Timestamp"] = pd.to_datetime(df_ts["Timestamp"], errors="coerce")
df_ts = df_ts.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

time_col = "Timestamp"
channel_cols = [c for c in df_ts.columns if c != time_col]
if not channel_cols:
    st.error("No SEB channels found (only Timestamp column exists).")
    st.stop()

# force channels numeric to avoid str arithmetic
for c in channel_cols:
    s = df_ts[c].astype(str).str.strip().str.replace(",", ".", regex=False)
    df_ts[c] = pd.to_numeric(s, errors="coerce")

# remove channels that are all NaN (optional but safer)
channel_cols = [c for c in channel_cols if df_ts[c].notna().any()]
if not channel_cols:
    st.error("All SEB channels are NaN after numeric conversion.")
    st.stop()

selected_channels = st.sidebar.multiselect(
    "Select SEB Channels",
    channel_cols,
    default=channel_cols
)

if not selected_channels:
    st.warning("No SEB channel selected.")
    st.stop()

day_start = df_ts["Timestamp"].min()
day_end = df_ts["Timestamp"].max()
st.markdown(f"**Day window:** {day_start} → {day_end}")

# ---------------- Load CGM ----------------

df_cgm_all = load_subject_cgm(subject)
df_cgm_day = None

if df_cgm_all is not None:
    df_cgm_day = df_cgm_all[
        (df_cgm_all["timestamp"] >= day_start) &
        (df_cgm_all["timestamp"] <= day_end)
    ].copy()

    if (df_cgm_day is None or df_cgm_day.empty) and show_cgm:
        st.warning("No CGM data in this day's time window.")
elif show_cgm:
    st.warning("No CGM file found or CGM file invalid for this subject.")

# ---------------- Clip + Rolling ----------------

st.sidebar.header("Denoising / Smoothing")

clip_min = st.sidebar.number_input(
    "Clip min",
    value=float(df_ts[selected_channels].min().min(skipna=True))
)
clip_max = st.sidebar.number_input(
    "Clip max",
    value=float(df_ts[selected_channels].max().max(skipna=True))
)

rolling_window = st.sidebar.slider(
    "Rolling window",
    min_value=1,
    max_value=500,
    value=1
)

df_plot = df_ts[selected_channels].clip(lower=clip_min, upper=clip_max)

if rolling_window > 1:
    # rolling mean smoothing
    df_plot = df_plot.rolling(window=rolling_window, min_periods=1).mean()

# ---------------- Normalize ----------------

if normalize:
    # per-channel normalization
    denom = (df_plot.max(skipna=True) - df_plot.min(skipna=True)).replace(0, 1.0)
    df_plot_norm = (df_plot - df_plot.min(skipna=True)) / denom

    if df_cgm_day is not None and not df_cgm_day.empty:
        g = df_cgm_day["glucose_mmol_L"]
        denom_g = (g.max() - g.min())
        df_cgm_day["glucose_norm"] = (g - g.min()) / (denom_g if denom_g != 0 else 1.0)
else:
    df_plot_norm = df_plot
    if df_cgm_day is not None and not df_cgm_day.empty:
        df_cgm_day["glucose_norm"] = df_cgm_day["glucose_mmol_L"]

# ---------------- Diet / drug timestamps ----------------

image_dir = day_path / "Diet"
df_images = pd.DataFrame()
if image_dir.exists():
    timestamps = []
    for img in image_dir.glob("*.jpg"):
        match = re.match(r"(\d{8}_\d{6})", img.stem)
        if match:
            ts = pd.to_datetime(match.group(1), format="%Y%m%d_%H%M%S", errors="coerce")
            if pd.notna(ts):
                timestamps.append(ts)
    if timestamps:
        df_images = pd.DataFrame({"Timestamp": timestamps})

# ---------------- Plot ----------------

fig = go.Figure()

# SEB channels
for ch in selected_channels:
    fig.add_trace(go.Scattergl(
        x=df_ts["Timestamp"],
        y=df_plot_norm[ch],
        mode="lines",
        name=ch,
        hovertemplate="Time: %{x}<br>Value: %{y:.4f}<extra></extra>"
    ))

# CGM
if show_cgm and df_cgm_day is not None and (not df_cgm_day.empty):
    fig.add_trace(go.Scattergl(
        x=df_cgm_day["timestamp"],
        y=df_cgm_day["glucose_norm"],
        mode="lines+markers",
        name="CGM",
        line=dict(color="orange"),
        yaxis="y2",
        hovertemplate="Time: %{x}<br>Glucose: %{y:.3f}<extra></extra>"
    ))

# Diet / med markers
if not df_images.empty:
    for ts in df_images["Timestamp"]:
        fig.add_vline(
            x=ts,
            line=dict(color="red", dash="dash"),
            opacity=0.6
        )

# Layout
fig.update_layout(
    title=f"{subject} — {day}",
    xaxis=dict(title="Time"),
    yaxis=dict(title="SEB Signals"),
    yaxis2=dict(title="CGM (normalized)", overlaying="y", side="right"),
    hovermode="x unified",
    height=650,
    margin=dict(l=60, r=60, t=80, b=50),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- Summaries ----------------

# CGM summary
if show_cgm:
    st.subheader("CGM Summary (Selected Day)")
    if df_cgm_day is not None and not df_cgm_day.empty:
        effective_duration = compute_effective_duration(df_cgm_day["timestamp"], threshold_sec=10 * 60)
        st.write({
            "Points": int(len(df_cgm_day)),
            "Mean (mmol/L)": round(float(df_cgm_day["glucose_mmol_L"].mean()), 2),
            "Min (mmol/L)": round(float(df_cgm_day["glucose_mmol_L"].min()), 2),
            "Max (mmol/L)": round(float(df_cgm_day["glucose_mmol_L"].max()), 2),
            "Effective Duration": effective_duration
        })
    else:
        st.write("No CGM data available for the selected day.")

# SEB summary
st.subheader("SEB Effective Duration")
seb_effective_duration = compute_effective_duration(df_ts["Timestamp"], threshold_sec=5)
st.write({
    "Duration": seb_effective_duration,
    "Points": int(len(df_ts))
})