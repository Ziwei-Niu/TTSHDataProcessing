from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

DATA_ROOT = Path("Organized Data")
PROCESSED_ROOT = Path("Processed_Organized Data")
CGM_ROOT = Path("cleaned_cgm_files")


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

    if "timestamp" not in df.columns:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "glucose_mmol_L" not in df.columns:
        return None
    s = df["glucose_mmol_L"].astype(str).str.strip().str.replace(",", ".", regex=False)
    df["glucose_mmol_L"] = pd.to_numeric(s, errors="coerce")

    df = df.dropna(subset=["timestamp", "glucose_mmol_L"]).sort_values("timestamp").reset_index(drop=True)
    return df


def compute_effective_duration(timestamps, threshold_sec=300):
    if len(timestamps) < 2:
        return pd.Timedelta(0)
    timestamps_sorted = timestamps.sort_values()
    deltas = timestamps_sorted.diff().iloc[1:]
    effective = deltas[deltas <= pd.Timedelta(seconds=threshold_sec)].sum()
    return effective


def find_first_seb_csv(folder: Path):
    seb_files = list(folder.glob("*_SEB.csv"))
    return seb_files[0] if seb_files else None


def load_seb_df(csv_path: Path, ts_col="Timestamp"):
    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        raise ValueError(f"Missing timestamp col: {ts_col}")
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # 强制所有通道为 numeric（避免 str-str 报错）
    for c in df.columns:
        if c != ts_col:
            s = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")

    return df


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

show_cgm = st.sidebar.checkbox("Show CGM Time Series", value=True)
normalize = st.sidebar.checkbox("Normalize Data for Trend Comparison", value=True)

st.title(f"Subject: {subject} — Day: {day}")

# ---------------- Load RAW SEB ----------------

day_path = DATA_ROOT / subject / day
raw_csv = find_first_seb_csv(day_path)
if raw_csv is None:
    st.error("No SEB CSV file found.")
    st.stop()

df_raw = load_seb_df(raw_csv, ts_col="Timestamp")

time_col = "Timestamp"
raw_channels = [c for c in df_raw.columns if c != time_col]
raw_channels = [c for c in raw_channels if df_raw[c].notna().any()]
if not raw_channels:
    st.error("All SEB channels are NaN after numeric conversion.")
    st.stop()

selected_channels = st.sidebar.multiselect(
    "Select SEB Channels",
    raw_channels,
    default=raw_channels
)
if not selected_channels:
    st.warning("Please select at least one channel.")
    st.stop()

day_start = df_raw["Timestamp"].min()
day_end = df_raw["Timestamp"].max()
st.markdown(f"**Day window:** {day_start} → {day_end}")

# ---------------- Load PROCESSED SEB (minimal output) ----------------
# 关键：processed CSV 只有 Timestamp + 通道（无 _denoised/_baseline_fixed）

proc_day_path = PROCESSED_ROOT / subject / day
df_proc = None
proc_csv = find_first_seb_csv(proc_day_path) if proc_day_path.exists() else None

if proc_csv is not None:
    df_proc = load_seb_df(proc_csv, ts_col="Timestamp")

    # 只保留 Timestamp + 选中的通道（存在才保留）
    keep_cols = ["Timestamp"] + [c for c in selected_channels if c in df_proc.columns]
    df_proc = df_proc[keep_cols].copy()

    if len(keep_cols) <= 1:
        st.warning("Processed file exists, but it doesn't contain the selected channels.")
        df_proc = None
else:
    st.info("No processed SEB file found under Processed_Organized Data for this subject/day.")

# ---------------- CGM ----------------

df_cgm_all = load_subject_cgm(subject)
df_cgm_day = None

if df_cgm_all is not None:
    df_cgm_day = df_cgm_all[
        (df_cgm_all["timestamp"] >= day_start) &
        (df_cgm_all["timestamp"] <= day_end)
    ].copy()
    if df_cgm_day.empty and show_cgm:
        st.warning("No CGM data in this day's time window.")
elif show_cgm:
    st.warning("No CGM file found for this subject.")

# ---------------- Clip ONLY (NO rolling) ----------------

st.sidebar.header("Value Range (Clip Only)")
clip_min_default = float(df_raw[selected_channels].min().min(skipna=True))
clip_max_default = float(df_raw[selected_channels].max().max(skipna=True))

clip_min = st.sidebar.number_input("Clip min", value=clip_min_default)
clip_max = st.sidebar.number_input("Clip max", value=clip_max_default)

# RAW clip
df_raw_clip = df_raw[["Timestamp"] + selected_channels].copy()
for c in selected_channels:
    df_raw_clip[c] = df_raw_clip[c].clip(lower=clip_min, upper=clip_max)

# PROC clip（存在才做）
df_proc_clip = None
if df_proc is not None:
    df_proc_clip = df_proc.copy()
    for c in selected_channels:
        if c in df_proc_clip.columns:
            df_proc_clip[c] = df_proc_clip[c].clip(lower=clip_min, upper=clip_max)

# ---------------- Normalize ----------------
# 建议：为了“形状对比”，RAW/PROC 各自归一化（你之前也倾向这个）
def normalize_df(df, cols):
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        vmin = out[c].min(skipna=True)
        vmax = out[c].max(skipna=True)
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0
        out[c] = (out[c] - vmin) / denom
    return out

if normalize:
    df_raw_plot = normalize_df(df_raw_clip, selected_channels)
    df_proc_plot = normalize_df(df_proc_clip, selected_channels) if df_proc_clip is not None else None

    if df_cgm_day is not None and not df_cgm_day.empty:
        g = df_cgm_day["glucose_mmol_L"]
        denom_g = (g.max() - g.min())
        df_cgm_day["glucose_norm"] = (g - g.min()) / (denom_g if denom_g != 0 else 1.0)
else:
    df_raw_plot = df_raw_clip
    df_proc_plot = df_proc_clip
    if df_cgm_day is not None and not df_cgm_day.empty:
        df_cgm_day["glucose_norm"] = df_cgm_day["glucose_mmol_L"]

# ---------------- Diet timestamps ----------------

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

# RAW（颜色自动分配）
for ch in selected_channels:
    fig.add_trace(go.Scattergl(
        x=df_raw_plot["Timestamp"],
        y=df_raw_plot[ch],
        mode="lines",
        name=f"{ch} (Raw)",
        hovertemplate="Time: %{x}<br>Raw: %{y:.3f}<extra></extra>"
    ))

# PROCESSED（同样实线，仅靠颜色区分；为了避免“完全同色”，加一个前缀）
if df_proc_plot is not None:
    for ch in selected_channels:
        if ch not in df_proc_plot.columns:
            continue
        fig.add_trace(go.Scattergl(
            x=df_proc_plot["Timestamp"],
            y=df_proc_plot[ch],
            mode="lines",
            name=f"{ch} (Processed)",
            hovertemplate="Time: %{x}<br>Processed: %{y:.3f}<extra></extra>"
        ))

# CGM
if show_cgm and df_cgm_day is not None and not df_cgm_day.empty:
    fig.add_trace(go.Scattergl(
        x=df_cgm_day["timestamp"],
        y=df_cgm_day["glucose_norm"],
        mode="lines+markers",
        name="CGM",
        line=dict(color="orange"),
        yaxis="y2",
        hovertemplate="Time: %{x}<br>Glucose: %{y:.2f}<extra></extra>"
    ))

# Diet markers
if not df_images.empty:
    for ts in df_images["Timestamp"]:
        fig.add_vline(x=ts, line=dict(color="red", dash="dash"), opacity=0.6)

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

st.subheader("SEB Effective Duration")
seb_effective_duration = compute_effective_duration(df_raw["Timestamp"], threshold_sec=5)
st.write({
    "Duration": seb_effective_duration,
    "Points": int(len(df_raw))
})