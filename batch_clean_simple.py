# -*- coding: utf-8 -*-
"""
SUPER SIMPLE SEB CLEANING + STEP CORRECTION (Python 3.8 compatible)
(Using Savitzky–Golay for smoothing)

Pipeline (per channel):
1) light denoise (savgol)                         --smooth_win --smooth_poly
2) short spike removal (Hampel -> NaN -> linear)  --spike_win --spike_sigma
3) long artifact removal (rolling std -> interp)  --long_win --long_std_k --long_min_len
4) STEP baseline correction (piecewise offset)    --step_win --step_thr_k --min_step_separation --max_steps

Output CSV:
Timestamp + processed channels ONLY (same channel names)

Notes:
- No warmup
- No extra metadata
- Requires scipy for savgol_filter.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter


# Utils
def mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 1e-12
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med)) + 1e-12


def savgol_smooth(x, win, poly):
    """
    Savitzky–Golay smoothing.
    - win must be odd and >= poly+2
    - If signal too short or invalid params, return original.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 5 or win <= 1:
        return x.copy()

    win = int(win)
    poly = int(poly)

    if win % 2 == 0:
        win += 1
    win = max(win, poly + 2)
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if (n % 2 == 1) else (n - 1)
    if win < 5 or win <= poly:
        return x.copy()

    # Fill NaNs temporarily for filtering
    xx = x.copy()
    if np.isnan(xx).any():
        s = pd.Series(xx)
        if s.notna().sum() == 0:
            return x.copy()
        xx = s.interpolate(limit_direction="both").to_numpy(dtype=float)

    try:
        return savgol_filter(xx, window_length=win, polyorder=poly, mode="interp").astype(float)
    except Exception:
        return x.copy()


#Hampel spike removal
def hampel_to_nan(x, win=21, sigma=3.0):
    """
    Mark short spikes as NaN using Hampel rule:
      |x[i] - median(window)| > sigma * MAD(window)
    """
    x = np.asarray(x, dtype=float).copy()
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    half = win // 2
    n = len(x)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size < 3:
            continue
        m = np.median(seg)
        s = mad(seg)
        if np.isfinite(x[i]) and abs(x[i] - m) > float(sigma) * s:
            x[i] = np.nan
    return x


def fill_nan_linear(x):
    s = pd.Series(np.asarray(x, dtype=float))
    if s.notna().sum() == 0:
        return s.to_numpy(dtype=float)
    return s.interpolate(limit_direction="both").to_numpy(dtype=float)


#long artifact removal (rolling std)
def remove_long_segments(y, win=15, std_thr_k=4.0, min_len=20):
    """
    Detect long abnormal segments using rolling std thresholding.
    Replace the entire segment by linear interpolation between segment boundaries.
    """
    y = np.asarray(y, dtype=float).copy()
    if y.size < 5:
        return y

    win = int(max(2, win))
    s = pd.Series(y)
    rstd = s.rolling(win, center=True, min_periods=1).std().to_numpy(dtype=float)

    base = np.nanmedian(rstd)
    if not np.isfinite(base) or base <= 0:
        return y

    mask = rstd > (float(std_thr_k) * base)
    n = len(mask)
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        j = i
        while j < n and mask[j]:
            j += 1

        if (j - i) >= int(min_len):
            left = i - 1
            right = j

            if left >= 0 and right < n and np.isfinite(y[left]) and np.isfinite(y[right]):
                y0 = y[left]
                y1 = y[right]
                denom = float(right - left)
                for k in range(i, j):
                    t = (k - left) / denom
                    y[k] = (1 - t) * y0 + t * y1
            else:
                y[i:j] = np.nan

        i = j

    if np.isnan(y).any():
        y = fill_nan_linear(y)

    return y


# STEP correction
def two_window_mean_diff(y, w):
    """
    score[i] = mean(y[i:i+w]) - mean(y[i-w:i])
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = int(max(1, w))
    if n == 0:
        return np.array([], dtype=float)
    if w == 1:
        return np.diff(y, prepend=y[0]).astype(float)

    c = np.concatenate(([0.0], np.cumsum(y)))
    score = np.zeros(n, dtype=float)

    for i in range(n):
        loL = max(0, i - w)
        hiL = i
        loR = i
        hiR = min(n, i + w)

        left_mean = (c[hiL] - c[loL]) / (hiL - loL) if hiL > loL else y[i]
        right_mean = (c[hiR] - c[loR]) / (hiR - loR) if hiR > loR else y[i]
        score[i] = right_mean - left_mean

    return score


def detect_steps(y, step_win=10, thr_k=5.0, min_separation=600, max_steps=3):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 10 or int(max_steps) <= 0:
        return np.array([], dtype=int)

    score = two_window_mean_diff(y, w=int(step_win))
    scale = mad(score)
    cand = np.where(np.isfinite(score) & (np.abs(score) > float(thr_k) * scale))[0]
    if cand.size == 0:
        return np.array([], dtype=int)

    order = np.argsort(-np.abs(score[cand]))
    cand = cand[order]

    selected = []
    for idx in cand:
        idx = int(idx)
        if selected and any(abs(idx - s) < int(min_separation) for s in selected):
            continue
        selected.append(idx)
        if len(selected) >= int(max_steps):
            break

    return np.array(sorted(selected), dtype=int)


def apply_step_correction(y, steps, win=30):
    y = np.asarray(y, dtype=float)
    n = len(y)
    steps = np.asarray(steps, dtype=int)
    if steps.size == 0:
        return y.copy()

    win = int(max(1, win))
    offsets = np.zeros(n, dtype=float)
    cum = 0.0
    prev = 0

    for s in steps:
        s = int(s)
        if s <= 0 or s >= n:
            continue

        lo1, hi1 = max(0, s - win), s
        lo2, hi2 = s, min(n, s + win)

        seg1 = y[lo1:hi1]
        seg2 = y[lo2:hi2]
        seg1 = seg1[np.isfinite(seg1)]
        seg2 = seg2[np.isfinite(seg2)]

        if seg1.size < 3 or seg2.size < 3:
            continue

        m1 = float(np.mean(seg1))
        m2 = float(np.mean(seg2))
        jump = m2 - m1

        offsets[prev:s] = cum
        cum += jump
        prev = s

    offsets[prev:] = cum
    return y - offsets


# Channel processing
def process_channel(y, args):
    y = np.asarray(y, dtype=float)

    # 1) savgol smooth
    y = savgol_smooth(y, win=args.smooth_win, poly=args.smooth_poly)

    # 2) short spikes -> NaN -> linear interp
    y = hampel_to_nan(y, win=args.spike_win, sigma=args.spike_sigma)
    y = fill_nan_linear(y)

    # 3) long artifacts
    y = remove_long_segments(
        y,
        win=args.long_win,
        std_thr_k=args.long_std_k,
        min_len=args.long_min_len
    )

    # 4) steps
    steps = detect_steps(
        y,
        step_win=args.step_win,
        thr_k=args.step_thr_k,
        min_separation=args.min_step_separation,
        max_steps=args.max_steps
    )
    y = apply_step_correction(y, steps, win=args.step_jump_win)

    return y


# File processin
def process_file(in_path, out_path, args):
    df = pd.read_csv(in_path)

    if "Timestamp" not in df.columns:
        print("[SKIP missing Timestamp]", in_path)
        return False

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        print("[SKIP empty after Timestamp parse]", in_path)
        return False

    channel_cols = [c for c in df.columns if c != "Timestamp"]
    for c in channel_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # group duplicate timestamps (mean)
    if channel_cols:
        df = df.groupby("Timestamp", as_index=False)[channel_cols].mean().sort_values("Timestamp").reset_index(drop=True)

    channel_cols = [c for c in df.columns if c != "Timestamp"]
    channel_cols = [c for c in channel_cols if df[c].notna().any()]
    if not channel_cols:
        print("[SKIP no valid channels]", in_path)
        return False

    out = pd.DataFrame({"Timestamp": df["Timestamp"]})
    for ch in channel_cols:
        out[ch] = process_channel(df[ch].to_numpy(dtype=float), args)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[OK]", in_path, "->", out_path)
    return True


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_root", required=True)
    ap.add_argument("--output_root", required=True)

    # keep your original params (same defaults)
    ap.add_argument("--smooth_win", type=int, default=3)          # now used by savgol
    ap.add_argument("--smooth_poly", type=int, default=2)         # new (savgol polyorder)

    ap.add_argument("--spike_win", type=int, default=21)
    ap.add_argument("--spike_sigma", type=float, default=3.0)
    ap.add_argument("--long_win", type=int, default=15)
    ap.add_argument("--long_std_k", type=float, default=4.0)
    ap.add_argument("--long_min_len", type=int, default=20)

    # step params
    ap.add_argument("--step_win", type=int, default=10)
    ap.add_argument("--step_thr_k", type=float, default=5.0)
    ap.add_argument("--min_step_separation", type=int, default=600)
    ap.add_argument("--max_steps", type=int, default=3)
    ap.add_argument("--step_jump_win", type=int, default=30)

    args = ap.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

    files = sorted(input_root.rglob("*_SEB.csv"))
    if not files:
        print("[WARN] no *_SEB.csv found under", input_root)
        return

    ok = 0
    skip = 0
    fail = 0

    for f in files:
        out_path = output_root / f.relative_to(input_root)
        try:
            if process_file(f, out_path, args):
                ok += 1
            else:
                skip += 1
        except Exception as e:
            print(f"[FAIL] {f} ({type(e).__name__}): {e}")
            fail += 1

    print(f"Done. OK={ok}, SKIP={skip}, FAIL={fail}, total={len(files)}")


if __name__ == "__main__":
    main()