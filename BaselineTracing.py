###
# This code is for manually adjusting the recorded points (baseline correction)
# The recorded data will have N_ANCHORS number of anchor points available for adjustment
# The horizontal coordinate of the anchors are fixed, so the points can only be adjusted vertically
# Once the anchors are adjusted based on domain knowledge (i.e. baseline corrected), the rest of the points in the gaps are interpolated
# The full interpolated values are saved as separate Trend_channel column in a separate csv file
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("Organized Data/P01/Day01/P01_Day01_251125_SEB.csv", parse_dates=["Timestamp"])
x = df["Timestamp"]
y = df["Channel 1"]

# Convert timestamps to float for interpolation
x_num = x.astype("int64") / 1e9

# ----------------------------
# Anchor points
# ----------------------------
N_ANCHORS = 240 # adjust anchors if needed
anchor_idx = np.linspace(0, len(df) - 1, N_ANCHORS).astype(int)

ax_x = x_num.iloc[anchor_idx].values
ax_y = y.iloc[anchor_idx].values.copy()  # make a mutable copy

# ----------------------------
# Plot
# ----------------------------
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(x, y, alpha=0.3, label="Ch1")

trend_line, = ax.plot(
    x.iloc[anchor_idx],
    ax_y,
    "-r",
    lw=2,
    label="Trend"
)

# Create scatter with pickable points
points = ax.scatter(
    x.iloc[anchor_idx],
    ax_y,
    s=10,
    color="red",
    picker=True  # enable picking
)

ax.set_title("Drag points vertically (X is fixed)")
ax.legend()

# ----------------------------
# Drag logic
# ----------------------------
active_idx = None

def on_pick(event):
    global active_idx
    if event.artist != points:
        return
    active_idx = event.ind[0]

def on_motion(event):
    global active_idx
    if active_idx is None or event.inaxes != ax:
        return

    # Update only the Y value
    ax_y[active_idx] = event.ydata

    # Update scatter offsets **in place**
    offsets = points.get_offsets()
    offsets[active_idx, 1] = event.ydata
    points.set_offsets(offsets)

    # Update trend line
    trend_line.set_ydata(ax_y)

    fig.canvas.draw_idle()

def on_release(event):
    global active_idx
    active_idx = None

def on_close(event):
    # Interpolate full trend over all timestamps
    full_trend = np.interp(
        x_num,
        ax_x,
        ax_y,
        left=np.nan,
        right=np.nan
    )
    df["Trend_Ch1"] = full_trend
    df.to_csv("annotated.csv", index=False) # change export data file nme if needed (prevent override)
    print("✅ Trendline saved!")

# Connect events
fig.canvas.mpl_connect("pick_event", on_pick)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("close_event", on_close)

plt.show()
