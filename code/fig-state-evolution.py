"""Single brain node + timeseries inset — State Evolution slide figure.

Saves: img/state_evolution_node.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

import bsplot
from tvbo import Dynamics

bsplot.style.use("tvbo")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(ROOT, "..", "img", "state_evolution_node.png"))

# --- Real Jansen-Rit (1995) timeseries — EEG proxy: PSP = y1 - y2 ---
_res = Dynamics.from_db("JansenRit").run(verbose=0)
_dt = _res.sample_period          # ms per step
# Find a 500 ms window with clear oscillations (highest std)
_win = int(500.0 / _dt)
_step = _win // 2
_starts = range(0, len(_res.data) - _win, _step)
_y1_full = _res.data[:, 1, 0, 0]
_y2_full = _res.data[:, 2, 0, 0]
_psp = _y1_full - _y2_full
_best = max(_starts, key=lambda s: _psp[s:s + _win].std())
_y1 = _y1_full[_best:_best + _win]
_y2 = _y2_full[_best:_best + _win]
signal = _y1 - _y2                # net pyramidal PSP ≈ EEG proxy
t = np.arange(len(signal)) * _dt / 1000.0  # seconds

NODE_COLOR = "#6880af"
zoom_r = 6  # box half-size in mm

# --- Layout ---
fig = plt.figure(figsize=(6, 2.8))
ax_brain = fig.add_axes([0.01, 0.0, 0.46, 1.0])
ax_ts = fig.add_axes([0.54, 0.18, 0.44, 0.70])

# --- Brain horizontal slice ---
bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax_brain,
    view="horizontal",
    slice_mm=10,
    cmap="gray",
)
ax_brain.axis("off")

# Read actual axis extent so we can place the node sensibly
xlim = ax_brain.get_xlim()
ylim = ax_brain.get_ylim()

# Right parietal cortex — peripheral grey matter ribbon
node_x = xlim[0] + 0.80 * (xlim[1] - xlim[0])   # right hemisphere
node_y = ylim[0] + 0.65 * (ylim[1] - ylim[0])    # mid-anterior

# Highlighted node marker
ax_brain.scatter(
    [node_x], [node_y],
    s=80, color=NODE_COLOR, zorder=6, edgecolors="white", linewidths=1.2,
)
rect = Rectangle(
    (node_x - zoom_r, node_y - zoom_r), 2 * zoom_r, 2 * zoom_r,
    linewidth=2.0, edgecolor=NODE_COLOR, facecolor="none", zorder=7,
)
ax_brain.add_patch(rect)
ax_brain.text(
    node_x, node_y - zoom_r - 3, "node $i$",
    color=NODE_COLOR, fontsize=7.5, ha="center", va="top",
)

# --- Timeseries ---
ax_ts.plot(t, signal, color=NODE_COLOR, lw=0.85)
ax_ts.set_xlabel("time [s]", fontsize=8)
ax_ts.set_ylabel(r"$y_1 - y_2$ [a.u.]", fontsize=8)
ax_ts.set_xlim(t[0], t[-1])
ax_ts.tick_params(labelsize=7)
for sp in ax_ts.spines.values():
    sp.set_color(NODE_COLOR)
    sp.set_linewidth(1.2)

# --- Connection lines from node box → timeseries ---
for xy_ax, xy_ts in [
    ((node_x + zoom_r, node_y + zoom_r), (0, 1)),
    ((node_x + zoom_r, node_y - zoom_r), (0, 0)),
]:
    fig.add_artist(
        ConnectionPatch(
            xyA=xy_ax, coordsA=ax_brain.transData,
            xyB=xy_ts, coordsB=ax_ts.transAxes,
            color=NODE_COLOR, linewidth=1.1, linestyle="--", alpha=0.55,
        )
    )

fig.savefig(OUT, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved → {OUT}")
