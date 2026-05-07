"""2D phase portraits (star layout): phase planes in the centre 2×2,
timeseries of x1 at the four corners.

Nullclines are removed (not reliably placed); equilibrium at the
analytically-known origin is highlighted in red.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from _bif_common import IMG, NODE_2D, SADDLE_2D, FOCUS_2D, CENTRE_2D, save
from tvbo import Dynamics

ACCENT = "#c85030"
OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

# (title, yaml, key, timeseries_duration)
CASES = [
    ("Stable node",  NODE_2D,   "node",   5.0),
    ("Saddle",       SADDLE_2D, "saddle", 2.0),
    ("Stable focus", FOCUS_2D,  "focus",  20.0),
    ("Centre",       CENTRE_2D, "centre", 10.0),
]

# Star layout — phase planes at inner 2×2, timeseries at corners
#
#  ┌──────────┬──────┬──────┬──────────┐
#  │ ts_node  │      │      │ ts_saddle│
#  ├──────────┼──────┼──────┼──────────┤
#  │          │ PP_N │ PP_S │          │
#  ├──────────┼──────┼──────┼──────────┤
#  │          │ PP_F │ PP_C │          │
#  ├──────────┼──────┼──────┼──────────┤
#  │ ts_focus │      │      │ ts_centre│
#  └──────────┴──────┴──────┴──────────┘
PP_POS = {"node": (1, 1), "saddle": (1, 2), "focus": (2, 1), "centre": (2, 2)}
TS_POS = {"node": (0, 0), "saddle": (0, 3), "focus": (3, 0), "centre": (3, 3)}

fig = plt.figure(figsize=(11, 11))
gs = fig.add_gridspec(
    4, 4,
    width_ratios=[1.3, 2.0, 2.0, 1.3],
    height_ratios=[1.3, 2.0, 2.0, 1.3],
    hspace=0.32, wspace=0.32,
)


def _clean_nullclines(ax):
    """Remove nullcline lines (label contains '= 0') and the axes legend."""
    for line in list(ax.get_lines()):
        if "= 0" in str(line.get_label()):
            line.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()


for title, yml, key, dur in CASES:
    dyn = Dynamics.from_string(yml)

    # ── Phase plane ──────────────────────────────────────────────────────────
    pp_ax = fig.add_subplot(gs[PP_POS[key]])
    dyn.plot("x1", "x2", kind="phaseplane", ax=pp_ax,
             grid_n=20, n_trajectories=4, duration=15)
    _clean_nullclines(pp_ax)
    # All four systems have their equilibrium analytically at the origin
    pp_ax.plot(0, 0, "o", color=ACCENT, ms=9, mec="white", mew=1.5,
               zorder=15, clip_on=False)
    pp_ax.set_title(title, fontsize=10)

    # ── Timeseries ───────────────────────────────────────────────────────────
    ts_ax = fig.add_subplot(gs[TS_POS[key]])
    dyn.plot("x1", kind="timeseries", duration=dur, dt=0.01, ax=ts_ax)
    ts_ax.set_title(f"{title}\n$x_1(t)$", fontsize=8.5)
    ts_ax.set_xlabel("$t$", fontsize=8)
    ts_ax.set_ylabel("$x_1$", fontsize=8)
    ts_ax.tick_params(labelsize=7)
    if ts_ax.get_legend() is not None:
        ts_ax.get_legend().remove()

save(fig, OUT)
