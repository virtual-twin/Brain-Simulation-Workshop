"""2D phase portraits in a 2×4 layout.

The middle two columns show the phase planes; the outer columns show the
corresponding x1 time series. Built-in fixed-point markers/nullclines are
disabled because these linear examples have a known equilibrium at the origin.
"""
from __future__ import annotations

import os
import numpy as np
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

# 2×4 layout: middle columns are phase planes, outer columns are time series.
#
#  ┌──────────┬──────┬──────────┬────────────┐
#  │ ts_node  │ PP_N │ PP_S     │ ts_saddle  │
#  ├──────────┼──────┼──────────┼────────────┤
#  │ ts_focus │ PP_F │ PP_C     │ ts_centre  │
#  └──────────┴──────┴──────────┴────────────┘
PP_POS = {"node": (0, 1), "saddle": (0, 2), "focus": (1, 1), "centre": (1, 2)}
TS_POS = {"node": (0, 0), "saddle": (0, 3), "focus": (1, 0), "centre": (1, 3)}

TRAJ_IC = {
    "node": [(-1.6, 1.2), (1.3, 0.8), (-0.9, -1.4)],
    "saddle": [(-1.8, 0.8), (1.8, -0.8), (-0.8, -1.8), (0.8, 1.8)],
    "focus": [(1.5, 0.0), (0.0, 1.5), (-1.2, 0.6)],
    "centre": [(1.5, 0.0), (0.0, 1.5), (-1.2, 0.6)],
}

fig = plt.figure(figsize=(12.2, 6.0))
gs = fig.add_gridspec(
    2, 4,
    width_ratios=[1.35, 2.0, 2.0, 1.35],
    height_ratios=[1.0, 1.0],
    hspace=0.34, wspace=0.34,
)


for title, yml, key, dur in CASES:
    dyn = Dynamics.from_string(yml)
    state_names = list(dyn.state_variables)
    base_ic = np.asarray(dyn.get_initial_values(), dtype=float).reshape(-1)
    i1 = state_names.index("x1")
    i2 = state_names.index("x2")

    # ── Phase plane ──────────────────────────────────────────────────────────
    pp_ax = fig.add_subplot(gs[PP_POS[key]])
    dyn.plot(
        "x1", "x2",
        kind="vectorfield",
        ax=pp_ax,
        grid_n=20,
        stream=True,
    )

    # Manual trajectories only. Avoid auto nullclines / fixed-point markers.
    for x1_0, x2_0 in TRAJ_IC[key]:
        u0 = base_ic.copy()
        u0[i1] = x1_0
        u0[i2] = x2_0
        ts = dyn.run(duration=15, dt=0.01, u_0=u0, save=False)
        traj = ts.data[:, :, 0, 0]
        pp_ax.plot(traj[:, i1], traj[:, i2], color="red", lw=1.6, alpha=0.95, zorder=8)

    # Known equilibrium at the origin only.
    pp_ax.plot(0, 0, "o", color=ACCENT, ms=9, mec="white", mew=1.5, zorder=15, clip_on=False)
    pp_ax.set_title(title, fontsize=10)

    # ── Timeseries ───────────────────────────────────────────────────────────
    ts_ax = fig.add_subplot(gs[TS_POS[key]])
    dyn.plot("x1", kind="timeseries", duration=dur, dt=0.01, ax=ts_ax)
    ts_ax.set_title(r"$x_1(t)$", fontsize=8.5)
    ts_ax.set_xlabel("$t$", fontsize=8)
    ts_ax.set_ylabel("$x_1$", fontsize=8)
    ts_ax.tick_params(labelsize=7)
    if ts_ax.get_legend() is not None:
        ts_ax.get_legend().remove()

save(fig, OUT)
