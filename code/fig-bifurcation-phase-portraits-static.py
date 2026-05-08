"""Static 2×6 phase-portrait figure with bifurcation diagrams.

Layout (per row):  eq | bif | pp || pp | bif | eq

Row 0 — node / saddle:  SN bifurcation diagram marks which branch each lives on.
Row 1 — focus / centre: Hopf bifurcation diagram marks stable-focus vs Hopf point.
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from _bif_common import (
    IMG,
    NODE_2D, SADDLE_2D, FOCUS_2D, CENTRE_2D,
    SADDLE_NODE, SADDLE_NODE_CONT,
    HOPF, HOPF_CONT,
    _bif, save, _simulate_trials_via_experiment,
)
from tvbo import Dynamics

ACCENT = "#c85030"
TRAJ_C = "red"
N_TRAJ = 5
OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

CASES = [
    ("Stable node", NODE_2D,   "node",   5.0,  11),
    ("Saddle",      SADDLE_2D, "saddle", 2.0,  22),
    ("Stable focus", FOCUS_2D, "focus",  20.0, 33),
    ("Centre",      CENTRE_2D, "centre", 10.0, 44),
]

EQUATIONS = {
    "node":   r"$\dot{x}_1 = -0.35\,x_1$" "\n" r"$\dot{x}_2 = -0.18\,x_2$",
    "saddle": r"$\dot{x}_1 = 0.1\,(x_1 - x_2)$" "\n" r"$\dot{x}_2 = -0.1\,(x_1 + x_2)$",
    "focus":  r"$\dot{x}_1 = -0.12\,x_1 - 0.8\,x_2$" "\n" r"$\dot{x}_2 = 0.8\,x_1 - 0.12\,x_2$",
    "centre": r"$\dot{x}_1 = -0.8\,x_2$" "\n" r"$\dot{x}_2 = 0.8\,x_1$",
}

# Which bifurcation diagram to show.
# SN: a_marker marks the operating point; y_branch = +√a (stable) or -√a (unstable)
# Hopf: a_marker marks where on the a-axis we are; y_branch = None → auto
BIF_INFO = {
    "node":   ("SN",    0.8, "stable node",  +1),   # upper branch: x*=+√a
    "saddle": ("SN",    0.8, "saddle",        -1),   # lower branch: x*=-√a
    "focus":  ("Hopf", -0.5, "stable focus",   0),
    "centre": ("Hopf",  0.0, "Hopf point",     0),
}

# Layout: eq | bif | pp || pp | bif | eq
MOSAIC = [
    ["node_eq",  "node_bif",  "node_pp",  "saddle_pp",  "saddle_bif",  "saddle_eq"],
    ["focus_eq", "focus_bif", "focus_pp", "centre_pp",  "centre_bif",  "centre_eq"],
]
fig, axes = plt.subplot_mosaic(
    MOSAIC,
    figsize=(17.0, 7.5),
    gridspec_kw={"width_ratios": [0.45, 1.2, 2.0, 2.0, 1.2, 0.45]},
    layout="compressed",
)

# ── Run continuations once, reuse for both panels in each row ──────────────
sn_result   = _bif(SADDLE_NODE, SADDLE_NODE_CONT)
hopf_result = _bif(HOPF,        HOPF_CONT)

for title, yml, key, dur, seed in CASES:
    dyn = Dynamics.from_string(yml)
    runs = _simulate_trials_via_experiment(dyn, duration=dur, dt=0.01, n_trials=N_TRAJ, seed=seed)

    # ── Equation panel ──
    eq_ax = axes[f"{key}_eq"]
    eq_ax.axis("off")
    eq_ax.text(0.5, 0.5, EQUATIONS[key],
               transform=eq_ax.transAxes, ha="center", va="center",
               fontsize=8.5, linespacing=1.9)

    # ── Phase portrait ──
    pp_ax = axes[f"{key}_pp"]
    dyn.plot("x1", "x2", kind="vectorfield", ax=pp_ax, grid_n=20, stream=True)
    for run in runs:
        pp_ax.plot(run["series"]["x1"], run["series"]["x2"],
                   color=TRAJ_C, lw=0.9, alpha=0.9, zorder=8)
    pp_ax.plot(0, 0, "o", color=ACCENT, ms=9, mec="white", mew=1.5, zorder=15, clip_on=False)
    pp_ax.set_title(title, fontsize=10)

    # ── Bifurcation diagram ──
    bif_kind, a_marker, label, branch_sign = BIF_INFO[key]
    bif_ax = axes[f"{key}_bif"]

    if bif_kind == "SN":
        sn_result.plot(VOI="x", ax=bif_ax)
        bif_ax.set_title(r"$\dot x = a - x^2$", fontsize=8.5)
        bif_ax.set_xlabel("$a$", fontsize=8)
        bif_ax.set_ylabel("$x^*$", fontsize=8)
        # Place label ON the branch: x* = ±√a
        import math
        y_label = branch_sign * math.sqrt(a_marker)
        va = "bottom" if branch_sign >= 0 else "top"
    else:
        hopf_result.plot(VOI="x1", ax=bif_ax)
        bif_ax.set_title(r"$\dot{\mathbf{x}} = $ Hopf NF", fontsize=8.5)
        bif_ax.set_xlabel("$a$", fontsize=8)
        bif_ax.set_ylabel(r"$|x_1|$", fontsize=8)
        ylim = bif_ax.get_ylim()
        y_label = ylim[0] + 0.85 * (ylim[1] - ylim[0])
        va = "center"

    bif_ax.axvline(a_marker, color=ACCENT, lw=1.3, ls="--", zorder=5)
    bif_ax.plot(a_marker, y_label, "o", color=ACCENT, ms=7, mec="white", mew=1.2, zorder=6)
    bif_ax.text(a_marker + 0.06, y_label, label,
                ha="left", va=va, fontsize=7, color=ACCENT)
    bif_ax.tick_params(labelsize=7)
    legend = bif_ax.get_legend()
    if legend is not None:
        legend.remove()

save(fig, OUT)
