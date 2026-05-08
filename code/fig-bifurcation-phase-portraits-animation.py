"""Animated 2D phase portraits with sequentially added trajectories."""
from __future__ import annotations

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from _bif_common import (
    IMG,
    NODE_2D,
    SADDLE_2D,
    FOCUS_2D,
    CENTRE_2D,
    _sample_initial_values,
    _simulate_trials_via_experiment,
)
from tvbo import Dynamics

ACCENT = "#c85030"
TRAJ_C = "red"
OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".gif"))
T_FINAL = 10.0
DT = 0.01
N_TRAJ = 5
SAMPLES_PER_TRAJ = 60


def _slower_node_yaml():
    return NODE_2D.replace("rhs: -2*x1", "rhs: -0.35*x1").replace("rhs: -x2", "rhs: -0.18*x2", 1)


def _slower_saddle_yaml():
    return SADDLE_2D.replace("rhs: x1 - x2", "rhs: 0.10*(x1 - x2)").replace("rhs: -x1 - x2", "rhs: 0.10*(-x1 - x2)")


def _slower_focus_yaml():
    return FOCUS_2D.replace("rhs: -0.3*x1 - x2", "rhs: -0.12*x1 - 0.8*x2").replace("rhs: x1 - 0.3*x2", "rhs: 0.8*x1 - 0.12*x2")


def _slower_centre_yaml():
    return CENTRE_2D.replace("rhs: -x2", "rhs: -0.8*x2").replace("rhs: x1", "rhs: 0.8*x1", 1)

CASES = [
    ("Stable node", _slower_node_yaml(), "node", 11, 0.90),
    ("Saddle", _slower_saddle_yaml(), "saddle", 22, 0.45),
    ("Stable focus", _slower_focus_yaml(), "focus", 33, 0.75),
    ("Centre", _slower_centre_yaml(), "centre", 44, 0.75),
]

EQUATIONS = {
    "node":   r"$\dot{x}_1 = -0.35\,x_1$" "\n" r"$\dot{x}_2 = -0.18\,x_2$",
    "saddle": r"$\dot{x}_1 = 0.1\,(x_1 - x_2)$" "\n" r"$\dot{x}_2 = -0.1\,(x_1 + x_2)$",
    "focus":  r"$\dot{x}_1 = -0.12\,x_1 - 0.8\,x_2$" "\n" r"$\dot{x}_2 = 0.8\,x_1 - 0.12\,x_2$",
    "centre": r"$\dot{x}_1 = -0.8\,x_2$" "\n" r"$\dot{x}_2 = 0.8\,x_1$",
}

# Layout: eq | ts | pp || pp | ts | eq
MOSAIC = [
    ["node_eq",  "node_ts",  "node_pp",  "saddle_pp",  "saddle_ts",  "saddle_eq"],
    ["focus_eq", "focus_ts", "focus_pp", "centre_pp",  "centre_ts",  "centre_eq"],
]
fig, axes = plt.subplot_mosaic(
    MOSAIC,
    figsize=(17.0, 7.5),
    gridspec_kw={"width_ratios": [0.45, 1.2, 2.0, 2.0, 1.2, 0.45]},
    layout="compressed",
)

artists = []
data_by_key = {}

for title, yml, key, seed, shrink in CASES:
    dyn = Dynamics.from_string(yml)
    initial_values_list = _sample_initial_values(dyn, n_trials=N_TRAJ, seed=seed, shrink=shrink)
    runs = _simulate_trials_via_experiment(
        dyn,
        duration=T_FINAL,
        dt=DT,
        n_trials=N_TRAJ,
        seed=seed,
        initial_values_list=initial_values_list,
    )
    data_by_key[key] = runs

    pp_ax = axes[f"{key}_pp"]
    dyn.plot("x1", "x2", kind="vectorfield", ax=pp_ax, grid_n=20, stream=True)
    pp_ax.plot(0, 0, "o", color=ACCENT, ms=9, mec="white", mew=1.5, zorder=15, clip_on=False)
    pp_ax.set_title(title, fontsize=10)

    ts_ax = axes[f"{key}_ts"]
    ts_ax.set_title(r"$x_1(t)$", fontsize=8.5)
    ts_ax.set_xlabel("$t$", fontsize=8)
    ts_ax.set_ylabel("$x_1$", fontsize=8)
    ts_ax.tick_params(labelsize=7)
    ts_ax.set_xlim(0, T_FINAL)

    eq_ax = axes[f"{key}_eq"]
    eq_ax.axis("off")
    eq_ax.text(
        0.5, 0.5, EQUATIONS[key],
        transform=eq_ax.transAxes,
        ha="center", va="center",
        fontsize=8.5, linespacing=1.9,
    )

    yvals = [run["series"]["x1"] for run in runs]
    ymin = min(float(y.min()) for y in yvals)
    ymax = max(float(y.max()) for y in yvals)
    pad = 0.08 * max(ymax - ymin, 1e-6)
    ts_ax.set_ylim(ymin - pad, ymax + pad)

    phase_lines = []
    time_lines = []
    for _ in runs:
        phase_line, = pp_ax.plot([], [], color=TRAJ_C, lw=0.9, alpha=0.9, zorder=8)
        time_line, = ts_ax.plot([], [], color=TRAJ_C, lw=0.9, alpha=0.9)
        phase_lines.append(phase_line)
        time_lines.append(time_line)
    artists.append((phase_lines, time_lines))

example_run = next(iter(data_by_key.values()))[0]
sample_idx = np.linspace(1, len(example_run["time"]) - 1, SAMPLES_PER_TRAJ, dtype=int)
total_frames = N_TRAJ * SAMPLES_PER_TRAJ


def _update(frame):
    active_trial = min(frame // SAMPLES_PER_TRAJ, N_TRAJ - 1)
    active_step = sample_idx[frame % SAMPLES_PER_TRAJ]
    changed = []
    for (_, _, key, _, _), (phase_lines, time_lines) in zip(CASES, artists):
        runs = data_by_key[key]
        for idx, run in enumerate(runs):
            if idx < active_trial:
                stop = len(run["time"])
            elif idx == active_trial:
                stop = active_step
            else:
                stop = 0

            phase_lines[idx].set_data(run["series"]["x1"][:stop], run["series"]["x2"][:stop])
            time_lines[idx].set_data(run["time"][:stop], run["series"]["x1"][:stop])
            changed.extend((phase_lines[idx], time_lines[idx]))
    return changed


anim = FuncAnimation(fig, _update, frames=total_frames, interval=80, blit=True)
anim.save(OUT, writer=PillowWriter(fps=15))
print("wrote", OUT)
plt.close(fig)