"""Animated 2D phase portraits with sequentially added trajectories."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from _bif_common import (
    IMG,
    NODE_2D,
    SADDLE_2D,
    FOCUS_2D,
    CENTRE_2D,
    _simulate_trials_via_experiment,
)
from tvbo import Dynamics

ACCENT = "#c85030"
TRAJ_C = "red"
OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".gif"))

CASES = [
    ("Stable node", NODE_2D, "node", 5.0, 11),
    ("Saddle", SADDLE_2D, "saddle", 2.0, 22),
    ("Stable focus", FOCUS_2D, "focus", 20.0, 33),
    ("Centre", CENTRE_2D, "centre", 10.0, 44),
]

PP_POS = {"node": (0, 1), "saddle": (0, 2), "focus": (1, 1), "centre": (1, 2)}
TS_POS = {"node": (0, 0), "saddle": (0, 3), "focus": (1, 0), "centre": (1, 3)}

fig = plt.figure(figsize=(12.2, 6.0))
gs = fig.add_gridspec(
    2,
    4,
    width_ratios=[1.35, 2.0, 2.0, 1.35],
    height_ratios=[1.0, 1.0],
    hspace=0.34,
    wspace=0.34,
)

artists = []
data_by_key = {}

for title, yml, key, dur, seed in CASES:
    dyn = Dynamics.from_string(yml)
    runs = _simulate_trials_via_experiment(dyn, duration=dur, dt=0.01, n_trials=5, seed=seed)
    data_by_key[key] = runs

    pp_ax = fig.add_subplot(gs[PP_POS[key]])
    dyn.plot("x1", "x2", kind="vectorfield", ax=pp_ax, grid_n=20, stream=True)
    pp_ax.plot(0, 0, "o", color=ACCENT, ms=9, mec="white", mew=1.5, zorder=15, clip_on=False)
    pp_ax.set_title(title, fontsize=10)

    ts_ax = fig.add_subplot(gs[TS_POS[key]])
    ts_ax.set_title(r"$x_1(t)$", fontsize=8.5)
    ts_ax.set_xlabel("$t$", fontsize=8)
    ts_ax.set_ylabel("$x_1$", fontsize=8)
    ts_ax.tick_params(labelsize=7)
    ts_ax.set_xlim(0, dur)

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

samples_per_trial = 40
frame_count = len(CASES[0][0:0])
sample_idx = None
for runs in data_by_key.values():
    sample_idx = None
    break
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None
sample_idx = None

example_run = next(iter(data_by_key.values()))[0]
sample_idx = [int(i) for i in plt.np.linspace(1, len(example_run["time"]) - 1, samples_per_trial)]
total_frames = 5 * samples_per_trial


def _update(frame):
    active_trial = frame // samples_per_trial
    active_step = sample_idx[frame % samples_per_trial]
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