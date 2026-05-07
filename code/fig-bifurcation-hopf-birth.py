"""GIF: Hopf birth — phase plane + 3D bifurcation diagram animation."""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401 — needed for bsplot style
from matplotlib.animation import FuncAnimation, PillowWriter
from _bif_common import IMG, HOPF, HOPF_CONT, _bif, _simulate_via_experiment
from tvbo import Dynamics
from tvbo.analysis.bifurcation import compute_voi

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".gif"))
ACCENT = "#c85030"
TRAJ_C = "red"
T_FINAL = 10.0
DT = 0.01

n_frames = 24
cont = _bif(HOPF, HOPF_CONT)
values = np.linspace(-2.0, 1.5, n_frames)

trajectories = []
for value in values:
    dyn = Dynamics.from_string(HOPF)
    dyn.parameters["a"].value = float(value)
    _, series = _simulate_via_experiment(dyn, duration=T_FINAL, dt=DT)
    trajectories.append(series)

voi = cont._resolve_voi("x1")
sv2 = "x2"
params = cont.df["param"].values if not cont.df.empty and "param" in cont.df.columns else np.array([])
voi_eq = (
    compute_voi(cont.df, voi, state_var_index=cont.state_var_index).values
    if len(params)
    else np.array([])
)

fig = plt.figure(figsize=(11, 4.8))
ax_phase = fig.add_subplot(1, 2, 1)
ax_bif = fig.add_subplot(1, 2, 2, projection="3d")
cont.plot_3d(ax=ax_bif, VOI=voi)

y_bb = 0.0
if ax_bif.lines:
    ydata = ax_bif.lines[0].get_ydata()
    if len(ydata):
        y_bb = float(ydata[0])

marker = ax_bif.scatter(
    [params[0] if len(params) else 0.0],
    [y_bb],
    [voi_eq[0] if len(voi_eq) else 0.0],
    marker="o",
    s=80,
    color=TRAJ_C,
    edgecolors="white",
    linewidths=1.0,
    zorder=20,
)
ring_artist = [None]
circle_artist = [None]
theta = np.linspace(0.0, 2.0 * np.pi, 240)


def _plot_phase(value, series, ring):
    ax_phase.clear()
    dyn = Dynamics.from_string(HOPF)
    dyn.parameters["a"].value = float(value)
    dyn.plot("x1", "x2", kind="vectorfield", ax=ax_phase, grid_n=18, stream=True)
    ax_phase.plot(series["x1"], series["x2"], color=TRAJ_C, lw=1.3, alpha=0.95, zorder=9)
    ax_phase.plot(0, 0, "o", color=ACCENT, ms=7, mec="white", mew=1.0, zorder=12)
    if ring is not None and value > 0:
        radius = np.sqrt(value)
        circle_artist[0], = ax_phase.plot(
            radius * np.cos(theta),
            radius * np.sin(theta),
            color=TRAJ_C,
            lw=2.0,
            alpha=0.95,
            zorder=11,
        )
    else:
        circle_artist[0] = None
    ax_phase.set_title(rf"$a = {value:+.2f}$", color=TRAJ_C)


def _update(frame_idx):
    value = float(values[frame_idx])
    series = trajectories[frame_idx]

    if len(params):
        i = int(np.abs(params - value).argmin())
        z_val = float(voi_eq[i]) if len(voi_eq) else 0.0
        marker._offsets3d = ([value], [y_bb], [z_val])

    if ring_artist[0] is not None:
        ring_artist[0].remove()
        ring_artist[0] = None
    ring = cont._lc_ring_at_param(value, voi, sv2, y_bb)
    if ring is not None:
        x_ring, y_ring, z_ring = ring
        ring_artist[0], = ax_bif.plot(
            x_ring,
            y_ring,
            z_ring,
            "-",
            color=TRAJ_C,
            linewidth=2.0,
            zorder=21,
            alpha=0.95,
        )

    _plot_phase(value, series, ring)
    return [marker]


anim = FuncAnimation(fig, _update, frames=len(values), interval=120, blit=False)
anim.save(OUT, writer=PillowWriter(fps=8))
print("wrote", OUT)
plt.close(fig)
