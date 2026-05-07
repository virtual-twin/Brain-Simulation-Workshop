"""GIF: Hopf birth with SimulationExperiment-backed trajectories."""
from __future__ import annotations

import os

import numpy as np
from matplotlib.animation import PillowWriter

from _bif_common import IMG, HOPF, HOPF_CONT, _bif
from tvbo import Dynamics

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".gif"))

cont = _bif(HOPF, HOPF_CONT)
dyn = Dynamics.from_string(HOPF)
values = np.linspace(-1.0, 1.0, 60)


def hopf_orbit_initial_values(a):
    if a <= 0:
        return None
    return {"x1": np.sqrt(a), "x2": 0.0}

anim = cont.animate(
    dyn,
    "a",
    values,
    "x1",
    "x2",
    kind="vectorfield",
    VOI="x1",
    interval=180,
    figsize=(11, 4.8),
    title_fmt=r"$a = {value:+.2f}$",
    grid_n=22,
    stream=True,
    stream_density=1.05,
    stream_arrowsize=1.05,
    alpha=0.72,
    lw=0.9,
    simulation=True,
    simulation_duration=10.0,
    simulation_dt=0.01,
    simulation_backend="tvboptim",
    simulation_initial_values=hopf_orbit_initial_values,
    trajectory_kwargs={"color": "red", "lw": 1.3, "alpha": 0.95, "zorder": 9},
    orbit_kwargs={"color": "red", "lw": 2.0, "alpha": 0.95, "zorder": 11},
    marker_kwargs={"color": "red"},
)
anim.save(OUT, writer=PillowWriter(fps=6))
print("wrote", OUT)
