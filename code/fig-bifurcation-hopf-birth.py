"""GIF: Hopf birth — phase plane + 3D bifurcation diagram animation."""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401 — needed for bsplot style
from matplotlib.animation import PillowWriter
from _bif_common import IMG, HOPF, HOPF_CONT, _bif
from tvbo import Dynamics

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".gif"))

n_frames = 24
dyn = Dynamics.from_string(HOPF)
cont = _bif(HOPF, HOPF_CONT)
values = np.linspace(-2.0, 1.5, n_frames)
anim = cont.animate(
    dyn,
    "a",
    values,
    "x1",
    "x2",
    VOI="x1",
    grid_n=18,
    n_trajectories=3,
    duration=15,
    figsize=(11, 4.8),
    title_fmt=r"$a = {value:+.2f}$",
)
anim.save(OUT, writer=PillowWriter(fps=8))
print("wrote", OUT)
