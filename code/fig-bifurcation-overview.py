"""Overview: time series · phase plane · bifurcation diagram (Hopf NF)."""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from _bif_common import IMG, HOPF, HOPF_CONT, _bif, save
from tvbo import Dynamics

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))

dyn_stable = Dynamics.from_string(HOPF)
dyn_stable.parameters["a"].value = -0.5
dyn_stable.plot("x1", kind="timeseries", duration=10, dt=0.01, ax=ax[0])
dyn_osc = Dynamics.from_string(HOPF)
dyn_osc.parameters["a"].value = +0.5
dyn_osc.plot("x1", kind="timeseries", duration=10, dt=0.01, ax=ax[0])
ax[0].set_title("Time series ($a=\\pm 0.5$)")

Dynamics.from_string(HOPF).plot(
    "x1",
    "x2",
    kind="phaseplane",
    ax=ax[1],
    grid_n=22,
    n_trajectories=3,
    duration=15,
)
ax[1].set_title("Phase plane ($a=0.5$)")

cont = _bif(HOPF, HOPF_CONT)
cont.plot(VOI="x1", ax=ax[2])
ax[2].set_title("Bifurcation diagram")

fig.tight_layout()
save(fig, OUT)
