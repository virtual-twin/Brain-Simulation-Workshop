"""Overview: time series · phase plane · bifurcation diagram (Hopf NF)."""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from _bif_common import IMG, HOPF, HOPF_CONT, _bif, save, _simulate_via_experiment, _simulate_trials_via_experiment
from tvbo import Dynamics

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))

dyn_stable = Dynamics.from_string(HOPF)
dyn_stable.parameters["a"].value = -0.5
t, series = _simulate_via_experiment(dyn_stable, duration=10, dt=0.01)
ax[0].plot(t, series["x1"], lw=1.4, color="C0")
dyn_osc = Dynamics.from_string(HOPF)
dyn_osc.parameters["a"].value = +0.5
t, series = _simulate_via_experiment(dyn_osc, duration=10, dt=0.01)
ax[0].plot(t, series["x1"], lw=1.4, color="C1")
ax[0].set_title("Time series ($a=\\pm 0.5$)")
ax[0].set_xlabel("$t$")
ax[0].set_ylabel("$x_1$")

dyn_phase = Dynamics.from_string(HOPF)
dyn_phase.plot(
    "x1",
    "x2",
    kind="vectorfield",
    ax=ax[1],
    grid_n=22,
    stream=True,
)
for run in _simulate_trials_via_experiment(dyn_phase, duration=15, dt=0.01, n_trials=3, seed=7):
    ax[1].plot(run["series"]["x1"], run["series"]["x2"], color="red", lw=1.0, alpha=0.9)
ax[1].plot(0, 0, "o", color="#c85030", ms=7, mec="white", mew=1.0, zorder=10)
ax[1].set_title("Phase plane ($a=0.5$)")

cont = _bif(HOPF, HOPF_CONT)
cont.plot(VOI="x1", ax=ax[2])
ax[2].set_title("Bifurcation diagram")

fig.tight_layout()
save(fig, OUT)
