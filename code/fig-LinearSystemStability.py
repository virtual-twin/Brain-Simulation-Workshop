import bsplot
from tvbo import DynamicalSystem, SimulationExperiment
from tvbo.datamodel.schema import Exploration, ExplorationAxis

import bsplot
import numpy as np
from tvbo import DynamicalSystem, SimulationExperiment
from tvbo.datamodel.schema import Event, Exploration, ExplorationAxis
import os
from matplotlib.ticker import FormatStrFormatter


bsplot.style.use("tvbo")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(ROOT, "..", "img", 'figures', 'bifurcation'))


g = DynamicalSystem(
    state_variables={"x": {"equation": {"rhs": "a*x"}, "initial_value": 1.0}},
    parameters={"a": {"value": 1.0}},
)

exp = SimulationExperiment(dynamics=g)

exp.explorations = Exploration(
    name="a_sweep",
    space=ExplorationAxis(parameter="a", explored_values=[1.0, 0, -1.0]),
)

exp.integration.duration = 4.0
exp.integration.step_size = 0.01

res = exp.run("tvboptim")

fig = res.explorations["a_sweep"].plot(overlay=True)
ax = fig.axes[0]
ax.set(xlabel=r"Time $t$", ylabel=r"$x(t)$", xlim=(0, 4), ylim=(-.1, 4))
ax.axhline(0, color="red", ls="--", lw=0.7)
ax.set_title("")

ax.annotate("$a<0$:Stable\n$a>0$:Unstable", xy=(2.5, 2.7), xytext=(3, 3))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

fig.set_figheight(3.5)
fig.set_figwidth(4.5)

fig.savefig(os.path.join(OUT, "fig-stability.png"), dpi=300, bbox_inches="tight")

# %%

exp.dynamics.state_variables["x"].initial_value = 0.0
exp.dynamics.state_variables["x"].equation.rhs = "a*x + perturbation"

exp.events["perturbation"] = Event(
    name="perturbation",
    event_type="stimulus",
    duration=0.1,
    parameters={
        "onset": {"value": 1.0},
        "width": {"value": 0.1},
        "amplitude": {"value": 1.0},
    },
    equation={
        "rhs": "Piecewise((amplitude, (t >= onset) & (t < onset + width)), (0.0, True))"
    },
)
res = exp.run("tvboptim")

fig = res.explorations["a_sweep"].plot(overlay=True)
ax = fig.axes[0]
ax.set(xlabel=r"Time $t$", ylabel=r"$x(t)$", xlim=(0, 4), ylim=(-.1, 2))
ax.axhline(0, color="red", ls="--", lw=0.7)

ax.axvline(exp.events["perturbation"].parameters['onset'].value, color="blue", ls="--", lw=0.7, label="Perturbation onset")

ax.set_title("")

ax.annotate("$a<0$:Stable\n$a>0$:Unstable", xy=(2.5, 2.7), xytext=(3, 3))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

inset = ax.inset_axes([0.4, 0.7, 0.3, 0.3])
exp.events["perturbation"].plot(ax=inset)

fig.set_figheight(3.5)
fig.set_figwidth(4.5)
fig.savefig(os.path.join(OUT, "fig-stability-perturbation.png"), dpi=300, bbox_inches="tight")