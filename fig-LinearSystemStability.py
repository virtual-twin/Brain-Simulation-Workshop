import bsplot
from tvbo import DynamicalSystem, SimulationExperiment
from tvbo.datamodel.schema import Exploration, ExplorationAxis

import bsplot
import numpy as np
from tvbo import DynamicalSystem, SimulationExperiment
from tvbo.datamodel.schema import Event, Exploration, ExplorationAxis

bsplot.style.use("tvbo")

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

fig.set_figheight(3.5)
fig.set_figwidth(4.5)

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

fig.set_figheight(3.5)
fig.set_figwidth(4.5)
fig