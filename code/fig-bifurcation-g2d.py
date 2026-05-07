"""Generic2dOscillator: bifurcation diagram in I with exploration time series."""
from __future__ import annotations

import os
from _bif_common import IMG, G2D_CONT, G2D_EXPLORATION, BACKEND, save
from tvbo import Dynamics, SimulationExperiment
from tvbo.classes.continuation import Continuation

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

dyn = Dynamics.from_ontology("Generic2dOscillator")
cont = Continuation.from_string(G2D_CONT)
exp = SimulationExperiment(
    dynamics=dyn,
    continuations=[cont],
    explorations=[G2D_EXPLORATION],
)
fig = exp.plot(
    figsize=(10.0, 3.8),
    run_kwargs={
        "format": BACKEND,
        "exploration_kwargs": {"duration": 1000, "dt": 0.01},
    },
)
save(fig, OUT)
