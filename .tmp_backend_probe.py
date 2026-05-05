import importlib.util
from pathlib import Path
import numpy as np

from tvbo import Coupling, Dynamics, Network, SimulationExperiment

p = Path("code/fig-bifurcation-slides.py")
s = importlib.util.spec_from_file_location("fig_bifurcation_slides", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
effective_input, voltage, strength, dt, ramp_end = m._g2d_native_network_trajectory(segment_duration=8.0, dt=0.5)
print("native finite", np.isfinite(effective_input).all(), np.isfinite(voltage).all(), effective_input.shape, voltage.min(), voltage.max())

network = Network.from_db("DesikanKilliany")
dynamics = Dynamics.from_db("Generic2dOscillator")
dynamics.parameters["d"].value = 0.05
coupling = Coupling.from_ontology("Linear")
coupling.parameters["a"].value = 1e-6
exp = SimulationExperiment(dynamics=dynamics, network=network, coupling=coupling)
exp.integration.duration = 4.0
exp.integration.step_size = 0.5
try:
    result = exp.run("tvboptim", mode="simulation")
    raw = getattr(result, "raw", result)
    print("tvboptim ok", type(result).__name__, type(raw).__name__)
except Exception as exc:
    print("tvboptim failed", type(exc).__name__, str(exc).splitlines()[0])

ns = exp.execute("tvboptim")
opt_network = ns.create_network(
    weights=exp.network.weights,
    dynamics_params={"I": 0.0},
)
sim = ns.run_simulation(opt_network, t1=4.0, dt=0.5, run_main=False)
print("state keys", list(sim.state.keys()))
print("initial dynamics shape", sim.state.initial_state.dynamics.shape)
print("state dynamics keys", list(sim.state.dynamics.keys()))
sim.state.initial_state.dynamics = sim.state.initial_state.dynamics.at[0].set(0.5)
opt_result = sim.model_fn(sim.state)
print("manual result", type(opt_result).__name__, opt_result.data.shape)

node_i = np.linspace(-2.0, 6.0, exp.network.number_of_nodes)
opt_network = ns.create_network(
    weights=exp.network.weights,
    dynamics_params={"I": node_i},
)
sim = ns.run_simulation(opt_network, t1=8.0, dt=0.5, run_main=False)
result = sim.model_fn(sim.state)
print("nodewise I ok", np.isfinite(np.asarray(result.data)).all(), np.asarray(result.data).shape)
