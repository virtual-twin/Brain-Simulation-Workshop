"""Shared constants and helpers for bifurcation figure scripts."""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import bsplot  # noqa: F401
from tvbo import Dynamics, SimulationExperiment
from tvbo.classes.continuation import Continuation

bsplot.style.use("tvbo")

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(ROOT, "..", "img", "figures", "bifurcation"))
os.makedirs(IMG, exist_ok=True)

TUTORIAL_FIG_48 = (
    "/Users/leonmartin_bih/tools/tvbo/dev/BifurcationTutorial/slides/img/fig_48.png"
)

BACKEND = "bifurcationkit.jl"

# ---------------------------------------------------------------------------
# YAML model / continuation strings
# ---------------------------------------------------------------------------

LINEAR = """
name: LinearScalar
parameters:
  a:
    name: a
    value: -0.5
state_variables:
  x:
    name: x
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x, t)
      rhs: a*x
    initial_value: 1.0
"""

HOPF = r"""
name: HopfNF
parameters:
  a:
    name: a
    value: 0.5
  w:
    name: w
    value: 6.283185307179586
state_variables:
  x1:
    name: x1
    domain:
      lo: -1.5
      hi: 1.5
    equation:
      lhs: Derivative(x1, t)
      rhs: (a - x1**2 - x2**2)*x1 - w*x2
    initial_value: 0.05
  x2:
    name: x2
    domain:
      lo: -1.5
      hi: 1.5
    equation:
      lhs: Derivative(x2, t)
      rhs: (a - x1**2 - x2**2)*x2 + w*x1
    initial_value: 0.0
"""

HOPF_CONT = """
name: hopf_in_a
dynamics: HopfNF
free_parameters:
  - name: a
    domain:
      lo: -1.0
      hi: 1.0
max_steps: 400
ds: 0.01
bothside: true
branches:
  - name: po_from_hopf
    source_point: "hopf:all"
    bothside: true
"""

SADDLE_NODE = """
name: SaddleNode
parameters:
  a:
    name: a
    value: 1.0
state_variables:
  x:
    name: x
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x, t)
      rhs: a - x**2
    initial_value: 1.0
"""
SADDLE_NODE_CONT = """
name: saddle_node_cont
dynamics: SaddleNode
free_parameters:
  - name: a
    domain: { lo: -2.0, hi: 2.0 }
max_steps: 400
ds: 0.01
bothside: true
"""

HYSTERESIS = """
name: Hysteresis
parameters:
  a:
    name: a
    value: 0.0
state_variables:
  x:
    name: x
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x, t)
      rhs: a + x - x**3
    initial_value: 1.0
"""
HYSTERESIS_CONT = """
name: hysteresis_cont
dynamics: Hysteresis
free_parameters:
  - name: a
    domain: { lo: -1.5, hi: 1.5 }
max_steps: 400
ds: 0.01
bothside: true
"""

SADDLE_2D = """
name: Saddle2D
state_variables:
  x1:
    name: x1
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x1, t)
      rhs: x1 - x2
    initial_value: 1.0
  x2:
    name: x2
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x2, t)
      rhs: -x1 - x2
    initial_value: 1.0
"""
NODE_2D = """
name: StableNode2D
state_variables:
  x1:
    name: x1
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x1, t)
      rhs: -2*x1
    initial_value: 1.0
  x2:
    name: x2
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x2, t)
      rhs: -x2
    initial_value: 1.0
"""
FOCUS_2D = """
name: StableFocus2D
state_variables:
  x1:
    name: x1
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x1, t)
      rhs: -0.3*x1 - x2
    initial_value: 1.0
  x2:
    name: x2
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x2, t)
      rhs: x1 - 0.3*x2
    initial_value: 1.0
"""
CENTRE_2D = """
name: Centre2D
state_variables:
  x1:
    name: x1
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x1, t)
      rhs: -x2
    initial_value: 1.0
  x2:
    name: x2
    domain:
      lo: -2.0
      hi: 2.0
    equation:
      lhs: Derivative(x2, t)
      rhs: x1
    initial_value: 1.0
"""

G2D_CONT = """
name: g2d_in_I
dynamics: Generic2dOscillator
free_parameters:
  - name: I
    domain: { lo: -10.0, hi: 20.0 }
max_steps: 500
ds: 0.05
bothside: true
branches:
  - name: po_from_hopf
    source_point: "hopf:all"
    bothside: true
"""

G2D_EXPLORATION = {
    "name": "g2d_I_timeseries",
    "space": [
        {
            "parameter": "I",
            "explored_values": [-10.0, 5.0, 20.0],
        }
    ],
    "observable": {"function": "V"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save(fig, out, **kw):
    fig.savefig(out, dpi=500, bbox_inches="tight", **kw)
    print("wrote", out)
    plt.close(fig)


def _simulate_via_experiment(dyn, duration, dt=0.01, initial_values=None):
  """Run a local Dynamics object through SimulationExperiment.run().

  Returns
  -------
  time : np.ndarray
    Time vector with the initial point prepended at t=0.
  series : dict[str, np.ndarray]
    One array per state variable, also prepended with the initial value.
  """
  model = dyn.copy()
  if initial_values:
    for name, value in initial_values.items():
      model.state_variables[name].initial_value = float(value)

  # These workshop examples are didactic linear systems. Keep domain bounds
  # for plotting extents, but do not let the simulation backend treat them as
  # hard state bounds, otherwise unstable saddles get artificially clipped.
  for sv in model.state_variables.values():
    sv.domain = None

  init_map = {
    name: float(sv.initial_value) if sv.initial_value is not None else 0.0
    for name, sv in model.state_variables.items()
  }

  exp = SimulationExperiment(dynamics=model)
  exp.integration.duration = duration
  exp.integration.step_size = dt
  res = exp.run("tvboptim").integration

  time = np.asarray(res.time, dtype=float)
  series = {
    name: np.asarray(res.data.sel(variable=name)).squeeze()
    for name in model.state_variables
  }

  if time.size == 0 or time[0] > 0.0:
    time = np.r_[0.0, time]
    for name, value in init_map.items():
      series[name] = np.r_[value, series[name]]

  return time, series


def _sample_initial_values(dyn, n_trials=5, seed=0, shrink=0.9):
  """Sample initial conditions from state-variable domains."""
  rng = np.random.default_rng(seed)
  trials = []
  for _ in range(n_trials):
    trial = {}
    for name, sv in dyn.state_variables.items():
      if getattr(sv, "domain", None) and sv.domain.lo is not None and sv.domain.hi is not None:
        lo = float(sv.domain.lo)
        hi = float(sv.domain.hi)
      else:
        value = float(sv.initial_value) if sv.initial_value is not None else 0.0
        lo, hi = value - 1.0, value + 1.0
      mid = 0.5 * (lo + hi)
      half = 0.5 * (hi - lo) * shrink
      trial[name] = rng.uniform(mid - half, mid + half) if half > 0 else mid
    trials.append(trial)
  return trials


def _simulate_trials_via_experiment(dyn, duration, dt=0.01, n_trials=5, seed=0, initial_values_list=None):
  """Run multiple experiment-backed trials from sampled initial conditions."""
  if initial_values_list is None:
    initial_values_list = _sample_initial_values(dyn, n_trials=n_trials, seed=seed)
  runs = []
  for initial_values in initial_values_list:
    time, series = _simulate_via_experiment(dyn, duration=duration, dt=dt, initial_values=initial_values)
    runs.append({"time": time, "series": series, "initial_values": initial_values})
  return runs


def _bif(dyn_yaml, cont_yaml):
    """Run a continuation and return the single Continuation result."""
    dyn = Dynamics.from_string(dyn_yaml)
    cont = Continuation.from_string(cont_yaml)
    exp = SimulationExperiment(dynamics=dyn, continuations=[cont])
    name = next(iter(exp.continuations))
    return exp.run(BACKEND).continuations[name]


def _plot_scalar_flow(dyn_yaml, parameter_value, ax, xlim=(-2.0, 2.0)):
    dynamics = Dynamics.from_string(dyn_yaml)
    dynamics.parameters["a"].value = parameter_value
    rhs = dynamics.execute(format="python")
    state_names = list(dynamics.state_variables)
    state_index = state_names.index("x")
    base_state = np.asarray(dynamics.get_initial_values(), dtype=float).reshape(-1)

    state_values = np.linspace(xlim[0], xlim[1], 401)
    flow_values = []
    for state_value in state_values:
        state_vector = base_state.copy()
        state_vector[state_index] = state_value
        derivative = np.asarray(rhs(state_vector, 0.0), dtype=float).reshape(-1)
        flow_values.append(derivative[state_index])
    flow_values = np.asarray(flow_values)

    ax.plot(state_values, flow_values, color="0.2", lw=1.4)
    ax.axhline(0, color="0.65", lw=0.9)
    arrow_positions = np.linspace(xlim[0] + 0.25, xlim[1] - 0.25, 11)
    arrow_flow = np.interp(arrow_positions, state_values, flow_values)
    arrow_lengths = 0.22 * np.sign(arrow_flow)
    ax.quiver(
        arrow_positions,
        np.zeros_like(arrow_positions),
        arrow_lengths,
        np.zeros_like(arrow_positions),
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.006,
        color="C0",
        zorder=4,
    )

    root_points = []
    for index in range(len(state_values) - 1):
        left_flow = flow_values[index]
        right_flow = flow_values[index + 1]
        if left_flow == 0:
            root_points.append(state_values[index])
        elif left_flow * right_flow < 0:
            left_state = state_values[index]
            right_state = state_values[index + 1]
            root_points.append(
                left_state
                - left_flow * (right_state - left_state) / (right_flow - left_flow)
            )
    for root_point in root_points:
        delta = 1e-3
        left_state = base_state.copy()
        right_state = base_state.copy()
        left_state[state_index] = root_point - delta
        right_state[state_index] = root_point + delta
        left_derivative = np.asarray(rhs(left_state, 0.0), dtype=float).reshape(-1)[
            state_index
        ]
        right_derivative = np.asarray(rhs(right_state, 0.0), dtype=float).reshape(-1)[
            state_index
        ]
        stable = (right_derivative - left_derivative) < 0
        ax.plot(
            root_point,
            0,
            "o",
            ms=4.5,
            mfc="#c85030" if stable else "white",
            mec="#c85030",
            mew=1.0,
            zorder=5,
        )

    ax.set(xlim=xlim, ylim=(-1.15, 1.15), xlabel="$x$", ylabel=r"$\dot x$")
    ax.set_title(rf"slice at $a={parameter_value:g}$", fontsize=9)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(labelsize=7)
