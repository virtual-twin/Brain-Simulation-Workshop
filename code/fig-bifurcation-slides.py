"""Render all figures + GIFs used in the Bifurcation section of slides.qmd.

Everything is computed by TVBO itself: ``Dynamics.plot``, ``Continuation.plot``,
``Continuation.plot_3d``. No schematic/synthetic curves. All YAML is literal
(no f-string templating). Styled with ``bsplot.style.use('tvbo')``.

Continuation figures need the ``bifurcationkit.jl`` backend (Julia).
"""

# %%
from __future__ import annotations

import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageOps
import bsplot  # noqa: F401
from tvbo import Coupling, Dynamics, Network, SimulationExperiment
from tvbo.classes.continuation import Continuation

bsplot.style.use("tvbo")
ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(ROOT, "..", "img"))
TUTORIAL_FIG_48 = "/Users/leonmartin_bih/tools/tvbo/dev/BifurcationTutorial/slides/img/fig_48.png"
os.makedirs(IMG, exist_ok=True)

BACKEND = "bifurcationkit.jl"


def save(fig, name, **kw):
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=500, bbox_inches="tight", **kw)
    print("wrote", out)
    plt.close(fig)


# ===========================================================================
# Literal YAML for every system used below.
# ===========================================================================

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

PITCHFORK = """
name: Pitchfork
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
      rhs: a*x - x**3
    initial_value: 0.01
"""
PITCHFORK_TRIVIAL = """
name: Pitchfork
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
      rhs: a*x - x**3
    initial_value: 0.0
"""
PITCHFORK_CONT = """
name: pitchfork_cont
dynamics: Pitchfork
free_parameters:
  - name: a
    domain: { lo: -1.0, hi: 2.0 }
max_steps: 400
ds: 0.005
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

# ---- 2D portraits ---------------------------------------------------------
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
            root_points.append(left_state - left_flow * (right_state - left_state) / (right_flow - left_flow))
    for root_point in root_points:
        delta = 1e-3
        left_state = base_state.copy()
        right_state = base_state.copy()
        left_state[state_index] = root_point - delta
        right_state[state_index] = root_point + delta
        left_derivative = np.asarray(rhs(left_state, 0.0), dtype=float).reshape(-1)[state_index]
        right_derivative = np.asarray(rhs(right_state, 0.0), dtype=float).reshape(-1)[state_index]
        stable = (right_derivative - left_derivative) < 0
        ax.plot(
            root_point,
            0,
            "o",
            ms=4.5,
            mfc="0.2" if stable else "white",
            mec="0.2",
            mew=1.0,
            zorder=5,
        )

    ax.set(xlim=xlim, ylim=(-1.15, 1.15), xlabel="$x$", ylabel=r"$\dot x$")
    ax.set_title(fr"slice at $a={parameter_value:g}$", fontsize=9)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(labelsize=7)


# ===========================================================================
# 1. Overview: time series · phase plane · bifurcation diagram (Hopf NF)
# ===========================================================================
def fig_overview():
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
        traj_dt=0.01,
    )
    ax[1].set_title("Phase plane ($a=0.5$)")

    cont = _bif(HOPF, HOPF_CONT)
    cont.plot(VOI="x1", ax=ax[2])
    ax[2].set_title("Bifurcation diagram")

    fig.tight_layout()
    save(fig, "bifurcation_overview.png")


# ===========================================================================
# 2. Linear stability: real solutions x(t) = x0 e^{at} from TVBO
# ===========================================================================
def fig_linear_stability():
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    dyn = Dynamics.from_string(LINEAR)
    x0 = 0.25
    styles = {
        -1.0: {"color": "C0", "ls": "-"},
        -0.5: {"color": "C1", "ls": "-"},
        0.0: {"color": "0.15", "ls": "--"},
        0.5: {"color": "C2", "ls": "-"},
        1.0: {"color": "C3", "ls": "-"},
    }
    for a, style in styles.items():
        dyn.parameters["a"].value = a
        dyn.plot("x", kind="timeseries", duration=3.0, dt=0.01, u_0=[x0], ax=ax)
        line = ax.lines[-1]
        line.set(color=style["color"], linestyle=style["ls"], linewidth=2.0)
        if a == 0.0:
            line.set_label(r"$a=0$ ($\lambda=0$)")
            line.set_linewidth(2.6)
        else:
            line.set_label(fr"$a={a:g}$")
    ax.set(
        xlabel="$t$",
        ylabel="$x(t)$",
        ylim=(0.0, 5.4),
        title=fr"$\dot x = a\,x,\quad x(0)={x0:g},\quad \lambda=a$",
    )
    ax.annotate(
        r"neutral trajectory: $\lambda=a=0$",
        xy=(1.75, x0),
        xytext=(0.7, 1.3),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "0.2"},
        color="0.2",
        fontsize=8,
    )
    ax.legend(title="growth rate", loc="upper left", fontsize=7, title_fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "linear_stability.png")


# ===========================================================================
# 3. 2D phase portraits: node · saddle · focus · centre  (real TVBO plot)
# ===========================================================================
def fig_phase_portraits():
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))
    cases = [
        ("Stable node", NODE_2D),
        ("Saddle", SADDLE_2D),
        ("Stable focus", FOCUS_2D),
        ("Centre", CENTRE_2D),
    ]
    for ax, (label, yml) in zip(axes.flat, cases):
        Dynamics.from_string(yml).plot(
            "x1",
            "x2",
            kind="phaseplane",
            ax=ax,
            grid_n=20,
            n_trajectories=4,
            duration=15,
            traj_dt=0.01,
        )
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    save(fig, "phase_portraits_2d.png")


# ===========================================================================
# 4. Regime-change building blocks: continuations + local scalar flows
# ===========================================================================
def fig_normal_forms():
  fig, ax = plt.subplots(2, 3, figsize=(11.5, 5.6), height_ratios=[1.15, 1.0])

  examples = [
    {
      "title": r"Saddle-node: $\dot x = a - x^2$",
      "continuations": [(SADDLE_NODE, SADDLE_NODE_CONT)],
      "flow": SADDLE_NODE,
      "parameter_value": 0.6,
    },
    {
      "title": r"Pitchfork: $\dot x = a x - x^3$",
      "continuations": [
        (PITCHFORK, PITCHFORK_CONT),
        (PITCHFORK_TRIVIAL, PITCHFORK_CONT),
      ],
      "flow": PITCHFORK,
      "parameter_value": 0.8,
    },
    {
      "title": r"Hysteresis: $\dot x = a + x - x^3$",
      "continuations": [(HYSTERESIS, HYSTERESIS_CONT)],
      "flow": HYSTERESIS,
      "parameter_value": 0.2,
    },
  ]

  for column, example in enumerate(examples):
    bifurcation_ax = ax[0, column]
    flow_ax = ax[1, column]
    for dyn_yaml, cont_yaml in example["continuations"]:
      _bif(dyn_yaml, cont_yaml).plot(VOI="x", ax=bifurcation_ax)
    bifurcation_ax.axvline(example["parameter_value"], color="0.35", lw=1.1, ls=":")
    bifurcation_ax.set_title(example["title"], fontsize=10)
    bifurcation_ax.set_xlabel("$a$", fontsize=9)
    bifurcation_ax.set_ylabel("$x$" if column == 0 else "", fontsize=9)
    bifurcation_ax.tick_params(labelsize=7)
    legend = bifurcation_ax.get_legend()
    if legend is not None:
      handles, labels = bifurcation_ax.get_legend_handles_labels()
      legend.remove()
      if column == 0:
        bifurcation_ax.legend(
          handles,
          labels,
          loc="upper left",
          fontsize=6,
          frameon=False,
          handlelength=1.4,
          markerscale=0.55,
          borderaxespad=0.2,
        )
    _plot_scalar_flow(example["flow"], example["parameter_value"], flow_ax)
    flow_ax.set_ylabel(r"$\dot x$" if column == 0 else "", fontsize=9)

  flow_legend = [
    Line2D([0], [0], color="0.2", lw=1.4, label=r"curve: $\dot x=f(x)$"),
    Line2D([0], [0], color="0.65", lw=0.9, label=r"baseline: $\dot x=0$"),
    Line2D(
      [0], [0], color="C0", marker=r"$\rightarrow$", linestyle="None",
      markersize=11, label="arrows: flow direction",
    ),
    Line2D(
      [0], [0], marker="o", color="0.2", mfc="0.2", mec="0.2",
      linestyle="None", markersize=5, label="filled dot: stable FP",
    ),
    Line2D(
      [0], [0], marker="o", color="0.2", mfc="white", mec="0.2",
      linestyle="None", markersize=5, label="open dot: unstable FP",
    ),
  ]
  fig.legend(
    handles=flow_legend,
    loc="lower center",
    ncol=5,
    fontsize=7,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
    columnspacing=1.2,
  )
  fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
  save(fig, "normal_forms.png")


# ===========================================================================
# 5. Hopf bifurcation, real 3D limit-cycle tube
# ===========================================================================
def fig_hopf_3d():
    cont = _bif(HOPF, HOPF_CONT)
    out = cont.plot_3d(VOI="x1")
    fig = out.figure if hasattr(out, "figure") else out
    save(fig, "hopf_3d.png")


# ===========================================================================
# 6. Continuation example: inverted tutorial schematic for white slides
# ===========================================================================
def fig_continuation():
    image = Image.open(TUTORIAL_FIG_48).convert("RGBA")
    rgb = Image.new("RGB", image.size, "black")
    rgb.paste(image, mask=image.split()[-1])
    out = os.path.join(IMG, "continuation_pseudo_arclength_white.png")
    ImageOps.invert(rgb).save(out)
    print("wrote", out)


# ===========================================================================
# 7. Generic2dOscillator: real bifurcation diagram in I
# ===========================================================================
def fig_g2d_bifurcation():
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
    save(fig, "g2d_bifurcation.png")


def _g2d_tvboptim_network_trajectory(segment_duration=45.0, dt=0.5):
  network = Network.from_db("DesikanKilliany")
  dynamics = Dynamics.from_db("Generic2dOscillator")
  dynamics.parameters["d"].value = 0.05
  coupling = Coupling.from_ontology("Linear")
  coupling.parameters["a"].value = 1e-6

  exp = SimulationExperiment(dynamics=dynamics, network=network, coupling=coupling)
  exp.integration.duration = segment_duration
  exp.integration.step_size = dt
  tvboptim = exp.execute("tvboptim")

  n_nodes = exp.network.number_of_nodes
  rng = np.random.default_rng(7)
  initial_state = np.zeros((len(exp.dynamics.state_variables), n_nodes), dtype=float)
  for state_index, state_variable in enumerate(exp.dynamics.state_variables.values()):
    domain = state_variable.domain
    initial_state[state_index] = rng.uniform(float(domain.lo), float(domain.hi), size=n_nodes)

  ramp_inputs = np.linspace(-4.0, 4.8, 28)
  hold_inputs = np.full(24, ramp_inputs[-1])
  input_values = np.concatenate([ramp_inputs, hold_inputs])

  voltage_segments = []
  effective_input_segments = []
  weights = np.asarray(exp.network.weights, dtype=float)
  strength = weights.sum(axis=1)
  strength_low, strength_high = np.quantile(strength, [0.05, 0.95])
  relative_strength = np.clip((strength - strength_low) / (strength_high - strength_low), 0.0, 1.0)
  weights_norm = weights / np.maximum(strength[:, None], 1.0)
  structural_drive_scale = 6.4
  dynamic_drive_scale = 0.75

  for input_value in input_values:
    opt_network = tvboptim.create_network(
      weights=weights,
      dynamics_params={"I": float(input_value)},
    )
    simulation = tvboptim.run_simulation(opt_network, t1=segment_duration, dt=dt, run_main=False)
    simulation.state.initial_state.dynamics = jnp.asarray(initial_state)
    result = simulation.model_fn(simulation.state)
    data = np.asarray(result.data)
    voltage = data[:, 0, :]
    voltage_segments.append(voltage)
    incoming_voltage = (weights_norm @ voltage.T).T
    dynamic_drive = dynamic_drive_scale * (incoming_voltage - incoming_voltage.mean(axis=1, keepdims=True))
    structural_drive = structural_drive_scale * relative_strength[None, :]
    effective_input_segments.append(input_value + structural_drive + dynamic_drive)

    initial_state = data[-1]

  return (
    np.concatenate(effective_input_segments, axis=0),
    np.concatenate(voltage_segments, axis=0),
    relative_strength,
    dt,
    len(ramp_inputs) * int(segment_duration / dt),
  )


def gif_g2d_network_bifurcation(n_frames=96):
    cont = Continuation.from_string(G2D_CONT)
    exp = SimulationExperiment(dynamics=Dynamics.from_ontology("Generic2dOscillator"), continuations=[cont])
    continuation = exp.run(BACKEND).continuations["g2d_in_I"]
    effective_input, voltage, relative_strength, dt, ramp_end = _g2d_tvboptim_network_trajectory()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    continuation.plot(VOI="V", ax=ax)
    ax.set_title("Network trajectories on the Generic2dOscillator regime map", fontsize=12)
    ax.set_xlabel(r"effective input $I_{\mathrm{eff}}$")
    ax.set_ylabel("$V$")
    ax.set_xlim(-10.5, 16.5)
    ax.set_ylim(-2.2, 3.6)

    strength_norm = Normalize(vmin=0.0, vmax=1.0)
    colors = plt.colormaps["viridis"](strength_norm(relative_strength))
    trajectory_lines = [
        ax.plot([], [], color=color, lw=0.95, alpha=0.64)[0]
        for color in colors
    ]
    current_points = ax.scatter([], [], s=46, c=[], cmap="viridis", norm=strength_norm,
            edgecolor="white", linewidth=0.45, zorder=5)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, ha="left", va="top",
          fontsize=9, color="0.2")
    ax.text(0.02, 0.04, "each colored trace = one coupled node", transform=ax.transAxes,
          ha="left", va="bottom", fontsize=8, color="0.35")

    colorbar = fig.colorbar(ScalarMappable(norm=strength_norm, cmap="viridis"), ax=ax, pad=0.02, shrink=0.82)
    colorbar.set_label("relative SC in-strength")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        legend.set_title("continuation")
        for text in legend.get_texts():
            text.set_fontsize(7)
        legend.get_title().set_fontsize(8)

    frames = np.unique(np.concatenate([
        np.linspace(8, ramp_end, int(0.62 * n_frames)),
        np.linspace(ramp_end + 1, len(effective_input) - 1, n_frames - int(0.62 * n_frames)),
    ]).astype(int))
    tail_steps = int(260 / dt)

    def update(frame_index):
        start = max(0, frame_index - tail_steps)
        for node, line in enumerate(trajectory_lines):
            line.set_data(effective_input[start:frame_index, node], voltage[start:frame_index, node])
        current_points.set_offsets(np.column_stack([effective_input[frame_index], voltage[frame_index]]))
        current_points.set_array(relative_strength)
        time_text.set_text(fr"TVBO/tvboptim network simulation: $t={frame_index * dt:.0f}$")
        return [*trajectory_lines, current_points, time_text]

    anim = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    out = os.path.join(IMG, "g2d_network_bifurcation.gif")
    anim.save(out, writer=PillowWriter(fps=16), dpi=120)
    plt.close(fig)
    print("wrote", out)


# ===========================================================================
# 8. GIF: Hopf birth — phase plane + 3D bifurcation diagram (built-in)
# ===========================================================================
def gif_hopf_birth(n_frames=24):
    import numpy as np

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
    out = os.path.join(IMG, "hopf_birth.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    print("wrote", out)


# %%

if __name__ == "__main__":
    fig_overview()
    fig_linear_stability()
    fig_phase_portraits()
    fig_normal_forms()
    fig_hopf_3d()
    fig_continuation()
    fig_g2d_bifurcation()
    gif_g2d_network_bifurcation()
    gif_hopf_birth()
    print("done")
