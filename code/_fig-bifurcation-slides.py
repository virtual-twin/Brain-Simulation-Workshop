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
TUTORIAL_FIG_48 = (
    "/Users/leonmartin_bih/tools/tvbo/dev/BifurcationTutorial/slides/img/fig_48.png"
)
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
            line.set_label(rf"$a={a:g}$")
    ax.set(
        xlabel="$t$",
        ylabel="$x(t)$",
        ylim=(0.0, 5.4),
        title=rf"$\dot x = a\,x,\quad x(0)={x0:g},\quad \lambda=a$",
    )
    ax.annotate(
        r"neutral trajectory: $\lambda=a=0$",
        xy=(1.75, x0),
        xytext=(0.7, 1.3),
        arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "0.2"},
        color="0.2",
        fontsize=8,
    )
    ax.legend(
        title="growth rate",
        loc="upper left",
        fontsize=7,
        title_fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    save(fig, "linear_stability.png")


# ===========================================================================
# 3. 2D phase portraits: node · saddle · focus · centre  (real TVBO plot)
# ===========================================================================
def fig_phase_portraits():
    fig, axes = plt.subplots(2, 2, layout="compressed")
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
        )
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    save(fig, "phase_portraits_2d.png")


# ===========================================================================
# 4. Regime-change building blocks: continuations + local scalar flows
# ===========================================================================
def fig_normal_forms():
    fig, ax = plt.subplots(2, 2, figsize=(8.0, 5.6), layout="compressed")

    examples = [
        {
            "title": r"Saddle-node: $\dot x = a - x^2$",
            "continuations": [(SADDLE_NODE, SADDLE_NODE_CONT)],
            "flow": SADDLE_NODE,
            "parameter_value": 0.6,
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
            [0],
            [0],
            color="C0",
            marker=r"$\rightarrow$",
            linestyle="None",
            markersize=11,
            label="arrows: flow direction",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#c85030",
            mfc="#c85030",
            mec="#c85030",
            linestyle="None",
            markersize=5,
            label="filled dot: stable FP",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#c85030",
            mfc="white",
            mec="#c85030",
            linestyle="None",
            markersize=5,
            label="open dot: unstable FP",
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


def _g2d_tvboptim_network_trajectory(
    segment_duration=45.0, dt=0.25, warmup_duration=260.0
):
    network = Network.from_db("DesikanKilliany")
    dynamics = Dynamics.from_db("Generic2dOscillator")
    dynamics.parameters["d"].value = 0.35
    sample_domains = {
        name: (float(state_variable.domain.lo), float(state_variable.domain.hi))
        for name, state_variable in dynamics.state_variables.items()
    }
    dynamics.state_variables["W"].domain.lo = -60.0
    dynamics.state_variables["W"].domain.hi = 20.0
    coupling = Coupling.from_ontology("Linear")
    coupling.parameters["a"].value = 1e-8

    exp = SimulationExperiment(dynamics=dynamics, network=network, coupling=coupling)
    exp.integration.duration = segment_duration
    exp.integration.step_size = dt
    tvboptim = exp.execute("tvboptim")

    n_nodes = exp.network.number_of_nodes
    rng = np.random.default_rng(7)
    initial_state = np.zeros((len(exp.dynamics.state_variables), n_nodes), dtype=float)
    for state_index, state_variable in enumerate(exp.dynamics.state_variables.values()):
        lo, hi = sample_domains[state_variable.name]
        initial_state[state_index] = rng.uniform(lo, hi, size=n_nodes)

    ramp_inputs = np.linspace(-7.0, 2.0, 28)
    hold_inputs = np.full(24, ramp_inputs[-1])
    input_values = np.concatenate([ramp_inputs, hold_inputs])

    voltage_segments = []
    w_segments = []
    effective_input_segments = []
    weights = np.asarray(exp.network.weights, dtype=float)
    strength = weights.sum(axis=1)
    strength_low, strength_high = np.quantile(strength, [0.05, 0.95])
    relative_strength = np.clip(
        (strength - strength_low) / (strength_high - strength_low), 0.0, 1.0
    )
    structural_drive_scale = 8.0

    warmup_input = input_values[0] + structural_drive_scale * relative_strength
    warmup_network = tvboptim.create_network(
        weights=weights, dynamics_params={"I": warmup_input}
    )
    warmup = tvboptim.run_simulation(
        warmup_network, t1=warmup_duration, dt=dt, run_main=False
    )
    warmup.state.initial_state.dynamics = jnp.asarray(initial_state)
    initial_state = np.asarray(warmup.model_fn(warmup.state).data)[-1]

    for input_value in input_values:
        node_input = input_value + structural_drive_scale * relative_strength
        opt_network = tvboptim.create_network(
            weights=weights,
            dynamics_params={"I": node_input},
        )
        simulation = tvboptim.run_simulation(
            opt_network, t1=segment_duration, dt=dt, run_main=False
        )
        simulation.state.initial_state.dynamics = jnp.asarray(initial_state)
        result = simulation.model_fn(simulation.state)
        data = np.asarray(result.data)
        voltage = data[:, 0, :]
        w_var = data[:, 1, :]
        voltage_segments.append(voltage)
        w_segments.append(w_var)
        effective_input_segments.append(np.broadcast_to(node_input, voltage.shape))

        initial_state = data[-1]

    return (
        np.concatenate(effective_input_segments, axis=0),
        np.concatenate(voltage_segments, axis=0),
        np.concatenate(w_segments, axis=0),
        relative_strength,
        dt,
        len(ramp_inputs) * int(segment_duration / dt),
    )


def gif_g2d_network_bifurcation(n_frames=108):
    """Three-panel animation: bifurcation diagram (trajectory tails),
    V-W phase plane (rings emerge on LC), and brain surface."""
    from bsplot.graph import create_network, plot_network_on_surface
    from matplotlib.collections import LineCollection

    cont = Continuation.from_string(G2D_CONT)
    exp = SimulationExperiment(
        dynamics=Dynamics.from_ontology("Generic2dOscillator"), continuations=[cont]
    )
    continuation = exp.run(BACKEND).continuations["g2d_in_I"]
    effective_input, voltage, w_data, relative_strength, dt, ramp_end = (
        _g2d_tvboptim_network_trajectory()
    )

    # Network + bsplot graph
    network = Network.from_db("DesikanKilliany")
    centers = network.get_centers()
    n_nodes = voltage.shape[1]
    weights = np.asarray(network.weights, dtype=float)
    centers_dict = {i: tuple(map(float, centers[i])) for i in range(n_nodes)}
    G = create_network(
        centers_dict,
        weights,
        labels=list(range(n_nodes)),
        threshold_percentile=92,
        directed=False,
    )
    for u, v_, d in G.edges(data=True):
        d["weight"] = float(np.log1p(d["weight"]))

    # Top-view projection: x_mni → axes-x, y_mni → axes-y
    node_xy = np.array([centers_dict[i][:2] for i in range(n_nodes)])

    # Per-node degree for coloring
    degree = np.array([G.degree(i) for i in range(n_nodes)], dtype=float)
    deg_norm = Normalize(vmin=float(degree.min()), vmax=float(degree.max()))
    cmap_deg = plt.colormaps["plasma"]

    # Voltage colormap (brain panel)
    v_min = float(np.percentile(voltage, 1))
    v_max = float(np.percentile(voltage, 99))
    v_norm = Normalize(vmin=v_min, vmax=v_max)
    cmap_v = plt.colormaps["viridis"]

    # W range for phase plane
    w_min = float(np.percentile(w_data, 1))
    w_max = float(np.percentile(w_data, 99))

    base_size = 30 + 100 * relative_strength

    # Pre-compute filled (FP) and hollow (LC) face-color arrays
    filled_colors = cmap_deg(deg_norm(degree)).copy()   # (n_nodes, 4) RGBA
    hollow_colors = filled_colors.copy()
    hollow_colors[:, 3] = 0.0   # transparent fill for LC nodes
    edge_colors = filled_colors.copy()   # degree-mapped edges, always opaque
    edge_colors[:, 3] = 1.0

    # Three-panel layout: bifurcation | phase plane | brain surface
    fig = plt.figure(figsize=(16.5, 4.8), facecolor="white")
    gs = fig.add_gridspec(
        1,
        5,
        width_ratios=[0.025, 1.0, 1.0, 1.0, 0.025],
        wspace=0.12,
        left=0.05,
        right=0.97,
        top=0.93,
        bottom=0.12,
    )
    cax_deg = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[0, 2])
    ax_brain = fig.add_subplot(gs[0, 3])
    cax_v = fig.add_subplot(gs[0, 4])

    # --- Brain panel ---
    plot_network_on_surface(
        G,
        ax=ax_brain,
        template="fsLR",
        density="32k",
        hemi=("lh", "rh"),
        view="top",
        surface_alpha=0.25,
        show_nodes=False,
        edge_radius=0.25,
        edge_cmap="viridis",
        edge_data_key="weight",
        edge_scale={"weight": 4, "mode": "log"},
    )
    ax_brain.set_title("Desikan-Killiany cortex", fontsize=11)
    ax_brain.set_axis_off()
    node_scat = ax_brain.scatter(
        node_xy[:, 0],
        node_xy[:, 1],
        s=base_size,
        c=np.zeros(n_nodes),
        cmap=cmap_v,
        norm=v_norm,
        edgecolor="white",
        linewidth=0.4,
        zorder=20,
    )

    # --- Bifurcation panel ---
    continuation.plot(VOI="V", ax=ax_map)
    ax_map.set_title("regime map", fontsize=11)
    ax_map.set_xlabel(r"effective input $I_{\mathrm{eff}}$")
    ax_map.set_ylabel("$V$")
    ax_map.set_xlim(-10.5, 16.5)
    ax_map.set_ylim(-2.2, 3.6)
    legend = ax_map.get_legend()
    if legend is not None:
        legend.set_frame_on(False)
        for text in legend.get_texts():
            text.set_fontsize(7)

    tail_steps = max(1, int(180 / dt))
    tail_colors = cmap_deg(deg_norm(degree))
    tail_lc = LineCollection(
        [], colors=tail_colors, linewidths=0.7, alpha=0.45, zorder=4
    )
    ax_map.add_collection(tail_lc)
    node_points = ax_map.scatter(
        np.zeros(n_nodes),
        np.zeros(n_nodes),
        s=22,
        c=degree,
        cmap=cmap_deg,
        norm=deg_norm,
        edgecolors=edge_colors,
        linewidth=0.5,
        zorder=6,
        alpha=0.95,
    )

    # --- Phase plane panel (V vs W — trajectories form rings on LC) ---
    ax_phase.set_title("phase plane", fontsize=11)
    ax_phase.set_xlabel("$V$")
    ax_phase.set_ylabel("$W$")
    v_pad = 0.12 * (v_max - v_min)
    w_pad = 0.12 * (w_max - w_min)
    ax_phase.set_xlim(v_min - v_pad, v_max + v_pad)
    ax_phase.set_ylim(w_min - w_pad, w_max + w_pad)
    phase_lc = LineCollection(
        [], colors=tail_colors, linewidths=0.6, alpha=0.40, zorder=4
    )
    ax_phase.add_collection(phase_lc)
    node_phase = ax_phase.scatter(
        np.zeros(n_nodes),
        np.zeros(n_nodes),
        s=18,
        c=degree,
        cmap=cmap_deg,
        norm=deg_norm,
        edgecolors=edge_colors,
        linewidth=0.5,
        zorder=6,
        alpha=0.95,
    )

    time_text = fig.text(
        0.5,
        0.985,
        "",
        ha="center",
        va="top",
        fontsize=10,
        color="0.15",
    )

    cbar_v = fig.colorbar(ScalarMappable(norm=v_norm, cmap=cmap_v), cax=cax_v)
    cbar_v.set_label("$V(t)$  (activity)")
    cbar_deg = fig.colorbar(ScalarMappable(norm=deg_norm, cmap=cmap_deg), cax=cax_deg)
    cbar_deg.set_label("node degree")
    cax_deg.yaxis.set_ticks_position("left")
    cax_deg.yaxis.set_label_position("left")

    # 80 % of frames cover the ramp for slower traversal of parameter space
    frames = np.unique(
        np.concatenate(
            [
                np.linspace(8, ramp_end, int(0.80 * n_frames)),
                np.linspace(
                    ramp_end + 1, len(voltage) - 1, n_frames - int(0.80 * n_frames)
                ),
            ]
        ).astype(int)
    )

    lc_threshold = 0.5  # V amplitude threshold for LC regime detection

    def update(frame_index):
        v_now = voltage[frame_index]
        w_now = w_data[frame_index]
        node_scat.set_array(v_now)
        node_points.set_offsets(np.column_stack([effective_input[frame_index], v_now]))
        start = max(0, frame_index - tail_steps)

        # Bifurcation diagram tails
        segs = [
            np.column_stack(
                [
                    effective_input[start : frame_index + 1, i],
                    voltage[start : frame_index + 1, i],
                ]
            )
            for i in range(n_nodes)
        ]
        tail_lc.set_segments(segs)

        # Phase plane (V-W) tails — form rings when nodes are on LC
        phase_segs = [
            np.column_stack(
                [
                    voltage[start : frame_index + 1, i],
                    w_data[start : frame_index + 1, i],
                ]
            )
            for i in range(n_nodes)
        ]
        phase_lc.set_segments(phase_segs)
        node_phase.set_offsets(np.column_stack([v_now, w_now]))

        # Regime detection: filled dot = stable FP, outline only = limit cycle
        v_window = voltage[start : frame_index + 1]
        amp = v_window.max(axis=0) - v_window.min(axis=0)
        is_lc = amp > lc_threshold
        fc = np.where(is_lc[:, None], hollow_colors, filled_colors)
        node_points.set_facecolors(fc)
        node_phase.set_facecolors(fc)

        mean_I = float(effective_input[frame_index].mean())
        time_text.set_text(
            rf"TVBO + tvboptim — Desikan-Killiany — $t={frame_index * dt:.0f}$ ms,  "
            rf"$\langle I_{{\mathrm{{eff}}}}\rangle={mean_I:+.2f}$"
        )
        return [node_scat, node_points, tail_lc, time_text, phase_lc, node_phase]

    # Render frames manually with a single shared palette.
    # PillowWriter quantizes per-frame → colorbar gradients shimmer; there
    # is no global-palette option in anim.save / PillowWriter.
    import io

    out = os.path.join(IMG, "g2d_network_bifurcation.gif")
    rendered = []
    for fi in frames:
        update(int(fi))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white", dpi=120)
        buf.seek(0)
        rendered.append(Image.open(buf).convert("RGB").copy())
        buf.close()
    img_w, img_h = rendered[0].size
    big = Image.new("RGB", (img_w * len(rendered), img_h))
    for i, f in enumerate(rendered):
        big.paste(f, (i * img_w, 0))
    pal_img = big.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    quantized = [
        f.quantize(palette=pal_img, dither=Image.Dither.NONE) for f in rendered
    ]
    quantized[0].save(
        out,
        save_all=True,
        append_images=quantized[1:],
        duration=int(1000 / 10),
        loop=0,
        disposal=2,
        optimize=False,
    )
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
# Individual figure scripts live alongside this file with the fig- prefix.
# Run them directly, e.g.:
#   uv run code/fig-bifurcation-overview.py
#
# To regenerate everything at once, run each script via subprocess:

if __name__ == "__main__":
    import subprocess
    import sys

    scripts = [
        "fig-bifurcation-overview.py",
        "fig-bifurcation-linear-stability.py",
        "fig-bifurcation-phase-portraits.py",
        "fig-bifurcation-normal-forms.py",
        "fig-bifurcation-hopf-3d.py",
        "fig-bifurcation-continuation.py",
        "fig-bifurcation-g2d.py",
        "fig-bifurcation-g2d-network.py",
        "fig-bifurcation-hopf-birth.py",
    ]
    code_dir = os.path.dirname(os.path.abspath(__file__))
    for script in scripts:
        print(f"--- running {script} ---")
        subprocess.run(
            [sys.executable, os.path.join(code_dir, script)],
            check=True,
            cwd=code_dir,
        )
    print("done")
