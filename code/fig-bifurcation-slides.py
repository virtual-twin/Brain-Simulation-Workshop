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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageOps
import bsplot  # noqa: F401
from tvbo import Dynamics, SimulationExperiment
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
    gif_hopf_birth()
    print("done")
