"""Render all figures + GIFs used in the Bifurcation section of slides.qmd.

Everything is computed by TVBO itself: ``Dynamics.plot``, ``Continuation.plot``,
``Continuation.plot_3d``. No schematic/synthetic curves. All YAML is literal
(no f-string templating). Styled with ``bsplot.style.use('tvbo')``.

Continuation figures need the ``bifurcationkit.jl`` backend (Julia).
"""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import bsplot  # noqa: F401
bsplot.style.use("tvbo")

from tvbo import Dynamics, SimulationExperiment
from tvbo.classes.continuation import Continuation

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(ROOT, "..", "img"))
os.makedirs(IMG, exist_ok=True)

BACKEND = "bifurcationkit.jl"


def save(fig, name, **kw):
    out = os.path.join(IMG, name)
    fig.savefig(out, dpi=150, bbox_inches="tight", **kw)
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


def _bif(dyn_yaml, cont_yaml):
    """Run a continuation and return the single Continuation result."""
    dyn = Dynamics.from_string(dyn_yaml)
    cont = Continuation.from_string(cont_yaml)
    exp = SimulationExperiment(dynamics=dyn, continuations=[cont])
    name = next(iter(exp.continuations))
    return exp.run(BACKEND).continuations[name]


# ===========================================================================
# 1. Overview: time series · phase plane · bifurcation diagram (Hopf NF)
# ===========================================================================
def fig_overview():
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.0))

    dyn_stable = Dynamics.from_string(HOPF)
    dyn_stable.parameters["a"].value = -0.5
    dyn_stable.plot("x1", kind="timeseries", duration=4000, dt=0.5, ax=ax[0])
    dyn_osc = Dynamics.from_string(HOPF)
    dyn_osc.parameters["a"].value = +0.5
    dyn_osc.plot("x1", kind="timeseries", duration=4000, dt=0.5, ax=ax[0])
    ax[0].set_title("Time series ($a=\\pm 0.5$)")

    Dynamics.from_string(HOPF).plot(
        "x1", "x2", kind="phaseplane", ax=ax[1],
        grid_n=22, n_trajectories=3, duration=15,
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
    for a in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        dyn.parameters["a"].value = a
        dyn.plot("x", kind="timeseries", duration=4.0, dt=0.01, ax=ax)
    ax.set(xlabel="$t$", ylabel="$x(t)$",
           title=r"$\dot x = a\,x,\quad x(0)=1$")
    fig.tight_layout()
    save(fig, "linear_stability.png")


# ===========================================================================
# 3. 2D phase portraits: node · saddle · focus · centre  (real TVBO plot)
# ===========================================================================
def fig_phase_portraits():
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.2))
    cases = [
        ("Stable node", NODE_2D),
        ("Saddle",      SADDLE_2D),
        ("Stable focus", FOCUS_2D),
        ("Centre",      CENTRE_2D),
    ]
    for ax, (label, yml) in zip(axes.flat, cases):
        Dynamics.from_string(yml).plot(
            "x1", "x2", kind="phaseplane", ax=ax,
            grid_n=20, n_trajectories=4, duration=15,
        )
        ax.set_title(label, fontsize=10)
    fig.tight_layout()
    save(fig, "phase_portraits_2d.png")


# ===========================================================================
# 4. Codim-1 normal forms: real continuations
# ===========================================================================
def fig_normal_forms():
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
    _bif(SADDLE_NODE, SADDLE_NODE_CONT).plot(VOI="x", ax=ax[0])
    ax[0].set_title(r"Saddle-node: $\dot x = a - x^2$")
    _bif(PITCHFORK, PITCHFORK_CONT).plot(VOI="x", ax=ax[1])
    ax[1].set_title(r"Pitchfork: $\dot x = a x - x^3$")
    _bif(HYSTERESIS, HYSTERESIS_CONT).plot(VOI="x", ax=ax[2])
    ax[2].set_title(r"Hysteresis: $\dot x = a + x - x^3$")
    fig.tight_layout()
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
# 6. Continuation example: real bifurcation diagram of the hysteresis system
# ===========================================================================
def fig_continuation():
    cont = _bif(HYSTERESIS, HYSTERESIS_CONT)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cont.plot(VOI="x", ax=ax)
    ax.set_title("Pseudo-arclength continuation along an S-curve")
    fig.tight_layout()
    save(fig, "continuation_schematic.png")


# ===========================================================================
# 7. Generic2dOscillator: real bifurcation diagram in I
# ===========================================================================
def fig_g2d_bifurcation():
    dyn = Dynamics.from_ontology("Generic2dOscillator")
    cont = Continuation.from_string(G2D_CONT)
    exp = SimulationExperiment(dynamics=dyn, continuations=[cont])
    g2d = exp.run(BACKEND).continuations["g2d_in_I"]
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    g2d.plot(VOI="V", ax=ax)
    ax.set_title("Generic2dOscillator: codim-1 in $I$")
    fig.tight_layout()
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
        dyn, "a", values, "x1", "x2", VOI="x1",
        grid_n=18, n_trajectories=3, duration=15,
        figsize=(11, 4.8),
        title_fmt=r"$a = {value:+.2f}$",
    )
    out = os.path.join(IMG, "hopf_birth.gif")
    anim.save(out, writer=PillowWriter(fps=8))
    print("wrote", out)


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
