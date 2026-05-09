"""Generate spring-mass figures and animations for the dynamical systems slides.

Outputs are written to ``img/figures/dynamical_systems`` with the prefix
``fig-dynamical_systems_springs_*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

import bsplot
from tvbo import Dynamics, SimulationExperiment
from tvbo.datamodel.schema import Parameter


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIG_DIR = PROJECT / "img" / "figures" / "dynamical_systems"
PREFIX = "fig-dynamical_systems_springs"
MEDIA_FIGSIZE = (7.0, 4.5)
MEDIA_DPI = 180
PHASE_ICS = [(-2.0, 0.0), (-1.0, 0.015), (2.0, -0.01)]

sys.path.insert(0, str(PROJECT / "notebooks"))
from spring_animation import animate_spring


bsplot.style.use("tvbo")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def make_system() -> Dynamics:
    return Dynamics(
        name="SpringMass",
        state_variables=[
            {
                "name": "x",
                "equation": {"rhs": "v"},
                "initial_value": 2.0,
            },
            {
                "name": "v",
                "equation": {"rhs": "-(k/m) * x"},
                "initial_value": 0.0,
            },
        ],
        parameters=[
            {"name": "k", "value": 0.0001},
            {"name": "m", "value": 1.0},
        ],
    )


def run_system(system: Dynamics):
    return SimulationExperiment(dynamics=system).run()


def x_series(result) -> np.ndarray:
    return np.asarray(result.data.sel(variable="x")).ravel()


def v_series(result) -> np.ndarray:
    return np.asarray(result.data.sel(variable="v")).ravel()


def save_animation(ani, stem: str, fps: int = 24) -> None:
    out_thumb = FIG_DIR / f"{PREFIX}_{stem}_thumb.png"
    out_gif = FIG_DIR / f"{PREFIX}_{stem}.gif"
    ani.save(
        str(out_gif),
        writer=PillowWriter(fps=fps),
        dpi=MEDIA_DPI,
    )
    with Image.open(out_gif) as gif:
        gif.seek(0)
        gif.convert("RGBA").save(out_thumb)
    print(f"Saved -> {out_thumb}")
    print(f"Saved -> {out_gif}")


def make_damped_system() -> Dynamics:
    system = make_system()
    system.state_variables["v"]["equation"]["rhs"] = "-(k/m) * x - (c/m) * v + g_eff"
    system.parameters["c"] = Parameter(name="c", value=0.006)
    system.parameters["g_eff"] = Parameter(name="g_eff", value=0.00015)
    return system


def run_phase_trajectories(system: Dynamics, initials=PHASE_ICS):
    results = []
    for x0, v0 in initials:
        model = system.copy(deep=True)
        model.state_variables["x"]["initial_value"] = x0
        model.state_variables["v"]["initial_value"] = v0
        results.append(run_system(model))
    return results


def animate_phase_plane(system: Dynamics, results, title: str, equilibrium: tuple[float, float]) -> FuncAnimation:
    fig, ax = plt.subplots(figsize=MEDIA_FIGSIZE)

    xs = [x_series(result) for result in results]
    vs = [v_series(result) for result in results]
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    frame_idx = np.linspace(0, min(len(x) for x in xs) - 1, 240, dtype=int)

    x_pad = 0.35 * max(max(np.ptp(x) for x in xs), 1.0)
    v_pad = 0.35 * max(max(np.ptp(v) for v in vs), 0.02)
    xlim = (min(float(x.min()) for x in xs) - x_pad, max(float(x.max()) for x in xs) + x_pad)
    ylim = (min(float(v.min()) for v in vs) - v_pad, max(float(v.max()) for v in vs) + v_pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    system.plot(kind="vectorfield", ax=ax, alpha=0.35, grid_n=28)
    ax.plot(*equilibrium, "o", color="#c85030", ms=8, mec="white", mew=1.2, zorder=12)
    ax.set_title(title)
    ax.set_xlabel("x (position)")
    ax.set_ylabel("v (velocity)")

    lines = []
    dots = []
    for idx, (x, v) in enumerate(zip(xs, vs)):
        color = colors[idx % len(colors)]
        line, = ax.plot([], [], lw=2.0, color=color, alpha=0.9, zorder=8)
        dot, = ax.plot([], [], "o", ms=6, color=color, mec="white", mew=0.8, zorder=10)
        lines.append((line, x, v))
        dots.append((dot, x, v))

    def update(frame):
        stop = frame_idx[frame]
        changed = []
        for line, x, v in lines:
            line.set_data(x[:stop + 1], v[:stop + 1])
            changed.append(line)
        for dot, x, v in dots:
            dot.set_data([x[stop]], [v[stop]])
            changed.append(dot)
        return changed

    fig.tight_layout()
    return FuncAnimation(fig, update, frames=len(frame_idx), interval=40, blit=True)


def generate_single() -> None:
    system = make_system()
    system.state_variables["x"]["initial_value"] = 2.0
    result = run_system(system)

    ani = animate_spring(
        [x_series(result)],
        orientation="horizontal",
        anchor_pos=-3.0,
        coil_amplitude=0.15,
        mass_sizes=0.3,
        labels=["$x$"],
        titles=["Spring oscillator"],
        figsize=MEDIA_FIGSIZE,
        n_periods=2,
    )
    save_animation(ani, "single")


def generate_initial_conditions() -> None:
    init_xs = [-1.0, -2.0, -3.0]
    results = []
    for x0 in init_xs:
        system = make_system()
        system.state_variables["x"]["initial_value"] = x0
        results.append(run_system(system))

    ani = animate_spring(
        [x_series(result) for result in results],
        labels=[f"$x_0={x0}$" for x0 in init_xs],
        titles=[f"$x_0 = {x0}$" for x0 in init_xs],
        mass_sizes=0.4,
        orientation="vertical",
        anchor_pos=3.5,
        figsize=MEDIA_FIGSIZE,
        n_periods=2,
    )
    save_animation(ani, "ics")


def generate_phase_space() -> None:
    fig, ax = plt.subplots(figsize=MEDIA_FIGSIZE)

    for init_x in [0.5, 1.0, 1.5, 2.0]:
        system = make_system()
        system.state_variables["x"]["initial_value"] = init_x
        system.plot(kind="phase", ax=ax, cmap="grey_r")

    make_system().plot(kind="vectorfield", ax=ax, alpha=0.3)
    ax.set_xlabel("x (position)")
    ax.set_ylabel("v (velocity)")
    fig.tight_layout()
    out = FIG_DIR / f"{PREFIX}_phase_space.png"
    with mpl.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(out, dpi=MEDIA_DPI)
    plt.close(fig)
    print(f"Saved -> {out}")


def generate_phase_animation() -> None:
    system = make_system()
    results = run_phase_trajectories(system)
    ani = animate_phase_plane(system, results, "Spring oscillator phase plane", (0.0, 0.0))
    save_animation(ani, "phase_plane")


def generate_damped_phase_animation() -> None:
    system = make_damped_system()
    system.state_variables["x"]["initial_value"] = -2

    results = run_phase_trajectories(system)
    x_eq = system.parameters["m"]["value"] * system.parameters["g_eff"]["value"] / system.parameters["k"]["value"]
    ani = animate_phase_plane(system, results, "Damped oscillator phase plane", (x_eq, 0.0))
    save_animation(ani, "realism_phase_plane")


def generate_mass() -> None:
    system = make_system()
    m_values = [0.5, 1.0, 2.0]
    x0_fixed = -2.0
    k_fixed = system.parameters["k"]["value"]

    results = []
    for mass in m_values:
        model = make_system()
        model.state_variables["x"]["initial_value"] = x0_fixed
        model.state_variables["v"]["initial_value"] = 0.0
        model.parameters["m"]["value"] = mass
        results.append(run_system(model))

    sizes = [0.4 * (mass ** 2) for mass in m_values]
    titles = [rf"$m={mass},\, \omega={(k_fixed / mass) ** 0.5:.4f}$" for mass in m_values]

    ani = animate_spring(
        [x_series(result) for result in results],
        labels=[f"$m={mass}$" for mass in m_values],
        titles=titles,
        mass_sizes=sizes,
        orientation="vertical",
        anchor_pos=3.5,
        figsize=MEDIA_FIGSIZE,
        n_periods=2,
    )
    save_animation(ani, "mass")


def generate_realism() -> None:
    ideal = make_system()
    realistic = make_damped_system()

    ideal.state_variables["x"]["initial_value"] = -2.0
    ideal.state_variables["v"]["initial_value"] = 0.0
    realistic.state_variables["x"]["initial_value"] = -2.0
    realistic.state_variables["v"]["initial_value"] = 0.0

    ideal_result = run_system(ideal)
    realistic_result = run_system(realistic)


    g_eff = realistic.parameters["g_eff"]["value"]
    k = realistic.parameters["k"]["value"]
    m = realistic.parameters["m"]["value"]
    x_eq_real = m * g_eff / k

    ani = animate_spring(
        [x_series(ideal_result), x_series(realistic_result)],
        labels=["toy model", "damped + load"],
        titles=["Toy model", "Damped + load"],
        equilibrium=[0.0, x_eq_real],
        mass_sizes=0.4,
        orientation="vertical",
        anchor_pos=3.5,
        figsize=MEDIA_FIGSIZE,
        n_periods=4,
    )
    save_animation(ani, "realism")


def main() -> None:
    generate_single()
    generate_initial_conditions()
    generate_phase_space()
    generate_phase_animation()
    generate_mass()
    generate_realism()
    generate_damped_phase_animation()


if __name__ == "__main__":
    main()