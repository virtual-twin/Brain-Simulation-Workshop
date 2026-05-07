"""Generate spring-mass figures and animations for the dynamical systems slides.

Outputs are written to ``img/figures/dynamical_systems`` with the prefix
``fig-dynamical_systems_springs_*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bsplot
from tvbo import Dynamics, SimulationExperiment
from tvbo.datamodel.schema import Parameter


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
FIG_DIR = PROJECT / "img" / "figures" / "dynamical_systems"
PREFIX = "fig-dynamical_systems_springs"

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


def save_animation(ani, stem: str, fps: int = 30) -> None:
    out_thumb = FIG_DIR / f"{PREFIX}_{stem}_thumb.png"
    out_mp4 = FIG_DIR / f"{PREFIX}_{stem}.mp4"
    ani._init_draw()
    ani._func(0)
    ani._fig.savefig(out_thumb, dpi=300, bbox_inches="tight")
    ani.save(
        str(out_mp4),
        writer="ffmpeg",
        fps=fps,
        dpi=150,
        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"],
    )
    print(f"Saved -> {out_thumb}")
    print(f"Saved -> {out_mp4}")


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
        n_periods=1,
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
        n_periods=1,
    )
    save_animation(ani, "ics")


def generate_phase_space() -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))

    for init_x in [0.5, 1.0, 1.5, 2.0]:
        system = make_system()
        system.state_variables["x"]["initial_value"] = init_x
        system.plot(kind="phase", ax=ax, cmap="grey_r")

    make_system().plot(kind="vectorfield", ax=ax, alpha=0.3)
    ax.set_xlabel("x (position)")
    ax.set_ylabel("v (velocity)")
    fig.tight_layout()
    out = FIG_DIR / f"{PREFIX}_phase_space.png"
    fig.savefig(out, dpi=500, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


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

    sizes = [0.4 * (mass ** (1 / 3)) for mass in m_values]
    titles = [rf"$m={mass},\, \omega={(k_fixed / mass) ** 0.5:.4f}$" for mass in m_values]

    ani = animate_spring(
        [x_series(result) for result in results],
        labels=[f"$m={mass}$" for mass in m_values],
        titles=titles,
        mass_sizes=sizes,
        orientation="vertical",
        anchor_pos=3.5,
        n_periods=1,
    )
    save_animation(ani, "mass")


def generate_realism() -> None:
    ideal = make_system()
    realistic = make_system()
    realistic.state_variables["v"]["equation"]["rhs"] = "-(k/m) * x - (c/m) * v + g_eff"
    realistic.parameters["c"] = Parameter(name="c", value=0.006)
    realistic.parameters["g_eff"] = Parameter(name="g_eff", value=0.00015)

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
        n_periods=3,
    )
    save_animation(ani, "realism")


def main() -> None:
    generate_single()
    generate_initial_conditions()
    generate_phase_space()
    generate_mass()
    generate_realism()


if __name__ == "__main__":
    main()