"""Generic2dOscillator regime map for the stimulation chapter (slide 7, Step 3).

Brute-force bifurcation diagram of the *database-default* Generic2dOscillator:
sweep the input current ``I``, integrate each setting to steady state, and record
the min/max of ``V`` over the settled tail. Where they coincide the node sits on a
stable fixed point; where they split it is on a limit cycle (the periodic-orbit
envelope). The lower Hopf is the input at which the envelope opens (~3.5).

Two operating points are marked to match the Step-3 time series:
- ``I = 0``  -> rest (stable focus)
- ``I = 5``  -> during the pulse (effective input ``I + perturbation``), inside the
  limit-cycle envelope -> a limit cycle is born for the pulse's duration.

Output: img/figures/stimulation/fig-g2d-bifurcation-hopf.png
"""
from __future__ import annotations

import contextlib
import io
import os

import matplotlib.pyplot as plt
import numpy as np

from tvbo import Dynamics, SimulationExperiment

ROOT = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.basename(__file__).replace(".py", ".png")
OUTDIR = os.path.join(ROOT, "..", "img", "figures", "stimulation")
os.makedirs(OUTDIR, exist_ok=True)

I_SWEEP = np.linspace(-2, 18, 41)
OSC_TOL = 0.05          # V peak-to-peak above this counts as a limit cycle
REST_I, PULSE_I = 0.0, 5.0


def _v_envelope(I: float) -> tuple[float, float]:
    """Return (V_min, V_max) over the settled tail for a single node at input ``I``."""
    dyn = Dynamics.from_db("Generic2dOscillator")
    dyn.parameters["I"].value = float(I)
    # widen W so a driven limit cycle is not clamped by the bounded solver
    dyn.state_variables["W"].domain.lo = -60.0
    dyn.state_variables["W"].domain.hi = 60.0
    exp = SimulationExperiment(dynamics=dyn)
    exp.integration.duration = 2000.0
    exp.integration.step_size = 0.5
    with contextlib.redirect_stdout(io.StringIO()):
        res = exp.run()
    V = np.asarray(res.integration.sel(variable="V").data.values).squeeze()
    tail = V[len(V) // 2:]
    return float(tail.min()), float(tail.max())


def main() -> None:
    vmin = np.empty_like(I_SWEEP)
    vmax = np.empty_like(I_SWEEP)
    for k, I in enumerate(I_SWEEP):
        vmin[k], vmax[k] = _v_envelope(I)

    osc = (vmax - vmin) > OSC_TOL
    hopf = I_SWEEP[osc][0] if osc.any() else None

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    fp = ~osc
    ax.plot(I_SWEEP[fp], vmax[fp], "k-", lw=2.2, label="stable fixed point")
    ax.fill_between(I_SWEEP, vmin, vmax, where=osc, color="tab:blue", alpha=0.25,
                    label="limit-cycle envelope")
    ax.plot(I_SWEEP[osc], vmax[osc], color="tab:blue", lw=1.4)
    ax.plot(I_SWEEP[osc], vmin[osc], color="tab:blue", lw=1.4)

    if hopf is not None:
        ax.axvline(hopf, color="purple", ls="--", lw=1.0)
        ax.text(hopf + 0.2, ax.get_ylim()[0] + 0.2, f"Hopf ≈ {hopf:.1f}",
                color="purple", fontsize=11)

    i_rest = int(np.argmin(np.abs(I_SWEEP - REST_I)))
    ax.scatter([REST_I], [vmax[i_rest]], color="green", zorder=5, s=70)
    ax.annotate("rest\nI = 0", (REST_I, vmax[i_rest]), textcoords="offset points",
                xytext=(-6, 14), color="green", fontsize=11, ha="center", fontweight="bold")

    i_pulse = int(np.argmin(np.abs(I_SWEEP - PULSE_I)))
    ax.scatter([PULSE_I, PULSE_I], [vmin[i_pulse], vmax[i_pulse]], color="red", zorder=5, s=70)
    ax.annotate("pulse\nI + 5", (PULSE_I, vmax[i_pulse]), textcoords="offset points",
                xytext=(10, 6), color="red", fontsize=11, fontweight="bold")

    ax.set_xlabel("effective input  I")
    ax.set_ylabel("V")
    ax.set_title("Generic2dOscillator: regimes vs input")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    out = os.path.join(OUTDIR, FILENAME)
    fig.savefig(out, dpi=130)
    print(f"saved {out}  (Hopf ≈ {hopf}, osc range "
          f"{I_SWEEP[osc].min() if osc.any() else None}..{I_SWEEP[osc].max() if osc.any() else None})")


if __name__ == "__main__":
    main()
