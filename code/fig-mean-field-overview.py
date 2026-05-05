"""Mean-field overview: 2x2 figure linking spiking cells to a 3-population NMM.

Top row    : (A) jaxley HH network of 3 cell populations (P/E/I).
             (B) Spike raster + smoothed population rates r_P, r_E, r_I.
Bottom row : (C) Jansen-Rit 3-population schematic.
             (D) Jansen-Rit mean-field activity from TVBO+tvboptim.

The schematic and the NMM are the same model (Jansen-Rit, 3 populations:
pyramidal cells, excitatory interneurons, inhibitory interneurons).
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

import jaxley as jx
from jaxley.channels import HH
from jaxley.connect import connect
from jaxley.synapses import IonotropicSynapse

from tvbo.classes.experiment import SimulationExperiment
from tvbo.classes.dynamics import Dynamics

import bsplot
bsplot.style.use("tvbo")


# ---- constants --------------------------------------------------------------
P_COLOR = "#7a4ea8"   # pyramidal cells (purple)
E_COLOR = "#c8553d"   # excitatory interneurons (red)
I_COLOR = "#3d6fc8"   # inhibitory interneurons (blue)
SYN_COLOR = "#bbbbbb"

N_P, N_E, N_I = 4, 3, 3
SWC_FILE = "/Users/leonmartin_bih/work_data/toolboxes/jaxley/docs/tutorials/data/morph.swc"
SCHEMATIC = "/Users/leonmartin_bih/projects/TVB-O/tvb-o-ptim/code/fig/3PopMeanField.png"
OUT = "/Users/leonmartin_bih/projects/TVB-O/tvb-ontology-optim-workshop/img/mean_field_overview.png"

T_MAX_MS = 400.0
DT_MS = 0.025


# ---- spiking network --------------------------------------------------------
def build_spiking_network():
    cell = jx.read_swc(SWC_FILE, ncomp=1)
    n_cells = N_P + N_E + N_I
    net = jx.Network([cell] * n_cells)
    net.insert(HH())

    # Spatial layout: 3 columns (P | E | I)
    col_x = {"P": 0, "E": 1500, "I": 3000}
    spacing = 900
    p_idx = list(range(N_P))
    e_idx = list(range(N_P, N_P + N_E))
    i_idx = list(range(N_P + N_E, n_cells))

    for k, i in enumerate(p_idx):
        net.cell(i).move(col_x["P"], k * spacing)
    for k, i in enumerate(e_idx):
        net.cell(i).move(col_x["E"], k * spacing + spacing / 2)
    for k, i in enumerate(i_idx):
        net.cell(i).move(col_x["I"], k * spacing + spacing / 2)

    rng = np.random.default_rng(0)

    def wire(pre, post, p_conn, gS):
        for a in pre:
            for b in post:
                if a == b:
                    continue
                if rng.random() < p_conn:
                    connect(net.cell(a).soma.branch(0).comp(0),
                            net.cell(b).soma.branch(0).comp(0),
                            IonotropicSynapse())

    # P drives E and I; E drives P; I inhibits P. Sparse recurrent within E.
    wire(p_idx, e_idx, 0.7, 5e-4)
    wire(p_idx, i_idx, 0.6, 5e-4)
    wire(e_idx, p_idx, 0.5, 5e-4)
    wire(i_idx, p_idx, 0.6, 5e-4)
    wire(e_idx, e_idx, 0.3, 3e-4)

    return net, p_idx, e_idx, i_idx


def simulate_spiking(net, t_max_ms=T_MAX_MS, dt_ms=DT_MS):
    n_cells = net.shape[0]
    rng = np.random.default_rng(1)
    base = rng.uniform(0.18, 0.30, size=n_cells)
    n_t = int(t_max_ms / dt_ms)
    noise = 0.06 * rng.standard_normal((n_cells, n_t))
    i_inj = base[:, None] + noise
    for k in range(n_cells):
        net.cell(k).soma.branch(0).comp(0).stimulate(i_inj[k])
        net.cell(k).soma.branch(0).comp(0).record("v")
    v = jx.integrate(net, delta_t=dt_ms, t_max=t_max_ms)
    t_out = np.linspace(0, t_max_ms, v.shape[1])
    return t_out, np.asarray(v)


def detect_spikes(t_ms, v, threshold=-20.0):
    spikes = []
    for k in range(v.shape[0]):
        crossings = np.where((v[k, :-1] < threshold) & (v[k, 1:] >= threshold))[0]
        spikes.append(t_ms[crossings])
    return spikes


def population_rate(spikes_subset, t_ms, bin_ms=4.0):
    """Per-cell average firing rate, smoothed. Reflect edges to avoid drop."""
    edges = np.arange(t_ms[0], t_ms[-1] + bin_ms, bin_ms)
    centers = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(len(centers))
    for s in spikes_subset:
        c, _ = np.histogram(s, bins=edges)
        counts += c
    n = max(len(spikes_subset), 1)
    rate = counts / (n * bin_ms / 1000.0)  # Hz per cell
    # Pad reflectively before smoothing to avoid roll-off at the edges
    pad = 6
    padded = np.concatenate([rate[:pad][::-1], rate, rate[-pad:][::-1]])
    smooth = gaussian_filter1d(padded, sigma=2.0)[pad:-pad]
    # Trim first/last bins to drop histogram edge artifacts
    return centers[3:-3], smooth[3:-3]


# ---- mean field (Jansen-Rit) ------------------------------------------------
def simulate_mean_field(duration_ms=T_MAX_MS):
    """Jansen-Rit produces a clear alpha-band oscillation by default."""
    model = Dynamics.from_db("JansenRit")
    exp = SimulationExperiment(dynamics=model)
    exp.integration.duration = duration_ms
    result = exp.run("tvboptim")
    return result


# ---- figure ----------------------------------------------------------------
def draw_jr_schematic(ax):
    """Replicate the 3-population Jansen-Rit cartoon with matching P/E/I colors.

    PC at top (triangle), E at bottom-left (star/octagon), I at bottom-right (circle).
    Excitatory connections: arrow heads. Inhibitory: round (circle) heads.
    """
    from matplotlib.patches import RegularPolygon, Circle, FancyArrowPatch

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    pc_xy = (0.0, 0.55)
    e_xy = (-0.85, -0.55)
    i_xy = (0.85, -0.55)

    # Pyramidal cells: triangle pointing up
    ax.add_patch(RegularPolygon(pc_xy, numVertices=3, radius=0.42,
                                orientation=0, facecolor=P_COLOR,
                                edgecolor="none", zorder=3))
    ax.text(pc_xy[0], pc_xy[1] - 0.02, "PC", color="white",
            ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)

    # Excitatory interneurons: 8-point star (octagon look-alike)
    ax.add_patch(RegularPolygon(e_xy, numVertices=8, radius=0.32,
                                orientation=np.pi / 8, facecolor=E_COLOR,
                                edgecolor="none", zorder=3))
    ax.text(e_xy[0], e_xy[1], "E", color="white",
            ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

    # Inhibitory interneurons: circle
    ax.add_patch(Circle(i_xy, radius=0.27, facecolor=I_COLOR,
                        edgecolor="none", zorder=3))
    ax.text(i_xy[0], i_xy[1], "I", color="white",
            ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)

    def _arrow(a, b, color, inhibitory=False, rad=0.0):
        style = "-|>" if not inhibitory else "-"
        arrow = FancyArrowPatch(
            a, b, arrowstyle=style, mutation_scale=18,
            color=color, linewidth=2.5,
            connectionstyle=f"arc3,rad={rad}", zorder=2,
        )
        ax.add_patch(arrow)
        if inhibitory:
            # round inhibitory terminal at the head
            ax.add_patch(Circle(b, radius=0.06, facecolor=color,
                                edgecolor="none", zorder=3))

    # PC -> E (excitatory feed-forward, outer curve on the left)
    _arrow((pc_xy[0] - 0.30, pc_xy[1] - 0.20),
           (e_xy[0] + 0.25, e_xy[1] + 0.25),
           P_COLOR, rad=0.30)
    # PC -> I (excitatory feed-forward, outer curve on the right)
    _arrow((pc_xy[0] + 0.30, pc_xy[1] - 0.20),
           (i_xy[0] - 0.25, i_xy[1] + 0.20),
           P_COLOR, rad=-0.30)
    # E -> PC (excitatory feedback, inner curve on the left)
    _arrow((e_xy[0] + 0.30, e_xy[1] + 0.10),
           (pc_xy[0] - 0.20, pc_xy[1] - 0.30),
           E_COLOR, rad=-0.30)
    # I -> PC (inhibitory feedback, inner curve on the right)
    _arrow((i_xy[0] - 0.30, i_xy[1] + 0.10),
           (pc_xy[0] + 0.20, pc_xy[1] - 0.30),
           I_COLOR, rad=0.30, inhibitory=True)


def make_figure():
    print("[1/4] building spiking network ...")
    net, p_idx, e_idx, i_idx = build_spiking_network()

    print("[2/4] simulating spiking network ...")
    t_ms, v = simulate_spiking(net)
    spikes = detect_spikes(t_ms, v)
    rP_t, rP = population_rate([spikes[i] for i in p_idx], t_ms)
    rE_t, rE = population_rate([spikes[i] for i in e_idx], t_ms)
    rI_t, rI = population_rate([spikes[i] for i in i_idx], t_ms)

    print("[3/4] running TVBO Jansen-Rit (tvboptim backend) ...")
    mf = simulate_mean_field(duration_ms=T_MAX_MS)
    mf_time = np.asarray(mf.time)
    # JR state vars: y0=PC, y1=EIN, y2=IIN. EEG-like signal commonly y1-y2.
    raw = np.asarray(mf.data)
    arr = raw.squeeze()
    # arr shape (n_t, n_state)
    y_ein = arr[:, 1]
    y_iin = arr[:, 2]
    y_pc = y_ein - y_iin  # PSP at pyramidal cells (EEG-like)

    print("[4/4] composing figure ...")
    fig = plt.figure(figsize=(11.5, 7.5))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.4],
                  height_ratios=[1.0, 1.0], wspace=0.25, hspace=0.35)

    # --- A: cell network -----------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    net.vis(ax=axA, detail="full",
            color="none",
            synapse_color=SYN_COLOR,
            cell_plot_kwargs={"linewidth": 0.0, "alpha": 0.0},
            synapse_plot_kwargs={"linewidth": 0.5, "alpha": 0.4},
            synapse_scatter_kwargs={"s": 3})
    for i in p_idx:
        net.cell(i).vis(ax=axA, color=P_COLOR, linewidth=0.5)
    for i in e_idx:
        net.cell(i).vis(ax=axA, color=E_COLOR, linewidth=0.5)
    for i in i_idx:
        net.cell(i).vis(ax=axA, color=I_COLOR, linewidth=0.5)
    axA.set_title("A. Spiking cells (jaxley, HH)\nP (purple) · E (red) · I (blue)",
                  fontsize=10)
    axA.set_xticks([]); axA.set_yticks([])
    for sp in axA.spines.values():
        sp.set_visible(False)

    # --- B: raster + rates ---------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    y_offset = 0
    for i in p_idx:
        axB.vlines(spikes[i], y_offset, y_offset + 0.8, color=P_COLOR, linewidth=0.6)
        y_offset += 1
    for i in e_idx:
        axB.vlines(spikes[i], y_offset, y_offset + 0.8, color=E_COLOR, linewidth=0.6)
        y_offset += 1
    for i in i_idx:
        axB.vlines(spikes[i], y_offset, y_offset + 0.8, color=I_COLOR, linewidth=0.6)
        y_offset += 1
    axB.set_xlim(0, T_MAX_MS)
    axB.set_ylim(0, y_offset)
    axB.set_xlabel("time [ms]")
    axB.set_ylabel("cell #")
    axB.set_title("B. Spike raster + smoothed population rates", fontsize=10)

    axBr = axB.twinx()
    axBr.plot(rP_t, rP, color=P_COLOR, linewidth=1.6, label=r"$r_P$")
    axBr.plot(rE_t, rE, color=E_COLOR, linewidth=1.6, label=r"$r_E$")
    axBr.plot(rI_t, rI, color=I_COLOR, linewidth=1.6, label=r"$r_I$")
    axBr.set_ylabel("rate [Hz/cell]")
    axBr.legend(loc="upper right", fontsize=8, frameon=False)

    # --- C: 3-pop schematic (Jansen-Rit, matching colors) -------------------
    axC = fig.add_subplot(gs[1, 0])
    draw_jr_schematic(axC)
    axC.set_title("C. Jansen-Rit 3-population schematic", fontsize=10)

    # --- D: JR mean-field traces (z-scored so all 3 visible) ----------------
    def _z(x):
        x = np.asarray(x)
        s = x.std()
        return (x - x.mean()) / (s if s > 0 else 1.0)
    axD = fig.add_subplot(gs[1, 1])
    axD.plot(mf_time, _z(y_pc), color=P_COLOR, linewidth=1.6, label=r"$y_1 - y_2$ (PC)")
    axD.plot(mf_time, _z(y_ein), color=E_COLOR, linewidth=1.6, label=r"$y_1$ (EIN)")
    axD.plot(mf_time, _z(y_iin), color=I_COLOR, linewidth=1.6, label=r"$y_2$ (IIN)")
    axD.set_xlim(0, mf_time[-1])
    axD.set_xlabel("time [ms]")
    axD.set_ylabel("activity (z-scored)")
    axD.set_title("D. Jansen-Rit mean field (TVBO, tvboptim)", fontsize=10)
    axD.legend(loc="upper right", fontsize=8, frameon=False)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    make_figure()
