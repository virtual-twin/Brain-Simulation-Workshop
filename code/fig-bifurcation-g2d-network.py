"""GIF: G2D network trajectory — bifurcation map, phase plane, brain surface."""
from __future__ import annotations

import io
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from PIL import Image
from bsplot.graph import create_network, plot_network_on_surface
from _bif_common import IMG, G2D_CONT, BACKEND
from tvbo import Coupling, Dynamics, Network, SimulationExperiment
from tvbo.classes.continuation import Continuation
from tvbo.datamodel.schema import Distribution, Range
from tvbo.datamodel.schema import Exploration as _Exploration

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".gif"))


def _g2d_tvboptim_network_trajectory(
    segment_duration=45.0, dt=0.25, warmup_duration=260.0
):
    network = Network.from_db("DesikanKilliany")
    dynamics = Dynamics.from_db("Generic2dOscillator")
    dynamics.parameters["d"].value = 0.35
    sample_domains = {
        name: (float(sv.domain.lo), float(sv.domain.hi))
        for name, sv in dynamics.state_variables.items()
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
    for state_index, sv in enumerate(exp.dynamics.state_variables.values()):
        lo, hi = sample_domains[sv.name]
        initial_state[state_index] = rng.uniform(lo, hi, size=n_nodes)

    ramp_inputs = np.linspace(-7.0, 2.0, 28)
    hold_inputs = np.full(24, ramp_inputs[-1])
    input_values = np.concatenate([ramp_inputs, hold_inputs])

    voltage_segments, w_segments, effective_input_segments = [], [], []
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
            weights=weights, dynamics_params={"I": node_input}
        )
        simulation = tvboptim.run_simulation(
            opt_network, t1=segment_duration, dt=dt, run_main=False
        )
        simulation.state.initial_state.dynamics = jnp.asarray(initial_state)
        result = simulation.model_fn(simulation.state)
        data = np.asarray(result.data)
        voltage_segments.append(data[:, 0, :])
        w_segments.append(data[:, 1, :])
        effective_input_segments.append(np.broadcast_to(node_input, data[:, 0, :].shape))
        initial_state = data[-1]

    return (
        np.concatenate(effective_input_segments, axis=0),
        np.concatenate(voltage_segments, axis=0),
        np.concatenate(w_segments, axis=0),
        relative_strength,
        dt,
        len(ramp_inputs) * int(segment_duration / dt),
    )


def _precompute_phase_plane_orbits(mean_I_eff_vals, n_trials=25, duration=20.0, step_size=0.1):
    """Precompute fine-dt G2D phase plane orbits via SimulationExperiment explorations.

    Uses n_trials random initial conditions sampled from state variable distributions
    (Uniform over the SV domains).  Returns (I_keys, V_arr, W_arr) where V_arr and
    W_arr have shape (n_I, n_trials, n_time).
    """
    I_keys = np.linspace(float(mean_I_eff_vals.min()), float(mean_I_eff_vals.max()), 14).tolist()

    dyn = Dynamics.from_db("Generic2dOscillator")
    dyn.parameters["d"].value = 0.35
    dyn.state_variables["W"].domain.lo = -60.0
    dyn.state_variables["W"].domain.hi = 20.0
    # Attach uniform distributions so the template samples random ICs per trial
    dyn.state_variables["V"].distribution = Distribution(name="Uniform")          # uses V domain [-2, 4]
    dyn.state_variables["W"].distribution = Distribution(                          # expanded W domain
        name="Uniform", domain=Range(lo=-60.0, hi=20.0)
    )

    exp = SimulationExperiment(dynamics=dyn, network=Network(number_of_nodes=1))
    exp.integration.duration = duration
    exp.integration.step_size = step_size

    # One exploration: I sweep × n_trials random ICs.  No observable → full state
    # data returned with shape (n_time, n_states, n_nodes) = (n_time, 2, 1).
    # After vmap over ICs and stacking over I values the result is
    # (n_I, n_trials, n_time, 2, 1).
    exp.explorations["phase_plane"] = _Exploration(
        name="phase_plane",
        space=[{"parameter": "I", "explored_values": I_keys}],
        n_trials=n_trials,
    )

    result = exp.run("tvboptim")
    data = np.asarray(result.explorations["phase_plane"].results)
    # shape: (n_I, n_trials, n_time, n_states, n_nodes)
    V_arr = data[:, :, :, 0, 0]   # (n_I, n_trials, n_time) — V state variable
    W_arr = data[:, :, :, 1, 0]   # (n_I, n_trials, n_time) — W state variable

    return np.array(I_keys), V_arr, W_arr


# ---------------------------------------------------------------------------
# Run continuation + trajectory
# ---------------------------------------------------------------------------
cont = Continuation.from_string(G2D_CONT)
exp = SimulationExperiment(
    dynamics=Dynamics.from_ontology("Generic2dOscillator"), continuations=[cont]
)
continuation = exp.run(BACKEND).continuations["g2d_in_I"]
effective_input, voltage, w_data, relative_strength, dt, ramp_end = (
    _g2d_tvboptim_network_trajectory()
)
# Precompute fine-dt phase plane orbits (default step_size ≈ 0.1, not the coarse 0.25
# used by the network simulation). The I_keys span the mean effective-input range.
phase_I_keys, phase_V_arr, phase_W_arr = _precompute_phase_plane_orbits(
    effective_input.mean(axis=1)
)

# ---------------------------------------------------------------------------
# Network graph
# ---------------------------------------------------------------------------
network = Network.from_db("DesikanKilliany")
centers = network.get_centers()
n_nodes = voltage.shape[1]
weights = np.asarray(network.weights, dtype=float)
centers_dict = {i: tuple(map(float, centers[i])) for i in range(n_nodes)}
G = create_network(
    centers_dict, weights, labels=list(range(n_nodes)),
    threshold_percentile=92, directed=False,
)
for u, v_, d in G.edges(data=True):
    d["weight"] = float(np.log1p(d["weight"]))

node_xy = np.array([centers_dict[i][:2] for i in range(n_nodes)])
degree = np.array([G.degree(i) for i in range(n_nodes)], dtype=float)
deg_norm = Normalize(vmin=float(degree.min()), vmax=float(degree.max()))
cmap_deg = plt.colormaps["plasma"]

v_min = float(np.percentile(voltage, 1))
v_max = float(np.percentile(voltage, 99))
v_norm = Normalize(vmin=v_min, vmax=v_max)
cmap_v = plt.colormaps["viridis"]

w_min = float(np.percentile(w_data, 1))
w_max = float(np.percentile(w_data, 99))

base_size = 30 + 100 * relative_strength

filled_colors = cmap_deg(deg_norm(degree)).copy()
hollow_colors = filled_colors.copy()
hollow_colors[:, 3] = 0.0
edge_colors = filled_colors.copy()
edge_colors[:, 3] = 1.0

# ---------------------------------------------------------------------------
# Figure layout: degree-cbar | bifurcation | phase-plane | brain | V-cbar
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16.5, 4.8), facecolor="white")
gs = fig.add_gridspec(
    1, 5,
    width_ratios=[0.025, 1.0, 1.0, 1.0, 0.025],
    wspace=0.12,
    left=0.05, right=0.97, top=0.93, bottom=0.12,
)
cax_deg = fig.add_subplot(gs[0, 0])
ax_map = fig.add_subplot(gs[0, 1])
ax_phase = fig.add_subplot(gs[0, 2])
ax_brain = fig.add_subplot(gs[0, 3])
cax_v = fig.add_subplot(gs[0, 4])

# Brain panel
plot_network_on_surface(
    G, ax=ax_brain, template="fsLR", density="32k",
    hemi=("lh", "rh"), view="top", surface_alpha=0.25,
    show_nodes=False, edge_radius=0.25,
    edge_cmap="viridis", edge_data_key="weight",
    edge_scale={"weight": 4, "mode": "log"},
)
ax_brain.set_title("Desikan-Killiany cortex", fontsize=11)
ax_brain.set_axis_off()
node_scat = ax_brain.scatter(
    node_xy[:, 0], node_xy[:, 1], s=base_size,
    c=np.zeros(n_nodes), cmap=cmap_v, norm=v_norm,
    edgecolor="white", linewidth=0.4, zorder=20,
)

# Bifurcation panel
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
tail_lc = LineCollection([], colors=tail_colors, linewidths=0.7, alpha=0.45, zorder=4)
ax_map.add_collection(tail_lc)
node_points = ax_map.scatter(
    np.zeros(n_nodes), np.zeros(n_nodes), s=22,
    c=degree, cmap=cmap_deg, norm=deg_norm,
    edgecolors=edge_colors, linewidth=0.5, zorder=6, alpha=0.95,
)

# Phase plane panel
ax_phase.set_title("phase plane", fontsize=11)
ax_phase.set_xlabel("$V$")
ax_phase.set_ylabel("$W$")
# Limits driven by the SV domains used in precomputation
ax_phase.set_xlim(-2.5, 4.5)
ax_phase.set_ylim(-65, 25)
# Background: fine-dt precomputed orbits — neutral grey, updated when I changes
phase_bg_lc = LineCollection([], colors="0.40", linewidths=0.55, alpha=0.35, zorder=4)
ax_phase.add_collection(phase_bg_lc)
# Foreground: current network node positions in V-W space
node_phase = ax_phase.scatter(
    np.zeros(n_nodes), np.zeros(n_nodes), s=18,
    c=degree, cmap=cmap_deg, norm=deg_norm,
    edgecolors=edge_colors, linewidth=0.5, zorder=6, alpha=0.95,
)

time_text = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=10, color="0.15")

cbar_v = fig.colorbar(ScalarMappable(norm=v_norm, cmap=cmap_v), cax=cax_v)
cbar_v.set_label("$V(t)$  (activity)")
cbar_deg = fig.colorbar(ScalarMappable(norm=deg_norm, cmap=cmap_deg), cax=cax_deg)
cbar_deg.set_label("node degree")
cax_deg.yaxis.set_ticks_position("left")
cax_deg.yaxis.set_label_position("left")

# Frame indices: 80% on the ramp for slower parameter traversal
n_frames = 108
frames = np.unique(
    np.concatenate([
        np.linspace(8, ramp_end, int(0.80 * n_frames)),
        np.linspace(ramp_end + 1, len(voltage) - 1, n_frames - int(0.80 * n_frames)),
    ]).astype(int)
)

lc_threshold = 0.5


def update(frame_index):
    v_now = voltage[frame_index]
    w_now = w_data[frame_index]
    node_scat.set_array(v_now)
    node_points.set_offsets(np.column_stack([effective_input[frame_index], v_now]))
    start = max(0, frame_index - tail_steps)

    tail_lc.set_segments([
        np.column_stack([effective_input[start:frame_index + 1, i],
                         voltage[start:frame_index + 1, i]])
        for i in range(n_nodes)
    ])

    # Phase plane background: pick the precomputed orbits closest to current mean I_eff
    mean_I = float(effective_input[frame_index].mean())
    i_idx = int(np.argmin(np.abs(phase_I_keys - mean_I)))
    segs_V = phase_V_arr[i_idx]   # (n_trials, n_time)
    segs_W = phase_W_arr[i_idx]   # (n_trials, n_time)
    phase_bg_lc.set_segments([
        np.column_stack([segs_V[t], segs_W[t]])
        for t in range(segs_V.shape[0])
    ])

    # Node positions in V-W space (current time step from network simulation)
    node_phase.set_offsets(np.column_stack([v_now, w_now]))

    amp = voltage[start:frame_index + 1].max(axis=0) - voltage[start:frame_index + 1].min(axis=0)
    is_lc = amp > lc_threshold
    fc = np.where(is_lc[:, None], hollow_colors, filled_colors)
    node_points.set_facecolors(fc)
    node_phase.set_facecolors(fc)

    time_text.set_text(
        rf"TVBO + tvboptim — Desikan-Killiany — $t={frame_index * dt:.0f}$ ms,  "
        rf"$\langle I_{{\mathrm{{eff}}}}\rangle={mean_I:+.2f}$"
    )


# Render with shared palette to avoid per-frame colorbar shimmer
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
quantized = [f.quantize(palette=pal_img, dither=Image.Dither.NONE) for f in rendered]
quantized[0].save(
    OUT, save_all=True, append_images=quantized[1:],
    duration=int(1000 / 10), loop=0, disposal=2, optimize=False,
)
plt.close(fig)
print("wrote", OUT)
