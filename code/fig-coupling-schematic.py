"""Coupling-function + network-creation schematic.

Two-panel horizontal figure (10 × 3.5 inches):
  [BigBrain + Tractogram + Parcellation nodes overlay] | [Coupling diagram]

Saves: img/coupling_schematic.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import networkx as nx

import bsplot
from bsplot import templates
from tvbo import Network

bsplot.style.use("tvbo")

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(ROOT, "..", "img", "coupling_schematic.png"))

_TCK  = "/Users/leonmartin_bih/tools/bsplot/docs/data/dTOR_10K_sample_subsampled_10k.tck"
_NODE_TSV = "/Users/leonmartin_bih/tools/tvbo/tvbo/database/networks/bids/dk_average/atlas-DesikanKilliany_nodeindices.tsv"

NODE_C = "#6880af"   # target node i
NBR_C  = "#aac4e0"   # neighbour nodes j
BOX_C  = "#d6e4f0"
WARM_C = "#f0dfd6"
DARK_C = "#1a1a2e"
WHITE  = "#ffffff"

# ── Load SC + node data ────────────────────────────────────────────────────
sc    = Network.from_db(atlas="DesikanKilliany", rec="dTOR")
W     = np.log1p(sc.matrix("weight"))
nodes = sc.nodes   # list of Node objects; index = matrix_index

# ── Select 8 right-hemisphere cortical nodes ──────────────────────────────
node_tsv = pd.read_csv(_NODE_TSV, sep="\t")
_SUBCORT = {"CER", "TH", "CA", "PU", "PA", "AM", "HI", "AC"}
rh_rows = node_tsv[
    node_tsv["label"].str.startswith("R.") &
    ~node_tsv["label"].str.replace(r"^R\.", "", regex=True).isin(_SUBCORT)
].copy()

# Greedy furthest-point sampling: pick 8 maximally spread-out nodes
_all_idx  = rh_rows["matrix_index"].tolist()
_all_pos  = np.array([[nodes[mi].position.y, nodes[mi].position.z] for mi in _all_idx])
_picked   = [0]  # start from first node
for _ in range(7):
    dists = np.min(
        np.linalg.norm(_all_pos[:, None] - _all_pos[np.array(_picked)][None], axis=-1),
        axis=1,
    )
    dists[_picked] = -1
    _picked.append(int(np.argmax(dists)))
sel_matrix_idx = sorted([_all_idx[p] for p in _picked])

# Sagittal projection (y, z) for the 8 selected nodes
sel_pos_2d = np.array(
    [[nodes[mi].position.y, nodes[mi].position.z] for mi in sel_matrix_idx]
)

# SC sub-matrix for the 8 nodes
sub_W = W[np.ix_(sel_matrix_idx, sel_matrix_idx)]
np.fill_diagonal(sub_W, 0)

# ── Layout ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 3), layout='compressed')
gs  = fig.add_gridspec(
    1, 2, width_ratios=[3.5, 4.5],
    left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.06,
)
ax_brain = fig.add_subplot(gs[0])
ax_coup  = fig.add_subplot(gs[1])


# ── Panel 1: BigBrain slice + Tractogram + 8 selected nodes ──────────────
# (a) Tractogram first (lower zorder)
bsplot.streamlines.plot_tractogram(
    _TCK, ax=ax_brain, view="lateral",
    subsample=10_000, alpha=0.35, linewidth=0.10, cmap="viridis",
)

# (b) BigBrain behind tractogram


n_nodes = 8
TARGET  = 0

ax_brain.axis("off")
ax_brain.set_title("Parcellation + Tractogram", fontsize=9, color=DARK_C, pad=3)

# ── Build abstract network from the 8 selected nodes ─────────────────────
G = nx.DiGraph()
G.add_nodes_from(range(n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        if sub_W[i, j] > np.percentile(sub_W[sub_W > 0], 55):
            G.add_edge(j, i, weight=float(sub_W[i, j]))

# Guarantee at least 3 strong edges into TARGET
_incoming = sorted(
    [(k, float(sub_W[k, TARGET])) for k in range(n_nodes) if k != TARGET],
    key=lambda x: -x[1],
)[:3]
for src, w in _incoming:
    if not G.has_edge(src, TARGET):
        G.add_edge(src, TARGET, weight=w)

TARGET      = 0
node_colors = [NODE_C if k == TARGET else NBR_C for k in range(n_nodes)]
node_sizes  = [400 if k == TARGET else 200 for k in range(n_nodes)]

ACCENT_C = "#c85030"   # warm red for strong incoming arrows

# Normalise edge weights for display; guaranteed-strong edges get min width 2.5
w_vals = np.array([G[u][v]["weight"] for u, v in G.edges()])
w_max  = w_vals.max() if w_vals.size else 1.0
edge_colors = [ACCENT_C if v == TARGET else "#bbbbbb" for u, v in G.edges()]
widths = [
    max(2.5, G[u][v]["weight"] / w_max * 5.0) if v == TARGET else 0.6
    for u, v in G.edges()
]

# ── Draw abstract network overlay ON the brain axes ───────────────────────
real_pos = {k: (nodes[sel_matrix_idx[k]].position.y, nodes[sel_matrix_idx[k]].position.z)
            for k in range(n_nodes)}

# Separate strong (→TARGET) from weak edges
strong_edges = [(u, v) for u, v in G.edges() if v == TARGET]
weak_edges   = [(u, v) for u, v in G.edges() if v != TARGET]
strong_widths = [max(2.5, G[u][v]["weight"] / w_max * 5.0) for u, v in strong_edges]

# Weak edges first
nx.draw_networkx_edges(G, real_pos, ax=ax_brain, edgelist=weak_edges,
                       edge_color="#bbbbbb", width=0.6, arrows=True, arrowsize=8,
                       connectionstyle="arc3,rad=0.1")

# Strong edges: single accent pass
nx.draw_networkx_edges(G, real_pos, ax=ax_brain, edgelist=strong_edges,
                       edge_color=ACCENT_C,
                       width=strong_widths,
                       arrows=True, arrowsize=12,
                       connectionstyle="arc3,rad=0.1")

for patch in ax_brain.patches:
    patch.set_zorder(8)

# Draw nodes on top (zorder 10)
nx.draw_networkx_nodes(G, real_pos, ax=ax_brain, node_color=node_colors,
                       node_size=node_sizes, linewidths=1.4,
                       edgecolors=DARK_C)
ax_brain.collections[-1].set_zorder(10)

# Labels centred inside nodes
ix, iy = real_pos[TARGET]
ax_brain.text(ix, iy, "$i$", ha="center", va="center",
              fontsize=8, color="white", fontweight="bold", zorder=12)
for nb in [u for u, v in G.in_edges(TARGET)][:3]:
    nbx, nby = real_pos[nb]
    ax_brain.text(nbx, nby, "$j$", ha="center", va="center",
                  fontsize=7, color="white", zorder=12)

bsplot.plot_slice(templates.bigbrain, ax=ax_brain, view="lateral",
                  slice_mm=-10, alpha=0.85, zorder=-1)
xlow, xhigh = ax_brain.get_xlim()
ax_brain.set_xlim(xlow, xhigh + 10)

# ── Panel 2: Coupling computation diagram ─────────────────────────────────
ax_coup.set_xlim(0, 10)
ax_coup.set_ylim(0, 6)
ax_coup.axis("off")
ax_coup.set_title(r"Coupling $C_i$", fontsize=9, color=DARK_C, pad=3)

def _circ(ax, xy, label, fc=BOX_C, ec=DARK_C, fs=11, fc_txt=DARK_C):
    ax.annotate(
        label, xy=xy, ha="center", va="center", fontsize=fs,
        bbox=dict(boxstyle="circle,pad=0.40", fc=fc, ec=ec, lw=1.4),
        color=fc_txt,
    )

def _box(ax, xy, label, fc="#6880af", ec="#3a5080", fs=10, fc_txt=WHITE):
    ax.annotate(
        label, xy=xy, ha="center", va="center", fontsize=fs,
        bbox=dict(boxstyle="round,pad=0.42", fc=fc, ec=ec, lw=1.4),
        color=fc_txt,
    )

def _arr(ax, xy_from, xy_to, color="#555555", lw=0.9, rad=0.0,
         label=None, label_off=(0, 0.22), label_color="#555555"):
    ax.annotate(
        "", xy=xy_to, xytext=xy_from,
        arrowprops=dict(
            arrowstyle="->", color=color, lw=lw,
            connectionstyle=f"arc3,rad={rad}",
        ),
    )
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2 + label_off[0]
        my = (xy_from[1] + xy_to[1]) / 2 + label_off[1]
        ax.text(mx, my, label, fontsize=9.5, color=label_color,
                ha="center", va="bottom", style="italic")

# 4 source nodes — arc curvature encodes delay τ_ij, linewidth encodes weight A_ij
src_ys   = [5.0, 3.9, 2.7, 1.5]
src_x    = 0.9
pre_x    = 2.85
sum_x    = 5.2
post_x   = 7.2
out_x    = 9.4
mid_y    = 3.1
j_labels = [r"$j_1$", r"$j_2$", r"$j_3$", r"$j_4$"]
tau_arcs = [0.0, -0.10, 0.0, 0.28]    # j→pre curvature  ∝ delay τ_ij
w_lws    = [0.6, 1.3, 2.5, 1.0]       # pre→Σ linewidth   ∝ weight A_ij

# θ^c parameter arrow
ax_coup.annotate(
    r"$\theta^c$",
    xy=(post_x, mid_y + 0.75), xytext=(post_x, mid_y + 1.75),
    ha="center", va="center", fontsize=10, color="#af8068",
    arrowprops=dict(arrowstyle="->", color="#af8068", lw=0.9),
)

for idx, (yi, lbl, tau, wlw) in enumerate(
        zip(src_ys, j_labels, tau_arcs, w_lws)):
    _circ(ax_coup, (src_x, yi), lbl, fc=NBR_C, ec=DARK_C, fc_txt=WHITE)
    _box(ax_coup, (pre_x, yi), r"$f_c^{\rm pre}$")

    # j → f_c^pre: curvature encodes delay τ_ij (more curved = longer delay)
    ann_tau = r"$\tau_{ij}$" if idx == 3 else None
    _arr(ax_coup, (src_x + 0.42, yi), (pre_x - 0.58, yi),
         rad=tau, label=ann_tau, label_off=(0, 0.32))

    # f_c^pre → Σ: linewidth encodes weight A_ij (thicker = stronger)
    rad_fan = -0.15 if yi > mid_y else (0.15 if yi < mid_y else 0.0)
    ann_wgt = r"$A_{ij}$" if idx == 2 else None
    # Exactly between j2 (y=3.9) and j3 (y=2.7) → target y=3.3; midpoint y=2.9 → offset=0.4
    wgt_off = (-0.3, 0.4)
    _arr(ax_coup, (pre_x + 0.58, yi), (sum_x - 0.65, mid_y),
         lw=wlw, rad=rad_fan, label=ann_wgt, label_off=wgt_off,
         label_color=ACCENT_C)

# Encoding guide at bottom
ax_coup.text(0.5, 0.02,
    r"line width $\propto$ weight $A_{ij}$   $\cdot$   curvature $\propto$ delay $\tau_{ij}$",
    fontsize=8.5, color="#888888", transform=ax_coup.transAxes,
    ha="center", va="bottom")

# Σ, f_c^post, C_i
_box(ax_coup, (sum_x, mid_y), r"$\Sigma_j$",
     fc=WARM_C, ec="#af8068", fc_txt=DARK_C)
_arr(ax_coup, (sum_x + 0.64, mid_y), (post_x - 0.64, mid_y))
_box(ax_coup, (post_x, mid_y), r"$f_c^{\rm post}$")
_arr(ax_coup, (post_x + 0.68, mid_y), (out_x - 0.52, mid_y))
ax_coup.annotate(
    r"$C_i$", xy=(out_x, mid_y), ha="center", va="center",
    fontsize=13, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.5", fc=NODE_C, ec="#3a5080", lw=2.0),
    color=WHITE,
)

plt.show()
# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved → {OUT}")
