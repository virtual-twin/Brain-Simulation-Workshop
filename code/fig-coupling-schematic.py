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

BOX_C  = "#d6e4f0"
WARM_C = "#f0dfd6"
DARK_C = "#1a1a2e"
WHITE  = "#ffffff"

# ── Load SC + node data ────────────────────────────────────────────────────
sc    = Network.from_db(atlas="DesikanKilliany", rec="dTOR")
W     = np.log1p(sc.matrix("weight"))
nodes = sc.nodes   # list of Node objects; index = matrix_index

# ── Build ctab color map (one colour per unique region, ignoring hemisphere) ─
node_tsv  = pd.read_csv(_NODE_TSV, sep="\t")
regions   = node_tsv["label"].str.replace(r"^[LR]\.", "", regex=True)
unique_rg = sorted(regions.unique())
cmap_ctab = plt.colormaps.get_cmap("gist_rainbow").resampled(len(unique_rg))
rg_to_color = {rg: cmap_ctab(i) for i, rg in enumerate(unique_rg)}

# ── Select 8 right-hemisphere cortical nodes ──────────────────────────────
_SUBCORT = {"CER", "TH", "CA", "PU", "PA", "AM", "HI", "AC"}
rh_rows = node_tsv[
    node_tsv["label"].str.startswith("R.") &
    ~node_tsv["label"].str.replace(r"^R\.", "", regex=True).isin(_SUBCORT)
].copy()

rng = np.random.default_rng(42)
sel_rows = rh_rows.sample(8, random_state=42).sort_values("matrix_index")
sel_matrix_idx = sel_rows["matrix_index"].tolist()
sel_labels     = sel_rows["label"].str.replace(r"^[LR]\.", "", regex=True).tolist()
sel_colors     = [rg_to_color[lbl] for lbl in sel_labels]

# Sagittal projection (y, z) for the 8 selected nodes
sel_pos_2d = np.array(
    [[nodes[mi].position.y, nodes[mi].position.z] for mi in sel_matrix_idx]
)

# SC sub-matrix for the 8 nodes
sub_W = W[np.ix_(sel_matrix_idx, sel_matrix_idx)]
np.fill_diagonal(sub_W, 0)

# ── Layout ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 3.5))
gs  = fig.add_gridspec(
    1, 2, width_ratios=[3.5, 4.5],
    left=0.01, right=0.99, top=0.88, bottom=0.06, wspace=0.06,
)
ax_brain = fig.add_subplot(gs[0])
ax_coup  = fig.add_subplot(gs[1])


# ── Panel 1: BigBrain slice + Tractogram + 8 selected nodes ──────────────
# (a) BigBrain background
bsplot.plot_slice(templates.bigbrain, ax=ax_brain, view="sagittal",
                  slice_mm=0, alpha=0.85)

# (b) Tractogram on top
bsplot.streamlines.plot_tractogram(
    _TCK, ax=ax_brain, view="sagittal",
    subsample=1500, alpha=0.35, linewidth=0.10, cmap="viridis",
)

# (c) Only the 8 selected right-hemisphere nodes
ax_brain.scatter(
    sel_pos_2d[:, 0], sel_pos_2d[:, 1],
    c=sel_colors, s=55, zorder=5,
    linewidths=0.5, edgecolors="white", alpha=0.95,
)

ax_brain.axis("off")
ax_brain.set_title("Parcellation + Tractogram", fontsize=9, color=DARK_C, pad=3)

# ── Build abstract network from the 8 selected nodes ─────────────────────
n_nodes = 8
G = nx.DiGraph()
G.add_nodes_from(range(n_nodes))
for i in range(n_nodes):
    for j in range(n_nodes):
        if sub_W[i, j] > np.percentile(sub_W[sub_W > 0], 55):
            G.add_edge(j, i, weight=float(sub_W[i, j]))

TARGET      = 0
TARGET_C    = sel_colors[TARGET]
node_colors = sel_colors
node_sizes  = [220 if k == TARGET else 100 for k in range(n_nodes)]
edge_colors = [TARGET_C if v == TARGET else "#cccccc" for u, v in G.edges()]
widths      = [(G[u][v]["weight"] * 0.7 if v == TARGET else 0.3) for u, v in G.edges()]

# ── Draw abstract network overlay ON the brain axes ───────────────────────
# Use real (y, z) positions of the 8 selected nodes
real_pos = {k: (nodes[sel_matrix_idx[k]].position.y, nodes[sel_matrix_idx[k]].position.z)
            for k in range(n_nodes)}

nx.draw_networkx_nodes(G, real_pos, ax=ax_brain, node_color=node_colors,
                       node_size=node_sizes)
ax_brain.collections[-1].set_zorder(10)
nx.draw_networkx_edges(G, real_pos, ax=ax_brain, edge_color=edge_colors,
                       width=widths, arrows=True, arrowsize=8,
                       connectionstyle="arc3,rad=0.1")

# Label node i and j neighbours
ix, iy = real_pos[TARGET]
ax_brain.text(ix, iy + 4, "$i$", ha="center", va="bottom",
              fontsize=9, color="white", fontweight="bold", zorder=11)
neighbours_in = [u for u, v in G.in_edges(TARGET)][:2]
for nb in neighbours_in:
    nbx, nby = real_pos[nb]
    ax_brain.text(nbx, nby + 4, "$j$", ha="center", va="bottom",
                  fontsize=8, color="white", style="italic", zorder=11)


# ── Panel 2: Coupling computation diagram ─────────────────────────────────
ax_coup.set_xlim(0, 10)
ax_coup.set_ylim(0, 6)
ax_coup.axis("off")
ax_coup.set_title(r"Coupling $C_i$", fontsize=9, color=DARK_C, pad=3)

# Pick j-node colors from sel_colors (first 3 that are neighbours of TARGET)
j_nodes  = [u for u, v in G.in_edges(TARGET)]
j_colors = [sel_colors[j] for j in j_nodes[:3]] + [BOX_C] * max(0, 3 - len(j_nodes))

# Box style helpers
def _circ(ax, xy, label, fc=BOX_C, ec=DARK_C, fs=8, fc_txt=DARK_C):
    ax.annotate(
        label, xy=xy, ha="center", va="center", fontsize=fs,
        bbox=dict(boxstyle="circle,pad=0.25", fc=fc, ec=ec, lw=1.2),
        color=fc_txt,
    )

def _box(ax, xy, label, fc="#6880af", ec="#3a5080", fs=7.5, fc_txt=WHITE):
    ax.annotate(
        label, xy=xy, ha="center", va="center", fontsize=fs,
        bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.2),
        color=fc_txt,
    )

def _arr(ax, xy_from, xy_to, color="#555555", lw=0.9, rad=0.0, label=None, label_off=(0, 0.18)):
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
        ax.text(mx, my, label, fontsize=7, color="#555555",
                ha="center", va="bottom", style="italic")

# Source nodes j1, j2, j3
src_ys = [4.8, 3.0, 1.2]
src_x  = 0.9
pre_x  = 2.85
sum_x  = 5.2
post_x = 7.2
out_x  = 9.4
mid_y  = 3.0

# θ^c parameter arrow (drops in from top into f_c^post)
ax_coup.annotate(
    r"$\theta^c$",
    xy=(post_x, mid_y + 0.6), xytext=(post_x, mid_y + 1.5),
    ha="center", va="center", fontsize=8, color="#af8068",
    arrowprops=dict(arrowstyle="->", color="#af8068", lw=0.9),
)

for yi, lbl, jc in zip(src_ys, [r"$j_1$", r"$j_2$", r"$j_3$"], j_colors):
    _circ(ax_coup, (src_x, yi), lbl, fc=jc, ec=DARK_C, fc_txt=WHITE)
    _box(ax_coup, (pre_x, yi), r"$f_c^{\rm pre}$")
    _arr(ax_coup, (src_x + 0.28, yi), (pre_x - 0.45, yi))
    if yi == src_ys[0]:
        _arr(ax_coup, (pre_x + 0.42, yi), (sum_x - 0.5, mid_y),
             rad=-0.25, label=r"$A_{ij}$", label_off=(0.1, 0.12))
    elif yi == src_ys[2]:
        _arr(ax_coup, (pre_x + 0.42, yi), (sum_x - 0.5, mid_y),
             rad=0.25, label=r"$\tau_{ij}$", label_off=(0.1, -0.30))
    else:
        _arr(ax_coup, (pre_x + 0.42, yi), (sum_x - 0.5, mid_y))

# Sum box
_box(ax_coup, (sum_x, mid_y), r"$\Sigma_j$",
     fc=WARM_C, ec="#af8068", fc_txt=DARK_C)
_arr(ax_coup, (sum_x + 0.48, mid_y), (post_x - 0.48, mid_y))

# f_c^post
_box(ax_coup, (post_x, mid_y), r"$f_c^{\rm post}$")
_arr(ax_coup, (post_x + 0.52, mid_y), (out_x - 0.38, mid_y))

# C_i output (uses ctab color of target node)
import matplotlib.colors as mcolors
tc_hex = mcolors.to_hex(TARGET_C)
tc_dark = mcolors.to_hex([max(0, c - 0.25) for c in TARGET_C[:3]])
ax_coup.annotate(
    r"$C_i$", xy=(out_x, mid_y), ha="center", va="center",
    fontsize=10, fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.4", fc=tc_hex, ec=tc_dark, lw=2.0),
    color=WHITE,
)


# ── Save ───────────────────────────────────────────────────────────────────
fig.savefig(OUT, dpi=300, bbox_inches="tight", transparent=True)
print(f"Saved → {OUT}")
