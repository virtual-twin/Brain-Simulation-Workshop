from nibabel.freesurfer.io import read_annot
import matplotlib.pyplot as plt
from bsplot import data, style
from bsplot.data.surface import get_surface_geometry
from bsplot.graph import (
    create_network,
    get_centers_from_surface_parc,
    plot_network_on_surface,
)
import numpy as np
import bsplot
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np

from importlib import reload
import bsplot.surface
from bsplot import surface as surface
from bsplot.data import hcp_mmp1_fslr_lh
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# %%

# Load surface geometry and parcellation
vertices_lh, faces_lh = get_surface_geometry(
    template="fsaverage", hemi="lh", density="164k"
)
vertices_rh, faces_rh = get_surface_geometry(
    template="fsaverage", hemi="rh", density="164k"
)

labels_lh, colors_lh, names_lh = read_annot(
    "/Applications/freesurfer/7.4.1/subjects/fsaverage/label/lh.aparc.annot"
)
labels_rh, colors_rh, names_rh = read_annot(
    "/Applications/freesurfer/7.4.1/subjects/fsaverage/label/rh.aparc.annot"
)
centers_lh = get_centers_from_surface_parc(vertices_lh, labels_lh)
centers_rh = get_centers_from_surface_parc(vertices_rh, labels_rh)

centers_all = {**centers_lh, **centers_rh}


sc = np.load(
    "/Users/leonmartin_bih/work_data/toolboxes/tvboptim/src/tvboptim/data/connectivity/dk_average/data.npz"
)


# === TVB abbreviated labels → FreeSurfer aparc names ===
sc_abbrev_to_aparc = {
    "BSTS": "bankssts",
    "CACG": "caudalanteriorcingulate",
    "CMFG": "caudalmiddlefrontal",
    "CU": "cuneus",
    "EC": "entorhinal",
    "FG": "fusiform",
    "IPG": "inferiorparietal",
    "ITG": "inferiortemporal",
    "ICG": "isthmuscingulate",
    "LOG": "lateraloccipital",
    "LOFG": "lateralorbitofrontal",
    "LG": "lingual",
    "MOFG": "medialorbitofrontal",
    "MTG": "middletemporal",
    "PHIG": "parahippocampal",
    "PaCG": "paracentral",
    "POP": "parsopercularis",
    "POR": "parsorbitalis",
    "PTR": "parstriangularis",
    "PCAL": "pericalcarine",
    "PoCG": "postcentral",
    "PCG": "posteriorcingulate",
    "PrCG": "precentral",
    "PCU": "precuneus",
    "RACG": "rostralanteriorcingulate",
    "RMFG": "rostralmiddlefrontal",
    "SFG": "superiorfrontal",
    "SPG": "superiorparietal",
    "STG": "superiortemporal",
    "SMG": "supramarginal",
    "FP": "frontalpole",
    "TP": "temporalpole",
    "TTG": "transversetemporal",
    "IN": "insula",
}
subcortical_abbrevs = {"CER", "TH", "CA", "PU", "PA", "HI", "AM", "AC"}

# Build aparc name → label index lookup
name_to_idx = {
    (n.decode() if isinstance(n, bytes) else str(n)): idx
    for idx, n in enumerate(names_lh)
}

# Parse SC labels, get cortical entries with their aparc label index
parsed = [str(l).split(".") for l in sc["region_labels"]]
cortical_entries = [
    (i, hemi, sc_abbrev_to_aparc[abbrev], name_to_idx[sc_abbrev_to_aparc[abbrev]])
    for i, (hemi, abbrev) in enumerate(parsed)
    if abbrev not in subcortical_abbrevs
]

# Sort: LH first (by aparc label index), then RH (by aparc label index)
cortical_entries.sort(key=lambda x: (x[1] != "L", x[3]))
reorder_idx = np.array([e[0] for e in cortical_entries])

# Extract reordered cortical-only SC submatrix
sc_cortical = sc["weights"][np.ix_(reorder_idx, reorder_idx)]

# Build centers keyed by sequential index 0..67 (matching sc_cortical rows/cols)
# Labels are also 0..67 so create_network can find centers[label] correctly
centers_src = {"L": centers_lh, "R": centers_rh}
centers_matched = {
    i: centers_src[hemi][aparc_idx]
    for i, (_, hemi, _, aparc_idx) in enumerate(cortical_entries)
}
labels_matched = list(range(len(cortical_entries)))  # 0..67, same as centers keys

print(
    f"Cortical SC: {sc_cortical.shape}, centers: {len(centers_matched)}, labels: {len(labels_matched)}"
)
n_lh = sum(1 for e in cortical_entries if e[1] == "L")
print(
    f"LH: {n_lh} (indices 0..{n_lh-1}), RH: {len(cortical_entries)-n_lh} (indices {n_lh}..{len(cortical_entries)-1})"
)

# %%

G = create_network(
    centers_matched,
    sc_cortical,
    labels=labels_matched,
    threshold_percentile=85,
    directed=False,
)

for node in G.nodes():
    G.nodes[node]["strength"] = sum(d["weight"] for _, _, d in G.edges(node, data=True))

node_to_aparc = {
    i: (hemi, aparc_idx) for i, (_, hemi, _, aparc_idx) in enumerate(cortical_entries)
}

overlay_lh = np.full(len(vertices_lh), np.nan)
overlay_rh = np.full(len(vertices_rh), np.nan)

for node_id in G.nodes():
    hemi, aparc_idx = node_to_aparc[node_id]
    strength = G.nodes[node_id]["strength"]
    if hemi == "L":
        overlay_lh[labels_lh == aparc_idx] = strength
    else:
        overlay_rh[labels_rh == aparc_idx] = strength

# %%

view = "horizontal"

fig, ax = plt.subplots(figsize=(5, 5))
bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax,
    view=view,
    slice_mm=10,
    cmap="gray",
    zorder=-1,
)
ax.axis("off")
fig.suptitle("MRI", fontsize=30)
fig.savefig(
    os.path.join(ROOT, "..", "img", "figures", "basic_idea_imaging_workflow1.png"),
    dpi=300,
    bbox_inches="tight",
)


# %%

fig, ax = plt.subplots(figsize=(5, 5), layout="compressed")

tractogram_path = (
    "/Users/leonmartin_bih/tools/bsplot/docs/data/dTOR_10K_sample_subsampled_10k.tck"
)

bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax,
    view=view,
    slice_mm=10,
    cmap="gray",
    zorder=1,
)


bsplot.streamlines.plot_tractogram(
    tractogram_path,
    view=view,
    alpha=0.7,
    linewidth=0.1,
    subsample=10_000,
    zorder=2,
    ax=ax,
)

ax.axis("off")
fig.suptitle(
    "Tractography",
    fontsize=30,
)

fig.savefig(
    os.path.join(ROOT, "..", "img", "figures", "basic_idea_imaging_workflow2.png"),
    dpi=300,
    bbox_inches="tight",
)


# %%

fig, ax = plt.subplots(figsize=(5, 5), layout="compressed")


bsplot.streamlines.plot_tractogram(
    tractogram_path,
    view=view,
    alpha=0.7,
    linewidth=0.1,
    subsample=10_000,
    zorder=1,
    ax=ax,
)

surface.plot_surf(
    view="top",
    surface="fsaverage",
    surface_density="164k",
    overlay=overlay_lh,
    ax=ax,
    hemi="lh",
    cmap="vik",
    parcellated=True,
    zorder=2,
)

bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax,
    view=view,
    slice_mm=10,
    cmap="gray",
    zorder=-1,
)

ax.axis("off")
fig.suptitle(
    "Parcellation",
    fontsize=30,
)
fig.savefig(
    os.path.join(ROOT, "..", "img", "figures", "basic_idea_imaging_workflow3.png"),
    dpi=300,
    bbox_inches="tight",
)


fig, ax = plt.subplots(figsize=(5, 5), layout="compressed")

view = "top"
_, _, mappables = plot_network_on_surface(
    G,
    ax=ax,
    template="fsaverage",
    density="164k",
    hemi="lh",
    view=view,
    surface_alpha=1,
    node_radius=2,
    node_color="auto",
    node_data_key="strength",
    node_cmap="vik",
    edge_radius=0.1,
    edge_cmap="vik",
    edge_data_key="weight",
    edge_scale={"weight": 0, "mode": "exp"},
    node_scale={"strength": 2, "mode": "exp"},
    overlay=overlay_lh,
    cmap="vik",
    threshold=0,
    parcellated=True,
)
bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=ax,
    view="horizontal",
    slice_mm=10,
    cmap="gray",
    zorder=-1,
)
ax.axis("off")
fig.suptitle(
    "Network",
    fontsize=30,
)
fig.savefig(
    os.path.join(ROOT, "..", "img", "figures", "basic_idea_imaging_workflow4.png"),
    dpi=300,
    bbox_inches="tight",
)
