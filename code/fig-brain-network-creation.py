import bsplot
import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
from bsplot.surface import plot_surf
from tvbo import Network
import numpy as np
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

view='lateral'

fig, axs = plt.subplots(nrows=4, layout="compressed")

tractogram_path = (
    "/Users/leonmartin_bih/tools/bsplot/docs/data/dTOR_10K_sample_subsampled_10k.tck"
)

bsplot.plot_slice(
    bsplot.templates.bigbrain,
    ax=axs[0],
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
    ax=axs[0],
    cmap="viridis",
)

axs[0].axis("off")


annot2 = "/Users/leonmartin_bih/projects/TVB-O/tvb-ontology-optim-workshop/data/lh.aparc.annot"
# Read FreeSurfer annot: labels (per-vertex ids), colortable, names
labels, ctab, names = nib.freesurfer.io.read_annot(annot2)
cmap = ListedColormap(ctab[:, 0:3] / 255.0)  # use RGB from ctab, ignore alpha
bsplot.surface.plot_surf(
    surface="fsaverage",
    overlay=labels,
    surface_kwargs={"surface_density": "164k"},
    hemi="rh",
    view="lateral",
    parcellated=True,
    cmap=cmap,
    surface_suffix="pial",
    ax=axs[1],
)
norm = plt.Normalize(vmin=0, vmax=len(names))

axs[1].axis("off")


net = Network.from_db(atlas='DesikanKilliany', rec='avgMatrix')

net.plot_brain_surface(hemi='rh', ax=axs[2], node_cmap='viridis',edge_cmap='viridis', surface_alpha=0.1)

axs[3].imshow(np.log1p(net.matrix('weight')), cmap='viridis', interpolation='none')
axs[3].set_xlabel('Node $j$')
axs[3].set_ylabel('Node $i$')

axs[3].set_xticks([0, len(net.nodes)])
axs[3].set_yticks([0, len(net.nodes)])

axs[0].set_title('Tractogram')
axs[1].set_title('Parcellation')
axs[2].set_title('Brain Network')
axs[3].set_title('Connectivity Matrix $A_{ij}$')
plt.show()
fig.savefig(os.path.join(ROOT_DIR, 'img', "fig-network_visualization_rows.png"), dpi=300, bbox_inches="tight")