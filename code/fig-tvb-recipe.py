from tvbo import Network, Observation
import matplotlib.pyplot as plt
from PIL import Image
from tvbo import Coupling
import bsplot
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(ROOT, 'img', "fig-tvb-recipe.png")

bsplot.style.use('tvbo')

nmm_schema = os.path.join(ROOT, "img", "figures", "jansen1995", "3PopMeanField.png")

fig, axs = plt.subplots(ncols=4, layout='compressed')
net = Network.from_db(atlas='DesikanKilliany', rec='avgMatrix')
net.plot_brain_surface(ax=axs[0], surface_alpha=0.1)

axs[0].set_title("Connectome")
img = Image.open(nmm_schema)
axs[1].imshow(img)
axs[1].set_axis_off()
axs[1].set_title("Local dynamics")


Coupling.from_db('HyperbolicTangent').plot(ax=axs[2])

axs[2].set_title("Coupling")
axs[2].set_xlabel("")
axs[2].set_ylabel("")
axs[2].set_xticks([])
axs[2].set_yticks([])

Observation.from_db('BOLD_TVB').plot(ax=axs[3])
axs[3].set_title("Observation")
axs[3].set_xlabel("")
axs[3].set_ylabel("")
axs[3].set_xticks([])
axs[3].set_yticks([])

fig.savefig(OUT, dpi=300, bbox_inches='tight')