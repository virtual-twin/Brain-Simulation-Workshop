from tvbo import Network
import os
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageOps
import bsplot  # noqa: F401
from tvbo import Coupling, Dynamics, Network, SimulationExperiment
from tvbo.classes.continuation import Continuation

bsplot.style.use("tvbo")
ROOT = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.abspath(os.path.join(ROOT, "..", "img"))

dk = Network.from_db(
  atlas="DesikanKilliany",
  rec="avgMatrix",
  )

fig = dk.plot_overview(log_weights=True)

out = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))
fig.savefig(out, dpi=500, bbox_inches="tight")