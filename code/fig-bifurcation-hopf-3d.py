"""Hopf bifurcation: real 3D limit-cycle tube from continuation."""
from __future__ import annotations

import os
from _bif_common import IMG, HOPF, HOPF_CONT, _bif, save

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

cont = _bif(HOPF, HOPF_CONT)
out = cont.plot_3d(VOI="x1")
fig = out.figure if hasattr(out, "figure") else out
save(fig, OUT)
