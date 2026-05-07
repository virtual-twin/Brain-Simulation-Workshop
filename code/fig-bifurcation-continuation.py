"""Continuation schematic: inverted tutorial figure for white slides."""
from __future__ import annotations

import os
from PIL import Image, ImageOps
from _bif_common import IMG, TUTORIAL_FIG_48

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

image = Image.open(TUTORIAL_FIG_48).convert("RGBA")
rgb = Image.new("RGB", image.size, "black")
rgb.paste(image, mask=image.split()[-1])
ImageOps.invert(rgb).save(OUT)
print("wrote", OUT)
