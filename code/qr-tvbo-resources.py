"""Generate styled QR codes for the TVB-O resources (platform, PyPI packages).
Run with a python that has `qrcode` (the workshop .venv). Outputs to img/qr/."""
import os
import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

OUT = os.path.join(os.path.dirname(__file__), "..", "img", "qr")
os.makedirs(OUT, exist_ok=True)

LINKS = {
    "qr-tvbo-platform": "https://tvbo.charite.de/",
    "qr-tvbo-pypi": "https://pypi.org/project/tvbo/",
    "qr-tvb-ext-ontology-pypi": "https://pypi.org/project/tvb-ext-ontology/",
}

for name, url in LINKS.items():
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=20, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer())
    path = os.path.join(OUT, name + ".png")
    img.save(path)
    print("wrote", os.path.relpath(path, os.path.join(OUT, "..", "..")), "->", url)
