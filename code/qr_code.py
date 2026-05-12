import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from IPython.display import display

url = "http://tvbo.charite.de/survey/start/c7a2e1f3-4b8d-4e96-a0f5-2d3c9b1e7a04"
qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=20, border=2)
qr.add_data(url)
qr.make(fit=True)
img = qr.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer())

img.save("survey_qr.png")


url2 = "https://virtual-twin.github.io/Brain-Simulation-Workshop"
qr2 = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=20, border=2)
qr2.add_data(url2)
qr2.make(fit=True)
img2 = qr2.make_image(image_factory=StyledPilImage, module_drawer=RoundedModuleDrawer())

img2.save("website_qr.png")