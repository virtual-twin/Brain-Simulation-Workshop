function attachTitleVisual() {
  const titleSlide = document.querySelector(".reveal #title-slide");
  if (!titleSlide || titleSlide.querySelector("brain-network-visual")) {
    return;
  }

  const visual = document.createElement("brain-network-visual");
  visual.className = "slides-title-visual";
  visual.setAttribute("mesh-src", "data/brain_mesh.json");
  visual.setAttribute("tracts-src", "data/tracts_preview.json");
  visual.setAttribute("aria-hidden", "true");

  titleSlide.classList.add("has-brain-network-visual");
  titleSlide.prepend(visual);
}

customElements.whenDefined("brain-network-visual").then(attachTitleVisual);