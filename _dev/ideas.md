# Slides — open TODOs & ideas

Forward-looking backlog for the `SummerSchool2026` deck (`slides.qmd` + `slides/_NN-*.qmd`).
One clean-revealjs deck assembled from `{{< include >}}`-ed partials; chapters **1, 2, 3, 5, 7**
are live, **4** (Clinical) and **6** (Parameter Exploration) are commented out in `slides.qmd`.
Priorities: **P1** important · **P2** clarity/polish · **P3** nice-to-have.

---

## Content gaps

- [ ] **Ch 7 "Stimulation with Bayesian inference" is a stub** (`_07-stimulation.qmd`, a
  `<!-- TODO -->` + notebook link + bullets). Build the stimulus + inference slides
  (degeneracy ridge, priors-as-hypotheses, posterior vs multi-start), or retitle it
  explicitly "hands-on only". **(P1)**
- [ ] **Ch 4 Clinical is empty** (`_04-clinical.qmd`, excluded). Either build it
  (epilepsy/VEP, stroke, personalized whole-brain case studies — ch2's Schirner/Koller
  material is "basic research", clinical is genuinely missing), or delete the file + its
  commented include. **(P2)**
- [ ] **Ch 3 "Setup & First Examples" is one link** (`_03-setup.qmd`, ~12 lines). The
  title promises a *first example* — add a tiny runnable snippet (3-line TVBO
  simulate-and-plot). **(P2)**
- [ ] **Ch 6 (Parameter Exploration) is the most polished chapter but excluded.** Decide:
  ship it as an optional standalone deck, keep a trimmed 5-slide version in, or leave out.
  Right now it's ~580 lines of dead include. **(P2)**

## Pedagogy & delivery

- [ ] **Speaker notes only exist for ch 5 (Bifurcation).** Add `::: {.notes}` to the other
  chapters (1, 2, 3, 7) so the deck is teachable by others / a self-reminder. **(P2)**
- [ ] **No section recap / transition slides.** Each `# N.` divider drops straight into
  content; a one-line "where we are / why this next" on each divider would help. **(P3)**

## Design & readability (now an online / Zoom course)

- [ ] **4 white-background slides** in ch1 dynamical systems
  (`{background-color="#ffffff"}` on Initial Conditions / Phase Space / Parameters–Mass /
  Towards Realism) break the teal `#6880af` identity. Theme them consistently or signpost
  the spring-mass run as deliberately light. **(P3)**
- [ ] **Tiny scrollable spec panels** — the Jansen-Rit model spec (`_01`, `font-size:0.45em`,
  `overflow-y:scroll`) and the uncoupled/coupled Kuramoto panels (`0.48em`). Unreadable on
  a Zoom share. Trim to the essentials, or move the full spec to a backup slide. **(P2)**
- [ ] **Document the raw-HTML flex layouts** — `_01`'s `<div style="display:flex…">`
  uncoupled/coupled/math-framework slides exist because reveal's `.columns` auto-stretch
  collapses the GIF column. Add a short comment so nobody "helpfully" converts them and
  breaks the GIFs. **(P3)**
- [ ] **`se-dia` diagram** (`_02`, ~150 lines inline CSS+HTML) is brittle. Render it once to
  an SVG/PNG asset, or move the style block to `slides.css`. **(P3)**

## Accessibility

- [ ] **Inconsistent alt text** — many `![](img/...)` have none. Add concise alt text,
  especially for GIFs and load-bearing diagrams. **(P3)**
- [ ] **Color-only encoding** in some figures (regime maps, CMA-ES ellipses) — check
  legibility in grayscale / for color-vision deficiency. **(P3)**
- [ ] **Static PDF export** — verify the heavily-`.incremental` slides still read when
  exported (fragments collapsed). **(P3)**

## Citations, links, assets

- [ ] **Notebook links point at `.qmd` source** (`_05`, `_07`); `_06` uses `.html`.
  Decide which resolves on the deployed site and standardize. *(Leon to test which works.)*
  **(P2)**
- [ ] **Citations used as bare captions** — `![[@TuringCommunity2024]](…)`,
  `![[@Schirner2023]](…)`. Works, but a short descriptive caption + citation reads better. **(P3)**
- [ ] **Orphan code snippet** — `_05` bifurcation hands-on shows
  `result = exp.run("bifurcationkit.jl")` with no surrounding context. Fold into the
  bullet list or expand. **(P3)**
- [ ] **Figure provenance** — bifurcation figures come from `code/fig-bifurcation-*.py`
  (the phase-portrait pair now renders matched at 250 dpi); section-1/section-4 figures
  from `notebooks/1_intro_demos.qmd` / `notebooks/4_inference_demos.qmd`. Keep this map
  somewhere discoverable so figures can be regenerated. **(P3)**

## Build / maintenance

- [ ] **Guard the FAIR SVGs** — `img/fair/*.svg` were once silently truncated. Add a quick
  check (make target / pre-commit) that the committed SVGs parse as XML and end in
  `</svg>`. **(P3)**
- [ ] **Upstream tvbo bug** — `quiet=` is accepted but unused; the
  `tvbo-tvboptim-experiment.py.mako` banner prints unconditionally (worked around with
  `redirect_stdout` in `_01`). File an issue for a real `quiet` guard. **(P3)**
- [ ] **CI parity** — `.gitlab-ci.yml` pins Quarto 1.7.24; confirm the include-based deck
  renders the single `slides.html` in CI (partials are `_`-prefixed, only `slides.qmd` is in
  the `render:` list — should be fine, verify on first push). **(P3)**

---

## Next up (suggested order)

1. Ch 7 Bayesian section — build or retitle. *(biggest visible gap)*
2. Decide ch 4 (build vs delete) and ch 6 (keep / trim / standalone).
3. Add a runnable first example to ch 3.
4. Speaker notes for ch 1, 2, 3, 7.
5. Trim the 0.45em scrollable spec panels for Zoom readability.
6. Resolve the `.qmd` vs `.html` notebook links once tested.
