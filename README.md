# BrainSimulation-Workshop

Materials for the Track 4 workshop *"Towards translational and mechanistic whole-brain simulations with The Virtual Brain"* (FAIR Brain Data Science Bootcamp, 12–13 May 2026).

## Repository layout

- `slides.qmd` — RevealJS workshop slides (root).
- `agenda/agenda.qmd` — printable workshop agenda (renders to PDF / DOCX).
- `notebooks/*.qmd` — hands-on session sources (rendered to `.ipynb` for participants).
- `bibliography.bib` — shared references for both documents.
- `_quarto.yml` — project-level Quarto config (bibliography, link defaults).
- `pyproject.toml` / `uv.lock` / `.python-version` — pinned Python environment.
- `img/` — logos and figures.

## Python environment (uv)

The Python environment is managed with [uv](https://docs.astral.sh/uv/). The lockfile and Python pin are committed, so the environment is reproducible across machines.

```bash
# one-time, after cloning
uv sync
```

This reads `pyproject.toml` + `uv.lock` and creates a `.venv/` with the exact package versions.

To add a new dependency:

```bash
uv add <package>          # updates pyproject.toml and uv.lock
```

## Rendering

[Quarto](https://quarto.org) (>= 1.8) must be installed separately. Run Quarto through `uv run` so it picks up the project's `.venv` Python:

```bash
# slides — RevealJS HTML
uv run quarto render slides.qmd
uv run quarto preview slides.qmd      # live reload during editing

# agenda — PDF + DOCX
uv run quarto render agenda/agenda.qmd

# hands-on notebooks — convert qmd → ipynb (no execution)
uv run quarto convert notebooks/02a_mean_field_models.qmd
# …or convert all of them at once:
for f in notebooks/*.qmd; do uv run quarto convert "$f"; done
```

The `.qmd` files under `notebooks/` are the source of truth; the generated `.ipynb` files are gitignored and produced on demand for the hands-on sessions.

## Authoring code on slides

Quarto chunk fences control whether a code block is executed:

- ` ```{python} ` — runs and shows code + output
- ` ```python ` (plain markdown, no braces) — shows code only, never runs
- ` ```{python} ` with `#| eval: false` — styled like an executable chunk but skipped
- `#| echo: false` — runs but hides the source
- `#| output: false` — runs but hides the output
