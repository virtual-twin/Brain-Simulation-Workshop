"""Acoustic stimulation of a whole-brain model from a real audio data file.

A real recording (.wav) is turned into a loudness envelope and passed to a
*data-driven* stimulus `Event` (`dataLocation` + `sampling_rate` + `onset`).
The tvboptim backend interpolates the waveform at simulation time and injects it
into the **primary auditory cortex** (Desikan-Killiany transverse-temporal gyrus,
L.TTG / R.TTG) of a Reduced-Wong-Wang whole-brain model with a Balloon-Windkessel
BOLD observation.

The figure shows (1) auditory-cortex BOLD tracking the music, (2) the resting
model FC fit to the empirical FC shipped with the DK avgMatrix connectome
(global coupling G line-searched), and (3) the connectivity change for music vs.
an energy-matched **constant control** — isolating the effect of the sound's
*temporal structure* rather than the drive level.

Pure native-tvbo spec (RWW + DK connectome + BOLD) + the tvboptim codegen backend.

Output: img/figures/stimulation/fig-acoustic-stimulation.png
        img/figures/stimulation/acoustic_stimulus.npy  (+ _control.npy)
"""
from __future__ import annotations

import contextlib
import io
import os

import bsplot
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tvbo
from bsplot.surface import plot_surf
from matplotlib.colors import Normalize
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import resample

from tvbo import SimulationExperiment
from tvbo.datamodel.schema import Event, Parameter
from tvboptim.data import load_structural_connectivity

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(ROOT, "..", "img", "figures", "stimulation")
os.makedirs(OUTDIR, exist_ok=True)
MUSIC_PATH = os.path.abspath(os.path.join(OUTDIR, "acoustic_stimulus.npy"))
CTRL_PATH = os.path.abspath(os.path.join(OUTDIR, "acoustic_control.npy"))
FIG_PATH = os.path.join(OUTDIR, "fig-acoustic-stimulation.png")
EMP_FC_H5 = os.path.join(os.path.dirname(tvbo.__file__), "database", "networks",
                         "tpl-MNI152NLin2009cAsym_rec-avgMatrix_atlas-DesikanKilliany_desc-SCFC_relmat.h5")

WAV = "/Users/leonmartin_bih/projects/TVB-O/tvbo-manuscript/code/data/Mariana Zwarg Sexteto Universal - Nascentes - 02 Viva Hermeto.wav"
TRANSIENT = 24000.0      # ms burn-in (discarded). Must exceed the 20 s BOLD HRF kernel
                         # so prepend_history is fully populated and the valid-mode HRF
                         # convolution stays causally aligned (BOLD lags, never precedes).
DUR = 120000.0           # ms recorded
SR = 0.1                 # stimulus samples per ms (100 Hz)
TR = 1000.0              # ms BOLD repetition time (subsample_bold = data[::TR/dt])
AMP = 0.4                # peak drive added to the auditory input current
G = 0.2                  # global coupling, tuned so resting FC matches empirical FC
AUD = [32, 81]           # L.TTG / R.TTG = primary auditory cortex (DK)
AUD_C = "#1b9e2b"        # auditory-cortex marker colour
SKIP_TR = 5              # drop first BOLD volumes (HRF warm-up)
VIK = bsplot.colors.get_cmap("vik")   # Crameri diverging map for the ΔFC


def make_stimuli():
    """Return (raw_for_display, music_envelope, constant_control)."""
    rate, data = wavfile.read(WAV)
    x = data.astype(float)
    x = x.mean(1) if x.ndim > 1 else x
    x = x[: int(rate * DUR / 1000.0)]
    w = resample(x, int(DUR * SR))
    env = uniform_filter1d(np.abs(w), 50)        # rectify + smooth -> loudness
    env = (AMP * env / np.abs(env).max()).astype("float32")
    raw = (w / np.abs(w).max()).astype("float32")
    ctrl = np.full(env.shape, float(env.mean()), dtype="float32")   # energy-matched, no structure
    return raw, env, ctrl


def bold(data_path: str | None) -> np.ndarray:
    """RWW + BOLD on the DK connectome; drive auditory cortex from `data_path` (None = rest)."""
    with contextlib.redirect_stdout(io.StringIO()):
        exp = SimulationExperiment.from_db("RWW_BOLD_FC_Optimization")
    exp.integration.transient_time = TRANSIENT
    exp.integration.duration = DUR
    exp.network.coupling["FastLinearCoupling"].parameters["G"].value = G
    exp.optimizations.clear()
    exp.explorations.clear()
    for k in list(exp.observations.keys()):
        if k != "bold":
            del exp.observations[k]
    with contextlib.suppress(Exception):
        exp.network.observations.clear()
    exp.dynamics.parameters["acoustic"] = Parameter(name="acoustic", value=0.0)
    exp.dynamics.derived_variables["x"].equation.rhs = (
        str(exp.dynamics.derived_variables["x"].equation.rhs) + " + acoustic"
    )
    if data_path is not None:
        exp.events["acoustic"] = Event(
            name="acoustic", event_type="stimulus", dataLocation=data_path,
            sampling_rate=SR, parameters={"onset": {"value": TRANSIENT}},
            nodes=AUD, weights=[1.0] * len(AUD),
        )
    with contextlib.redirect_stdout(io.StringIO()):
        res = exp.run("tvboptim")
    obs = res.observations["bold"]
    return np.asarray(obs.data).squeeze(), np.asarray(obs.ts).squeeze()


# TVB abbreviated DK labels → FreeSurfer aparc.annot names
_ABBR_TO_APARC = {
    "BSTS": "bankssts", "CACG": "caudalanteriorcingulate", "CMFG": "caudalmiddlefrontal",
    "CU": "cuneus", "EC": "entorhinal", "FG": "fusiform", "IPG": "inferiorparietal",
    "ITG": "inferiortemporal", "ICG": "isthmuscingulate", "LOG": "lateraloccipital",
    "LOFG": "lateralorbitofrontal", "LG": "lingual", "MOFG": "medialorbitofrontal",
    "MTG": "middletemporal", "PHIG": "parahippocampal", "PaCG": "paracentral",
    "POP": "parsopercularis", "POR": "parsorbitalis", "PTR": "parstriangularis",
    "PCAL": "pericalcarine", "PoCG": "postcentral", "PCG": "posteriorcingulate",
    "PrCG": "precentral", "PCU": "precuneus", "RACG": "rostralanteriorcingulate",
    "RMFG": "rostralmiddlefrontal", "SFG": "superiorfrontal", "SPG": "superiorparietal",
    "STG": "superiortemporal", "SMG": "supramarginal", "FP": "frontalpole",
    "TP": "temporalpole", "TTG": "transversetemporal", "IN": "insula",
}
_SUBCORTICAL = {"CER", "TH", "CA", "PU", "PA", "HI", "AM", "AC"}


def dk_surface_mapper():
    """Return build(vals, hemi) mapping per-region DK values onto fsaverage 'lh'/'rh' vertices."""
    from bsplot.data.surface import get_surface_geometry
    from nibabel.freesurfer.io import read_annot

    _, _, labels = load_structural_connectivity(name="dk_average")
    labels = np.array(labels)
    data_dir = os.path.join(ROOT, "data")
    annot = {"lh": read_annot(os.path.join(data_dir, "lh.aparc.annot")),
             "rh": read_annot(os.path.join(data_dir, "rh.aparc.annot"))}
    name2idx = {h: {(n.decode() if isinstance(n, bytes) else str(n)): i
                    for i, n in enumerate(annot[h][2])} for h in ("lh", "rh")}
    nverts = {h: len(get_surface_geometry(template="fsaverage", hemi=h, density="164k")[0])
              for h in ("lh", "rh")}

    def build(vals, hemi):
        want = "L" if hemi == "lh" else "R"
        labv = annot[hemi][0]
        ov = np.full(nverts[hemi], np.nan)
        for name, v in zip(labels, vals):
            side, abbrev = str(name).split(".")
            if side != want or abbrev in _SUBCORTICAL:
                continue
            aparc = _ABBR_TO_APARC.get(abbrev)
            if aparc is not None and (j := name2idx[hemi].get(aparc)) is not None:
                ov[labv == j] = v
        return ov

    return build


def main() -> None:
    raw, env, ctrl = make_stimuli()
    np.save(MUSIC_PATH, env)
    np.save(CTRL_PATH, ctrl)

    emp_FC = np.array(h5py.File(EMP_FC_H5, "r")["edges/fc/data"])
    iu = np.triu_indices(emp_FC.shape[0], 1)

    B_rest, _ = bold(None)
    B_music, ts = bold(MUSIC_PATH)           # ts: exact BOLD sample times (now correctly TR-spaced)
    B_ctrl, _ = bold(CTRL_PATH)
    FC_rest = np.corrcoef(B_rest[SKIP_TR:].T)
    FC_music = np.corrcoef(B_music[SKIP_TR:].T)
    FC_ctrl = np.corrcoef(B_ctrl[SKIP_TR:].T)
    dFC = FC_music - FC_ctrl                              # content-specific change
    a, b = FC_rest[iu], emp_FC[iu]
    msk = np.isfinite(a) & np.isfinite(b)
    fit = float(np.corrcoef(a[msk], b[msk])[0, 1])

    def mark_auditory(a):
        """Mark L/R primary auditory cortex as ticks on the axes (no crosshair lines)."""
        a.set_xticks(AUD); a.set_xticklabels(["L", "R"], fontsize=7)
        a.set_yticks(AUD); a.set_yticklabels(["L", "R"], fontsize=7)
        a.tick_params(colors=AUD_C, length=4, width=1.4)

    fig = plt.figure(figsize=(12.0, 8.4), layout="compressed")
    gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1.0, 1.0])

    # row 1 — auditory cortex BOLD follows the music loudness, lagging by the HRF.
    # The Balloon-Windkessel warm-up (initial-condition transient) is hidden so the
    # genuine hemodynamic lag is visible rather than the start-up swing.
    ax = fig.add_subplot(gs[0, :]); ax2 = ax.twinx()
    te = np.arange(len(env)) / SR / 1000.0
    tb_all = np.arange(len(ts)) * TR / 1000.0            # exact TR-spaced BOLD times (s)
    WARM = 12.0                                           # s of BOLD warm-up to hide
    keep = tb_all >= WARM
    tb = tb_all[keep]

    def zc(v):
        return (v - v.mean()) / v.std()

    ax2.fill_between(te, env / env.max(), color="#6a3d9a", alpha=0.18)
    ax2.plot(te, env / env.max(), color="#6a3d9a", lw=0.8)
    ax.plot(tb, zc(B_music[keep, AUD[0]]), color=AUD_C, lw=1.3, label="L auditory BOLD")
    ax.plot(tb, zc(B_music[keep, AUD[1]]), color="#0b6e0b", lw=1.3, ls="--", label="R auditory BOLD")
    ax.set_title("auditory cortex BOLD follows the music loudness (hemodynamic lag)")
    ax.set_xlabel("t [s]"); ax.set_ylabel("BOLD (z)")
    ax2.set_ylabel("loudness", color="#6a3d9a")
    ax.set_xlim(0, te[-1]); ax2.set_xlim(0, te[-1])
    ax.legend(loc="upper left", fontsize=7)

    # rows 2-3 — FC matrices (auditory marked on the axes) + the fit scatter
    def mat(cell, M, ttl, cmap, vlo, vhi, mark):
        a = fig.add_subplot(cell)
        im = a.imshow(M, cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        a.set_title(ttl, fontsize=10)
        if mark:
            mark_auditory(a)
        else:
            a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, shrink=0.78, pad=0.02)

    mat(gs[1, 0], emp_FC, "empirical FC", "magma", -0.3, 1.0, False)
    mat(gs[1, 1], FC_rest, f"model FC — rest  (r = {fit:.2f})", "magma", -0.3, 1.0, True)
    mat(gs[1, 2], FC_music, "model FC — listening", "magma", -0.3, 1.0, True)

    axs = fig.add_subplot(gs[2, 0])
    axs.scatter(emp_FC[iu], FC_rest[iu], s=3, alpha=0.3, color="#1565c0")
    axs.plot([-0.2, 1], [-0.2, 1], "k--", lw=0.7)
    axs.set_title(f"FC fit  (r = {fit:.2f})", fontsize=10)
    axs.set_xlabel("empirical FC"); axs.set_ylabel("model FC")

    mat(gs[2, 1], dFC, "ΔFC: listening − control", VIK, -0.6, 0.6, True)

    # bottom-right — the auditory seed's ΔFC on the fsaverage surface, both hemispheres lateral
    build_overlay = dk_surface_mapper()
    seed = dFC[AUD].mean(0)                               # ΔFC of every region to auditory seed
    vmx = float(np.nanmax(np.abs(seed)))
    norm = Normalize(-vmx, vmx)
    axp = fig.add_subplot(gs[2, 2]); axp.axis("off")
    axp.set_title("ΔFC to auditory seed", fontsize=10)
    for k, (hemi, lab) in enumerate((("lh", "L"), ("rh", "R"))):
        bax = axp.inset_axes([0.0 + 0.5 * k, 0.05, 0.5, 0.82])
        plot_surf(surface="fsaverage", overlay=build_overlay(seed, hemi), hemi=hemi,
                  view="lateral", cmap="vik", norm=norm, parcellated=True, ax=bax)
        bax.axis("off"); bax.set_title(lab, fontsize=8)
    fig.colorbar(plt.cm.ScalarMappable(cmap=VIK, norm=norm), ax=axp, shrink=0.55, pad=0.02)

    fig.text(0.5, 0.005,
             f"RWW (G={G}, tuned to empirical FC, r={fit:.2f}) + BOLD on Desikan-Killiany; music vs "
             "energy-matched constant control at primary auditory cortex (L/R ticks).",
             ha="center", fontsize=8, color="0.4")
    fig.savefig(FIG_PATH, dpi=120, bbox_inches="tight")
    print(f"saved {FIG_PATH}")
    print(f"FC fit r={fit:.3f}  L-R auditory FC: control {FC_ctrl[AUD[0], AUD[1]]:.2f} -> "
          f"music {FC_music[AUD[0], AUD[1]]:.2f}  | BOLD TR={TR:.0f} ms ({len(ts)} volumes), "
          f"tvbo ts spacing {(ts[1]-ts[0]):.0f} ms")


if __name__ == "__main__":
    main()
