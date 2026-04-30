"""Spring–mass oscillator animation utilities.

Provides a single high-level function :func:`animate_spring` that turns one or
more position time-series into a matplotlib animation with:

* one spring-panel per trajectory arranged side by side
* a shared time-series panel below all spring panels
* support for vertical (hanging) and horizontal (sliding) orientations
* per-trajectory mass size (visually encodes mass value)

Designed to be used from Jupyter notebooks and eventually integrated into
*bsplot* or *tvbo*.

Example
-------
>>> ani = animate_spring(
...     [x1, x2, x3],
...     labels=["$x_0=1$", "$x_0=2$", "$x_0=3$"],
...     orientation="vertical",
... )
>>> from IPython.display import HTML
>>> HTML(ani.to_jshtml())
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

__all__ = ["spring_curve", "animate_spring"]

# ---------------------------------------------------------------------------
# Default colour palette (Tableau-inspired)
# ---------------------------------------------------------------------------
import matplotlib as mpl


def _default_colors(n: int) -> list[str]:
    """Return first n colors from the current matplotlib color cycler."""
    cycle = mpl.rcParams['axes.prop_cycle'].by_key()['color']
    return [cycle[i % len(cycle)] for i in range(n)]


# ---------------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------------

def spring_curve(
    p0: float,
    p1: float,
    *,
    n_coils: int = 8,
    amplitude: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (transverse, axial) coordinates of a coil spring.

    The spring runs along the *axial* direction from ``p0`` to ``p1``.
    The coil zigzag is in the *transverse* direction.

    For a **vertical** spring the caller maps ``(transverse → x, axial → y)``.
    For a **horizontal** spring the caller maps ``(transverse → y, axial → x)``.

    Parameters
    ----------
    p0, p1:
        Start and end positions along the axial axis.
    n_coils:
        Number of full coil turns.
    amplitude:
        Half-width of the coil zigzag in transverse direction.

    Returns
    -------
    transverse : ndarray
    axial      : ndarray
    """
    n = n_coils * 20 + 2
    axial = np.linspace(p0, p1, n)
    transverse = np.zeros(n)
    transverse[1:-1] = amplitude * np.sin(
        np.linspace(0, n_coils * 2 * np.pi, n - 2)
    )
    return transverse, axial


# ---------------------------------------------------------------------------
# Main animation function
# ---------------------------------------------------------------------------

def animate_spring(
    trajectories: Sequence[np.ndarray],
    *,
    t: np.ndarray | None = None,
    labels: list[str] | None = None,
    titles: list[str] | None = None,
    colors: list[str] | None = None,
    mass_sizes: float | list[float] = 0.4,
    orientation: str = "vertical",
    anchor_pos: float = 3.5,
    equilibrium: float | list[float] = 0.0,
    n_coils: int = 8,
    coil_amplitude: float = 0.08,
    figsize: tuple[float, float] | None = None,
    n_frames: int = 200,
    interval: int = 30,
    repeat: bool = True,
    n_periods: float | None = None,
    ts_ylabel: str = "x",
    ts_xlabel: str = "time (normalised)",
    show_equilibrium: bool = True,
    anchor_thickness: float = 0.25,
    anchor_half_width: float = 0.6,
    spring_lw: float | list[float] = 1.5,
    mass_alpha: float = 0.85,
    mass_edgecolor: str = "k",
) -> animation.FuncAnimation:
    """Animate one or more spring–mass oscillators.

    Each trajectory is shown in its own spring panel; a shared time-series
    panel is placed below all spring panels.

    Parameters
    ----------
    trajectories:
        List of 1-D arrays, each giving the **position** of one mass over
        time.  All arrays are trimmed to the shortest length.
    t:
        Time axis (1-D array).  Defaults to ``np.linspace(0, 1, n_pts)``.
    labels:
        Legend label per trajectory.  Defaults to ``['$x_0$', '$x_1$', ...]``.
    titles:
        Title shown above each spring panel.  Defaults to *labels*.
    colors:
        Colour per trajectory.  Defaults to a built-in Tableau-inspired
        palette.
    mass_sizes:
        Side length of the square mass box in data coordinates — either a
        single float (same for all) or one value per trajectory.  Scaling
        this with a physical quantity (e.g. ``m ** (1/3)``) makes the mass
        visually comparable.
    orientation:
        ``'vertical'``  — spring hangs from ceiling, mass bounces up/down
        (position = y-coordinate of mass centre).
        ``'horizontal'`` — spring attached to left wall, mass slides
        left/right (position = x-coordinate of mass centre).
    anchor_pos:
        Coordinate of the fixed anchor: ceiling y (vertical) or wall x
        (horizontal).
    equilibrium:
        Rest position drawn as a dashed reference line.
    n_coils:
        Number of coil turns in the spring drawing.
    coil_amplitude:
        Half-width of the coil zigzag in transverse data coordinates.
    figsize:
        Figure size in inches.  Defaults to ``(3*n + 1.5, 6)`` for
        multiple springs or ``(7, 5)`` for a single spring.
    n_frames:
        Target number of animation frames (determines the time stride).
    interval:
        Delay between frames in milliseconds.
    ts_ylabel:
        Y-axis label for the shared time-series panel.
    ts_xlabel:
        X-axis label for the shared time-series panel.
    repeat:
        Whether to loop the animation continuously.  Defaults to ``True``.
    n_periods:
        If given, trim every trajectory to exactly this many oscillation
        periods so the loop restarts at the same phase.  The period is
        estimated from the first trajectory using zero-crossing detection.
        E.g. ``n_periods=1`` gives a perfectly seamless single-period loop.
    show_equilibrium:
        Whether to draw the red dashed equilibrium reference line.
    anchor_thickness:
        Visual thickness of the ceiling/wall slab in data coordinates.
    anchor_half_width:
        Half-width of the ceiling/wall slab in transverse data coordinates.
    spring_lw:
        Line width of the spring drawing.
    mass_alpha:
        Opacity of the mass rectangle.

    mass_edgecolor:
        Edge colour of the mass rectangle.
        Call ``.to_jshtml()`` for inline Jupyter display or
        ``.save(path)`` to write a video file.

    Examples
    --------
    >>> ani = animate_spring([x1, x2], labels=["$x_0=1$", "$x_0=2$"])
    >>> HTML(ani.to_jshtml())

    >>> # Make box size encode mass
    >>> m_vals = [0.5, 1.0, 2.0]
    >>> sizes   = [0.4 * (m ** (1/3)) for m in m_vals]
    >>> ani = animate_spring(trajectories, mass_sizes=sizes,
    ...                      titles=[f"m={m}" for m in m_vals])
    """
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    n = len(trajectories)
    n_pts = min(len(a) for a in trajectories)
    trajs = [np.asarray(a).ravel()[:n_pts] for a in trajectories]

    # Normalise equilibrium to a per-trajectory list
    if np.isscalar(equilibrium):
        equilibria = [float(equilibrium)] * n
    else:
        equilibria = [float(e) for e in equilibrium]

    # --- Optionally trim to whole periods for seamless looping ---
    if n_periods is not None:
        ref = trajs[0] - equilibria[0]
        # Find zero-crossings (positive-going) after the first sample
        crossings = np.where((ref[:-1] < 0) & (ref[1:] >= 0))[0] + 1
        if len(crossings) >= int(n_periods) + 1:
            period_end = crossings[int(n_periods)]  # index of n_periods-th crossing
            n_pts = period_end
            trajs = [a[:n_pts] for a in trajs]

    if t is None:
        t = np.linspace(0, 1, n_pts)

    _colors = list(colors) if colors else _default_colors(n)
    _labels = labels or [f"$x_{{{i}}}$" for i in range(n)]
    _titles = titles or _labels

    # Normalise mass_sizes to a per-trajectory list
    if np.isscalar(mass_sizes):
        sizes = [float(mass_sizes)] * n
    else:
        sizes = [float(s) for s in mass_sizes]

    # Normalise spring_lw to a per-trajectory list
    if np.isscalar(spring_lw):
        lws = [float(spring_lw)] * n
    else:
        lws = [float(lw) for lw in spring_lw]

    stride = max(1, n_pts // n_frames)

    # -----------------------------------------------------------------------
    # Figure layout
    # -----------------------------------------------------------------------
    if n == 1:
        _figsize = figsize or (7, 5)
        fig, (ax_sp, ax_t) = plt.subplots(
            2, 1, figsize=_figsize, gridspec_kw={"height_ratios": [2, 1]}
        )
        fig.tight_layout(pad=2)
        spring_axes = [ax_sp]
    else:
        _figsize = figsize or (3 * n + 1.5, 6)
        keys = [chr(ord("a") + i) for i in range(n)]
        mosaic = [keys, ["t"] * n]
        fig, axes_dict = plt.subplot_mosaic(
            mosaic,
            figsize=_figsize,
            gridspec_kw={"height_ratios": [2, 1]},
        )
        fig.tight_layout(pad=2.5)
        spring_axes = [axes_dict[k] for k in keys]
        ax_t = axes_dict["t"]

    # -----------------------------------------------------------------------
    # Shared axis limits for spring panels
    # -----------------------------------------------------------------------
    amp_max = max(np.abs(a).max() for a in trajs)
    size_max = max(sizes)

    if orientation == "vertical":
        y_hi = anchor_pos + anchor_thickness + 0.2
        y_lo = min(equilibria) - amp_max - size_max - 0.4
        sp_xlim = (-1.0, 1.0)
        sp_ylim = (y_lo, y_hi)
    else:
        x_lo = anchor_pos - anchor_thickness - 0.2
        x_hi = max(equilibria) + amp_max + size_max + 0.4
        sp_xlim = (x_lo, x_hi)
        sp_ylim = (-1.0, 1.0)

    # -----------------------------------------------------------------------
    # Build spring panels
    # -----------------------------------------------------------------------
    spring_lines: list = []
    mass_patches: list = []

    for ax, traj, col, title, side, lw, eq in zip(spring_axes, trajs, _colors, _titles, sizes, lws, equilibria):
        ax.set_xlim(*sp_xlim)
        ax.set_ylim(*sp_ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=8 if n > 1 else 10, color=col)

        if orientation == "vertical":
            ax.fill_between(
                [-anchor_half_width, anchor_half_width],
                anchor_pos,
                anchor_pos + anchor_thickness,
                color="gray",
            )
            ax.axhline(anchor_pos, color="gray", lw=2)
            if show_equilibrium:
                ax.axhline(eq, color="red", lw=0.8, ls="--", alpha=0.4)
        else:
            ax.fill_betweenx(
                [-anchor_half_width, anchor_half_width],
                anchor_pos - anchor_thickness,
                anchor_pos,
                color="gray",
            )
            ax.axvline(anchor_pos, color="gray", lw=2)
            if show_equilibrium:
                ax.axvline(eq, color="red", lw=0.8, ls="--", alpha=0.4)

        sl, = ax.plot([], [], "k-", lw=lw)
        mr = Rectangle(
            (-side / 2, -side / 2), side, side,
            fc=col, ec=mass_edgecolor, zorder=5, alpha=mass_alpha,
        )
        ax.add_patch(mr)
        spring_lines.append(sl)
        mass_patches.append(mr)

    # -----------------------------------------------------------------------
    # Time-series panel
    # -----------------------------------------------------------------------
    ts_ymin = min(a.min() for a in trajs) * 1.15
    ts_ymax = max(a.max() for a in trajs) * 1.15
    ax_t.set_xlim(float(t[0]), float(t[-1]))
    ax_t.set_ylim(ts_ymin, ts_ymax)
    ax_t.set_xlabel(ts_xlabel)
    ax_t.set_ylabel(ts_ylabel)
    if show_equilibrium:
        for eq in set(equilibria):
            ax_t.axhline(eq, color="red", lw=0.8, ls="--", alpha=0.4)

    ts_lines: list = []
    ts_dots: list = []
    for traj, col, lbl in zip(trajs, _colors, _labels):
        ax_t.plot(t, traj, color=col, lw=0.8, alpha=0.25)
        al, = ax_t.plot([], [], color=col, lw=1.5, label=lbl)
        td, = ax_t.plot([], [], "o", color=col, ms=5)
        ts_lines.append(al)
        ts_dots.append(td)
    ax_t.legend(fontsize=8, loc="upper right")

    # -----------------------------------------------------------------------
    # Per-frame helpers
    # -----------------------------------------------------------------------

    def _spring_xy(mass_pos: float, side: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (xs, ys) of spring from anchor to mass surface."""
        if orientation == "vertical":
            tip = mass_pos + side / 2          # top edge of mass
            transv, axial = spring_curve(
                anchor_pos, tip, n_coils=n_coils, amplitude=coil_amplitude
            )
            return transv, axial               # x=transverse, y=axial
        else:
            tip = mass_pos - side / 2          # left edge of mass
            transv, axial = spring_curve(
                anchor_pos, tip, n_coils=n_coils, amplitude=coil_amplitude
            )
            return axial, transv               # x=axial, y=transverse

    def _mass_xy(mass_pos: float, side: float) -> tuple[float, float]:
        """Bottom-left corner of mass rectangle."""
        if orientation == "vertical":
            return (-side / 2, mass_pos - side / 2)
        else:
            return (mass_pos - side / 2, -side / 2)

    # -----------------------------------------------------------------------
    # FuncAnimation
    # -----------------------------------------------------------------------
    all_artists = (*spring_lines, *mass_patches, *ts_lines, *ts_dots)

    def init():
        for sl, mr, side in zip(spring_lines, mass_patches, sizes):
            sl.set_data([], [])
            mr.set_xy((-side / 2, -side / 2))
        for al, td in zip(ts_lines, ts_dots):
            al.set_data([], [])
            td.set_data([], [])
        return all_artists

    def update(frame):
        i = frame * stride
        for sl, mr, traj, side in zip(spring_lines, mass_patches, trajs, sizes):
            xi = float(traj[i])
            sl.set_data(*_spring_xy(xi, side))
            mr.set_xy(_mass_xy(xi, side))
        for al, td, traj in zip(ts_lines, ts_dots, trajs):
            xi = float(traj[i])
            al.set_data(t[:i + 1], traj[:i + 1])
            td.set_data([t[i]], [xi])
        return all_artists

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_pts // stride,
        init_func=init,
        blit=True,
        interval=interval,
        repeat=repeat,
    )
    plt.close(fig)
    return ani
