"""Linear stability: real solutions x(t) = x0 e^{at} from TVBO."""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from _bif_common import IMG, LINEAR, save
from tvbo import Dynamics

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

fig, ax = plt.subplots(figsize=(5.2, 3.4))
dyn = Dynamics.from_string(LINEAR)
x0 = 0.25
styles = {
    -1.0: {"color": "C0", "ls": "-"},
    -0.5: {"color": "C1", "ls": "-"},
    0.0: {"color": "0.15", "ls": "--"},
    0.5: {"color": "C2", "ls": "-"},
    1.0: {"color": "C3", "ls": "-"},
}
for a, style in styles.items():
    dyn.parameters["a"].value = a
    dyn.plot("x", kind="timeseries", duration=3.0, dt=0.01, u_0=[x0], ax=ax)
    line = ax.lines[-1]
    line.set(color=style["color"], linestyle=style["ls"], linewidth=2.0)
    if a == 0.0:
        line.set_label(r"$a=0$ ($\lambda=0$)")
        line.set_linewidth(2.6)
    else:
        line.set_label(rf"$a={a:g}$")
ax.set(
    xlabel="$t$",
    ylabel="$x(t)$",
    ylim=(0.0, 5.4),
    title=rf"$\dot x = a\,x,\quad x(0)={x0:g},\quad \lambda=a$",
)
ax.annotate(
    r"neutral trajectory: $\lambda=a=0$",
    xy=(1.75, x0),
    xytext=(0.7, 1.3),
    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "0.2"},
    color="0.2",
    fontsize=8,
)
ax.legend(
    title="growth rate",
    loc="upper left",
    fontsize=7,
    title_fontsize=8,
    frameon=False,
)
fig.tight_layout()
save(fig, OUT)
