"""Normal forms: saddle-node and hysteresis continuations + scalar flows."""
from __future__ import annotations

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from _bif_common import (
    IMG,
    SADDLE_NODE, SADDLE_NODE_CONT,
    HYSTERESIS, HYSTERESIS_CONT,
    _bif, _plot_scalar_flow, save,
)

OUT = os.path.join(IMG, os.path.basename(__file__).replace(".py", ".png"))

fig, ax = plt.subplots(2, 2, figsize=(8.0, 5.6), layout="compressed")

examples = [
    {
        "title": r"Saddle-node: $\dot x = a - x^2$",
        "continuations": [(SADDLE_NODE, SADDLE_NODE_CONT)],
        "flow": SADDLE_NODE,
        "parameter_value": 0.6,
    },
    {
        "title": r"Hysteresis: $\dot x = a + x - x^3$",
        "continuations": [(HYSTERESIS, HYSTERESIS_CONT)],
        "flow": HYSTERESIS,
        "parameter_value": 0.2,
    },
]

for column, example in enumerate(examples):
    bifurcation_ax = ax[0, column]
    flow_ax = ax[1, column]
    for dyn_yaml, cont_yaml in example["continuations"]:
        _bif(dyn_yaml, cont_yaml).plot(VOI="x", ax=bifurcation_ax)
    bifurcation_ax.axvline(example["parameter_value"], color="0.35", lw=1.1, ls=":")
    bifurcation_ax.set_title(example["title"], fontsize=10)
    bifurcation_ax.set_xlabel("$a$", fontsize=9)
    bifurcation_ax.set_ylabel("$x$" if column == 0 else "", fontsize=9)
    bifurcation_ax.tick_params(labelsize=7)
    legend = bifurcation_ax.get_legend()
    if legend is not None:
        handles, labels = bifurcation_ax.get_legend_handles_labels()
        legend.remove()
        if column == 0:
            bifurcation_ax.legend(
                handles,
                labels,
                loc="upper left",
                fontsize=6,
                frameon=False,
                handlelength=1.4,
                markerscale=0.55,
                borderaxespad=0.2,
            )
    _plot_scalar_flow(example["flow"], example["parameter_value"], flow_ax)
    flow_ax.set_ylabel(r"$\dot x$" if column == 0 else "", fontsize=9)

flow_legend = [
    Line2D([0], [0], color="0.2", lw=1.4, label=r"curve: $\dot x=f(x)$"),
    Line2D([0], [0], color="0.65", lw=0.9, label=r"baseline: $\dot x=0$"),
    Line2D(
        [0], [0], color="C0", marker=r"$\rightarrow$", linestyle="None",
        markersize=11, label="arrows: flow direction",
    ),
    Line2D(
        [0], [0], marker="o", color="#c85030", mfc="#c85030", mec="#c85030",
        linestyle="None", markersize=5, label="filled dot: stable FP",
    ),
    Line2D(
        [0], [0], marker="o", color="#c85030", mfc="white", mec="#c85030",
        linestyle="None", markersize=5, label="open dot: unstable FP",
    ),
]
fig.legend(
    handles=flow_legend,
    loc="lower center",
    ncol=5,
    fontsize=7,
    frameon=False,
    bbox_to_anchor=(0.5, 0.005),
    columnspacing=1.2,
)
fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
save(fig, OUT)
