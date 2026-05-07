from tvbo import SimulationStudy
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.basename(__file__).replace(".py", ".png")
jr = SimulationStudy.from_file('/Users/leonmartin_bih/tools/tvbo/docs/Replication/Jansen1995/Jansen1995_extracted.yaml')

exp = jr.get_experiment(1)

res = exp.run()

fig = res.explorations['c_sweep_fig3'].plot()

fig.set_figwidth(5)
fig.set_figheight(4)
fig.tight_layout()
fig.align_labels()

fig.suptitle("Jansen-Rit Regimes ($C$)", fontsize=16)

fig.savefig(
    os.path.join(ROOT, "..", "img", "figures", "jansen1995", FILENAME),
    dpi=300,
    bbox_inches="tight",
)


