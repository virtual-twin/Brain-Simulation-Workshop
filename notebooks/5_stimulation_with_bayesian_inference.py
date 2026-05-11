# %% Imports
import os

N_DEVICES = 25
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={N_DEVICES}"

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import optax
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density
from scipy.stats import gaussian_kde, iqr

from tvboptim.execution import ParallelExecution
from tvboptim.experimental.network_dynamics import Network, prepare, solve as nd_solve
from tvboptim.experimental.network_dynamics.dynamics.tvb import Generic2dOscillator
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseGraph
from tvboptim.experimental.network_dynamics.external_input import PulseInput
from tvboptim.experimental.network_dynamics.solvers import Heun
from tvboptim.types import DataAxis, GridAxis, Space, Parameter, collect_parameters
from tvboptim.optim import OptaxOptimizer, SavingLossCallback, SavingParametersCallback, MultiCallback

# %% Shared settings
T1 = 150.0
DT = 0.2
ONSET = 10.0
DURATION = 1.0

TRUE_AMPLITUDE = 0.4
TRUE_EXCITABILITY = 0.1
OBS_NOISE_SIGMA = 0.1    # data generation noise
LIKELIHOOD_SIGMA = 0.2   # fixed in likelihood — intentionally flat along ridge so priors steer

PRIOR_AMP_MEAN = 0.2
PRIOR_EXC_MEAN = 0.0

# Scenario priors: all share the same means; hypotheses constrain one parameter (tight σ).
# Standard deviations:                No hyp.   H1       H2
PRIOR_AMP_STD = {"A": 0.2, "B": 0.1, "C": 0.2}   # H1: tight amp → excitability explains
PRIOR_EXC_STD = {"A": 0.1, "B": 0.1, "C": 0.05}  # H2: tight exc → stimulus explains

SCENARIO_LABELS = {
    "A": "No hypothesis\n(wide priors on both)",
    "B": f"H1: Higher excitability\namp ~ N({PRIOR_AMP_MEAN}, {PRIOR_AMP_STD['B']})",
    "C": f"H2: Stronger stimulus\nexc ~ N({PRIOR_EXC_MEAN}, {PRIOR_EXC_STD['C']})",
}

DYNAMICS_PARAMS = dict(a=-1.5, b=-15.0, c=0.0, d=0.015, e=3.0, f=1.0, tau=4.0)
weights = jnp.zeros((1, 1))

SCENARIO_KEYS = ["A", "B", "C"]
COLORS = ["tab:blue", "tab:green", "tab:orange"]


# %% Helpers

def build_network(amplitude, excitability):
    """Deterministic forward model network."""
    return Network(
        dynamics=Generic2dOscillator(**DYNAMICS_PARAMS, I=excitability, VARIABLES_OF_INTEREST=("V",)),
        coupling={"instant": LinearCoupling(incoming_states="V", G=0.0)},
        graph=DenseGraph(weights),
        external_input={"stimulus": PulseInput(onset=ONSET, duration=DURATION, amplitude=amplitude)},
    )


def make_model(scenario_key):
    """Return a numpyro model with scenario-specific prior widths baked in as closure constants.

    Prior std values must be Python floats at JIT-compile time — passing live distribution
    objects into mcmc.run() silently breaks tracing and causes priors to be ignored.
    """
    amp_std = float(PRIOR_AMP_STD[scenario_key])
    exc_std = float(PRIOR_EXC_STD[scenario_key])

    def model(v_obs, solve_fn, config, obs_idx):
        amplitude    = numpyro.sample("amplitude",    dist.Normal(PRIOR_AMP_MEAN, amp_std))
        excitability = numpyro.sample("excitability", dist.Normal(PRIOR_EXC_MEAN, exc_std))
        config.external.stimulus.amplitude = amplitude
        config.dynamics.I = excitability
        v_pred = solve_fn(config).ys[obs_idx, 0, 0]
        numpyro.sample("obs", dist.Normal(v_pred, LIKELIHOOD_SIGMA), obs=v_obs)

    return model


def make_loss(solve_fn):
    """MSE loss against observed data, closed over solve_fn."""
    def loss(config):
        return jnp.mean((solve_fn(config).ys[obs_idx, 0, 0] - v_obs) ** 2)
    return loss


def run_mcmc(model_fn, seed, label, num_warmup=500, num_samples=2000, num_chains=1):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    net = build_network(TRUE_AMPLITUDE, TRUE_EXCITABILITY)
    sf, cfg = prepare(net, Heun(), t0=0.0, t1=T1, dt=DT)
    nuts = NUTS(
        model_fn,
        max_tree_depth=10,
        dense_mass=True,        # learns ridge correlation → explores along it
        target_accept_prob=0.8,
    )
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(jax.random.key(seed), v_obs, sf, cfg, obs_idx)
    mcmc.print_summary()
    return mcmc.get_samples(group_by_chain=False)


def _draw_landscape(ax, vmax=None):
    """Draw MSE heatmap + 10th-percentile contour + ground truth marker; return pcm."""
    pcm = ax.pcolormesh(amp_vals, excit_vals, mse_grid,
                        cmap="cividis_r", vmax=vmax or vmax_clip)
    ax.contour(amp_vals, excit_vals, mse_grid,
               levels=[float(jnp.percentile(mse_grid, 10))],
               colors="white", linewidths=1.0, linestyles="--")
    ax.scatter(TRUE_AMPLITUDE, TRUE_EXCITABILITY,
               color="k", marker="*", s=150, zorder=6, label="ground truth")
    ax.set_xlabel("Stimulus amplitude")
    ax.set_ylabel("Excitability (I)")
    return pcm


# %% Generate observed data
# Noiseless signal as base ensures MLE at true params. Ridge points produce near-identical
# noiseless signals, so the likelihood is flat along the ridge and priors steer freely.
net_det = build_network(TRUE_AMPLITUDE, TRUE_EXCITABILITY)
sf_det, cfg_det = prepare(net_det, Heun(), t0=0.0, t1=T1, dt=DT)
v_det_true = sf_det(cfg_det).ys[:, 0, 0]
ts = sf_det(cfg_det).ts

obs_idx = jnp.arange(0, len(ts), 15)
ts_obs = ts[obs_idx]
v_obs = v_det_true[obs_idx] + OBS_NOISE_SIGMA * jax.random.normal(
    jax.random.key(42), (len(obs_idx),)
)

# %% Plot observed data
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(ts, v_det_true, color="tab:blue", lw=1.5, label="noiseless signal")
ax.scatter(ts_obs, v_obs, s=12, color="k", zorder=3, label="observations (+noise)")
ax.axvspan(ONSET, ONSET + DURATION, alpha=0.2, color="tab:red", label="stimulus")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("V")
ax.set_title(f"Observed data  (ground truth: amp={TRUE_AMPLITUDE}, I={TRUE_EXCITABILITY})")
ax.legend()
fig.tight_layout()
plt.show()

# %% Background landscape (deterministic MSE scan)
N_AMP = N_DEVICES
N_EXCIT = N_DEVICES

net_scan = build_network(TRUE_AMPLITUDE, TRUE_EXCITABILITY)
sf_scan, cfg_scan = prepare(net_scan, Heun(), t0=0.0, t1=T1, dt=DT)

cfg_scan.external.stimulus.amplitude = GridAxis(0.0, 1.0, N_AMP)
cfg_scan.dynamics.I = GridAxis(-0.1, 0.5, N_EXCIT)
space = Space(cfg_scan, mode="product")

par = ParallelExecution(
    model=lambda cfg: sf_scan(cfg).ys[:, 0, 0],
    space=space, n_pmap=N_DEVICES, n_vmap=N_DEVICES,
)
scan_results = par.run()
print("Landscape scan complete.")

df_scan = scan_results.to_dataframe()
amp_vals   = jnp.array(sorted(df_scan["external.stimulus.amplitude"].unique()))
excit_vals = jnp.array(sorted(df_scan["dynamics.I"].unique()))

mse_grid = jnp.zeros((len(excit_vals), len(amp_vals)))
for row_idx, row in df_scan.iterrows():
    mse   = float(jnp.mean((scan_results[row_idx] - v_det_true) ** 2))
    i_amp = int(jnp.argmin(jnp.abs(amp_vals   - row["external.stimulus.amplitude"])))
    i_ex  = int(jnp.argmin(jnp.abs(excit_vals - row["dynamics.I"])))
    mse_grid = mse_grid.at[i_ex, i_amp].set(mse)

vmax_clip = float(jnp.percentile(mse_grid, 75))

# %% Plot N closest matches on landscape + their timeseries
N_MATCHES = 8
cmap_matches = plt.get_cmap("tab10")

sorted_idx = jnp.argsort(mse_grid.flatten())
matches = []
for flat_idx in sorted_idx:
    if len(matches) >= N_MATCHES:
        break
    i_ex  = int(flat_idx) // N_AMP
    i_amp = int(flat_idx) % N_AMP
    amp   = float(amp_vals[i_amp])
    excit = float(excit_vals[i_ex])
    if abs(amp - TRUE_AMPLITUDE) < 0.05 and abs(excit - TRUE_EXCITABILITY) < 0.05:
        continue
    net_m = build_network(amp, excit)
    sf_m, cfg_m = prepare(net_m, Heun(), t0=0.0, t1=T1, dt=DT)
    matches.append((amp, excit, sf_m(cfg_m).ys[:, 0, 0]))

fig, (ax_map, ax_ts) = plt.subplots(2, 1, figsize=(9, 8),
                                     gridspec_kw={"height_ratios": [1.4, 1]})

pcm = _draw_landscape(ax_map)
fig.colorbar(pcm, ax=ax_map, label="MSE(V)")
for idx, (amp, excit, _) in enumerate(matches):
    ax_map.scatter(amp, excit, color=cmap_matches(idx), s=60, zorder=5,
                   edgecolors="white", linewidths=0.5)
ax_map.set_title("Degeneracy landscape — closest matches marked")
ax_map.legend(fontsize=8)

ax_ts.plot(ts, v_det_true, color="black", lw=2,
           label=f"Ground truth (amp={TRUE_AMPLITUDE}, I={TRUE_EXCITABILITY})", zorder=10)
ax_ts.axvspan(ONSET, ONSET + DURATION, alpha=0.15, color="tab:red")
for idx, (amp, excit, v_m) in enumerate(matches):
    ax_ts.plot(ts, v_m, color=cmap_matches(idx), alpha=0.75, lw=1.2,
               label=f"amp={amp:.2f}, I={excit:.2f}")
ax_ts.set_ylabel("V")
ax_ts.set_xlabel("Time (ms)")
ax_ts.set_title("Near-identical timeseries from different mechanisms")
ax_ts.legend(fontsize=7, ncol=2)
fig.tight_layout()
plt.show()

# %% Sanity check: verify different σ steers log-probability before full sampling
_net_chk = build_network(TRUE_AMPLITUDE, TRUE_EXCITABILITY)
_sf_chk, _cfg_chk = prepare(_net_chk, Heun(), t0=0.0, t1=T1, dt=DT)
_args = (v_obs, _sf_chk, _cfg_chk, obs_idx)

# model_C has tight excitability prior → high-|I| points should score worse
_model_C = make_model("C")
lp_low,  _ = log_density(_model_C, _args, {}, {"amplitude": 0.5, "excitability": 0.0})
lp_high, _ = log_density(_model_C, _args, {}, {"amplitude": 0.1, "excitability": 0.3})
print(f"model_C  log p(amp=0.50, I=0.0):  {lp_low:.1f}")
print(f"model_C  log p(amp=0.10, I=0.3):  {lp_high:.1f}")
print(f"Difference: {lp_low - lp_high:.1f}  (should be >> 0 if tight exc prior is active)")

# %% Run the three inference scenarios
MCMC_KWARGS = dict(num_warmup=500, num_samples=4000, num_chains=1)

all_samples = [
    run_mcmc(make_model("A"), seed=0, label="A: No hypothesis (uninformative priors)", **MCMC_KWARGS),
    run_mcmc(make_model("B"), seed=1, label="H1: Higher excitability (e.g. PET, gene expression)", **MCMC_KWARGS),
    run_mcmc(make_model("C"), seed=2, label="H2: Stronger stimulus (e.g. device logs, protocol)", **MCMC_KWARGS),
]

# %% Joint posteriors overlaid on landscape
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, samples, key in zip(axes, all_samples, SCENARIO_KEYS):
    pcm = _draw_landscape(ax)
    ax.scatter(samples["amplitude"], samples["excitability"],
               s=4, alpha=0.25, color="white", rasterized=True)
    ax.axvline(PRIOR_AMP_MEAN, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(PRIOR_EXC_MEAN, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.set_title(SCENARIO_LABELS[key], fontsize=9)
    ax.legend(fontsize=8)

axes[0].set_ylabel("Excitability (I)")
fig.colorbar(pcm, ax=axes[-1], label="MSE(V)")
fig.suptitle("Same observation, different hypotheses — priors steer mechanistic interpretation",
             fontsize=11)
fig.tight_layout()
plt.show()

# %% Marginal posteriors
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex="row")

param_meta = [
    ("amplitude",    TRUE_AMPLITUDE,    "Stimulus amplitude",
     [dist.Normal(PRIOR_AMP_MEAN, PRIOR_AMP_STD[k]) for k in SCENARIO_KEYS],
     amp_vals),
    ("excitability", TRUE_EXCITABILITY, "Excitability (I)",
     [dist.Normal(PRIOR_EXC_MEAN, PRIOR_EXC_STD[k]) for k in SCENARIO_KEYS],
     excit_vals),
]

for row, (param, true_val, xlabel, priors, grid) in enumerate(param_meta):
    x = jnp.linspace(float(grid.min()) - 0.05, float(grid.max()) + 0.05, 300)
    for col, (samples, key, prior, color) in enumerate(
        zip(all_samples, SCENARIO_KEYS, priors, COLORS)
    ):
        ax  = axes[row, col]
        kde = gaussian_kde(samples[param])
        ax.plot(x, jnp.exp(prior.log_prob(x)), color="gray", ls="--", lw=1.5, label="Prior")
        ax.plot(x, kde(x), color=color, lw=2, label="Posterior")
        ax.fill_between(x, kde(x), alpha=0.25, color=color)
        ax.axvline(true_val, color="k", lw=1.5, label=f"True={true_val}")
        if row == 0:
            ax.set_title(SCENARIO_LABELS[key], fontsize=8)
        if col == 0:
            ax.set_ylabel(xlabel, fontsize=9)
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=7)

fig.suptitle("Marginal posteriors — hypotheses select different mechanistic explanations", fontsize=11)
fig.tight_layout()
plt.show()

# %% Posterior predictive
# Semi-transparent traces from posterior draws → uncertainty spread;
# posterior-mean prediction as solid line → point summary.
N_PP_DRAWS = 2 * N_DEVICES  # must be divisible into N_DEVICES × n_vmap

pp_traces = {}
pp_means  = {}

for key, samples in zip(SCENARIO_KEYS, all_samples):
    n_total  = len(samples["amplitude"])
    draw_idx = jnp.linspace(0, n_total - 1, N_PP_DRAWS).astype(int)

    net_pp = build_network(TRUE_AMPLITUDE, TRUE_EXCITABILITY)
    sf_pp, cfg_pp = prepare(net_pp, Heun(), t0=0.0, t1=T1, dt=DT)

    cfg_pp.external.stimulus.amplitude = DataAxis(samples["amplitude"][draw_idx])
    cfg_pp.dynamics.I                  = DataAxis(samples["excitability"][draw_idx])
    space_pp = Space(cfg_pp, mode="zip")

    par_pp = ParallelExecution(
        model=lambda cfg: sf_pp(cfg).ys[:, 0, 0],
        space=space_pp, n_pmap=N_DEVICES, n_vmap=N_PP_DRAWS // N_DEVICES,
    )
    results = par_pp.run()
    pp_traces[key] = jnp.array([results[i] for i in range(N_PP_DRAWS)])
    pp_means[key]  = (float(jnp.mean(samples["amplitude"])),
                      float(jnp.mean(samples["excitability"])))

print("Posterior predictive simulations complete.")

# %% Plot posterior predictive
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)

for ax, key, color in zip(axes, SCENARIO_KEYS, COLORS):
    for i in range(N_PP_DRAWS):
        ax.plot(ts, pp_traces[key][i], color=color, alpha=0.1, lw=0.8)
    amp_mean, exc_mean = pp_means[key]
    r_mean = nd_solve(build_network(amp_mean, exc_mean), Heun(), t0=0.0, t1=T1, dt=DT)
    ax.plot(ts, r_mean.ys[:, 0, 0], color=color, lw=2.0,
            label=f"posterior mean (amp={amp_mean:.2f}, I={exc_mean:.2f})")
    ax.scatter(ts_obs, v_obs, s=12, color="k", zorder=5, label="observations")
    ax.axvspan(ONSET, ONSET + DURATION, alpha=0.15, color="tab:red")
    ax.set_ylabel("V")
    ax.set_title(SCENARIO_LABELS[key], fontsize=10)
    ax.legend(fontsize=8)

axes[-1].set_xlabel("Time (ms)")
fig.suptitle("Posterior predictive — hypotheses yield different mechanistic fits", fontsize=11)
fig.tight_layout()
plt.show()

# %% Multi-start optimisation
# Gradient descent from prior-sampled starts reveals:
# (a) each run converges to a single point on the ridge — no uncertainty
# (b) the starting distribution (prior) determines which ridge segment is found
# (c) there is no principled uncertainty quantification
N_SAMPLES     = 12 * N_DEVICES
N_OPTIM_STEPS = 1000

key_opt     = jax.random.key(99)
opt_results = {}

for scenario in SCENARIO_KEYS:
    key_opt, k_amp, k_exc = jax.random.split(key_opt, 3)
    amp_starts = PRIOR_AMP_MEAN + PRIOR_AMP_STD[scenario] * jax.random.normal(k_amp, (N_SAMPLES,))
    exc_starts = PRIOR_EXC_MEAN + PRIOR_EXC_STD[scenario] * jax.random.normal(k_exc, (N_SAMPLES,))

    net_multi = build_network(PRIOR_AMP_MEAN, PRIOR_EXC_MEAN)
    sf_multi, cfg_multi = prepare(net_multi, Heun(), t0=0.0, t1=T1, dt=DT)

    cfg_multi.external.stimulus.amplitude = DataAxis(amp_starts)
    cfg_multi.dynamics.I = DataAxis(exc_starts)
    space_opt = Space(cfg_multi, mode="zip")

    # optimizer = OptaxOptimizer(make_loss(sf_multi), optax.adamax(learning_rate=0.1))
    optimizer = OptaxOptimizer(make_loss(sf_multi), optax.adam(learning_rate=0.1))

    def run_optim(config):
        config.external.stimulus.amplitude = Parameter(config.external.stimulus.amplitude)
        config.dynamics.I = Parameter(config.dynamics.I)
        p_fit, _ = optimizer.run(config, max_steps=N_OPTIM_STEPS, chunk_size=N_OPTIM_STEPS)
        return jnp.array([
            collect_parameters(p_fit.external.stimulus.amplitude),
            collect_parameters(p_fit.dynamics.I),
        ])

    par_opt = ParallelExecution(
        model=run_optim, space=space_opt,
        n_pmap=N_DEVICES, n_vmap=N_SAMPLES // N_DEVICES,
    )
    results_opt = par_opt.run()

    final_params = jnp.array([results_opt[i] for i in range(N_SAMPLES)])
    opt_results[scenario] = {
        "amplitude":    final_params[:, 0],
        "excitability": final_params[:, 1],
    }
    print(f"Scenario {scenario}: {N_SAMPLES} optimisations complete.")


# %% Plot: multi-start optimisation results on landscape
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax, key, color in zip(axes, SCENARIO_KEYS, COLORS):
    pcm = _draw_landscape(ax)
    ax.scatter(opt_results[key]["amplitude"], opt_results[key]["excitability"],
               s=15, alpha=0.6, color=color, edgecolors="white", linewidths=0.3,
               zorder=5, label=f"optimised ({N_SAMPLES} runs)")
    ax.axvline(PRIOR_AMP_MEAN, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.axhline(PRIOR_EXC_MEAN, color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.set_title(SCENARIO_LABELS[key], fontsize=9)
    ax.legend(fontsize=7)

axes[0].set_ylabel("Excitability (I)")
fig.colorbar(pcm, ax=axes[-1], label="MSE(V)")
fig.suptitle(
    f"Multi-start optimisation ({N_OPTIM_STEPS} Adam steps) — starting points sampled from priors",
    fontsize=11,
)
fig.tight_layout()
plt.show()

# %% Marginal comparison: Bayesian posteriors vs optimisation endpoints
# Filter optimisation results to landscape range (converged runs only)
amp_range = (float(amp_vals.min()),   float(amp_vals.max()))
exc_range = (float(excit_vals.min()), float(excit_vals.max()))

opt_converged = {}
for key in SCENARIO_KEYS:
    amp  = opt_results[key]["amplitude"]
    exc  = opt_results[key]["excitability"]
    mask = (
        (amp >= amp_range[0]) & (amp <= amp_range[1]) &
        (exc >= exc_range[0]) & (exc <= exc_range[1])
    )
    opt_converged[key] = {
        "amplitude":    amp[mask],
        "excitability": exc[mask],
        "pct": float(mask.sum()) / len(mask) * 100,
    }

fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex="row")

param_meta_cmp = [
    ("amplitude",    TRUE_AMPLITUDE,    "Stimulus amplitude",
     [dist.Normal(PRIOR_AMP_MEAN, PRIOR_AMP_STD[k]) for k in SCENARIO_KEYS],
     amp_range),
    ("excitability", TRUE_EXCITABILITY, "Excitability (I)",
     [dist.Normal(PRIOR_EXC_MEAN, PRIOR_EXC_STD[k]) for k in SCENARIO_KEYS],
     exc_range),
]

for row, (param, true_val, xlabel, priors, hist_range) in enumerate(param_meta_cmp):
    x = jnp.linspace(hist_range[0] - 0.05, hist_range[1] + 0.05, 300)
    for col, (samples, key, prior, color) in enumerate(
        zip(all_samples, SCENARIO_KEYS, priors, COLORS)
    ):
        ax        = axes[row, col]
        kde_bayes = gaussian_kde(samples[param])
        opt_vals  = opt_converged[key][param]

        ax.plot(x, jnp.exp(prior.log_prob(x)), color="gray", ls="--", lw=1.5, label="Prior")
        ax.plot(x, kde_bayes(x), color=color, lw=2, label="Bayesian posterior")
        ax.fill_between(x, kde_bayes(x), alpha=0.15, color=color)
        if len(opt_vals) >= 1:
            ax.hist(opt_vals, bins=25, range=hist_range, density=True,
                    color=color, alpha=0.35, edgecolor=color, linewidth=0.8,
                    label=f"Optimisation ({opt_converged[key]['pct']:.0f}% converged)")
        ax.axvline(true_val, color="k", lw=1.5, label=f"True={true_val}")
        if row == 0:
            ax.set_title(SCENARIO_LABELS[key], fontsize=8)
        if col == 0:
            ax.set_ylabel(xlabel, fontsize=9)
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=9)
        ax.legend(fontsize=6)

fig.suptitle("Bayesian posterior (solid) vs optimisation endpoints (histogram)", fontsize=11)
fig.tight_layout()
plt.show()

# %% Single-trajectory optimisation — detailed view with loss curve
N_ADAM_STEPS = 250

net_opt = build_network(PRIOR_AMP_MEAN, PRIOR_EXC_MEAN)
sf_opt, cfg_opt = prepare(net_opt, Heun(), t0=0.0, t1=T1, dt=DT)

cfg_opt.external.stimulus.amplitude = Parameter(PRIOR_AMP_MEAN)
cfg_opt.dynamics.I                  = Parameter(PRIOR_EXC_MEAN)

cbs = MultiCallback([SavingLossCallback(), SavingParametersCallback()])
optimizer = OptaxOptimizer(make_loss(sf_opt), optax.adamax(learning_rate=0.01), callback=cbs)
cfg_final, fitting_data = optimizer.run(cfg_opt, max_steps=N_ADAM_STEPS)

opt_amp = float(collect_parameters(cfg_final.external.stimulus.amplitude))
opt_exc = float(collect_parameters(cfg_final.dynamics.I))
print(f"Optimised: amplitude={opt_amp:.4f}, excitability={opt_exc:.4f}")

param_history = fitting_data["parameters"].save
amp_traj     = jnp.array([float(collect_parameters(p.external.stimulus.amplitude)) for p in param_history])
exc_traj     = jnp.array([float(collect_parameters(p.dynamics.I)) for p in param_history])
loss_history = fitting_data["loss"]["save"].values

fig, (ax_land, ax_loss) = plt.subplots(1, 2, figsize=(12, 5),
                                        gridspec_kw={"width_ratios": [1.2, 1]})

pcm = _draw_landscape(ax_land)
fig.colorbar(pcm, ax=ax_land, label="MSE(V)")
ax_land.plot(amp_traj, exc_traj, color="red", lw=1.5, alpha=0.7)
ax_land.scatter(amp_traj[0], exc_traj[0], color="red", marker="o", s=80, zorder=7,
                label=f"start (amp={PRIOR_AMP_MEAN}, I={PRIOR_EXC_MEAN})")
ax_land.scatter(opt_amp, opt_exc, color="red", marker="X", s=120, zorder=7,
                label=f"optimised (amp={opt_amp:.2f}, I={opt_exc:.2f})")
ax_land.set_title("Gradient descent trajectory on degeneracy landscape")
ax_land.legend(fontsize=7)

ax_loss.plot(loss_history, color="red", lw=1.5)
ax_loss.set_xlabel("Adam step")
ax_loss.set_ylabel("MSE loss")
ax_loss.set_title("Optimisation loss")

fig.suptitle("Point optimisation finds one solution — no uncertainty, no hypothesis integration",
             fontsize=11)
fig.tight_layout()
plt.show()

# %% Summary table: Bayesian posterior vs multi-start optimisation statistics
SHORT_NAMES = {"A": "No hypothesis", "B": "H1: Higher excitability", "C": "H2: Stronger stimulus"}

col_headers = [
    "Scenario", "Parameter",
    "Bayes mean (std)", "Bayes median (IQR)",
    "Optim mean (std)", "Optim median (IQR)",
]

rows = []
for key, samples in zip(SCENARIO_KEYS, all_samples):
    for param in ["amplitude", "excitability"]:
        bayes_s = np.array(samples[param])
        opt_s   = np.array(opt_converged[key][param])
        has_opt = len(opt_s) > 0
        rows.append([
            SHORT_NAMES[key], param,
            f"{np.mean(bayes_s):.3f} ({np.std(bayes_s):.3f})",
            f"{np.median(bayes_s):.3f} ({iqr(bayes_s):.3f})",
            f"{np.mean(opt_s):.3f} ({np.std(opt_s):.3f})"     if has_opt else "—",
            f"{np.median(opt_s):.3f} ({iqr(opt_s):.3f})"      if has_opt else "—",
        ])

fig, ax = plt.subplots(figsize=(8, 2), dpi=200)
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.auto_set_column_width(col=list(range(len(col_headers))))
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#404040")
        cell.set_text_props(color="white", fontweight="bold")
    elif r > 0:
        color = COLORS[(r - 1) // 2]
        cell.set_facecolor(color)
        cell.set_alpha(0.25)
fig.suptitle("Summary: Bayesian posterior vs multi-start optimisation  (true: amp=0.4, I=0.1)",
             fontsize=11)
fig.tight_layout()
plt.show()

# %%
