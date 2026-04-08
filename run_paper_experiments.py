"""
run_paper_experiments.py
========================
Replicates all paper figures and tables using the production
`semi_supervised_gmm` package.

Strategy
--------
The simulation experiment runners, data generators, and plot functions live in
`discovered_materials/code/simulations.py`.  The core model-fitting calls
inside those runners use module-level functions (em_semisup, em_supervised,
etc.) that return bare GMMParams objects.

The production package returns (GMMParams, n_iter, converged) tuples and
exposes improved algorithms (Cholesky caching, analytic Fisher, precomputed
scatter).  We bridge the gap with thin adapter functions that strip the tuple
and forward kwargs, then monkeypatch `simulations` before running experiments.

Usage
-----
    python3 run_paper_experiments.py [--experiments all|e1|e2|e3|e4|e10 ...]

Outputs go to `discovered_materials/papers/figures/` as before.
"""

import argparse
import sys
import os

# Make both packages importable
REPO = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(REPO, "discovered_materials", "code")
sys.path.insert(0, REPO)
sys.path.insert(0, SIM_DIR)

# ---------------------------------------------------------------------------
# 1.  Import production package
# ---------------------------------------------------------------------------
import numpy as np
from semi_supervised_gmm._em import (
    em_semisup  as _prod_em_semisup,
    em_supervised as _prod_em_supervised,
    em_conf_weighted as _prod_em_conf_weighted,
    posterior as _prod_posterior,
)
from semi_supervised_gmm._lambda import (
    grid_search_lambda as _prod_grid_search,
    gradient_lambda    as _prod_gradient_lambda,
)
from semi_supervised_gmm._diagnostics import (
    alignment_A0      as _prod_alignment_A0,
    score_residual_g0 as _prod_score_residual_g0,
)

# ---------------------------------------------------------------------------
# 2.  Adapter functions (strip tuple → bare GMMParams to match old API)
# ---------------------------------------------------------------------------

def _em_semisup(X_pos, X_neg, X_u, lam=1.0, **kwargs):
    params, _, _ = _prod_em_semisup(X_pos, X_neg, X_u, lam=lam, warn=False, **kwargs)
    return params

def _em_supervised(X_pos, X_neg, **kwargs):
    params, _, _ = _prod_em_supervised(X_pos, X_neg, warn=False, **kwargs)
    return params

def _em_conf_weighted(X_pos, X_neg, X_u, lam=1.0, alpha=1.0, **kwargs):
    params, _, _ = _prod_em_conf_weighted(
        X_pos, X_neg, X_u, lam=lam, alpha=alpha, warn=False, **kwargs
    )
    return params

def _grid_search_lambda(X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
                        lam_grid=None, **kwargs):
    return _prod_grid_search(
        X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
        lam_grid=lam_grid, **kwargs
    )

def _gradient_lambda(X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
                     lam_init=1.0, lr=0.5, n_steps=10, **kwargs):
    return _prod_gradient_lambda(
        X_pos_tr, X_neg_tr, X_u, X_pos_val, X_neg_val,
        lam_init=lam_init, lr=lr, n_steps=n_steps, **kwargs
    )

# ---------------------------------------------------------------------------
# 3.  Monkeypatch simulations module
# ---------------------------------------------------------------------------
import simulations as sim

sim.em_semisup          = _em_semisup
sim.em_supervised       = _em_supervised
sim.em_conf_weighted    = _em_conf_weighted
sim.posterior           = _prod_posterior
sim.grid_search_lambda  = _grid_search_lambda
sim.gradient_lambda     = _gradient_lambda
sim.alignment_A0        = _prod_alignment_A0
sim.score_residual_g0   = _prod_score_residual_g0
sim.alignment_coefficient_mu = lambda X_pos, X_neg, X_u, X_val, y_val, eps_cov=1e-6: (
    _prod_alignment_A0(
        X_pos, X_neg, X_u,
        X_val[y_val == 1], X_val[y_val == 0],
        eps_cov=eps_cov,
    )[0]
)

print("Production package patched into simulations module.")
print(f"Figures will be saved to: {sim.FIGDIR}\n")

# ---------------------------------------------------------------------------
# 4.  Experiment registry
# ---------------------------------------------------------------------------

def run_e1():
    print("=" * 60)
    print("E1: Diagnostic validity (A(0) predicts AUROC gain)")
    print("=" * 60)
    b_lab  = 1.0
    deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    all_A0, all_gain, all_g0n, all_delta = sim.experiment_diagnostic(
        B=40, N1=15, N0=15, Nu=300, N_val=200, d=2,
        b_labeled=b_lab, deltas=deltas, lam_fixed=0.5,
    )
    sim.plot_diagnostic(all_A0, all_gain, all_g0n, all_delta, b_labeled=b_lab)


def run_e2():
    print("=" * 60)
    print("E2: Lambda learning (gradient ascent vs grid search)")
    print("=" * 60)
    results, lam_ess, lam_grid, lam_grad = sim.experiment_lambda_learning(
        B=100, N1=40, N0=40, Nu=600, N_val=40, d=5,
    )
    sim.plot_lambda_learning(results, lam_ess, lam_grid, lam_grad)
    sim.print_lambda_table(results, lam_ess, lam_grid, lam_grad)


def run_e3():
    print("=" * 60)
    print("E3: Misspecification (||g_0|| as risk indicator)")
    print("=" * 60)
    table, sg0, sdeg, records = sim.experiment_misspec(
        B=200, N1=50, N0=50, Nu=1000, N_val=50, d=5,
    )
    sim.plot_misspec(sg0, sdeg, records)
    sim.print_misspec_table(table)


def run_e4():
    print("=" * 60)
    print("E4: Bias–variance tradeoff as a function of lambda")
    print("=" * 60)
    summary, lam_grid = sim.experiment_bias_variance(
        B=300, N1=20, N0=20, Nu=500, d=5,
    )
    sim.plot_bias_variance(summary, lam_grid)


def run_e5():
    print("=" * 60)
    print("E5: Local lambda trajectory")
    print("=" * 60)
    corrs, data = sim.experiment_local_lambda(
        B=60, N1=15, N0=15, Nu=300, N_val=200, d=2,
    )
    sim.plot_local_lambda(corrs, data)


def run_e6():
    print("=" * 60)
    print("E6: Lambda parameter trajectory")
    print("=" * 60)
    results, A0_mean, lam_grid = sim.experiment_lam_trajectory(
        B=80, N1=15, N0=15, Nu=300, N_val=200, d=2,
    )
    sim.plot_lam_trajectory(results, A0_mean, lam_grid)


def run_e7():
    print("=" * 60)
    print("E7: Decision accuracy (regime grid)")
    print("=" * 60)
    records = sim.experiment_decision_accuracy(
        B=100, N1=15, N0=15, Nu=300, N_val=200, d=2,
    )
    sim.plot_decision_accuracy(records)


def run_e8():
    print("=" * 60)
    print("E8: Confidence-weighting ablation")
    print("=" * 60)
    records = sim.experiment_conf_weighting_ablation()
    sim.plot_conf_ablation(records)


def run_e9():
    print("=" * 60)
    print("E9: Finite-sample regime grid")
    print("=" * 60)
    rows = sim.experiment_regime_grid(B=50)
    sim.plot_regime_grid(rows)


def run_e10():
    print("=" * 60)
    print("E10: High-dimensional geometry sweep")
    print("=" * 60)
    records = sim.experiment_dimension_sweep(
        N1=20, N0=20, Nu=500, N_val=200, N_test=3000, B=60,
        lam=0.5, alpha=2.0,
    )
    sim.plot_dimension_sweep(records)


def run_pr():
    print("=" * 60)
    print("PR: Parameter recovery under correct specification")
    print("=" * 60)
    rows = sim.experiment_parameter_recovery(B=200, N1=50, N0=50, d=5)
    sim.print_param_recovery_table(rows)
    sim.plot_param_recovery(rows)


def run_sens():
    print("=" * 60)
    print("Sens: Sensitivity to lambda and alpha")
    print("=" * 60)
    results = sim.experiment_sensitivity(B=100, N1=40, N0=40, Nu=600, N_val=40, d=5)
    sim.plot_sensitivity(results)


EXPERIMENTS = {
    "e1":   run_e1,
    "e2":   run_e2,
    "e3":   run_e3,
    "e4":   run_e4,
    "e5":   run_e5,
    "e6":   run_e6,
    "e7":   run_e7,
    "e8":   run_e8,
    "e9":   run_e9,
    "e10":  run_e10,
    "pr":   run_pr,
    "sens": run_sens,
}

# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Replicate paper figures using the production semi_supervised_gmm package."
    )
    parser.add_argument(
        "--experiments", nargs="+", default=["all"],
        help=(
            "Which experiments to run. "
            f"Options: all, {', '.join(EXPERIMENTS)}. "
            "Default: all."
        ),
    )
    args = parser.parse_args()

    to_run = list(EXPERIMENTS.keys()) if "all" in args.experiments else args.experiments

    invalid = set(to_run) - set(EXPERIMENTS)
    if invalid:
        print(f"Unknown experiments: {invalid}. Valid: {set(EXPERIMENTS)}")
        sys.exit(1)

    for key in to_run:
        EXPERIMENTS[key]()
        print()

    print(f"Done. Figures saved to {sim.FIGDIR}")
