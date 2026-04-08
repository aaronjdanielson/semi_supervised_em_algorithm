"""
real_data_experiments.py
========================
Cross-dataset validation of the semi-supervised EM diagnostic.

For each dataset we:
  1. Load and preprocess features / binary labels.
  2. Subsample labeled data (N1=N0=n_lab) to create the low-label regime;
     all remaining samples become unlabeled.
  3. Fit R2 (supervised) and R3 (semi-supervised, grid λ).
  4. Compute A_μ(0) (mean-subspace alignment coefficient).
  5. Make a binary decision D = 1[A_μ(0) > 0].
  6. Record oracle decision O = 1[gain > 0].
  7. Report per-dataset decision accuracy, gain, and regret.

Datasets (all publicly available):
  - Breast Cancer Wisconsin   (sklearn built-in)
  - Ionosphere                (UCI, fetched via sklearn)
  - Spambase                  (UCI)
  - Banknote Authentication   (UCI)
  - Sonar                     (UCI)

Usage:
  python real_data_experiments.py

Dependencies: numpy, scipy, scikit-learn, pandas, matplotlib
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import core algorithm from simulations.py (same directory)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from simulations import (
    em_supervised, em_semisup, em_conf_weighted, posterior,
    alignment_coefficient_mu,          # mean-subspace A(0)
    GMMParams,
)


# ---------------------------------------------------------------------------
# Neural learned λ(x): MLP confidence weighting
# ---------------------------------------------------------------------------

def neural_weights(X_pos, X_neg, X_u, hidden=(32,), alpha=2.0, seed=0):
    """
    Train a small MLP on labeled data to estimate P(z=1|x); use
    (max(p, 1-p))^alpha as a per-point weight for unlabeled observations.

    Unlike the Gaussian confidence weight, the MLP is nonparametric and works
    in high dimensions when the Gaussian assumption is violated.
    Returns a weight vector of shape (Nu,).
    """
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler as SS
    X_lab = np.vstack([X_pos, X_neg])
    y_lab = np.array([1]*len(X_pos) + [0]*len(X_neg))
    scaler = SS().fit(X_lab)
    clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=800,
                        random_state=seed, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=15)
    try:
        clf.fit(scaler.transform(X_lab), y_lab)
        p = clf.predict_proba(scaler.transform(X_u))[:, 1]
    except Exception:
        p = np.full(len(X_u), 0.5)
    conf = np.maximum(p, 1.0 - p)
    return conf ** alpha


def em_neural_weighted(X_pos, X_neg, X_u, lam=1.0, alpha=2.0,
                        hidden=(32,), seed=0, **kwargs):
    """
    Semi-supervised EM weighted by neural confidence estimate.
    Weights are computed once from the supervised MLP, then fixed (MM step).
    """
    w = neural_weights(X_pos, X_neg, X_u, hidden=hidden, alpha=alpha, seed=seed)
    lam_j = lam * w

    # One-shot weighted EM with fixed lam_j
    from simulations import _logpdf_mvn, _soft_counts as _sc
    eps_cov  = kwargs.get("eps_cov", 1e-6)
    resp_clip = kwargs.get("resp_clip", 1e-6)
    tol      = kwargs.get("tol", 5e-6)
    max_iter = kwargs.get("max_iter", 300)

    N1, d = X_pos.shape
    N0    = X_neg.shape[0]
    mu1   = X_pos.mean(0)
    mu0   = X_neg.mean(0)
    S1    = np.cov(X_pos, rowvar=False) + eps_cov * np.eye(d)
    S0    = np.cov(X_neg, rowvar=False) + eps_cov * np.eye(d)
    pi    = N1 / (N1 + N0)

    for _ in range(max_iter):
        lp1 = _logpdf_mvn(X_u, mu1, S1) + np.log(np.clip(pi, 1e-12, 1-1e-12))
        lp0 = _logpdf_mvn(X_u, mu0, S0) + np.log(np.clip(1-pi, 1e-12, 1-1e-12))
        m   = np.maximum(lp1, lp0)
        gam = np.clip(np.exp(lp1 - m - np.log(np.exp(lp1-m)+np.exp(lp0-m))),
                      resp_clip, 1-resp_clip)

        N1t = N1 + np.sum(lam_j * gam)
        N0t = N0 + np.sum(lam_j * (1-gam))
        Nt  = N1 + N0 + np.sum(lam_j)

        mu1n = (X_pos.sum(0) + ((lam_j*gam)[:,None]*X_u).sum(0)) / N1t
        mu0n = (X_neg.sum(0) + ((lam_j*(1-gam))[:,None]*X_u).sum(0)) / N0t

        Xp = X_pos - mu1n; Xu1 = X_u - mu1n
        S1n = (Xp.T@Xp + (Xu1*(lam_j*gam)[:,None]).T@Xu1)/N1t + eps_cov*np.eye(d)
        Xn = X_neg - mu0n; Xu0 = X_u - mu0n
        S0n = (Xn.T@Xn + (Xu0*(lam_j*(1-gam))[:,None]).T@Xu0)/N0t + eps_cov*np.eye(d)
        pi_n = N1t / Nt

        delta = max(np.max(np.abs(mu1n-mu1)), np.max(np.abs(mu0n-mu0)),
                    np.max(np.abs(S1n-S1)), np.max(np.abs(S0n-S0)), abs(pi_n-pi))
        mu1, mu0, S1, S0, pi = mu1n, mu0n, S1n, S0n, pi_n
        if delta < tol:
            break

    return GMMParams(pi=pi, mu0=mu0, mu1=mu1, Sigma0=S0, Sigma1=S1)

FIGDIR = Path(__file__).parent.parent / "papers" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

DATADIR = Path(__file__).parent.parent / "data"
DATADIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_breast_cancer():
    data = load_breast_cancer()
    X, y = data.data, data.target   # 1=malignant, 0=benign → keep as-is
    return X, y, "Breast Cancer"


def _load_ionosphere():
    """UCI Ionosphere — fetch or load from cache."""
    cache = DATADIR / "ionosphere.csv"
    if not cache.exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data"
        try:
            df = pd.read_csv(url, header=None)
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"  [ionosphere] download failed: {e}")
            return None, None, None
    df = pd.read_csv(cache, header=None)
    y = (df.iloc[:, -1] == "g").astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    # Drop column 1 (constant zero)
    X = np.delete(X, 1, axis=1)
    return X, y, "Ionosphere"


def _load_banknote():
    """UCI Banknote Authentication."""
    cache = DATADIR / "banknote.csv"
    if not cache.exists():
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00267/data_banknote_authentication.txt")
        try:
            df = pd.read_csv(url, header=None)
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"  [banknote] download failed: {e}")
            return None, None, None
    df = pd.read_csv(cache, header=None)
    X = df.iloc[:, :4].values.astype(float)
    y = df.iloc[:, 4].values.astype(int)
    return X, y, "Banknote Auth"


def _load_sonar():
    """UCI Sonar (mines vs rocks)."""
    cache = DATADIR / "sonar.csv"
    if not cache.exists():
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "undocumented/connectionist-bench/sonar/sonar.all-data")
        try:
            df = pd.read_csv(url, header=None)
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"  [sonar] download failed: {e}")
            return None, None, None
    df = pd.read_csv(cache, header=None)
    y = (df.iloc[:, -1] == "M").astype(int).values
    X = df.iloc[:, :-1].values.astype(float)
    return X, y, "Sonar"


def _load_spambase():
    """UCI Spambase."""
    cache = DATADIR / "spambase.csv"
    if not cache.exists():
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "spambase/spambase.data")
        try:
            df = pd.read_csv(url, header=None)
            df.to_csv(cache, index=False)
        except Exception as e:
            print(f"  [spambase] download failed: {e}")
            return None, None, None
    df = pd.read_csv(cache, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(int)
    return X, y, "Spambase"


DATASET_LOADERS = [
    _load_breast_cancer,
    _load_ionosphere,
    _load_banknote,
    _load_sonar,
    _load_spambase,
]


# ---------------------------------------------------------------------------
# PCA projection to low-d for Gaussian mixture validity
# ---------------------------------------------------------------------------

def pca_reduce(X, n_components=10):
    """Project X to top-k PCA components (standardized first)."""
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_comp = min(n_components, Xs.shape[1], Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=0)
    return pca.fit_transform(Xs)


# ---------------------------------------------------------------------------
# Single-dataset experiment
# ---------------------------------------------------------------------------

def run_one_dataset(
    X, y, name,
    n_lab=25,            # labeled per class
    n_val=30,            # validation per class (for AUROC)
    n_test=500,          # held-out test
    lam_grid=None,
    B=20,                # subsampling replications
    seed0=42,
    pca_dims=8,
):
    if lam_grid is None:
        lam_grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

    rng = np.random.default_rng(seed0)

    # Balance classes
    idx1 = np.where(y == 1)[0]
    idx0 = np.where(y == 0)[0]
    n_per_class = min(len(idx1), len(idx0))
    if n_per_class < n_lab + n_val + 10:
        print(f"  [{name}] too few samples per class ({n_per_class}), skipping")
        return None

    # PCA to make Gaussian assumption more plausible
    X_red = pca_reduce(X, n_components=pca_dims)

    records = []
    for b in range(B):
        # Random split per replication
        shuf1 = rng.permutation(len(idx1))
        shuf0 = rng.permutation(len(idx0))

        i1 = idx1[shuf1]
        i0 = idx0[shuf0]

        n_needed = n_lab + n_val + max(n_test // 2, 50)
        if len(i1) < n_needed or len(i0) < n_needed:
            n_lab_actual = max(5, min(n_lab, len(i1) - n_val - 20))
        else:
            n_lab_actual = n_lab

        lab1  = i1[:n_lab_actual]
        lab0  = i0[:n_lab_actual]
        val1  = i1[n_lab_actual:n_lab_actual + n_val]
        val0  = i0[n_lab_actual:n_lab_actual + n_val]
        rest1 = i1[n_lab_actual + n_val:]
        rest0 = i0[n_lab_actual + n_val:]

        n_test_half = min(n_test // 2, len(rest1) // 2, len(rest0) // 2)
        test1 = rest1[:n_test_half]
        test0 = rest0[:n_test_half]
        unl1  = rest1[n_test_half:]
        unl0  = rest0[n_test_half:]

        X_pos  = X_red[lab1]
        X_neg  = X_red[lab0]
        X_val  = np.vstack([X_red[val1], X_red[val0]])
        y_val  = np.array([1]*len(val1) + [0]*len(val0))
        X_test = np.vstack([X_red[test1], X_red[test0]])
        y_test = np.array([1]*n_test_half + [0]*n_test_half)
        X_unl  = np.vstack([X_red[unl1], X_red[unl0]])

        if len(X_unl) < 10:
            continue

        # R2: supervised
        try:
            sup = em_supervised(X_pos, X_neg)
            auc_sup = roc_auc_score(y_test, posterior(X_test, sup))
        except Exception:
            continue

        # Score residual norm |g_0| (pre-fitting misspecification indicator)
        try:
            from simulations import score_residual_g0
            _, g0_norm_per = score_residual_g0(X_pos, X_neg, X_unl)
        except Exception:
            g0_norm_per = np.nan

        # Alignment coefficient A_μ(0)
        try:
            A0 = alignment_coefficient_mu(X_pos, X_neg, X_unl, X_val, y_val)
        except Exception:
            A0 = np.nan

        # R3: semi-supervised, grid λ (global)
        best_lam, best_val_auc = lam_grid[0], -np.inf
        for lam in lam_grid:
            try:
                p = em_semisup(X_pos, X_neg, X_unl, lam=lam)
                val_auc = roc_auc_score(y_val, posterior(X_val, p))
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_lam = lam
            except Exception:
                pass

        try:
            p_ssl = em_semisup(X_pos, X_neg, X_unl, lam=best_lam)
            auc_ssl = roc_auc_score(y_test, posterior(X_test, p_ssl))
        except Exception:
            auc_ssl = auc_sup

        # R4: confidence-weighted (Gaussian) λ(x)
        try:
            best_lam_cw, best_val_auc_cw = best_lam, -np.inf
            for alpha in [1.0, 2.0, 3.0]:
                p = em_conf_weighted(X_pos, X_neg, X_unl, lam=best_lam, alpha=alpha)
                va = roc_auc_score(y_val, posterior(X_val, p))
                if va > best_val_auc_cw:
                    best_val_auc_cw = va
                    best_alpha_cw = alpha
            p_cw = em_conf_weighted(X_pos, X_neg, X_unl, lam=best_lam, alpha=best_alpha_cw)
            auc_cw = roc_auc_score(y_test, posterior(X_test, p_cw))
        except Exception:
            auc_cw = auc_sup

        # R5: neural learned λ(x) — MLP confidence weighting
        try:
            best_val_auc_nn, best_alpha_nn = -np.inf, 2.0
            for alpha in [1.0, 2.0]:
                p = em_neural_weighted(X_pos, X_neg, X_unl, lam=best_lam,
                                        alpha=alpha, seed=b)
                va = roc_auc_score(y_val, posterior(X_val, p))
                if va > best_val_auc_nn:
                    best_val_auc_nn = va
                    best_alpha_nn = alpha
            p_nn = em_neural_weighted(X_pos, X_neg, X_unl, lam=best_lam,
                                       alpha=best_alpha_nn, seed=b)
            auc_nn = roc_auc_score(y_test, posterior(X_test, p_nn))
        except Exception:
            auc_nn = auc_sup

        gain = auc_ssl - auc_sup
        D = int(A0 > 0) if not np.isnan(A0) else np.nan
        O = int(gain > 0)
        correct = int(D == O) if not np.isnan(D) else np.nan
        regret = abs(gain) * int(D != O) if not np.isnan(D) else np.nan

        records.append(dict(
            dataset=name, b=b,
            n_lab=n_lab_actual, n_unl=len(X_unl),
            A0=A0, g0_norm=g0_norm_per, lam_star=best_lam,
            auc_sup=auc_sup, auc_ssl=auc_ssl, auc_cw=auc_cw, auc_nn=auc_nn,
            gain_ssl=gain,
            gain_cw=auc_cw - auc_sup,
            gain_nn=auc_nn - auc_sup,
            D=D, O=O, correct=correct, regret=regret,
        ))

    return records


# ---------------------------------------------------------------------------
# Diagnostic: cross-dataset summary table
# ---------------------------------------------------------------------------

def print_cross_dataset_table(all_records):
    """Print the cross-dataset diagnostic decision table."""
    print("\n" + "=" * 80)
    print("CROSS-DATASET DIAGNOSTIC DECISION TABLE")
    print("=" * 80)
    print(f"{'Dataset':<22} {'N_lab':>6} {'|g0|/Nu':>8} "
          f"{'R3 gain':>9} {'R4 gain':>9} {'R5 gain':>9} "
          f"{'D acc':>7} {'n_rep':>6}")
    print("-" * 90)

    summary = []
    for name in sorted(set(r["dataset"] for r in all_records)):
        recs = [r for r in all_records if r["dataset"] == name
                and r["correct"] is not None and not np.isnan(r["correct"] or 0)]
        if not recs:
            continue
        n_lab      = int(np.median([r["n_lab"] for r in recs]))
        g0_med     = np.nanmedian([r.get("g0_norm", np.nan) for r in recs])
        frac_A_pos = np.mean([r["D"] for r in recs if r["D"] is not None])
        gain_ssl   = np.mean([r.get("gain_ssl", r.get("gain", 0)) for r in recs])
        gain_cw    = np.mean([r.get("gain_cw", 0) for r in recs])
        gain_nn    = np.mean([r.get("gain_nn", 0) for r in recs])
        acc        = np.mean([r["correct"] for r in recs if r["correct"] is not None])
        regret     = np.mean([r["regret"] for r in recs if r["regret"] is not None])
        print(f"  {name:<20} {n_lab:>6} {g0_med:>8.2f} "
              f"{gain_ssl:>+9.4f} {gain_cw:>+9.4f} {gain_nn:>+9.4f} "
              f"{acc:>7.2f} {len(recs):>6}")
        summary.append(dict(name=name, n_lab=n_lab, g0_norm=g0_med,
                             frac_A_pos=frac_A_pos,
                             gain_ssl=gain_ssl, gain_cw=gain_cw, gain_nn=gain_nn,
                             acc=acc, regret=regret, n_rep=len(recs)))

    print("-" * 80)
    print("-" * 90)
    overall_acc = np.mean([r["correct"] for r in all_records
                            if r["correct"] is not None
                            and not np.isnan(r["correct"] or 0)])
    overall_reg = np.mean([r["regret"] for r in all_records
                            if r["regret"] is not None
                            and not np.isnan(r["regret"] or 0)])
    print(f"  {'OVERALL':<20} {'':>6} {'':>8} {'':>8} {'':>10} "
          f"{overall_acc:>7.2f} {overall_reg:>8.5f} {len(all_records):>6}")
    # Stratify by |g0|: low vs high
    print("\n  Stratified by |g_0|/N_u (median split):")
    g0_all = [r.get("g0_norm", np.nan) for r in all_records]
    g0_med_global = np.nanmedian(g0_all)
    for label, mask_fn in [("Low  |g0| (< median)", lambda r: (r.get("g0_norm",np.nan) or 1) < g0_med_global),
                            ("High |g0| (>= median)", lambda r: (r.get("g0_norm",np.nan) or 0) >= g0_med_global)]:
        sub = [r for r in all_records if mask_fn(r) and r["correct"] is not None
               and not np.isnan(r["correct"] or 0)]
        if sub:
            acc_s = np.mean([r["correct"] for r in sub])
            reg_s = np.mean([r["regret"]  for r in sub if r["regret"] is not None])
            print(f"    {label:<30s}  acc={acc_s:.2f}  regret={reg_s:.5f}  n={len(sub)}")
    return summary


def plot_cross_dataset(all_records, summary):
    """
    Three-panel figure:
    Left:  scatter A(0) vs R3 gain, decision quadrants, colored by dataset
    Centre: grouped bar chart — R3/R4/R5 AUROC gain per dataset
    Right:  per-dataset decision accuracy bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle("Cross-dataset evaluation: four methods vs supervised baseline",
                 fontsize=12, fontweight="bold")

    datasets = sorted(set(r["dataset"] for r in all_records))
    cmap = plt.cm.tab10
    ds_color = {ds: cmap(i / max(len(datasets) - 1, 1)) for i, ds in enumerate(datasets)}

    # --- Left: scatter A(0) vs gain ---
    ax = axes[0]
    for ds in datasets:
        recs = [r for r in all_records if r["dataset"] == ds
                and not np.isnan(r.get("A0") or np.nan)]
        if not recs:
            continue
        gains = [r.get("gain_ssl", r.get("gain", 0)) for r in recs]
        ax.scatter([r["A0"] for r in recs], gains,
                   s=20, alpha=0.45, color=ds_color[ds], label=ds, linewidths=0)
    ax.axhline(0, color="k", lw=0.9, ls="--", alpha=0.5)
    ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.5)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.fill_betweenx([0, max(ylim)], min(xlim), 0, alpha=0.05, color="#f39c12")
    ax.fill_betweenx([min(ylim), 0], 0, max(xlim), alpha=0.05, color="#e74c3c")
    ax.set_xlabel(r"$\mathcal{A}_\mu(0)$", fontsize=11)
    ax.set_ylabel("AUROC gain (R3 − R2)", fontsize=11)
    ax.set_title("Diagnostic sign vs realized gain", fontsize=11)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.7)

    # --- Centre: violin+jitter R3/R4/R5 gain — shows full replication distribution ---
    ax = axes[1]
    datasets_ord = [s["name"] for s in summary]
    method_keys   = ["gain_ssl", "gain_cw", "gain_nn"]
    method_names  = ["R3: global λ", "R4: Gaussian λ(x)", "R5: neural λ(x)"]
    method_colors = ["#3498db", "#9b59b6", "#e67e22"]
    n_m = len(method_keys)
    spacing = 0.28
    rng_jit = np.random.default_rng(7)
    for ds_idx, ds in enumerate(datasets_ord):
        recs_ds = [r for r in all_records if r["dataset"] == ds]
        for m_idx, (key, nm, col) in enumerate(zip(method_keys, method_names, method_colors)):
            vals = np.array([r.get(key, 0) for r in recs_ds
                             if r.get(key) is not None and not np.isnan(r.get(key, np.nan))])
            pos = ds_idx + (m_idx - 1) * spacing
            if len(vals) > 2:
                parts = ax.violinplot([vals], positions=[pos], widths=spacing * 0.85,
                                      showmeans=False, showmedians=True, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(col); pc.set_alpha(0.50)
                parts["cmedians"].set_color("k"); parts["cmedians"].set_linewidth(1.5)
            if len(vals) > 0:
                jitter = rng_jit.normal(0, spacing * 0.06, len(vals))
                ax.scatter(pos + jitter, vals, s=8, alpha=0.30, color="k",
                           linewidths=0, zorder=3)
    # Legend proxies
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=col, alpha=0.7, label=nm)
               for nm, col in zip(method_names, method_colors)]
    ax.legend(handles=handles, fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(datasets_ord)))
    ax.set_xticklabels(datasets_ord, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC gain over supervised (R2)", fontsize=10)
    ax.set_title("Method comparison per dataset\n(all replications)", fontsize=11)

    # --- Right: decision accuracy ---
    ax = axes[2]
    names = [s["name"] for s in summary]
    accs  = [s["acc"] for s in summary]
    bars  = ax.bar(names, accs,
                   color=[ds_color[n] for n in names],
                   edgecolor="k", linewidth=0.6, alpha=0.85)
    ax.axhline(0.5, color="gray", lw=1.2, ls="--", label="Chance (50%)")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Decision accuracy  D = sign(A(0)) == oracle", fontsize=10)
    ax.set_title("Diagnostic decision accuracy", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=9)
    ax.legend(fontsize=9)

    plt.tight_layout()
    out = FIGDIR / "fig_real_data.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("REAL-DATA CROSS-DATASET DIAGNOSTIC EVALUATION")
    print("=" * 60)

    all_records = []
    for loader in DATASET_LOADERS:
        X, y, name = loader()
        if X is None:
            print(f"  Skipping {name} (unavailable)")
            continue
        print(f"\n  Loading {name}: {X.shape[0]} samples, {X.shape[1]} features, "
              f"class balance {y.mean():.2f}")
        recs = run_one_dataset(X, y, name, n_lab=25, B=20, seed0=42)
        if recs:
            all_records.extend(recs)
            print(f"    {len(recs)} replications completed")

    if not all_records:
        print("No datasets loaded. Check network access or place cached CSVs in data/.")
    else:
        summary = print_cross_dataset_table(all_records)
        plot_cross_dataset(all_records, summary)
        print("\nDone.")
