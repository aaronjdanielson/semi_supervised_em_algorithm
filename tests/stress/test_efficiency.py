"""
Efficiency benchmarks for semi_supervised_gmm.
Run directly (not via pytest): python3 tests/stress/test_efficiency.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import time

from semi_supervised_gmm._em import em_semisup, em_semisup_multi
from semi_supervised_gmm._diagnostics import alignment_A0
from semi_supervised_gmm._lambda import grid_search_lambda

rng = np.random.default_rng(0)


def bench(fn, n_runs=5):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times), result


def make_binary_data(N1, N0, Nu, d):
    mu1 = np.zeros(d); mu1[0] = 2.0
    mu0 = -mu1
    X_pos = rng.multivariate_normal(mu1, np.eye(d), N1)
    X_neg = rng.multivariate_normal(mu0, np.eye(d), N0)
    X_u   = rng.multivariate_normal(np.zeros(d), np.eye(d), Nu)
    return X_pos, X_neg, X_u


def make_multi_data(K, N_per, Nu, d):
    Xs, classes = [], np.arange(K)
    for k in range(K):
        mu = np.zeros(d); mu[k % d] = 3.0 * (k + 1)
        Xs.append(rng.multivariate_normal(mu, np.eye(d), N_per))
    X_u = rng.multivariate_normal(np.zeros(d), np.eye(d)*3, Nu)
    return Xs, X_u, classes


def print_table(title, rows, col_headers):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    col_w = 14
    header = "  " + "".join(h.ljust(col_w) for h in col_headers)
    print(header)
    print("  " + "-" * (col_w * len(col_headers)))
    for row in rows:
        print("  " + "".join(str(v).ljust(col_w) for v in row))


# ---------------------------------------------------------------------------
# Benchmark 1: Binary EM vs N
# ---------------------------------------------------------------------------
rows = []
d, Nu = 5, 1000
for N in [50, 200, 500, 2000]:
    X_pos, X_neg, X_u = make_binary_data(N, N, Nu, d)
    mean_t, std_t, (_, n_iter, _) = bench(
        lambda Xp=X_pos, Xn=X_neg, Xu=X_u:
            em_semisup(Xp, Xn, Xu, lam=1.0, warn=False)
    )
    rows.append([f"N={N}", f"{mean_t*1000:.1f}ms", f"±{std_t*1000:.1f}", f"iters={n_iter}"])
print_table("Binary EM: scaling with N (d=5, Nu=1000)", rows,
            ["N per class", "mean time", "std", "n_iter"])

# ---------------------------------------------------------------------------
# Benchmark 2: Binary EM vs d
# ---------------------------------------------------------------------------
rows = []
N1, N0, Nu = 100, 100, 500
for d in [2, 5, 10, 20, 50]:
    X_pos, X_neg, X_u = make_binary_data(N1, N0, Nu, d)
    eps = max(1e-6, d / N1 * 1e-4)
    mean_t, std_t, (_, n_iter, _) = bench(
        lambda Xp=X_pos, Xn=X_neg, Xu=X_u, e=eps:
            em_semisup(Xp, Xn, Xu, lam=1.0, eps_cov=e, warn=False)
    )
    rows.append([f"d={d}", f"{mean_t*1000:.1f}ms", f"±{std_t*1000:.1f}", f"iters={n_iter}"])
print_table("Binary EM: scaling with d (N=100, Nu=500)", rows,
            ["d", "mean time", "std", "n_iter"])

# ---------------------------------------------------------------------------
# Benchmark 3: Multiclass EM vs K
# ---------------------------------------------------------------------------
rows = []
N_per, Nu, d = 50, 500, 5
for K in [2, 3, 5, 10]:
    Xs, X_u, classes = make_multi_data(K, N_per, Nu, d)
    mean_t, std_t, (_, n_iter, _) = bench(
        lambda Xs=Xs, Xu=X_u, cls=classes:
            em_semisup_multi(Xs, Xu, cls, lam=1.0, warn=False)
    )
    rows.append([f"K={K}", f"{mean_t*1000:.1f}ms", f"±{std_t*1000:.1f}", f"iters={n_iter}"])
print_table("Multiclass EM: scaling with K (N=50/class, Nu=500, d=5)", rows,
            ["K", "mean time", "std", "n_iter"])

# ---------------------------------------------------------------------------
# Benchmark 4: Multiclass EM vs d (K=3)
# ---------------------------------------------------------------------------
rows = []
K, N_per, Nu = 3, 50, 300
for d in [2, 5, 10, 20, 50]:
    Xs, X_u, classes = make_multi_data(K, N_per, Nu, d)
    eps = max(1e-6, d / N_per * 1e-4)
    mean_t, std_t, (_, n_iter, _) = bench(
        lambda Xs=Xs, Xu=X_u, cls=classes, e=eps:
            em_semisup_multi(Xs, Xu, cls, lam=1.0, eps_cov=e, warn=False)
    )
    rows.append([f"d={d}", f"{mean_t*1000:.1f}ms", f"±{std_t*1000:.1f}", f"iters={n_iter}"])
print_table("Multiclass EM: scaling with d (K=3, N=50/class, Nu=300)", rows,
            ["d", "mean time", "std", "n_iter"])

# ---------------------------------------------------------------------------
# Benchmark 5: Diagnostics alignment_A0 vs d
# ---------------------------------------------------------------------------
rows = []
N1, N0, Nu = 50, 50, 300
for d in [2, 5, 10, 20]:
    X_pos, X_neg, X_u = make_binary_data(N1, N0, Nu, d)
    X_pos_val, X_neg_val, _ = make_binary_data(20, 20, 0, d)
    mean_t, std_t, _ = bench(
        lambda Xp=X_pos, Xn=X_neg, Xu=X_u, Xpv=X_pos_val, Xnv=X_neg_val:
            alignment_A0(Xp, Xn, Xu, Xpv, Xnv)
    )
    rows.append([f"d={d}", f"{mean_t*1000:.1f}ms", f"±{std_t*1000:.1f}"])
print_table("Diagnostics alignment_A0: scaling with d (N=50, Nu=300)", rows,
            ["d", "mean time", "std"])

# ---------------------------------------------------------------------------
# Benchmark 6: Grid search lambda
# ---------------------------------------------------------------------------
N1, N0, Nu, d = 50, 50, 500, 5
X_pos, X_neg, X_u = make_binary_data(N1, N0, Nu, d)
X_pos_val, X_neg_val, _ = make_binary_data(20, 20, 0, d)
lam_grid = np.logspace(-2, 2, 20)
mean_t, std_t, _ = bench(
    lambda Xp=X_pos, Xn=X_neg, Xu=X_u, Xpv=X_pos_val, Xnv=X_neg_val:
        grid_search_lambda(Xp, Xn, Xu, Xpv, Xnv, lam_grid=lam_grid)
)
print(f"\n{'='*60}")
print(f"  Grid search lambda (N=50, Nu=500, d=5, 20 grid points)")
print(f"{'='*60}")
print(f"  Mean: {mean_t*1000:.1f}ms  ±{std_t*1000:.1f}ms")

# ---------------------------------------------------------------------------
# Convergence check
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  Convergence: well-separated data (sep=3, d=5, N=50, Nu=300)")
print(f"{'='*60}")
X_pos, X_neg, X_u = make_binary_data(50, 50, 300, 5)
_, n_iter, converged = em_semisup(X_pos, X_neg, X_u, lam=1.0, warn=False)
print(f"  n_iter={n_iter}, converged={converged} (expect <50)")

# ---------------------------------------------------------------------------
# Memory check
# ---------------------------------------------------------------------------
import tracemalloc
tracemalloc.start()
X_pos, X_neg, X_u = make_binary_data(5000, 5000, 5000, 20)
_, _, _ = em_semisup(X_pos, X_neg, X_u, lam=1.0, warn=False)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\n{'='*60}")
print(f"  Memory: N=5000, d=20 binary EM")
print(f"{'='*60}")
print(f"  Peak: {peak / 1e6:.1f} MB")

print("\n\nDone.")
