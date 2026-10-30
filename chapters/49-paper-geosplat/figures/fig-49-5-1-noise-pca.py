"""Figure 49.5.1: GeoSplat's tangential kernel matrix K versus local PCA under position noise (Section 49.5).

20000 uniform points on the unit sphere, each moved by isotropic Gaussian noise of standard deviation sigma.
At 200 random points q (themselves noisy) we compute three normals and their angle to the true normal:
  local PCA:           smallest eigenvector of the covariance of the 30 nearest neighbours (centred at their mean);
  GeoSplat K (43)+(45): smallest eigenvector of K = (1/2) sum_j a_j (x_j - q)(x_j - q)^T with random-walk weights
                       a_j = (4/t) k_j / sum k, k_j = exp(-|x_j - q|^2 / t), t = 0.003 (about 30 effective neighbours);
  kernel PCA:          the same weights, centred at the weighted mean instead of q.
(a) median angle (degrees) against sigma.  (b) 90th percentile.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-5-1-noise-pca.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
N, t = 20000, 0.003
X0 = geo49.sphere_points(N, seed=3, pole=False)
rng = np.random.default_rng(0)
Q = rng.choice(N, 200, replace=False)
sigmas = np.array([0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])


def angle(n, n0):
    return np.degrees(np.arccos(min(1.0, abs(n @ n0))))


errs = {"local PCA (k = 30)": [], "GeoSplat K (centred at q)": [], "kernel PCA (centred at mean)": []}
for sig in sigmas:
    X = X0 + sig * rng.normal(size=X0.shape)
    e = {k: [] for k in errs}
    for i in Q:
        q, n0 = X[i], X0[i]
        e["local PCA (k = 30)"].append(angle(geo49.local_pca_normal(X, q, 30), n0))
        a = geo49.generator_weights(q, X, t)
        D = X - q
        K = 0.5 * np.einsum("j,ja,jb->ab", a, D, D)
        e["GeoSplat K (centred at q)"].append(angle(np.linalg.eigh(K)[1][:, 0], n0))
        w = a / a.sum()
        Dc = X - w @ X
        e["kernel PCA (centred at mean)"].append(angle(np.linalg.eigh(np.einsum("j,ja,jb->ab", w, Dc, Dc))[1][:, 0], n0))
    for k in errs:
        errs[k].append((np.median(e[k]), np.percentile(e[k], 90)))
errs = {k: np.array(v) for k, v in errs.items()}
for k, v in errs.items():
    print(k, np.round(v[:, 0], 2), np.round(v[:, 1], 2))
# self-checks: all three are below 1 degree without noise and grow to several degrees at sigma = 0.02;
# K is never better than both PCA variants by more than a factor 1.5 (no robustness advantage)
med = np.stack([v[:, 0] for v in errs.values()])
assert np.all(med[:, 0] < 1.0) and np.all(med[:, 4] > 3.0)
assert np.all(med[1] > med[[0, 2]].min(0) / 1.5)

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharex=True)
styles = [(C["aux"], "--", "s"), (C["main"], "-", "o"), (C["third"], "-.", "^")]
for j, title in enumerate(["(a) median angle error", "(b) 90th percentile"]):
    ax = axes[j]
    for (name, v), (col, ls, mk) in zip(errs.items(), styles):
        ax.plot(sigmas, v[:, j], color=col, ls=ls, marker=mk, ms=4, lw=1.5, label=name)
    ax.set_xlabel(r"noise $\sigma$ (unit sphere)")
    ax.set_ylabel("normal error (degrees)")
    ax.set_title(title, fontsize=10.5)
    ax.set_ylim(bottom=0)
axes[0].legend(loc="upper left", fontsize=7.5, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)
