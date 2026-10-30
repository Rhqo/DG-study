"""Figure 48.6.1: principal curvatures of a random point cloud on the torus T_{2,0.8} with the pipeline of [BLM22] §7.

Point cloud: N points of T_{2,0.8} (R = 2, r = 0.8), i.i.d. uniform with respect to area (seed 3), no noise.
Pipeline of [BLM22] Section 7.2 / 7.1 with one parameter k (= N_neigh): tangent planes by rho-weighted PCA of the
k nearest points, eps_i = distance to the k-th nearest point, equal masses, epsilon-WSFF-perp ((6.22), (7.2) with
its 1/eps), contraction with the PCA normal and restriction to the tangent plane (Proposition 48.2.1).
The PCA normal has no sign; only for the comparison with signed exact values we orient it outward with the known
geometry (the paper cannot and does not).
(a) N = 20000, k = 40: the two eigenvalues at 4000 of the points against u, with the exact curves
    -1/r and -cos u / (R + r cos u).
(b) mean absolute error of kappa_min (solid) and kappa_max (dashed) against k, for N = 20000 and N = 80000.

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-6-1-torus.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var48 as v  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8
KS = [20, 40, 80, 160, 320]


def run(N):
    u, w = v.torus_sample(N, R, r, seed=3)
    X = v.torus_point(u, w, R, r)
    Nex = v.torus_outward(u, w)
    ex = np.sort(np.stack(v.torus_kappa_outward(u, R, r), 1), 1)
    m = np.ones(N)
    sub = np.arange(0, N, N // 4000)[:4000]
    idx_all, dist_all = v.knn(X, max(KS))
    out = {}
    for k in KS:
        idx, dist = idx_all[:, :k], dist_all[:, :k]
        P, nrm = v.pca_planes(X, k, idx=idx, dist=dist)
        nrm = nrm * np.sign(np.einsum("ij,ij->i", nrm, Nex))[:, None]
        kap = np.zeros((len(sub), 2))
        for a, i in enumerate(sub):
            Mx = v.shape_perp_fast(X[i], P[i], nrm[i], X[idx[i]], P[idx[i]], m[idx[i]], dist[i, -1] * (1 + 1e-9))
            Q = v.tangent_basis(nrm[i])
            W = Q.T @ Mx @ Q
            kap[a] = np.linalg.eigvalsh(0.5 * (W + W.T))
        ang = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nrm[sub], Nex[sub]), -1, 1))).mean()
        out[k] = dict(u=u[sub], kap=kap, err=np.abs(kap - ex[sub]).mean(0), eps=dist[sub, -1].mean(),
                      Kerr=np.abs(kap.prod(1) - ex[sub].prod(1)).mean(), ang=ang)
    return out


R20, R80 = run(20000), run(80000)
e20 = np.array([R20[k]["err"] for k in KS])
e80 = np.array([R80[k]["err"] for k in KS])

# self-checks
assert abs(R20[40]["eps"] - 0.197) < 0.003 and abs(R80[40]["eps"] - 0.0987) < 0.002
assert np.allclose(e20[1], [0.153, 0.056], atol=0.004) and np.allclose(e80[1], [0.160, 0.056], atol=0.004)
assert abs(e20[1, 0] - e80[1, 0]) < 0.01                       # fixed k: the error does not decrease with N
assert np.argmin(e20[:, 0]) == 3 and np.argmin(e80[:, 0]) == 4  # optimum at larger k for the denser cloud
assert R20[40]["ang"] < 1.1 and R80[40]["ang"] < 0.6

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"wspace": 0.36, "width_ratios": [1.15, 1.0]})
ax = axes[0]
d = R20[40]
uu = np.mod(d["u"] + np.pi, 2 * np.pi) - np.pi
ax.plot(uu, d["kap"][:, 0], ".", ms=1.2, color=C["tangent"], alpha=0.5, rasterized=True)
ax.plot(uu, d["kap"][:, 1], ".", ms=1.2, color=C["normal"], alpha=0.5, rasterized=True)
g = np.linspace(-np.pi, np.pi, 400)
a1, a2 = v.torus_kappa_outward(g, R, r)
ax.plot(g, a1, color=C["main"], lw=1.1)
ax.plot(g, a2, color=C["main"], lw=1.1, ls="--")
ax.set_xticks([-np.pi, 0, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
ax.set_xlabel(r"$u$")
ax.set_ylabel(r"$\kappa_{\min},\ \kappa_{\max}$")
ax.set_ylim(-2.1, 1.5)
ax.set_title(r"(a) $N = 20000$, $k = 40$", fontsize=10.5)
ax = axes[1]
ax.plot(KS, e20[:, 0], "o-", color=C["tangent"], ms=4, lw=1.1, label=r"$\kappa_{\min}$, $N = 20000$")
ax.plot(KS, e80[:, 0], "s-", color=C["tangent"], mfc="white", ms=4, lw=1.1, label=r"$\kappa_{\min}$, $N = 80000$")
ax.plot(KS, e20[:, 1], "o--", color=C["normal"], ms=4, lw=1.1, label=r"$\kappa_{\max}$, $N = 20000$")
ax.plot(KS, e80[:, 1], "s--", color=C["normal"], mfc="white", ms=4, lw=1.1, label=r"$\kappa_{\max}$, $N = 80000$")
ax.set_xscale("log")
ax.set_xticks(KS)
ax.set_xticklabels([str(k) for k in KS])
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"mean $|\kappa - \kappa_{\mathrm{exact}}|$")
ax.set_ylim(0, 0.27)
ax.legend(loc="upper right", fontsize=7.5, frameon=False, handlelength=1.8, labelspacing=0.25)
ax.set_title(r"(b) error against $k$", fontsize=10.5)
for a in axes:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

for name, Rr in (("N=20000", R20), ("N=80000", R80)):
    for k in KS:
        print(name, k, "eps %.4f" % Rr[k]["eps"], "err", np.round(Rr[k]["err"], 4), "K err %.4f" % Rr[k]["Kerr"],
              "angle %.2f" % Rr[k]["ang"])
dgfig.save(fig, __file__)
