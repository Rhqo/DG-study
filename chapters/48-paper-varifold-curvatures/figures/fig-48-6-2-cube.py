"""Figure 48.6.2: the cube test of [BLM22] Figure 5, recomputed (our reproduction, not a copy of the paper's images).

Point cloud: the surface of [0, 1]^3 on the grid of spacing 1/60 (6 * 61^2 - 12 * 61 + 8 = 21602 points, the N of
[BLM22] §7.3), optionally with additive centred Gaussian noise of standard deviation sigma on every coordinate (seed 0).
Pipeline of [BLM22] §7.2 with one parameter k = N_neigh: rho-weighted PCA planes from the k nearest points,
eps_i = distance to the k-th nearest point, equal masses, epsilon-WSFF-perp, eigenvalues on the tangent plane.
Shown is S = |kappa_1| + |kappa_2| (independent of the sign of the normal), as in [BLM22] Figure 5 (d)-(f).
(a)-(d) S on the top face z = 1 (61 x 61 grid points, labelled by their noiseless positions) for
        (sigma, k) = (0, 40), (0.01, 40), (0.05, 40), (0.05, 150); common colour scale 0 ... 15 (clipped).
(e) mean of S against the distance t = 0 ... 20/60 to the nearest edge (points with the other in-face coordinate in [0.2, 0.8]).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-6-2-cube.py
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
X0 = v.cube_grid(61)
N = len(X0)
onb = np.isclose(X0, 0) | np.isclose(X0, 1)
fax = np.argmax(onb, axis=1)                                      # one axis on which the point is at 0 or 1
other = np.array([[c for c in range(3) if c != a] for a in range(3)])
fc = np.take_along_axis(X0, other[fax], axis=1)                   # in-face coordinates
dd = np.minimum(fc, 1 - fc)
tmin, tmax = dd.min(1), dd.max(1)
top = np.isclose(X0[:, 2], 1.0)
settings = [(0.0, 40), (0.01, 40), (0.05, 40), (0.05, 150)]


def run(sig, k):
    rng = np.random.default_rng(0)
    X = X0 + sig * rng.normal(size=X0.shape) if sig > 0 else X0.copy()
    idx, dist = v.knn(X, k)
    P, nrm = v.pca_planes(X, k, idx=idx, dist=dist)
    kap, eps = v.curvatures_knn(X, P, nrm, np.ones(N), k, idx=idx, dist=dist)
    return kap, dist[:, -1]


res = {s: run(*s) for s in settings}
ts = np.arange(21) / 60
prof, stats = {}, {}
for s in settings:
    kap, epsk = res[s]
    S, K = np.abs(kap).sum(1), kap.prod(1)
    prof[s] = np.array([S[np.isclose(tmin, t) & (tmax > 0.2)].mean() for t in ts])
    face, edge, corner = tmin > 0.1, np.isclose(tmin, 0) & (tmax > 0.1), tmax < 0.05
    stats[s] = dict(eps=epsk.mean(), Sf=S[face].mean(), Se=S[edge].mean(), Sc=S[corner].mean(),
                    Kc=K[corner].mean(), Kf=K[face].mean(), sumK=K.sum() / 60 ** 2)

# self-checks
assert N == 21602
assert abs(stats[(0.0, 40)]["eps"] - 0.0589) < 5e-4 and abs(stats[(0.05, 150)]["eps"] - 0.1305) < 2e-3
assert stats[(0.0, 40)]["Sf"] < 1e-9 and abs(stats[(0.0, 40)]["Se"] - 14.69) < 0.05
assert stats[(0.05, 40)]["Sf"] > 0.9 * stats[(0.05, 40)]["Se"]          # k = 40: edges lost in the noise
assert stats[(0.05, 150)]["Se"] > 1.3 * stats[(0.05, 150)]["Sf"]        # k = 150: edges visible again
assert abs(stats[(0.0, 40)]["sumK"] - 4.55) < 0.01                       # not 4 pi

fig = plt.figure(figsize=(5.4, 7.4))  # 2 x 2 heatmaps above the profile: legible at phone width
cmap = plt.get_cmap("viridis")
for c_, s in enumerate(settings):
    ax = fig.add_axes([0.03 + (c_ % 2) * 0.44, 0.71 - (c_ // 2) * 0.3, 0.38, 0.25])
    S = np.abs(res[s][0]).sum(1)
    img = np.zeros((61, 61))
    ij = np.round(X0[top][:, :2] * 60).astype(int)
    img[ij[:, 1], ij[:, 0]] = S[top]
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=15, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"(%s) $\sigma = %g$, $k = %d$" % ("abcd"[c_], s[0], s[1]), fontsize=12)
cax = fig.add_axes([0.89, 0.45, 0.022, 0.48])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 15), cmap=cmap), cax=cax)
cb.set_ticks([0, 5, 10, 15])
cb.ax.tick_params(labelsize=11)
ax = fig.add_axes([0.14, 0.07, 0.8, 0.26])
for s, mk, col in zip(settings, "osD^", (C["main"], C["tangent"], C["normal"], C["third"])):
    ax.plot(ts * 60, prof[s], mk + "-", color=col, ms=3.5, lw=1.1, label=r"$\sigma = %g$, $k = %d$" % s)
ax.set_xticks([0, 5, 10, 15, 20])
ax.tick_params(labelsize=11)
ax.set_xlabel(r"distance to the edge $t$ (grid steps of $1/60$)", fontsize=12)
ax.set_ylabel(r"$|\kappa_1| + |\kappa_2|$", fontsize=12)
ax.set_title(r"(e) profile across an edge", fontsize=12)
ax.legend(loc="upper right", fontsize=10.5, frameon=False, handlelength=1.8)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

for s in settings:
    print(s, {k_: round(float(v_), 4) for k_, v_ in stats[s].items()}, "profile", np.round(prof[s], 3).tolist())
dgfig.save(fig, __file__)
