"""Figure 48.5.1: the two convergence rates of [BLM22] Theorems 6.8 and 6.11 on the unit sphere.

V is the unit sphere (multiplicity 1), represented by a Fibonacci point cloud of 400000 points with exact tangent
planes (a proxy for the continuous V: its own discretisation error is below 0.02 for eps >= 0.04).
V_N is an i.i.d. area-uniform sample of N = 20000 or 80000 points with exact tangent planes, masses 4 pi / N.
At 60 sample points z of V_N (so x = z, |x - z| = 0, d_2 = 0) and for eps = 0.04 ... 0.4 we compute
  (a) the discrepancies max_ijk |A^{V_N,eps}(z) - A^{V,eps}(z)| (epsilon-WSFF, (6.9), c^{V,eps} with eta = rho) and
      max_ijk |A^{V_N,eps,perp}(z) - A^{V,eps,perp}(z)| (epsilon-WSFF-perp, (6.22)), averaged over the 60 points:
      the quantities bounded in (6.16) and (6.25);
  (b) the total error max_ijk |A^{V_N,eps,perp}(z) - A(z)| against the exact tensor A_ijk = -(P_ij z_k + P_ik z_j),
      and the bias max_ijk |A^{V,eps,perp} - A| of the proxy itself.
Kernels of [BLM22] Section 7.1: rho(t) = exp(-1/(1 - t^2)), natural xi = -t rho'/n.

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/figures/fig-48-5-1-two-rates.py
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
EPS = np.array([0.04, 0.06, 0.1, 0.15, 0.25, 0.4])
NP = 400000
Xp = v.fib_sphere(NP)
Pp = v.proj_from_normal(Xp)
mp = np.full(NP, 4 * np.pi / NP)


def exact_A(P, x):
    return -(np.einsum("ij,k->ijk", P, x) + np.einsum("ik,j->ijk", P, x))


def experiment(N):
    Z = np.random.default_rng(N).normal(size=(N, 3))
    X = Z / np.linalg.norm(Z, axis=1)[:, None]
    P = v.proj_from_normal(X)
    m = np.full(N, 4 * np.pi / N)
    idx = np.random.default_rng(1).choice(N, 60, replace=False)
    out = np.zeros((len(EPS), 3))
    for a, eps in enumerate(EPS):
        rows = []
        for i in idx:
            x = X[i]
            nb, nbp = v.ball(X, x, eps), v.ball(Xp, x, eps)
            Aw = v.wsff(x, X[nb], P[nb], m[nb], eps, 3, 2)
            Awp = v.wsff(x, Xp[nbp], Pp[nbp], mp[nbp], eps, 3, 2)
            Ap = v.wsff_perp(x, P[i], X[nb], P[nb], m[nb], eps, 3, 2)
            App = v.wsff_perp(x, P[i], Xp[nbp], Pp[nbp], mp[nbp], eps, 3, 2)
            rows.append([np.abs(Aw - Awp).max(), np.abs(Ap - App).max(), np.abs(Ap - exact_A(P[i], x)).max()])
        out[a] = np.mean(rows, axis=0)
    return out


R20, R80 = experiment(20000), experiment(80000)
bias = []
for eps in EPS:
    b = []
    for i in range(0, NP, NP // 60)[:60]:
        nbp = v.ball(Xp, Xp[i], eps)
        b.append(np.abs(v.wsff_perp(Xp[i], Pp[i], Xp[nbp], Pp[nbp], mp[nbp], eps, 3, 2) - exact_A(Pp[i], Xp[i])).max())
    bias.append(np.mean(b))
bias = np.array(bias)


def slope(e, y):
    return np.polyfit(np.log(e), np.log(y), 1)[0]


# self-checks
sW, sP = slope(EPS, R20[:, 0]), slope(EPS, R20[:, 1])
assert -2.2 < sW < -1.8 and -1.15 < sP < -0.85, (sW, sP)
assert -2.2 < slope(EPS, R80[:, 0]) < -1.8 and -1.15 < slope(EPS, R80[:, 1]) < -0.85
ratio = R80[:, :2] / R20[:, :2]                                  # four times more points: about half the error
assert np.all(ratio < 0.65) and 0.42 < ratio.mean() < 0.58, ratio
assert abs(slope(EPS[2:], bias[2:]) - 2) < 0.25                 # bias of the proxy ~ eps^2
assert np.argmin(R80[:, 2]) in (3, 4) and np.argmin(R20[:, 2]) in (3, 4)

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"wspace": 0.38})
ax = axes[0]
for R, mk, fc, lab in ((R20, "o", None, "20000"), (R80, "s", "white", "80000")):
    ax.loglog(EPS, R[:, 0], mk + "-", color=C["normal"], ms=4, lw=1.1, mfc=fc or C["normal"])
    ax.loglog(EPS, R[:, 1], mk + "-", color=C["tangent"], ms=4, lw=1.1, mfc=fc or C["tangent"])
ax.plot([], [], color=C["normal"], lw=1.2, label=r"$\varepsilon$-WSFF, slope $%.2f$" % sW)
ax.plot([], [], color=C["tangent"], lw=1.2, label=r"$\varepsilon$-WSFF$^\perp$, slope $%.2f$" % sP)
ax.plot([], [], "o", color=C["main"], ms=4, label=r"$N = 20000$")
ax.plot([], [], "s", color=C["main"], mfc="white", ms=4, label=r"$N = 80000$")
ax.legend(loc="upper right", fontsize=7.8, frameon=False, handlelength=1.4, labelspacing=0.25)
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\max_{ijk}|A^{V_N,\varepsilon} - A^{V,\varepsilon}|$")
ax.set_title(r"(a) discretization error", fontsize=10.5)

ax = axes[1]
ax.loglog(EPS, R20[:, 2], "o-", color=C["tangent"], ms=4, lw=1.1, label=r"$N = 20000$")
ax.loglog(EPS, R80[:, 2], "s-", color=C["tangent"], mfc="white", ms=4, lw=1.1, label=r"$N = 80000$")
ax.loglog(EPS[2:], bias[2:], "^--", color=C["accent"], ms=4, lw=1.0, label=r"bias of $V$")
ax.legend(loc="upper center", fontsize=7.8, frameon=False, handlelength=1.6, labelspacing=0.25)
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\max_{ijk}|A^{V_N,\varepsilon,\perp} - A|$")
ax.set_title(r"(b) total error, $\varepsilon$-WSFF$^\perp$", fontsize=10.5)
for a in axes:
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

np.set_printoptions(precision=4, suppress=False)
print("N=20000:", R20.round(4).tolist())
print("N=80000:", R80.round(4).tolist())
print("bias:", bias.round(4).tolist(), " slopes", round(sW, 2), round(sP, 2), " mean ratio", round(ratio.mean(), 3))
dgfig.save(fig, __file__)
