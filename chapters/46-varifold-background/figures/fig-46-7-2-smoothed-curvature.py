"""Figure 46.7.2: reading curvature from a point cloud by smoothing first variation and mass with the same kernel.

Point cloud on the ellipse (2 cos t, sin t): N points at equal arc-length spacing h = L/N, exact tangent lines P_j,
masses m_j = L/N. At each point x the vector  -(delta V * rho_eps)(x) / (||V|| * rho_eps)(x)  with
(delta V * rho_eps)(x) = sum_j m_j P_j grad rho_eps(x_j - x)  (Proposition 46.7.4) and the exp kernel of Figure 46.7.1.
(a) its length along the ellipse for N = 1024 and eps = 0.03, 0.1, 0.4, against the curvature kappa (black).
(b) the largest relative error  max |(|ratio| - kappa) / kappa|  over the points, as a function of eps, for N = 256 and
N = 1024 (log-log). Too small an eps sees too few neighbours; too large an eps averages the curvature away.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-7-2-smoothed-curvature.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
L = v.ellipse_length_total()
tg = np.linspace(0, 2 * np.pi, 200001)
sg = np.concatenate([[0.0], np.cumsum(0.5 * (v.ellipse_speed(tg[1:]) + v.ellipse_speed(tg[:-1])) * np.diff(tg))])
sg *= L / sg[-1]


def cloud(N):
    t = np.interp(np.arange(N) * L / N, sg, tg)
    return t, v.ellipse(t), v.dir_proj(v.ellipse_tangent(t)), np.full(N, L / N)


def ratio(x, X, P, m, eps):
    d = X - x[None, :]
    sel = np.linalg.norm(d, axis=1) < eps
    g = v.grad_kernel_eps(d[sel], eps)
    num = np.einsum("i,ijk,ik->j", m[sel], P[sel], g)
    den = np.sum(m[sel] * v.kernel_eps(d[sel], eps))
    return -num / den


def ratios(N, eps):
    t, X, P, m = cloud(N)
    return t, np.array([ratio(X[k], X, P, m, eps) for k in range(N)])


# self-checks: the vector points inward (towards the curvature vector); N = 1024, eps = 0.1 is within 2 percent
t1, R1 = ratios(1024, 0.1)
kap1 = v.ellipse_kappa(t1)
assert np.all(np.einsum("ij,ij->i", R1, v.ellipse_normal(t1)) > 0)
assert np.max(np.abs(np.linalg.norm(R1, axis=1) - kap1) / kap1) < 0.02

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.38})
ax = axes[0]
tt = np.linspace(0, np.pi, 400)
ax.plot(np.degrees(tt), v.ellipse_kappa(tt), color=C["main"], lw=1.6, label=r"$\kappa$")
for eps, col in ((0.03, C["normal"]), (0.1, C["tangent"]), (0.4, C["accent"])):
    t, R = ratios(1024, eps)
    half = t <= np.pi + 1e-9
    ax.plot(np.degrees(t[half]), np.linalg.norm(R[half], axis=1), color=col, lw=1.1, label=rf"$\varepsilon = {eps}$")
ax.set_xlabel(r"$t$ (degrees)")
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_ylim(0, 2.6)
ax.legend(loc="upper center", frameon=False, fontsize=8, ncol=2, handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
ax.set_title(r"(a) length of the ratio, $N = 1024$", fontsize=11)

ax = axes[1]
E = np.geomspace(0.02, 0.6, 28)
for N, col, mk in ((256, C["normal"], "s"), (1024, C["tangent"], "o")):
    errs = []
    for eps in E:
        t, R = ratios(N, eps)
        k = v.ellipse_kappa(t)
        errs.append(np.max(np.abs(np.linalg.norm(R, axis=1) - k) / k))
    ax.loglog(E, errs, mk + "-", color=col, ms=3, lw=1.0, label=f"$N = {N}$")
ax.set_xlabel(r"$\varepsilon$")
ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5"])
ax.minorticks_off()
ax.set_yticks([0.01, 0.1, 1])
ax.set_yticklabels(["0.01", "0.1", "1"])
ax.set_ylabel("largest relative error")
ax.set_ylim(2e-3, 3)
ax.legend(loc="upper right", frameon=False, fontsize=8, handlelength=1.4, handletextpad=0.3)
ax.set_title(r"(b) error against $\varepsilon$", fontsize=11)

dgfig.save(fig, __file__)
