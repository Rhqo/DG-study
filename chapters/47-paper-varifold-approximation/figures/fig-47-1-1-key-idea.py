"""Figure 47.1.1: the key idea of [BLM17] on a point cloud of the circle of radius 0.5.

N = 60 points x_j = 0.5 (cos t_j, sin t_j), t_j = 2 pi j / 60, exact tangent lines P_j, masses m_j = pi / 60.
At x = x_0 = (0.5, 0) with eps = 0.2 and the natural exp pair (Section 47.5) every neighbour contributes
c_j = m_j (-rho'(|x_j - x| / eps)) P_j (x_j - x) / |x_j - x|; the sum S = sum c_j is proportional to
-(delta V * rho_eps)(x) (formula (3.4) of [BLM17]); dividing by the smoothed mass gives the approximate mean
curvature H = (C_xi / C_rho) S / (eps sum_j m_j xi(|x_j - x| / eps)), which points to the centre.
(a) the cloud, the tangent lines and the ball B_eps(x). (b) zoom: the contributions c_j (blue, drawn at x_j) and their
sum S (black), both 3 times, and H (vermilion) 0.1 times.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-1-1-key-idea.py
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
from matplotlib.patches import Circle  # noqa: E402

import var47 as v  # noqa: E402

C = dgfig.COLORS
N, R, EPS = 60, 0.5, 0.2
pair = v.pairs(2)["natural exp"]
t = 2 * np.pi * np.arange(N) / N
X, T, Hx = v.circle(t, R)
P = v.proj_from_tangent(T)
m = np.full(N, np.pi / N)
x = X[0]

D = X - x
r = np.linalg.norm(D, axis=1)
nb = (r > 0) & (r < EPS)
u = D[nb] / r[nb, None]
cj = (m[nb] * (-pair.drho(r[nb] / EPS)))[:, None] * np.einsum("kab,kb->ka", P[nb], u)
S = cj.sum(0)
H, Hp = v.amc_single(x, P[0], X, P, m, EPS, pair, 1)

# self-checks: 6 neighbours, tangential parts cancel, H points to the centre with length close to 2
assert nb.sum() == 6
assert abs(S[1]) < 1e-14 and S[0] < 0
assert abs(H[1]) < 1e-12 and H[0] < 0 and abs(np.linalg.norm(H) - 2) < 0.05
assert np.allclose(H, Hp)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), gridspec_kw={"width_ratios": [1, 1.05], "wspace": 0.12})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(R * np.cos(tt), R * np.sin(tt), color=C["aux"], lw=0.6, ls="--")
for k in range(N):
    a, b = X[k] - 0.035 * T[k], X[k] + 0.035 * T[k]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C["tangent"], lw=0.9)
ax.plot(X[:, 0], X[:, 1], "o", color=C["main"], ms=2.2)
ax.add_patch(Circle(x, EPS, facecolor=C["region"], alpha=0.25, edgecolor=C["region"], lw=0.8))
ax.plot(*x, "o", color=C["normal"], ms=4)
ax.annotate(r"$x$", x, xytext=(8, -12), textcoords="offset points", fontsize=12)
ax.annotate(r"$B_\varepsilon(x)$", (x[0] + 0.02, x[1] + 0.17), xytext=(-62, 22), textcoords="offset points",
            fontsize=11, arrowprops=dict(arrowstyle="-", color=C["aux"], lw=0.6))
ax.set_aspect("equal")
ax.set_xlim(-0.62, 0.78)
ax.set_ylim(-0.62, 0.62)
ax.set_axis_off()
ax.set_title(r"(a) $N = 60$, $\varepsilon = 0.2$", fontsize=11)

ax = axes[1]
ax.plot(R * np.cos(tt), R * np.sin(tt), color=C["aux"], lw=0.6, ls="--")
ax.add_patch(Circle(x, EPS, facecolor=C["region"], alpha=0.18, edgecolor=C["region"], lw=0.8))
sel = np.abs(np.arctan2(X[:, 1], X[:, 0])) < 0.5
for k in np.where(sel)[0]:
    a, b = X[k] - 0.03 * T[k], X[k] + 0.03 * T[k]
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C["aux"], lw=0.7)
ax.plot(X[sel, 0], X[sel, 1], "o", color=C["main"], ms=3)
SC = 3.0
for p0, c in zip(X[nb], cj):
    ax.annotate("", xy=p0 + SC * c, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.2, mutation_scale=8))
sS, sH = SC, 0.1
ax.annotate("", xy=x + sH * H, xytext=x, arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.6,
                                                         mutation_scale=10))
ax.annotate("", xy=x + sS * S, xytext=x, arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=2.0, mutation_scale=10))
ax.text(x[0] + sS * S[0] / 2, x[1] + 0.018, r"$S$", fontsize=12, ha="center")
ax.text(x[0] + sH * H[0] + 0.02, x[1] - 0.045, r"$\vec H^{V}_{\rho,\xi,\varepsilon}$", fontsize=11,
        color=C["normal"], ha="center")
ax.plot(*x, "o", color=C["normal"], ms=4)
ax.text(x[0] + 0.012, x[1] - 0.03, r"$x$", fontsize=12)
ax.text(X[2, 0] + 0.02, X[2, 1], r"$c_j$", fontsize=12, color=C["tangent"])
ax.set_aspect("equal")
ax.set_xlim(0.20, 0.74)
ax.set_ylim(-0.24, 0.24)
ax.set_axis_off()
ax.set_title(r"(b) contributions $c_j$ and their sum", fontsize=11)

dgfig.save(fig, __file__)
print("S =", S, " H =", H, " |H| =", np.linalg.norm(H))
