"""Figure 47.4.1: approximate mean curvature versus its orthogonal version on a circle sampled with uneven density
and equal masses ([BLM17] Def 4.1, Def 4.6, Prop 4.7; the cloud of Example 46.2.7(b)).

Unit circle, N = 2000 points t_j = 2 pi u_j + 0.5 sin(2 pi u_j), u_j = (j + 1/2)/N, exact tangent lines, equal
masses 2 pi / N. The limit varifold is v(C, theta) with theta = 1/(1 + 0.5 cos 2 pi u): rectifiable but not integral,
and its generalized mean curvature is H = -x + (theta'/theta) T (Example 46.5.9; T = (-sin t, cos t)).
Natural exp pair, eps = 0.1.
(a) H^V (vermilion) and H^{V,perp} (blue) at the 24 sample points closest to the angles 2 pi k / 24; H^V leans towards the denser side (t = pi).
(b) components along the circle: normal part -<H^V, x> (black) and tangential part <H^V, T> (vermilion) against
t, with the prediction theta'/theta (grey dashed); the orthogonal version has normal part equal to that of H^V and
tangential part 0.

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/figures/fig-47-4-1-tangential-artifact.py
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

import var47 as v  # noqa: E402

C = dgfig.COLORS
N, EPS = 2000, 0.1
u = (np.arange(N) + 0.5) / N
t = 2 * np.pi * u + 0.5 * np.sin(2 * np.pi * u)
X, T, Hx = v.circle(t, 1.0)
P = v.proj_from_tangent(T)
m = np.full(N, 2 * np.pi / N)
H, Hp, cnt = v.amc(X, P, m, EPS, v.pairs(2)["natural exp"], 1)
tan = np.einsum("ka,ka->k", H, T)
nor = -np.einsum("ka,ka->k", H, X)
pred = 0.5 * np.sin(2 * np.pi * u) / (1 + 0.5 * np.cos(2 * np.pi * u)) ** 2      # theta'/theta
k = np.argmax(np.abs(pred))
assert abs(np.abs(pred).max() - 0.847) < 2e-3 and abs(t[k] - 2.73) < 0.03
assert np.abs(tan - pred).max() < 0.02                       # H^V tangential part ~ theta'/theta
assert np.abs(nor - 1).max() < 0.002                         # normal part ~ 1
assert np.abs(np.einsum("ka,ka->k", Hp, T)).max() < 1e-12    # orthogonal version has no tangential part
assert np.abs(-np.einsum("ka,ka->k", Hp, X) - nor).max() < 1e-12

sel = [int(np.argmin(np.abs(np.angle(np.exp(1j * (t - 2 * np.pi * (q + 0.5) / 24)))))) for q in range(24)]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.3})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(np.cos(tt), np.sin(tt), color=C["aux"], lw=0.6)
ax.plot(X[sel, 0], X[sel, 1], "o", color=C["main"], ms=2.5)
SC = 0.4
for j in sel:
    ax.annotate("", xy=X[j] + SC * Hp[j], xytext=X[j],
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.0, mutation_scale=7))
    ax.annotate("", xy=X[j] + SC * H[j], xytext=X[j],
                arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.0, mutation_scale=7))
ax.text(-1.32, 0.0, "dense", fontsize=9, ha="right", va="center", rotation=90)
ax.text(1.32, 0.0, "sparse", fontsize=9, ha="left", va="center", rotation=90)
ax.set_aspect("equal")
ax.set_xlim(-1.45, 1.45)
ax.set_ylim(-1.2, 1.2)
ax.set_axis_off()
ax.set_title(r"(a) $\vec H^V$ (vermilion), $\vec H^{V,\perp}$ (blue)", fontsize=10)

ax = axes[1]
ax.axhline(0, color=C["aux"], lw=0.5)
ax.plot(t, pred, "--", color=C["aux"], lw=1.4, label=r"$\theta'/\theta$")
ax.plot(t, tan, color=C["normal"], lw=1.2, label=r"$\langle \vec H^V, T\rangle$")
ax.plot(t, nor, color=C["main"], lw=1.2, label=r"$-\langle \vec H^V, x\rangle$")
ax.set_xlim(0, 2 * np.pi)
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
ax.set_xlabel(r"position $t$ on the circle")
ax.set_ylim(-1.0, 1.35)
ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=1)
ax.set_title(r"(b) components, $N = 2000$, $\varepsilon = 0.1$", fontsize=10)
dgfig.save(fig, __file__)
print("max |theta'/theta| =", np.abs(pred).max(), "at t =", t[k])
print("max |tan - pred| =", np.abs(tan - pred).max(), " normal range", nor.min(), nor.max())
print("max |H^V| =", np.linalg.norm(H, axis=1).max())
