"""Figure 37.4.1: a closed discrete curve is determined by its edge lengths and turning angles.

(a) the ellipse (2 cos t, sin t) sampled at 12 parameter values: edge vectors e_i (arrows, head to tail) and the turning
angles theta_i at the vertices; (b) curves rebuilt from (l_i, theta_i) by the discrete recipe (37.4.3): two choices of
(p_0, phi_0) (black, grey: they differ by a rigid motion) and the same angles with l_3 doubled (orange: the
closing condition (ii) fails).

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-4-1-reconstruction.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
n = 12
X, _ = ddg37.sample("ellipse", n)
E = dd.curve_edge_vectors(X, True)
ell = dd.curve_edge_lengths(X, True)
th = dd.turning_angles(X, True)
assert abs(th.sum() - 2 * np.pi) < 1e-12 and np.allclose(E.sum(0), 0, atol=1e-12)


def rebuild(ell, th, p0, phi0):
    """vertices gamma_0, ..., gamma_n from edge lengths and the turning angles theta_1, ..., theta_{n-1}."""
    phi = phi0 + np.concatenate([[0.0], np.cumsum(th[1:len(ell)])])
    T = np.stack([np.cos(phi), np.sin(phi)], 1)
    return p0 + np.vstack([[0.0, 0.0], np.cumsum(ell[:, None] * T, 0)])


phi_e0 = np.arctan2(E[0, 1], E[0, 0])
G1 = rebuild(ell, th, X[0], phi_e0)
assert np.allclose(G1[:n], X, atol=1e-12) and np.linalg.norm(G1[n] - G1[0]) < 1e-12
G2 = rebuild(ell, th, np.array([5.6, 0.4]), phi_e0 + 0.7)
ell3 = ell.copy()
ell3[3] *= 2
G3 = rebuild(ell3, th, np.array([0.0, -3.4]) + X[0], phi_e0)
gap = G3[n] - G3[0]
assert np.linalg.norm(gap) > 0.5

fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25]})

# (a) edge vectors head to tail and turning angles
s = np.linspace(0, 2 * np.pi, 300)
S = ddg37.point("ellipse", s)
ax.plot(S[:, 0], S[:, 1], color=C["aux"], lw=dgfig.LW["aux"], ls="--")
for i in range(n):
    ax.annotate("", xy=X[(i + 1) % n], xytext=X[i],
                arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4, mutation_scale=9, shrinkA=0, shrinkB=1))
ax.plot(X[:, 0], X[:, 1], "o", color=C["main"], ms=3, zorder=5)
for i in (0, 2, 3):
    a_in = np.degrees(np.arctan2(E[i - 1, 1], E[i - 1, 0]))
    ext = X[i] + 0.7 * E[i - 1] / ell[i - 1]
    ax.plot([X[i, 0], ext[0]], [X[i, 1], ext[1]], color=C["aux"], lw=0.8, ls=":")
    ax.add_patch(Arc(X[i], 1.0, 1.0, theta1=a_in, theta2=a_in + np.degrees(th[i]), color=C["accent"], lw=1.4))
ax.text(X[0, 0] + 0.22, X[0, 1] + 0.42, r"$\theta_0$", color=C["accent"], fontsize=11)
ax.text(X[3, 0] - 0.55, X[3, 1] + 0.22, r"$\theta_3$", color=C["accent"], fontsize=11)
ax.text(0.5 * (X[0, 0] + X[1, 0]) - 0.42, 0.5 * (X[0, 1] + X[1, 1]) - 0.12, r"$e_0$", color=C["tangent"], fontsize=11)
ax.text(X[0, 0] + 0.1, X[0, 1] - 0.3, r"$\gamma_0$", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-2.4, 3.0)
ax.set_ylim(-1.35, 1.45)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12, loc="left")

# (b) rebuilt curves
for G, col, ls in ((G1, C["main"], "-"), (G2, C["aux"], "-"), (G3, C["accent"], "--")):
    bx.plot(G[:, 0], G[:, 1], color=col, ls=ls, lw=1.6)
    bx.plot(G[:-1, 0], G[:-1, 1], "o", color=col, ms=2.5)
    bx.plot([G[0, 0]], [G[0, 1]], "o", color=col, ms=5, zorder=5)
bx.plot([G3[n, 0]], [G3[n, 1]], "s", color=C["accent"], ms=5, zorder=5)
bx.annotate("", xy=G3[0], xytext=G3[n], arrowprops=dict(arrowstyle="<->", color=C["normal"], lw=1.0,
                                                          shrinkA=3, shrinkB=3))
mid = 0.5 * (G3[0] + G3[n])
bx.text(mid[0] + 0.2, mid[1] - 0.35, "gap", color=C["normal"], fontsize=10)
bx.text(-0.6, -4.95, r"$\ell_3 \to 2\ell_3$", color=C["accent"], fontsize=10.5)
bx.set_aspect("equal")
bx.set_axis_off()
bx.set_title("(b)", fontsize=12, loc="left")

fig.tight_layout()
dgfig.save(fig, __file__)
