"""Figure 32.2.3: a discrete curve (10 vertices gamma_i = gamma(2 pi i/10) on the ellipse (2 cos t, sin t)),
its discrete tangents t_i and normals J t_i (Definition 32.2.11), and the tangent angle as a function of arc length:
smooth (a continuous function with slope kappa_s) versus discrete (a staircase that jumps at the vertices).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-2-3-discrete-tangents.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg32  # noqa: E402

C = dgfig.COLORS
J = ddg32.J
n = 10
X, tv = ddg32.sample("ellipse", n)
E = dd.curve_edge_vectors(X, True)
ell = dd.curve_edge_lengths(X, True)
T = dd.curve_tangents(X, True)
Nn = T @ J.T
assert np.allclose(np.linalg.norm(T, axis=1), 1) and np.allclose((T * Nn).sum(1), 0)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})

# (a) the polygon, vertices, discrete tangent and normal at the midpoint of each edge
s = np.linspace(0, 2 * np.pi, 400)
S = ddg32.point("ellipse", s)
ax.plot(S[:, 0], S[:, 1], color=C["aux"], lw=dgfig.LW["aux"], ls="--")
Xc = np.vstack([X, X[:1]])
ax.plot(Xc[:, 0], Xc[:, 1], color=C["main"], lw=dgfig.LW["main"])
ax.plot(X[:, 0], X[:, 1], "o", color=C["main"], ms=3.5, zorder=5)
M = X + 0.5 * E
for i in range(n):
    b = M[i] - 0.12 * Nn[i] - 0.19 * T[i]      # drawn just outside the edge, parallel to it
    ax.annotate("", xy=b + 0.38 * T[i], xytext=b, arrowprops=dict(arrowstyle="-|>", color=C["tangent"],
                                                                  lw=1.3, mutation_scale=8))
    ax.annotate("", xy=M[i] + 0.38 * Nn[i], xytext=M[i], arrowprops=dict(arrowstyle="-|>", color=C["normal"],
                                                                         lw=1.3, mutation_scale=8))
ax.text(X[0, 0] + 0.08, X[0, 1] - 0.18, r"$\gamma_0$", fontsize=11)
ax.text(X[1, 0] - 0.05, X[1, 1] + 0.12, r"$\gamma_1$", fontsize=11)
q = M[0] - 0.12 * Nn[0] + 0.19 * T[0]
ax.text(q[0] + 0.07, q[1] - 0.02, r"$\mathbf{t}_0$", color=C["tangent"], fontsize=11)
q = M[0] + 0.38 * Nn[0]
ax.text(q[0] - 0.36, q[1] - 0.10, r"$J\mathbf{t}_0$", color=C["normal"], fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-2.4, 2.6)
ax.set_ylim(-1.3, 1.4)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12)

# (b) tangent angle against arc length
m = 4000
tt = 2 * np.pi * np.arange(m + 1) / m
sp_ = ddg32.speed("ellipse", tt)
arc = np.r_[0, np.cumsum(0.5 * (sp_[1:] + sp_[:-1]) * np.diff(tt))]
phi = np.unwrap(np.arctan2(np.cos(tt), -2 * np.sin(tt)))          # angle of gamma'(t) = (-2 sin t, cos t)
assert abs(phi[-1] - phi[0] - 2 * np.pi) < 1e-9                   # one full turn (Theorem 0.2.7)
bx.plot(arc, phi, color=C["aux"], lw=1.2, ls="--")
ang = np.unwrap(np.arctan2(T[:, 1], T[:, 0]))
svert = np.r_[0, np.cumsum(ell)]                                  # arc length of the polygon at the vertices
for i in range(n):
    bx.plot([svert[i], svert[i + 1]], [ang[i], ang[i]], color=C["main"], lw=1.8)
    if i + 1 < n:
        bx.plot([svert[i + 1], svert[i + 1]], [ang[i], ang[i + 1]], color=C["accent"], lw=1.4)
jumps = np.diff(ang)
assert np.allclose(jumps, dd.turning_angles(X, True)[1:], atol=1e-12)
bx.annotate("jump at\nvertex 1", xy=(svert[1], 0.5 * (ang[0] + ang[1])), xytext=(svert[1] + 1.1, ang[0] - 0.55),
            fontsize=9.5, arrowprops=dict(arrowstyle="-", lw=0.6, color=C["accent"]), color=C["accent"])
bx.set_xlabel(r"arc length $s$")
bx.set_ylabel(r"tangent angle $\varphi$")
bx.set_yticks([np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi, 5 * np.pi / 2])
bx.set_yticklabels([r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$", r"$5\pi/2$"])
bx.set_title("(b)", fontsize=12)
for sp2 in ("top", "right"):
    bx.spines[sp2].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
