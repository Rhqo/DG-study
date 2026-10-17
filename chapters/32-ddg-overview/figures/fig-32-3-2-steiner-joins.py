"""Figure 32.3.2: the discrete Steiner formula (Proposition 32.3.7). Offset every edge of a convex polygon outward by
|eps| and join consecutive offset edges by (A) a circular arc, (B) a straight chord, (C) extending the edges until they
meet. (a) the whole polygon; (b) one vertex, with the length each join adds.

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-3-2-steiner-joins.py
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
S = ddg32.KIND_STYLE
J = ddg32.J


def joins(X, eps, m=40):
    """closed curves for the three joins (outward offset eps < 0 of a counterclockwise convex polygon)."""
    T = dd.curve_tangents(X, True)
    Nn = T @ J.T
    th = dd.turning_angles(X, True)
    n = len(X)
    A, B, Cc = [], [], []
    for i in range(n):
        pm, pp = X[i] + eps * Nn[i - 1], X[i] + eps * Nn[i]          # end of offset edge i-1, start of edge i
        a0 = np.arctan2(*(pm - X[i])[::-1])
        arc = X[i] + abs(eps) * np.stack([np.cos(a0 + np.linspace(0, th[i], m)),
                                          np.sin(a0 + np.linspace(0, th[i], m))], 1)
        assert np.allclose(arc[-1], pp)
        A.extend(arc)
        B.extend([pm, pp])
        Cc.append(X[i] + eps * dd.curve_vertex_normals(X, True)[i] / np.cos(th[i] / 2))   # miter point q_i
    close_ = lambda P: np.vstack([P, P[:1]])  # noqa: E731
    return close_(np.array(A)), close_(np.array(B)), close_(np.array(Cc)), th


X = np.array([[0.0, 0.0], [2.0, -0.1], [2.6, 1.1], [1.3, 2.0], [-0.3, 1.1]])
eps = -0.35
A, B, Cc, th = joins(X, eps)
assert np.all(th > 0) and abs(th.sum() - 2 * np.pi) < 1e-12
L = dd.curve_length(X, True)
LA = np.linalg.norm(np.diff(A, axis=0), axis=1).sum()
LB = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
LC = np.linalg.norm(np.diff(Cc, axis=0), axis=1).sum()
assert abs(LA - (L - eps * th.sum())) < 1e-3                                   # arc polygonized with 40 points
assert abs(LB - (L - eps * (2 * np.sin(th / 2)).sum())) < 1e-12
assert abs(LC - (L - eps * (2 * np.tan(th / 2)).sum())) < 1e-12

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.6), gridspec_kw={"width_ratios": [0.85, 1.25]})
Xc = np.vstack([X, X[:1]])
ax.fill(X[:, 0], X[:, 1], fc=C["region"], alpha=0.18, ec="none")
ax.plot(Xc[:, 0], Xc[:, 1], color=C["main"], lw=dgfig.LW["main"])
ax.plot(Cc[:, 0], Cc[:, 1], color=S["C"]["color"], ls=S["C"]["ls"], lw=1.6, label="(C) extend")
ax.plot(B[:, 0], B[:, 1], color=S["B"]["color"], ls=S["B"]["ls"], lw=1.4, label="(B) chord")
ax.plot(A[:, 0], A[:, 1], color=S["A"]["color"], ls=S["A"]["ls"], lw=1.3, label="(A) arc")
ax.legend(loc="upper center", frameon=False, fontsize=9.5, handlelength=2.4, ncol=1, bbox_to_anchor=(0.5, 1.02))
ax.set_aspect("equal")
ax.set_xlim(-1.0, 3.3)
ax.set_ylim(-0.75, 3.55)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12)

# (b) one vertex, zoomed: vertex 2 of the polygon above
i = 2
T = dd.curve_tangents(X, True)
Nn = T @ J.T
pm, pp = X[i] + eps * Nn[i - 1], X[i] + eps * Nn[i]
q = Cc[i]
bx.plot([X[i - 1, 0], X[i, 0], X[i + 1, 0]], [X[i - 1, 1], X[i, 1], X[i + 1, 1]], color=C["main"],
        lw=dgfig.LW["main"])
bx.plot(*X[i], "o", color=C["main"], ms=4)
bx.text(X[i, 0] - 0.16, X[i, 1] - 0.03, r"$\gamma_i$", fontsize=12)
for p, tt, sgn in ((pm, T[i - 1], -1), (pp, T[i], 1)):
    e = p + sgn * 0.75 * tt
    bx.plot([e[0], p[0]], [e[1], p[1]], color=C["aux"], lw=1.0)                # offset edges (shown in part)
    bx.plot([X[i, 0], p[0]], [X[i, 1], p[1]], color=C["aux"], lw=dgfig.LW["aux"], ls=":")   # |eps| J t
a0 = np.arctan2(*(pm - X[i])[::-1])
ang = a0 + np.linspace(0, th[i], 60)
arc = X[i] + abs(eps) * np.stack([np.cos(ang), np.sin(ang)], 1)
bx.plot(arc[:, 0], arc[:, 1], color=S["A"]["color"], lw=1.8)
bx.plot([pm[0], pp[0]], [pm[1], pp[1]], color=S["B"]["color"], ls=S["B"]["ls"], lw=1.6)
bx.plot([pm[0], q[0], pp[0]], [pm[1], q[1], pp[1]], color=S["C"]["color"], ls=S["C"]["ls"], lw=1.8)
bx.plot(*q, "o", color=S["C"]["color"], ms=3)
bx.annotate(r"$q_i$", xy=q, xytext=q + np.array([0.12, 0.05]), fontsize=12, color=S["C"]["color"])
bx.annotate(r"(A) $|\varepsilon|\,\theta_i$", xy=arc[38], xytext=X[i] + np.array([0.42, 0.42]), fontsize=10.5,
            color=S["A"]["color"], arrowprops=dict(arrowstyle="-", color=S["A"]["color"], lw=0.6))
mc = 0.5 * (pm + pp)
bx.annotate(r"(B) $2|\varepsilon|\sin\frac{\theta_i}{2}$", xy=mc, xytext=X[i] + np.array([-0.62, 0.42]),
            fontsize=10.5, color=S["B"]["color"], arrowprops=dict(arrowstyle="-", color=S["B"]["color"], lw=0.6))
mq = 0.5 * (q + pm)
bx.annotate(r"(C) $2|\varepsilon|\tan\frac{\theta_i}{2}$", xy=mq, xytext=X[i] + np.array([0.18, -0.55]),
            fontsize=10.5, color=S["C"]["color"], arrowprops=dict(arrowstyle="-", color=S["C"]["color"], lw=0.6))
mx = 0.5 * (X[i] + pm)
bx.text(mx[0] - 0.02, mx[1] - 0.12, r"$|\varepsilon|$", fontsize=11, color=C["aux"])
bx.set_aspect("equal")
bx.set_xlim(X[i, 0] - 0.75, X[i, 0] + 0.95)
bx.set_ylim(X[i, 1] - 0.7, X[i, 1] + 0.7)
bx.set_axis_off()
bx.set_title("(b)", fontsize=12)

fig.tight_layout()
dgfig.save(fig, __file__)
print(f"L = {L:.4f}, theta = {np.round(th, 4)}, L_A = {L - eps * th.sum():.4f}, L_B = {LB:.4f}, L_C = {LC:.4f}")
