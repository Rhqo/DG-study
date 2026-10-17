"""Figure 33.1.1: k-simplices for k = 0, 1, 2, 3 and barycentric coordinates in a triangle (Section 33.1).

(a) a point, a segment, a triangle and a tetrahedron. The tetrahedron (vertices (0,0,0), (1,0,0), (0.5,0.87,0),
    (0.5,0.29,0.82)) is drawn with an orthographic camera; the edge whose two faces both face away is dashed.
(b) the triangle x_0 = (0, 0), x_1 = (1, 0), x_2 = (0.35, 0.9) and the point p = 0.25 x_0 + 0.4 x_1 + 0.35 x_2.
    The sub-triangle opposite x_i (the triangle p x_j x_k) has area lambda_i * A, A = area of x_0 x_1 x_2.
Self-check: the barycentric coordinates of p (dgdec.barycentric_coordinates) are (0.25, 0.4, 0.35) and equal the
area ratios; the four vertices of the tetrahedron are affinely independent; exactly one edge is hidden.

Run from the project root: ``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-1-1-simplices.py``
"""

import dgmesh as dm
import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS

fig = plt.figure(figsize=(6.2, 5.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.75], width_ratios=[0.22, 1.0, 0.22])
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 1])


def dot(ax, p, label, off=(0.0, -0.13), color=None):
    ax.plot(*p, "o", color=color or C["main"], ms=4.5, zorder=6)
    ax.text(p[0] + off[0], p[1] + off[1], label, fontsize=12, ha="center", va="center", zorder=7,
            color=color or C["main"])


# ---------------------------------------------------------------- (a) the four simplices
dot(axa, (0.0, 0.4), r"$x_0$")
axa.text(0.0, -0.45, r"$k = 0$", ha="center", fontsize=12)

s1 = np.array([[0.5, 0.05], [1.3, 0.8]])
axa.plot(*s1.T, color=C["main"], lw=1.8, zorder=5)
dot(axa, s1[0], r"$x_0$")
dot(axa, s1[1], r"$x_1$", off=(0.0, 0.14))
axa.text(0.9, -0.45, r"$k = 1$", ha="center", fontsize=12)

s2 = np.array([[1.75, 0.05], [2.75, 0.1], [2.1, 0.9]])
axa.add_patch(Polygon(s2, closed=True, facecolor=C["region"], alpha=0.35, edgecolor="none", zorder=3))
axa.add_patch(Polygon(s2, closed=True, fill=False, edgecolor=C["main"], lw=1.4, zorder=4))
dot(axa, s2[0], r"$x_0$")
dot(axa, s2[1], r"$x_1$")
dot(axa, s2[2], r"$x_2$", off=(0.0, 0.14))
axa.text(2.25, -0.45, r"$k = 2$", ha="center", fontsize=12)

T3 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.87, 0], [0.5, 0.29, 0.82]], float)
assert np.linalg.matrix_rank(T3[1:] - T3[0]) == 3          # affinely independent (Definition 33.1.1)
faces3 = [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)]       # outward
c0 = T3.mean(0)
for f in faces3:
    a, b, c = T3[list(f)]
    assert np.cross(b - a, c - a) @ (a - c0) > 0
cam = dm.Ortho(elev=18, azim=-40)
front = [np.cross(T3[f[1]] - T3[f[0]], T3[f[2]] - T3[f[0]]) @ cam.view > 0 for f in faces3]
P = cam(T3) * 1.05
P = P - P.min(0) + np.array([3.2, 0.05])
hidden = []
for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
    adj = [fr for f, fr in zip(faces3, front) if a in f and b in f]
    (hidden if not any(adj) else []).append((a, b))
    if any(adj):
        axa.plot(*P[[a, b]].T, color=C["main"], lw=1.4, zorder=5)
assert len(hidden) == 1, hidden
for a, b in hidden:
    axa.plot(*P[[a, b]].T, color=C["aux"], lw=0.9, ls=(0, (3, 2.5)), zorder=4)
for f, fr in zip(faces3, front):
    if fr:
        axa.add_patch(Polygon(P[list(f)], closed=True, facecolor=C["region"], alpha=0.2, edgecolor="none",
                              zorder=2))
cen = P.mean(0)
for v in range(4):
    d = P[v] - cen
    dot(axa, P[v], rf"$x_{v}$", off=tuple(0.17 * d / np.linalg.norm(d)))
axa.text(P[:, 0].mean(), -0.45, r"$k = 3$", ha="center", fontsize=12)
axa.set_title("(a) $k$-simplices", fontsize=12)
dgfig.schematic_axes(axa, xlim=(-0.3, 4.6), ylim=(-0.65, 1.25))

# ---------------------------------------------------------------- (b) barycentric coordinates
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.35, 0.9]])
lam = np.array([0.25, 0.4, 0.35])
p = lam @ X
area = lambda a, b, c: 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))  # noqa: E731
A = area(*X)
tri_sub = [(p, X[1], X[2]), (X[0], p, X[2]), (X[0], X[1], p)]          # opposite x_0, x_1, x_2
sub = np.array([area(*t) for t in tri_sub])
bc = dd.barycentric_coordinates(p, X)
assert np.allclose(bc, lam) and np.allclose(sub / A, lam) and abs(bc.sum() - 1) < 1e-14

cols = [C["tangent"], C["third"], C["accent"]]
for t, c in zip(tri_sub, cols):
    axb.add_patch(Polygon(np.array(t), closed=True, facecolor=c, alpha=0.22, edgecolor="none", zorder=2))
axb.add_patch(Polygon(X, closed=True, fill=False, edgecolor=C["main"], lw=1.6, zorder=4))
for i in range(3):
    axb.plot([p[0], X[i][0]], [p[1], X[i][1]], color=C["aux"], lw=0.8, zorder=3)
for i, (t, c) in enumerate(zip(tri_sub, cols)):
    c0 = (np.mean(np.array(t), 0) * 2 + np.array(t)[[j for j in range(3) if not np.allclose(t[j], p)]].mean(0)) / 3
    axb.text(*c0, rf"$\lambda_{i}A$", color=c, fontsize=12, ha="center", va="center", zorder=8,
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
dot(axb, X[0], r"$x_0$", off=(-0.06, -0.06))
dot(axb, X[1], r"$x_1$", off=(0.06, -0.06))
dot(axb, X[2], r"$x_2$", off=(0.0, 0.07))
axb.plot(*p, "o", color=C["normal"], ms=5.5, zorder=9)
axb.text(p[0] - 0.05, p[1] + 0.05, r"$p$", fontsize=12.5, color=C["normal"], zorder=9, ha="center")
axb.text(0.5, -0.17, r"$p = 0.25\,x_0 + 0.4\,x_1 + 0.35\,x_2$", ha="center", fontsize=12)
axb.set_title("(b) barycentric coordinates", fontsize=12)
dgfig.schematic_axes(axb, xlim=(-0.1, 1.1), ylim=(-0.25, 1.0))

fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.03, hspace=0.18)
dgfig.save(fig, __file__)
