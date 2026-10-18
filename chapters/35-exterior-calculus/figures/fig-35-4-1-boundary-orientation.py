"""Figure 35.4.1: the Stokes orientation ("outward normal first") of boundaries in the plane
(Section 35.4, Definition 35.4.1, Example 35.4.2, Proposition 35.4.4).

(a) The annulus A = {1 <= r <= 2} (light blue). At 6 points of each boundary circle: the outward normal n of A
    (vermillion; on the inner circle it points toward the origin) and the tangent t with (n, t) positively oriented,
    i.e. t = J n (n rotated by +90 degrees) (blue). The outer circle runs counterclockwise, the inner one clockwise.
(b) The positively oriented triangle sigma = [x0, x1, x2], x0 = (0, 0), x1 = (2, 0.4), x2 = (0.7, 1.7) (light blue).
    On each edge: the outward normal (vermillion) at the midpoint and the edge arrow (blue) given by the Stokes
    orientation. Labels: the face opposite x_m with its sign (-1)^m (Proposition 35.4.4):
    (-1)^0 [x1, x2], (-1)^1 [x0, x2] = [x2, x0], (-1)^2 [x0, x1].
Self-checks: det(n, t) = +1 at every sampled point of (a); in (b) the outward normal of the face opposite x_m and the
direction of (-1)^m [..] satisfy det(n, edge direction) > 0, and the induced boundary agrees with
tools/dgdec.induced_boundary.
"""
import numpy as np

import dgfig
import dgdec as dd

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
J = np.array([[0.0, -1.0], [1.0, 0.0]])


def arrow(ax, p, v, color, lw=1.5, ms=10, zorder=5):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(np.asarray(p) + v), arrowstyle="-|>", mutation_scale=ms, color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=zorder))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.4))

# (a) annulus
ts = np.linspace(0, 2 * np.pi, 300)
ring = np.concatenate([np.stack([2 * np.cos(ts), 2 * np.sin(ts)], 1), np.stack([np.cos(ts[::-1]), np.sin(ts[::-1])], 1)])
axA.add_patch(Polygon(ring, closed=True, facecolor=C["region"], alpha=0.3, edgecolor="none", zorder=0))
for R, sgn in [(2.0, 1.0), (1.0, -1.0)]:
    axA.plot(R * np.cos(ts), R * np.sin(ts), color=C["main"], lw=1.2, zorder=1)
    for k in range(6):
        a = 2 * np.pi * k / 6 + np.pi / 12
        p = R * np.array([np.cos(a), np.sin(a)])
        n = sgn * p / R                       # outward normal of the annulus
        tv = J @ n                            # (n, t) positively oriented
        assert abs(np.linalg.det(np.column_stack([n, tv])) - 1) < 1e-12
        arrow(axA, p, 0.38 * n, C["normal"], lw=1.3, ms=9)
        arrow(axA, p, 0.55 * tv, C["tangent"], lw=1.5, ms=10)
axA.text(0, 0, r"$A$", ha="center", va="center", fontsize=12)
axA.text(1.55, -2.45, r"$\mathbf{n}$", color=C["normal"], fontsize=12)
axA.text(-1.9, -2.45, r"$t = J\mathbf{n}$", color=C["tangent"], fontsize=12)
dgfig.schematic_axes(axA, xlim=(-2.6, 2.6), ylim=(-2.65, 2.6))
axA.set_title(r"(a) annulus $1 \leq r \leq 2$", fontsize=12)

# (b) triangle
X = np.array([[0.0, 0.0], [2.0, 0.4], [0.7, 1.7]])
assert np.linalg.det(np.column_stack([X[1] - X[0], X[2] - X[0]])) > 0
axB.add_patch(Polygon(X, closed=True, facecolor=C["region"], alpha=0.3, edgecolor=C["main"], lw=1.2, zorder=0))
bd = dd.induced_boundary((0, 1, 2))       # [(+1, (1, 2)), (-1, (0, 2)), (+1, (0, 1))]
assert [(s_, f) for s_, f in bd] == [(1, (1, 2)), (-1, (0, 2)), (1, (0, 1))]
labels = {0: r"$(-1)^0[x_1, x_2]$", 1: r"$(-1)^1[x_0, x_2]$", 2: r"$(-1)^2[x_0, x_1]$"}
offs = {0: (0.62, 0.28), 1: (-0.75, 0.2), 2: (-0.2, -0.62)}
for m, (s_, (i, j)) in enumerate(bd):
    a, b = (X[i], X[j]) if s_ > 0 else (X[j], X[i])
    d = b - a
    mid = 0.5 * (a + b)
    opp = X[m]
    n = J @ d                                  # candidate normal; flip to point away from the opposite vertex
    n = -J @ d if np.dot(J @ d, opp - mid) > 0 else n
    n = n / np.linalg.norm(n)
    assert np.dot(n, opp - mid) < 0             # outward
    assert np.linalg.det(np.column_stack([n, d])) > 0   # outward normal first
    arrow(axB, a + 0.08 * d, 0.84 * d, C["tangent"], lw=1.6, ms=11)
    arrow(axB, mid, 0.42 * n, C["normal"], lw=1.3, ms=9)
    axB.text(mid[0] + offs[m][0], mid[1] + offs[m][1], labels[m], fontsize=11, ha="center")
for k, off in zip(range(3), [(-0.15, -0.22), (0.18, -0.1), (-0.05, 0.13)]):
    axB.plot(*X[k], "o", color=C["main"], ms=3, zorder=6)
    axB.text(X[k, 0] + off[0], X[k, 1] + off[1], rf"$x_{k}$", fontsize=12, ha="center")
axB.text(0.85, 0.65, r"$\sigma$", fontsize=12, ha="center")
dgfig.schematic_axes(axB, xlim=(-1.25, 2.75), ylim=(-0.85, 2.3))
axB.set_title(r"(b) triangle $[x_0, x_1, x_2]$", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.9, wspace=0.08)
dgfig.save(fig, __file__)
