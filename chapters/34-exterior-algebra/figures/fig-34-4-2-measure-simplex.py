"""Figure 34.4.2: constant forms measure the oriented edges and the oriented face of a triangle
(Section 34.4, Smooth vs. Discrete box and Implementation box).

Triangle x0 = (0, 0), x1 = (2, 0.5), x2 = (0.6, 1.6), oriented counterclockwise (x0, x1, x2).
(a) The 1-form alpha = e^1 + e^2 (level lines x + y = c, purple) measures the edge vectors of the oriented edges
    x0->x1, x1->x2, x2->x0: 2.5, -0.3, -2.2. They sum to 0 because the edge vectors sum to 0.
(b) The 2-form e^1 ^ e^2 measures the parallelogram on (x1 - x0, x2 - x0): det = 2.9 (dashed), and the triangle is
    half of it: integral = 2.9 / 2 = 1.45.
Self-checks: the three edge values and their sum, the determinant, and the triangle integral against
dgdec.derham1 and dgdec.derham2 (constant forms are integrated exactly).
"""
import numpy as np

import dgfig
import dgdec as dd

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
X = np.array([[0.0, 0.0], [2.0, 0.5], [0.6, 1.6]])
al = np.array([1.0, 1.0])
edges = [(0, 1), (1, 2), (2, 0)]
vals = [al @ (X[j] - X[i]) for i, j in edges]
assert np.allclose(vals, [2.5, -0.3, -2.2]) and abs(sum(vals)) < 1e-12
# dgdec.derham1 with E rows (i, j) integrates alpha over i -> j
E = np.array(edges)
assert np.allclose(dd.derham1(lambda x: al, X, E), vals)
det = (X[1] - X[0])[0] * (X[2] - X[0])[1] - (X[1] - X[0])[1] * (X[2] - X[0])[0]
assert abs(det - 2.9) < 1e-12
assert abs(dd.derham2(lambda x: 1.0, X, np.array([[0, 1, 2]]))[0] - det / 2) < 1e-12


def arrow(ax, p, q, color, lw=1.6, zorder=5):
    ax.add_patch(FancyArrowPatch(tuple(p), tuple(q), arrowstyle="-|>", mutation_scale=12, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 3.1))

# (a)
s = np.linspace(-1.0, 3.5, 2)
for c in np.arange(-1.0, 4.01, 0.5):
    axA.plot(s, c - s, color=C["covector"], lw=0.7 if c % 1 else 1.0, alpha=0.6, zorder=0)
axA.add_patch(Polygon(X, closed=True, facecolor=C["region"], alpha=0.25, edgecolor="none", zorder=1))
offs = [(0.25, -0.35), (0.65, 0.35), (-0.75, 0.25)]
for (i, j), val, off in zip(edges, vals, offs):
    arrow(axA, X[i], X[j], C["tangent"])
    m = 0.5 * (X[i] + X[j])
    axA.text(m[0] + off[0], m[1] + off[1], rf"${val:+.1f}$", fontsize=12, color=C["tangent"], ha="center")
for k, off in zip(range(3), [(-0.2, -0.25), (0.2, -0.05), (-0.1, 0.12)]):
    axA.text(X[k, 0] + off[0], X[k, 1] + off[1], rf"$x_{k}$", fontsize=12, ha="center")
axA.text(2.55, 1.75, r"$\alpha = e^1 + e^2$", color=C["covector"], fontsize=11, ha="center")
dgfig.schematic_axes(axA, xlim=(-0.9, 3.2), ylim=(-0.7, 2.2))
fig.text(0.25, 0.97, "(a) 1-form on the edges", ha="center", va="top", fontsize=12)

# (b)
Pg = np.array([X[0], X[1], X[1] + X[2] - X[0], X[2]])
axB.add_patch(Polygon(Pg, closed=True, facecolor="none", edgecolor=C["aux"], lw=0.9, ls="--", zorder=1))
axB.add_patch(Polygon(X, closed=True, facecolor=C["accent"], alpha=0.45, edgecolor=C["main"], lw=1.0, zorder=2))
arrow(axB, X[0], X[1], C["tangent"])
arrow(axB, X[0], X[2], C["third"])
ts = np.linspace(0.3, 1.25 * np.pi + 0.3, 40)
cen = X.mean(0)
arc = cen + 0.22 * np.stack([np.cos(ts), np.sin(ts)], 1)
axB.plot(*arc.T, color=C["main"], lw=1.0, zorder=6)
arrow(axB, arc[-4], arc[-1], C["main"], lw=1.0, zorder=6)
axB.text(1.2, 1.55, r"$\det = 2.9$", fontsize=11, color="#555555")
axB.text(0.55, -0.45, r"$\int_\sigma e^1\wedge e^2 = 1.45$", fontsize=12, color="#9A6200")
for k, off in zip(range(3), [(-0.2, -0.25), (0.2, -0.05), (-0.1, 0.12)]):
    axB.text(X[k, 0] + off[0], X[k, 1] + off[1], rf"$x_{k}$", fontsize=12, ha="center")
dgfig.schematic_axes(axB, xlim=(-0.6, 3.0), ylim=(-0.7, 2.4))
fig.text(0.75, 0.97, "(b) 2-form on the face", ha="center", va="top", fontsize=12)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.08)
dgfig.save(fig, __file__)
