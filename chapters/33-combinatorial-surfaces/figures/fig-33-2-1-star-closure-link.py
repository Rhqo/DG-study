"""Figure 33.2.1: star, simplicial closure and link in a planar complex (Section 33.2).

The complex: a vertex i = 0 at the origin, six vertices 1..6 on the unit circle (angles 60k degrees), six vertices
7..12 on the circle of radius 1.85 (angles 60k + 30 degrees); triangles {0, k, k+1} and {k, k+1, k+6}.
(a) St({0}), (b) Cl(St({0})), (c) Lk({0}), (d) St({e}) and Lk({e}) for the edge e = {0, 1}.
The sets are computed with dgdec.star / closure / link; the script asserts their sizes and that Lk({0}) is the
hexagon and Lk({e}) the two opposite vertices.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-2-1-star-closure-link.py``
"""

import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
BLUE, ORANGE, GRAY = C["tangent"], C["accent"], "#BBBBBB"

ang1 = np.deg2rad(60 * np.arange(6))
ang2 = ang1 + np.deg2rad(30)
X = np.vstack([[0.0, 0.0], np.c_[np.cos(ang1), np.sin(ang1)], 1.85 * np.c_[np.cos(ang2), np.sin(ang2)]])
ring = lambda k: 1 + k % 6  # noqa: E731
F = [(0, ring(k), ring(k + 1)) for k in range(6)] + [(ring(k), ring(k + 1), 7 + k) for k in range(6)]
K = dd.generate_complex(F)
assert dd.is_complex(K) and [len(dd.k_simplices(K, k)) for k in range(3)] == [13, 24, 12]

St0 = dd.star(K, 0)
Cl0 = dd.closure(K, St0)
Lk0 = dd.link(K, 0)
e = (0, 1)
Ste = dd.star(K, [e])
Lke = dd.link(K, [e])
assert len(St0) == 1 + 6 + 6 and len(Cl0) == 7 + 12 + 6 and not dd.is_complex(St0) and dd.is_complex(Cl0)
assert Lk0 == {(k,) for k in range(1, 7)} | {tuple(sorted((ring(k), ring(k + 1)))) for k in range(6)}
assert Lk0 == Cl0 - St0                                    # for a vertex: St(Cl(i)) = St(i)
assert Ste == {(0, 1), (0, 1, 2), (0, 1, 6)} and Lke == {(2,), (6,)}


def draw(ax, sets, title):
    """sets: list of (simplex set, color, filled?) drawn on top of the gray complex."""
    for f in F:
        ax.add_patch(Polygon(X[list(f)], closed=True, facecolor="#F2F2F2", edgecolor="none", zorder=0))
    for s in K:
        if len(s) == 2:
            ax.plot(*X[list(s)].T, color=GRAY, lw=0.8, zorder=1)
    ax.plot(*X.T, "o", color=GRAY, ms=3.2, zorder=1)
    for S, col in sets:
        for s in S:
            if len(s) == 3:
                ax.add_patch(Polygon(X[list(s)], closed=True, facecolor=col, alpha=0.30, edgecolor="none", zorder=2))
        for s in S:
            if len(s) == 2:
                ax.plot(*X[list(s)].T, color=col, lw=2.3, zorder=3, solid_capstyle="butt")
        for s in S:
            if len(s) == 1:
                ax.plot(*X[s[0]], "o", color=col, ms=6.5, zorder=4)
    ax.set_title(title, fontsize=12)
    dgfig.schematic_axes(ax, xlim=(-2.0, 2.0), ylim=(-1.95, 1.95))


fig, axs = plt.subplots(2, 2, figsize=(6.0, 6.2))
(a, b), (c, d) = axs
draw(a, [(St0, BLUE)], r"(a) $\mathrm{St}(i)$")
# open ends of the star: the hexagon edges and ring vertices are NOT in St(i)
for k in range(6):
    a.plot(*X[[ring(k), ring(k + 1)]].T, color=BLUE, lw=0.9, ls=(0, (2.5, 2)), zorder=3)
    a.plot(*X[ring(k)], "o", mfc="white", mec=BLUE, mew=1.3, ms=6.5, zorder=4)
draw(b, [(Cl0, BLUE)], r"(b) $\mathrm{Cl}(\mathrm{St}(i))$")
draw(c, [(Lk0, ORANGE)], r"(c) $\mathrm{Lk}(i)$")
c.plot(*X[0], "o", mfc="white", mec="black", mew=1.2, ms=6.5, zorder=5)
draw(d, [(Ste, BLUE), (Lke, ORANGE)], r"(d) $\mathrm{St}(e)$ and $\mathrm{Lk}(e)$")
for ax in (a, b, c):
    ax.text(X[0, 0] - 0.07, X[0, 1] - 0.36, r"$i$", fontsize=13, zorder=6)
d.text(0.45, -0.25, r"$e$", fontsize=13, color=BLUE, zorder=6)
d.text(*(X[0] + np.array([-0.05, -0.32])), r"$i$", fontsize=12, zorder=6)
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.04, hspace=0.1)
dgfig.save(fig, __file__)
