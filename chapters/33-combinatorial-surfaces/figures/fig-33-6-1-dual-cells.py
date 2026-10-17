"""Figure 33.6.1: the dual cells of a planar patch (Section 33.6).

The complex of Figure 33.2.1 (vertex 0 at the origin, six vertices on the unit circle, six on the circle of radius
1.85; twelve triangles). Dual vertices are drawn at the triangle barycenters, a drawing choice only: the dual
complex of this section is combinatorial (its geometry is Section 36.5).
(a) all dual vertices and dual edges; a dual edge of an interior edge joins the two dual vertices, a dual edge of a
    boundary edge is cut at the edge midpoint (dashed). The dual face 0* of the interior vertex 0 is filled.
(b) three correspondences: the triangle f = {1, 2, 7} and its dual vertex f*, the edge e = {0, 1} and its dual edge
    e*, the vertex 0 and its dual face 0*.
Self-check: the interior vertex 0 has a hexagonal dual face whose corners are the six triangles around 0 in the
cyclic order of its link; the dual edge e* joins the two triangles containing e; every dual vertex has degree 3
counting cut dual edges.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-6-1-dual-cells.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
ORANGE, BLUE = C["accent"], C["tangent"]
ang1 = np.deg2rad(60 * np.arange(6))
ang2 = ang1 + np.deg2rad(30)
X = np.vstack([[0.0, 0.0], np.c_[np.cos(ang1), np.sin(ang1)], 1.85 * np.c_[np.cos(ang2), np.sin(ang2)]])
ring = lambda k: 1 + k % 6  # noqa: E731
F = np.array([(0, ring(k), ring(k + 1)) for k in range(6)] + [(ring(k + 1), ring(k), 7 + k) for k in range(6)])
assert dm.oriented_consistently(F)
bc = X[F].mean(1)                                            # drawing positions of the dual vertices
ef = dm.edge_faces(F)
he = dd.HalfedgeMesh(F)
star0 = he.vertex_faces(0)                                   # the corners of the dual face 0*, in cyclic order
assert sorted(star0) == list(range(6)) and len(star0) == 6
for a, b in zip(star0, star0[1:] + star0[:1]):
    assert len(set(F[a]) & set(F[b]) & {0}) == 1 and len(set(F[a]) & set(F[b])) == 2
deg = np.zeros(len(F), int)
for fs in ef.values():
    for f in fs:
        deg[f] += 1
assert (deg == 3).all()                                       # trivalent, counting cut dual edges


def base(ax):
    for f in F:
        ax.add_patch(Polygon(X[f], closed=True, facecolor="#F1F1F1", edgecolor="#A8A8A8", lw=0.8, zorder=0))
    ax.plot(*X.T, "o", color="#888888", ms=3, zorder=1)


fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 3.5))
base(axa)
axa.add_patch(Polygon(bc[star0], closed=True, facecolor=ORANGE, alpha=0.25, edgecolor="none", zorder=1.5))
for e, fs in ef.items():
    if len(fs) == 2:
        axa.plot(*bc[list(fs)].T, color=ORANGE, lw=1.8, zorder=3)
    else:
        m = X[list(e)].mean(0)
        axa.plot(*np.array([bc[fs[0]], m]).T, color=ORANGE, lw=1.2, ls=(0, (3, 2)), zorder=3)
axa.plot(*bc.T, "o", color=ORANGE, ms=5.5, zorder=4, mec="#8a5a00", mew=0.6)
axa.text(0.0, 0.0, r"$0^\star$", fontsize=12, ha="center", va="center", color="#8a5a00", zorder=5)
axa.set_title("(a) dual vertices and dual edges", fontsize=11.5)
dgfig.schematic_axes(axa, xlim=(-2.0, 2.0), ylim=(-1.9, 1.95))

base(axb)
f = 6                                                        # the triangle {1, 2, 7}
assert sorted(F[f]) == [1, 2, 7]
axb.add_patch(Polygon(X[F[f]], closed=True, facecolor=BLUE, alpha=0.3, edgecolor=BLUE, lw=1.4, zorder=2))
axb.plot(*bc[f], "o", color=ORANGE, ms=8, zorder=5, mec="#8a5a00", mew=0.8)
axb.text(*(bc[f] + np.array([0.18, 0.14])), r"$f^\star$", fontsize=12, color="#8a5a00", zorder=6)
axb.text(*(X[F[f]].mean(0) + np.array([0.05, -0.28])), r"$f$", fontsize=12, color=BLUE, zorder=6)
e = (0, 1)
fe = ef[e]
axb.plot(*X[list(e)].T, color=BLUE, lw=2.6, zorder=3)
axb.plot(*bc[list(fe)].T, color=ORANGE, lw=2.6, zorder=4)
axb.text(0.58, -0.17, r"$e$", fontsize=12, color=BLUE, zorder=6)
axb.text(0.38, 0.32, r"$e^\star$", fontsize=12, color="#8a5a00", zorder=6)
axb.add_patch(Polygon(bc[star0], closed=True, facecolor=ORANGE, alpha=0.22, edgecolor=ORANGE, lw=1.0, zorder=1.5))
axb.plot(*X[0], "o", color="black", ms=6, zorder=6)
axb.text(-0.32, -0.08, r"$0$", fontsize=12, zorder=6)
axb.text(-0.62, -0.62, r"$0^\star$", fontsize=12, color="#8a5a00", zorder=6)
axb.set_title(r"(b) $f \leftrightarrow f^\star$, $e \leftrightarrow e^\star$, $0 \leftrightarrow 0^\star$",
              fontsize=11.5)
dgfig.schematic_axes(axb, xlim=(-2.0, 2.0), ylim=(-1.9, 1.95))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)
