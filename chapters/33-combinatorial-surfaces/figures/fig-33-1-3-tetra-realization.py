"""Figure 33.1.3: one abstract complex, two pictures (Section 33.1).

The abstract complex is the boundary of the tetrahedron: all nonempty proper subsets of {0, 1, 2, 3}
(4 vertices, 6 edges, 4 triangles; 14 simplices).
(a) a geometric realization in R^3: the regular tetrahedron of chapters/33-combinatorial-surfaces/lib/ch33.py
    (orthographic view, the one hidden edge dashed).
(b) a drawing in the plane: 1, 2, 3 at the corners of a big triangle, 0 at its centroid. The triangles {0,1,2},
    {0,2,3}, {0,1,3} tile the big triangle and {1,2,3} is the big triangle itself, so it overlaps the other three:
    the drawing is not a geometric simplicial complex in R^2.
Self-check: the 14 simplices of the two pictures are the same abstract complex; in (a) every pair of the four
triangles meets in a common edge; in (b) the triangle {1,2,3} contains the other three.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-1-3-tetra-realization.py``
"""

import pathlib
import sys

import dgmesh as dm
import dgdec as dd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
V, F = ch33.tetrahedron()
K = dd.complex_from_faces(F)
assert len(K) == 14 and dd.is_complex(K) and sorted(len(s) for s in K).count(3) == 4
for f in range(4):
    for g in range(f + 1, 4):
        assert len(set(F[f]) & set(F[g])) == 2          # any two triangles share an edge

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.4, 3.3))

# ---------------------------------------------------------------- (a) realization in R^3
# rotate (for drawing only) so that vertex 0 points up; a rotation keeps the realization a regular tetrahedron
a0, b0 = V[0] / np.linalg.norm(V[0]), np.array([0.0, 0.0, 1.0])
ax_ = np.cross(a0, b0)
Kx = np.array([[0, -ax_[2], ax_[1]], [ax_[2], 0, -ax_[0]], [-ax_[1], ax_[0], 0]])
Rot = np.eye(3) + Kx + Kx @ Kx * ((1 - a0 @ b0) / (ax_ @ ax_))
W = V @ Rot.T
assert np.allclose(Rot @ Rot.T, np.eye(3)) and np.allclose(W[0], [0, 0, 1])
cam = dm.Ortho(elev=15, azim=-20)
front = [np.cross(W[f[1]] - W[f[0]], W[f[2]] - W[f[0]]) @ cam.view > 0 for f in F]
P = cam(W)
hidden = []
for a, b in dm.edges(F):
    adj = [fr for f, fr in zip(F, front) if a in f and b in f]
    if any(adj):
        axa.plot(*P[[a, b]].T, color=C["main"], lw=1.4, zorder=5)
    else:
        hidden.append((a, b))
        axa.plot(*P[[a, b]].T, color=C["aux"], lw=0.9, ls=(0, (3, 2.5)), zorder=4)
assert len(hidden) == 1
for f, fr in zip(F, front):
    if fr:
        axa.add_patch(Polygon(P[list(f)], closed=True, facecolor=C["region"], alpha=0.28, edgecolor="none",
                              zorder=2))
cen = P.mean(0)
for v in range(4):
    d = P[v] - cen
    axa.plot(*P[v], "o", color=C["main"], ms=5, zorder=7)
    q = P[v] + 0.13 * d / np.linalg.norm(d)
    axa.text(*q, rf"${v}$", fontsize=13, ha="center", va="center", zorder=8)
axa.set_title(r"(a) a realization in $\mathbb{R}^3$", fontsize=12)
dgfig.schematic_axes(axa, xlim=(P[:, 0].min() - 0.25, P[:, 0].max() + 0.25),
                     ylim=(P[:, 1].min() - 0.25, P[:, 1].max() + 0.3))

# ---------------------------------------------------------------- (b) a planar drawing
Q = {1: np.array([0.0, 0.0]), 2: np.array([2.0, 0.0]), 3: np.array([1.0, 1.7])}
Q[0] = (Q[1] + Q[2] + Q[3]) / 3
big = np.array([Q[1], Q[2], Q[3]])
axb.add_patch(Polygon(big, closed=True, fill=False, hatch="///", edgecolor=C["accent"], lw=0.0, zorder=5))
axb.add_patch(Polygon(big, closed=True, fill=False, edgecolor=C["accent"], lw=3.0, zorder=2))
for f in [(0, 1, 2), (0, 2, 3), (0, 1, 3)]:
    T = np.array([Q[v] for v in f])
    axb.add_patch(Polygon(T, closed=True, facecolor=C["region"], alpha=0.35, edgecolor="none", zorder=3))
for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]:
    axb.plot(*np.array([Q[a], Q[b]]).T, color=C["main"], lw=1.3, zorder=4)
labs = {0: (0.15, 0.1), 1: (-0.12, -0.08), 2: (0.12, -0.08), 3: (0.0, 0.14)}
for v in range(4):
    axb.plot(*Q[v], "o", color=C["main"], ms=5, zorder=7)
    axb.text(*(Q[v] + np.array(labs[v])), rf"${v}$", fontsize=13, ha="center", va="center", zorder=8)
# the triangle {1,2,3} contains the other three (checked with barycentric coordinates)
for f in [(0, 1, 2), (0, 2, 3), (0, 1, 3)]:
    for v in f:
        assert np.all(dd.barycentric_coordinates(Q[v], big) >= -1e-12)
axb.text(1.0, -0.33, r"hatched: the triangle $\{1, 2, 3\}$", color="#8a5a00", fontsize=11, ha="center")
axb.set_title(r"(b) a drawing in $\mathbb{R}^2$", fontsize=12)
dgfig.schematic_axes(axb, xlim=(-0.3, 2.3), ylim=(-0.5, 1.95))

fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.12)
dgfig.save(fig, __file__)
