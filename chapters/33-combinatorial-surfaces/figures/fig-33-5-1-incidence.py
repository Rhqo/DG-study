"""Figure 33.5.1: one small mesh and its incidence matrices (Section 33.5).

The mesh: vertex 0 at the origin and vertices 1..5 on the unit circle (angles 90 + 72 (k - 1) degrees), triangles
f_t = (0, t+1, t+2) for t = 0..4 with 5 + 1 -> 1 (a closed fan). Edges are numbered in the lexicographic order of
dgdec.edge_list (e_0 = 01, ..., e_9 = 45).
(a) the mesh with vertex numbers (black), edge numbers (blue) and triangle numbers (gray).
(b) A_1 (5 x 10, triangles x edges) above A_0^T (6 x 10, vertices x edges), sharing the edge columns; a filled
    cell is a 1.
Self-check: row sums of A_0 are 2, row sums of A_1 are 3, column sums of A_1 are 2 for the five spokes and 1 for the
five rim edges, and A_0^T A_0 = D + W (Proposition 33.5.3).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-5-1-incidence.py``
"""

import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

C = dgfig.COLORS
ang = np.deg2rad(90 + 72 * np.arange(5))
X = np.vstack([[0, 0], np.c_[np.cos(ang), np.sin(ang)]])
F = np.array([(0, 1 + t, 1 + (t + 1) % 5) for t in range(5)])
E, index = dd.edge_list(F)
A0 = np.abs(dd.d0(F)).astype(int)
A1 = np.abs(dd.d1(F)).astype(int)
assert (A0.sum(1) == 2).all() and (A1.sum(1) == 3).all()
assert sorted(A1.sum(0).tolist()) == [1] * 5 + [2] * 5
W = np.zeros((6, 6), int)
for a, b in E:
    W[a, b] = W[b, a] = 1
assert np.array_equal(A0.T @ A0, np.diag(A0.sum(0)) + W)

fig = plt.figure(figsize=(6.8, 3.6))
axa = fig.add_axes([0.0, 0.02, 0.42, 0.86])
axb = fig.add_axes([0.47, 0.02, 0.52, 0.86])

# ---------------------------------------------------------------- (a) the mesh with numbers
for t, f in enumerate(F):
    axa.add_patch(Polygon(X[f], closed=True, facecolor="#E8E8E8", edgecolor="none", zorder=0))
    axa.text(*X[f].mean(0), rf"$f_{t}$", fontsize=11, color="#555555", ha="center", va="center")
for e, (a, b) in enumerate(E):
    axa.plot(*X[[a, b]].T, color="black", lw=1.2, zorder=1)
    m = (X[a] + X[b]) / 2
    off = 0.11 * (m / np.linalg.norm(m)) if np.linalg.norm(m) > 0.6 else np.array([0.0, 0.0])
    axa.text(*(m + off), rf"$e_{e}$", fontsize=10, color=C["tangent"], ha="center", va="center", zorder=3,
             bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.85))
for v in range(6):
    axa.plot(*X[v], "o", color="black", ms=4.5, zorder=4)
    q = X[v] * 1.17 if v else np.array([0.13, -0.12])
    axa.text(*q, str(v), fontsize=12, ha="center", va="center", zorder=5)
axa.set_title("(a) mesh", fontsize=12)
dgfig.schematic_axes(axa, xlim=(-1.3, 1.3), ylim=(-1.2, 1.35))

# ---------------------------------------------------------------- (b) A1 over A0^T, sharing the edge columns
gap = 1.2
M = [(A1, "f"), (A0.T, "")]
y0 = 0.0
for k, (Mat, rl) in enumerate(M):
    nr, nc = Mat.shape
    for r in range(nr):
        for c in range(nc):
            axb.add_patch(Rectangle((c, -(y0 + r) - 1), 1, 1, facecolor="black" if Mat[r, c] else "white",
                                    edgecolor="#BBBBBB", lw=0.6))
        axb.text(-0.35, -(y0 + r) - 0.5, rf"$f_{r}$" if rl else str(r), ha="right", va="center", fontsize=10.5,
                 color="#555555" if rl else "black")
    axb.text(nc + 0.35, -(y0 + nr / 2), r"$A_1$" if k == 0 else r"$A_0^{\mathsf{T}}$", fontsize=13,
             ha="left", va="center")
    y0 += nr + gap
for c in range(len(E)):
    axb.text(c + 0.5, 0.35, rf"$e_{c}$", ha="center", va="bottom", fontsize=9.5, color=C["tangent"], rotation=0)
axb.set_title("(b) incidence matrices", fontsize=12, pad=16)
dgfig.schematic_axes(axb, xlim=(-1.3, len(E) + 1.6), ylim=(-y0 + gap - 0.3, 1.2))
dgfig.save(fig, __file__)
