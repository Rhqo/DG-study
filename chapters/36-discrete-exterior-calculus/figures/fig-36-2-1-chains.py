"""Figure 36.2.1: chains and their boundaries on the grid of Example 36.1.2 (Section 36.2).

(a) the 1-chain c = [0,3] + [3,4] + [4,8] (a path from vertex 0 to vertex 8) and its boundary dc = [8] - [0]:
    the inner vertices 3 and 4 receive +1 from one edge and -1 from the next.
(b) the 2-chain b = [0,3,4] + [0,4,1] + [3,6,4] (three counterclockwise faces) and its boundary: the loop
    0 -> 3 -> 6 -> 4 -> 1 -> 0; the shared edges [0,4] and [3,4] appear twice with opposite signs and cancel.
Self-check: the boundaries are computed with lib/ch36.boundary_of_chain and with the matrices
dgdec.boundary1/boundary2, and the boundary of each boundary is zero.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-2-1-chains.py``
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

import ch36  # noqa: E402
import dgdec as dd  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

V, F = ch36.grid(2)
E, index = dd.edge_list(F)
B1, B2 = dd.boundary1(F, len(V)), dd.boundary2(F)
c = {(0, 3): 1, (3, 4): 1, (4, 8): 1}
dc = ch36.boundary_of_chain(c)
assert dc == {(8,): 1, (0,): -1}
cvec = np.zeros(len(E))
for (i, j), v in c.items():
    cvec[index[(i, j)]] = v
assert np.array_equal(B1 @ cvec, np.eye(len(V))[8] - np.eye(len(V))[0])
faces = [(0, 3, 4), (0, 4, 1), (3, 6, 4)]
rows = [F.tolist().index(list(t)) for t in faces]
b = {tuple(sorted(t)): dd.permutation_sign(t) for t in faces}
db = ch36.boundary_of_chain(b)
assert ch36.boundary_of_chain(db) == {}
bvec = np.zeros(len(F))
bvec[rows] = 1
dbv = B2 @ bvec
loop = [(0, 3), (3, 6), (6, 4), (4, 1), (1, 0)]
expect = np.zeros(len(E))
for i, j in loop:
    expect[index[(min(i, j), max(i, j))]] += 1 if i < j else -1
assert np.array_equal(dbv, expect) and not np.any(B1 @ dbv)
assert all(db.get(tuple(sorted(e)), 0) == (1 if e[0] < e[1] else -1) for e in loop) and len(db) == 5

C = dgfig.COLORS
BLUE, RED, GRAY = C["tangent"], C["normal"], "#888888"
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.5))
for ax in (axa, axb):
    ch36.draw_mesh2d(ax, V, F, face="#F7F7F7", edge="#D2D2D2", lw=0.7)
    for i, p in enumerate(V):
        ax.text(*(p + np.array([-0.06, -0.075])), str(i), fontsize=9, color="#666666", ha="center", va="center")

for (i, j) in c:
    ch36.edge_arrow(axa, V[i], V[j], BLUE, lw=2.2, head=12, frac=(0.08, 0.92))
for v, val, col in ((0, r"$-1$", RED), (8, r"$+1$", BLUE), (3, r"$0$", GRAY), (4, r"$0$", GRAY)):
    axa.plot(*V[v], "o", ms=11 if col != GRAY else 7, color="white", mec=col, mew=2.0, zorder=6)
    off = np.array([0.08, 0.075]) if v != 8 else np.array([-0.1, 0.03])
    axa.text(*(V[v] + off), val, fontsize=12, color=col, ha="center", va="center", zorder=7)
axa.text(0.5, -0.2, r"$\partial c = [8] - [0]$", fontsize=12, ha="center")
axa.set_title(r"(a) $c = [0,3] + [3,4] + [4,8]$", fontsize=11)

for r in rows:
    axb.add_patch(Polygon(V[F[r]], closed=True, facecolor=C["region"], alpha=0.35, edgecolor="none", zorder=1))
for i, j in loop:
    ch36.edge_arrow(axb, V[i], V[j], BLUE, lw=2.2, head=12, frac=(0.1, 0.9))
for i, j in ((0, 4), (3, 4)):
    axb.plot(*V[[i, j]].T, ls=(0, (3, 2)), color=GRAY, lw=1.4, zorder=3)
    m = 0.5 * (V[i] + V[j]) + np.array([0.0, 0.07])
    axb.text(*m, r"$+1-1$", fontsize=9.5, color=GRAY, ha="center", va="center", zorder=7,
             bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
axb.text(0.5, -0.2, r"$\partial b$ = the loop $0\to3\to6\to4\to1\to0$", fontsize=11, ha="center")
axb.set_title(r"(b) $b = [0,3,4] + [0,4,1] + [3,6,4]$", fontsize=11)
for ax in (axa, axb):
    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.27, 1.08)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)
