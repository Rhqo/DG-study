"""Figure 35.4.3: the boundary of the boundary of a cube is empty (Section 35.4, Proposition 35.4.3 and the
derivation of d d = 0).

The unit cube Q = [0, 1]^3 with the standard orientation. Each face {x^i = c} is parametrized by the remaining two
coordinates in increasing order; its Stokes orientation (outward normal first) is that parametrization times the sign
(-1)^(i-1) for c = 1 and (-1)^i for c = 0 (Proposition 35.4.3). The boundary loop of each face is drawn slightly
inside the face: face x = 1 (blue), y = 1 (green), z = 1 (orange). The vermillion arrows are the outward normals.
On each of the three edges shared by two visible faces the two loops run in opposite directions, so the edge cancels.
Self-checks: for all 6 faces the induced loop satisfies det(n, w1, w2) > 0 with (w1, w2) the oriented parameter
directions; each of the 12 edges of the cube lies on exactly two face loops and is traversed in opposite directions.
"""
import itertools

import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
E3 = np.eye(3)


def face_loop(i, c):
    """Corners of the face x^i = c in Stokes orientation, and its outward normal."""
    others = [j for j in range(3) if j != i]
    sign = (-1) ** i if c == 1 else (-1) ** (i + 1)    # 0-based i: (-1)^(i-1) in 1-based numbering for c = 1
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    if sign < 0:
        sq = sq[::-1]
    pts = []
    for a, b in sq:
        q = np.zeros(3)
        q[i] = c
        q[others[0]], q[others[1]] = a, b
        pts.append(q)
    n = E3[i] if c == 1 else -E3[i]
    # check: outward normal first
    w1, w2 = pts[1] - pts[0], pts[2] - pts[1]
    assert np.linalg.det(np.column_stack([n, w1, w2])) > 0
    return np.array(pts), n


edge_dirs = {}
for i, c in itertools.product(range(3), (0, 1)):
    pts, n = face_loop(i, c)
    for k in range(4):
        a, b = tuple(pts[k]), tuple(pts[(k + 1) % 4])
        key = tuple(sorted([a, b]))
        edge_dirs.setdefault(key, []).append(+1 if (a, b) == key else -1)
assert len(edge_dirs) == 12 and all(sorted(v) == [-1, 1] for v in edge_dirs.values())

fig = plt.figure(figsize=(4.8, 4.6))
ax = dgfig.axes3d(fig, elev=24, azim=35)
cols = {0: C["tangent"], 1: C["third"], 2: C["accent"]}
for i in range(3):
    pts, n = face_loop(i, 1)
    ax.add_collection3d(Poly3DCollection([pts], facecolor=C["surface"], alpha=0.45, edgecolor=C["main"], lw=0.8,
                                         zorder=1))
    cen = pts.mean(0)
    inset = cen + 0.8 * (pts - cen)
    for k in range(4):
        a, b = inset[k], inset[(k + 1) % 4]
        dgfig.arrow3d(ax, a + 0.12 * (b - a), 0.76 * (b - a), role=cols[i], lw=1.6, head=10, zorder=5)
    dgfig.arrow3d(ax, cen, 0.45 * n, role="normal", lw=1.4, head=10, zorder=6)
# hidden edges of the cube
for key in edge_dirs:
    if key[0] == (0.0, 0.0, 0.0):            # the three edges at the origin are hidden
        ax.plot(*zip(key[0], key[1]), color=C["aux"], lw=0.7, ls="--", zorder=0)
for i, lab in zip(range(3), ("x = 1", "y = 1", "z = 1")):
    pts, n = face_loop(i, 1)
    q = pts.mean(0) + 0.62 * n
    ax.text(*q, lab, color=cols[i], fontsize=11, ha="center", zorder=7)
fig.text(0.5, 0.05, r"each edge: two opposite arrows $\Rightarrow$ $\partial\partial Q = 0$", ha="center", fontsize=11)
dgfig.equal_aspect(ax, np.array([[0, 0, 0], [1.45, 1.45, 1.45]]), zoom=1.05)
dgfig.save(fig, __file__)
