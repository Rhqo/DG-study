"""Figure 33.4.1: the link of a vertex as a graph, with the degree of each link vertex (Section 33.4).

The number at a link vertex j is its degree in Lk(i), which equals the number of triangles containing the edge
{i, j} (Proposition 33.4.4(a)). Four pure 2-complexes, drawn in the plane around the vertex i at the origin:
(a) six triangles around i (link: a 6-cycle), (b) three triangles in a half-disk (link: a path),
(c) three triangles on the edge {i, j} (j = (1.2, 0); one of them drawn hatched as a flap above the plane),
(d) two fans of three triangles that meet only at i.
Self-check: the links and degrees are computed with dgdec.link; dgdec.vertex_link_type gives
cycle, path, other, other; the degree of j equals the number of triangles on {i, j}.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-4-1-link-degrees.py``
"""

import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
ORANGE, GRAY = C["accent"], "#E9E9E9"


def ring(n, a0, a1, r=1.0):
    t = np.deg2rad(np.linspace(a0, a1, n))
    return np.c_[r * np.cos(t), r * np.sin(t)]


cases = []
# (a) closed fan
X = np.vstack([[0, 0], ring(6, 0, 300)])
cases.append(("(a) cycle", X, [(0, 1 + k, 1 + (k + 1) % 6) for k in range(6)], {}, "cycle"))
# (b) half fan
X = np.vstack([[0, 0], ring(4, 0, 180)])
cases.append(("(b) path", X, [(0, k, k + 1) for k in range(1, 4)], {}, "path"))
# (c) three triangles on the edge {0, 1}
X = np.array([[0, 0], [1.2, 0], [0.55, 0.95], [0.55, -0.95], [0.95, 0.5]])
cases.append(("(c) a vertex of degree 3", X, [(0, 1, 2), (0, 1, 3), (0, 1, 4)], {2: "flap"}, "other"))
# (d) two fans that share only the vertex 0
X = np.vstack([[0, 0], ring(4, 20, 160), ring(4, 200, 340)])
cases.append(("(d) two components", X, [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 5, 6), (0, 6, 7), (0, 7, 8)], {},
              "other"))

fig, axs = plt.subplots(2, 2, figsize=(6.0, 5.6))
for ax, (title, X, F, style, expect) in zip(axs.ravel(), cases):
    K = dd.generate_complex(F)
    assert dd.vertex_link_type(K, 0) == expect
    Lk = dd.link(K, 0)
    ledges = [s for s in Lk if len(s) == 2]
    deg = {v: sum(v in e for e in ledges) for (v,) in [s for s in Lk if len(s) == 1]}
    for j, d in deg.items():                               # degree in Lk(0) = number of triangles on {0, j}
        assert d == sum(1 for f in F if 0 in f and j in f)
    for t, f in enumerate(F):
        hatch = style.get(t) == "flap"
        ax.add_patch(Polygon(X[list(f)], closed=True, facecolor="none" if hatch else GRAY, hatch="////" if hatch
                             else None, edgecolor="#9a9a9a", lw=0.9, zorder=1 if not hatch else 2))
    for e in ledges:
        ax.plot(*X[list(e)].T, color=ORANGE, lw=3.0, zorder=3, solid_capstyle="round")
    ax.plot(*X[0], "o", color="black", ms=6, zorder=5)
    ax.text(X[0, 0] - 0.12, X[0, 1] + 0.1, r"$i$", fontsize=13, zorder=6)
    for j, d in deg.items():
        ax.plot(*X[j], "o", color=ORANGE, ms=6.5, zorder=4)
        dirn = X[j] - X[0]
        q = X[j] + 0.22 * dirn / np.linalg.norm(dirn)
        ax.text(*q, str(d), fontsize=12, ha="center", va="center", color="#8a5a00", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=ORANGE, lw=0.7))
    if expect == "other" and title.startswith("(c)"):
        ax.text(X[1, 0] + 0.08, X[1, 1] - 0.2, r"$j$", fontsize=13)
    ax.set_title(title, fontsize=12)
    dgfig.schematic_axes(ax, xlim=(-1.45, 1.6), ylim=(-1.35, 1.35))
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.12)
dgfig.save(fig, __file__)
