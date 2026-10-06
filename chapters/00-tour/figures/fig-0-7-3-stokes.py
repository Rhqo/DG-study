"""Figure 0.7.3: why Stokes's theorem holds — interior edges cancel (Section 0.7). (schematic)

M = the rectangle [0, 3] × [0, 2], cut into 3 × 2 unit cells. Each cell carries its boundary loop with the
counterclockwise (Stokes) orientation, drawn slightly inside the cell. Along every interior edge the two neighbouring
loops run in opposite directions, so their contributions cancel; only the outer boundary ∂M (thick, counterclockwise)
survives. At one boundary point the outward normal (vermilion) and the boundary direction (blue) are drawn:
outward normal first, then the boundary direction, is a positively oriented pair.

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-7-3-stokes.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

C = dgfig.COLORS
LW = dgfig.LW

NX, NY = 3, 2
inset = 0.13


def cell_edges(i, j):
    """oriented edges (start, end) of the boundary of cell [i, i+1] × [j, j+1], counterclockwise"""
    a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
    return [(a, b), (b, c), (c, d), (d, a)]


# self-check (§12.1): summing all cell boundaries, interior edges cancel and the outer boundary remains
count = {}
for i in range(NX):
    for j in range(NY):
        for s, e in cell_edges(i, j):
            key = (min(s, e), max(s, e))
            count[key] = count.get(key, 0) + (1 if s < e else -1)
surviving = {k for k, val in count.items() if val != 0}
outer = set()
for i in range(NX):
    outer |= {((i, 0), (i + 1, 0)), ((i, NY), (i + 1, NY))}
for j in range(NY):
    outer |= {((0, j), (0, j + 1)), ((NX, j), (NX, j + 1))}
assert surviving == outer
# the outward normal followed by the boundary direction is positively oriented (det > 0)
nrm, tan = np.array([1.0, 0.0]), np.array([0.0, 1.0])
assert np.linalg.det(np.column_stack([nrm, tan])) > 0

fig, ax = plt.subplots(figsize=(5.6, 3.6))
dgfig.schematic_axes(ax, (-0.55, NX + 0.9), (-0.45, NY + 0.45))
ax.add_patch(plt.Rectangle((0, 0), NX, NY, facecolor=C["region"], alpha=0.22, lw=0))
for i in range(NX + 1):
    ax.plot([i, i], [0, NY], color=C["aux"], lw=0.6, ls=(0, (3, 2)))
for j in range(NY + 1):
    ax.plot([0, NX], [j, j], color=C["aux"], lw=0.6, ls=(0, (3, 2)))


def arrow(p0, p1, color, lw, ms=9, zorder=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=zorder))


for i in range(NX):
    for j in range(NY):
        x0, x1, y0, y1 = i + inset, i + 1 - inset, j + inset, j + 1 - inset
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        for (sx, sy), (ex, ey) in zip(corners[:-1], corners[1:]):
            m = (0.35 * sx + 0.65 * ex, 0.35 * sy + 0.65 * ey)
            ax.plot([sx, ex], [sy, ey], color=C["accent"], lw=1.0, zorder=4)
            arrow((sx, sy), m, C["accent"], 1.0)

# outer boundary
P = [(0, 0), (NX, 0), (NX, NY), (0, NY), (0, 0)]
ax.plot(*np.array(P).T, color=C["main"], lw=LW["main"] + 0.4, zorder=6)
for (sx, sy), (ex, ey) in zip(P[:-1], P[1:]):
    m = (0.42 * sx + 0.58 * ex, 0.42 * sy + 0.58 * ey)
    arrow((sx, sy), m, C["main"], LW["main"], ms=13)

q = np.array([NX, 0.55])
arrow(tuple(q), tuple(q + 0.6 * nrm), C["normal"], LW["vector"] + 0.4, ms=13, zorder=8)
arrow(tuple(q), tuple(q + 0.6 * tan), C["tangent"], LW["vector"] + 1.2, ms=14, zorder=8)
ax.plot(*q, "o", color=C["main"], ms=4, zorder=9)
ax.text(1.5, NY + 0.18, r"$\partial M$", fontsize=13, ha="center", va="bottom")
ax.text(1.5, 1.0, r"$M$", fontsize=13, ha="center", va="center", zorder=9,
        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
