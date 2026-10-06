"""Figure 0.7.2: dx∧dy(v, w) = det(v w) is the signed area of the parallelogram spanned by v and w (Section 0.7).

v = (2, 1/2), w = (1/2, 3/2): dx∧dy(v, w) = 2·(3/2) − (1/2)·(1/2) = 11/4 > 0 (counterclockwise, left panel),
dx∧dy(w, v) = −11/4 (clockwise, right panel). The area of the parallelogram is 11/4 in both panels.

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-7-2-wedge.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

C = dgfig.COLORS
LW = dgfig.LW

v = np.array([2.0, 0.5])
w = np.array([0.5, 1.5])


def wedge(a, b):
    return a[0] * b[1] - a[1] * b[0]


# self-checks (§12.1)
assert np.isclose(wedge(v, w), 11 / 4) and np.isclose(wedge(w, v), -11 / 4) and wedge(v, v) == 0
area = abs(np.cross(np.append(v, 0), np.append(w, 0))[2])
assert np.isclose(area, 11 / 4)

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.7))
for k, ax in enumerate(axs):
    a, b = (v, w) if k == 0 else (w, v)
    dgfig.schematic_axes(ax, (-0.35, 2.85), (-0.45, 2.25))
    ax.add_patch(Polygon([[0, 0], a, a + b, b], closed=True, facecolor=C["region"], alpha=0.3, lw=0))
    ax.plot(*np.array([a, a + b, b]).T, color=C["aux"], lw=0.8, ls=(0, (3, 2)))
    for vec, col, lab in ((a, C["tangent"], "first"), (b, C["third"], "second")):
        ax.add_patch(FancyArrowPatch((0, 0), tuple(vec), arrowstyle="-|>", mutation_scale=14, color=col,
                                     lw=LW["vector"] + 0.2, shrinkA=0, shrinkB=0, zorder=5))
    # orientation: arc from the first vector towards the second
    a0, a1 = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
    ts = np.linspace(a0, a1, 50)
    rr = 0.55
    ax.plot(rr * np.cos(ts[:-4]), rr * np.sin(ts[:-4]), color=C["main"], lw=1.0)
    ax.add_patch(FancyArrowPatch((rr * np.cos(ts[-5]), rr * np.sin(ts[-5])), (rr * np.cos(ts[-1]), rr * np.sin(ts[-1])),
                                 arrowstyle="-|>", mutation_scale=10, color=C["main"], lw=1.0))
    ax.plot(0, 0, "o", color=C["main"], ms=3.5, zorder=6)
    c = (a + b) / 2
    s = wedge(a, b)
    ax.text(c[0] + 0.05, c[1], r"$+11/4$" if s > 0 else r"$-11/4$", fontsize=12, ha="center", va="center")
axs[0].text(v[0] + 0.05, v[1] - 0.28, r"$v$", fontsize=13, color=C["tangent"])
axs[0].text(w[0] - 0.3, w[1] + 0.05, r"$w$", fontsize=13, color=C["third"])
axs[1].text(v[0] + 0.05, v[1] - 0.28, r"$v$", fontsize=13, color=C["third"])
axs[1].text(w[0] - 0.3, w[1] + 0.05, r"$w$", fontsize=13, color=C["tangent"])
axs[0].set_title(r"$dx\wedge dy\,(v, w)$", fontsize=12)
axs[1].set_title(r"$dx\wedge dy\,(w, v)$", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)
