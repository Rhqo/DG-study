"""Figure 34.1.2: what a 2-vector in R^2 remembers (Section 34.1, Proposition 34.1.6 and 34.1.9).

(a) Three ordered pairs built from Chapter 1's running basis b1 = (2, 1), b2 = (-1, 1):
    (b1, b2), (b1, b1 + b2) and (b2, -b1). Their change-of-basis matrices from (b1, b2) are
    I, [[1, 1], [0, 1]] and [[0, -1], [1, 0]], all with determinant 1, so all three give the same
    2-vector b1 ^ b2 = 3 e1 ^ e2 (signed area 3, counterclockwise). They are drawn translated apart.
(b) The reversed pair (b2, b1) gives -3 e1 ^ e2 (clockwise), and the degenerate pair (b1, 2 b1) gives 0.
Self-checks: the signed areas det[x y] are 3, 3, 3, -3, 0 and each curved arrow turns from the first
vector toward the second.
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
b1 = np.array([2.0, 1.0])
b2 = np.array([-1.0, 1.0])


def det2(x, y):
    return x[0] * y[1] - x[1] * y[0]


pairs_a = [(b1, b2, r"$(b_1,\,b_2)$"), (b1, b1 + b2, r"$(b_1,\,b_1+b_2)$"), (b2, -b1, r"$(b_2,\,-b_1)$")]
for x, y, _ in pairs_a:
    assert abs(det2(x, y) - 3.0) < 1e-12
assert abs(det2(b2, b1) + 3.0) < 1e-12 and abs(det2(b1, 2 * b1)) < 1e-12


def arrow(ax, base, vec, color, lw=1.5, zorder=5):
    ax.add_patch(FancyArrowPatch(tuple(base), tuple(base + vec), arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=zorder))


def turn_arc(ax, center, x, y, r=0.32, color="#000000"):
    """Curved arrow around center turning from the direction of x toward the direction of y (3/4 turn)."""
    s = np.sign(det2(x, y))
    a0 = np.arctan2(x[1], x[0])
    ts = a0 + s * np.linspace(0.15, 1.35 * np.pi, 60)
    pts = center + r * np.stack([np.cos(ts), np.sin(ts)], 1)
    ax.plot(*pts.T, color=color, lw=1.1, zorder=6)
    ax.add_patch(FancyArrowPatch(tuple(pts[-4]), tuple(pts[-1]), arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=1.1, shrinkA=0, shrinkB=0, zorder=6))
    # the arc really turns in the sense of x -> y
    assert s * det2(pts[10] - center, pts[11] - center) > 0


def parallelogram(ax, o, x, y, fill, lab, lab_xy):
    P = np.array([o, o + x, o + x + y, o + y])
    ax.add_patch(Polygon(P, closed=True, facecolor=fill, alpha=0.32, edgecolor="none", zorder=1))
    arrow(ax, o, x, C["tangent"])
    arrow(ax, o, y, C["third"])
    turn_arc(ax, o + 0.5 * (x + y), x, y)
    ax.text(*lab_xy, lab, ha="center", va="top", fontsize=11)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw=dict(width_ratios=[2.6, 1.0]))

offs = [np.array([1.0, 0.0]), np.array([4.4, 0.0]), np.array([11.4, 1.0])]
labx = [1.5, 5.9, 9.9]
for (x, y, lab), o, lx in zip(pairs_a, offs, labx):
    parallelogram(axA, o, x, y, C["region"], lab, (lx, -0.25))
dgfig.schematic_axes(axA, xlim=(-0.3, 11.9), ylim=(-1.0, 3.3))
fig.text(0.36, 0.95, r"(a) same 2-vector $3\,e_1\wedge e_2$", ha="center", va="top", fontsize=12)

# (b) reversed order and a degenerate pair
o = np.array([0.0, 0.0])
parallelogram(axB, o, b2, b1, C["accent"], r"$(b_2,\,b_1)$", (0.5, -0.35))
o2 = np.array([-1.0, -2.6])
arrow(axB, o2, 2 * b1, C["third"], lw=1.5, zorder=4)
arrow(axB, o2, b1, C["tangent"], lw=2.2, zorder=5)
axB.text(1.0, -3.3, r"$b_1\wedge 2b_1 = 0$", ha="center", va="top", fontsize=11)
dgfig.schematic_axes(axB, xlim=(-1.6, 3.4), ylim=(-3.9, 2.4))
fig.text(0.86, 0.95, r"(b) $-3\,e_1\wedge e_2$ and $0$", ha="center", va="top", fontsize=12)

fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.86, wspace=0.05)
dgfig.save(fig, __file__)
