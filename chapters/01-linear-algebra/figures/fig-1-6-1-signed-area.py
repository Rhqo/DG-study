"""그림 1.6.1: 부호 있는 넓이와 방향 (1.6절, 예 1.6.13, 연습 1.6.5).

(a) (b_1, b_2), b_1 = (2,1), b_2 = (-1,1): det = 3 > 0. b_1에서 b_2로 반시계 방향으로 돈다(양의 방향).
(b) (b_2, b_1): det = -3 < 0. 같은 평행사변형이지만 시계 방향(음의 방향).
(c) 밀기(shear): (b_1 + b_2, b_2)의 평행사변형(실선)과 (b_1, b_2)의 평행사변형(점선).
    밑변 b_2와 높이가 같으므로 넓이가 같다: det(b_1 + b_2, b_2) = det(b_1, b_2) = 3.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-6-1-signed-area.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

C = dgfig.COLORS
LW = dgfig.LW

b1 = np.array([2.0, 1.0])
b2 = np.array([-1.0, 1.0])
det2 = lambda a, b: a[0] * b[1] - a[1] * b[0]
assert det2(b1, b2) == 3 and det2(b2, b1) == -3
assert det2(b1 + b2, b2) == det2(b1, b2)


def arrow(ax, base, vec, color, lw=None, z=6):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


def turn(ax, a, b, color, rad):
    """a 방향에서 b 방향으로 도는 원호 화살표 (원점 근처)."""
    ang_a, ang_b = np.arctan2(a[1], a[0]), np.arctan2(b[1], b[0])
    p0 = 0.55 * np.array([np.cos(ang_a), np.sin(ang_a)])
    p1 = 0.55 * np.array([np.cos(ang_b), np.sin(ang_b)])
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                                 mutation_scale=10, color=color, lw=1.1, zorder=7))


def frame(ax, xlim, ylim, tag):
    dgfig.schematic_axes(ax, xlim, ylim)
    ax.plot(list(xlim), [0, 0], color=C["aux"], lw=0.5, zorder=0)
    ax.plot([0, 0], list(ylim), color=C["aux"], lw=0.5, zorder=0)
    ax.plot(0, 0, "o", color=C["main"], ms=2.5, zorder=8)
    ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11)


WB = dict(facecolor="white", edgecolor="none", pad=0.3)
XL, YL = (-1.5, 2.4), (-0.5, 3.15)
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.75))

# (a) 양의 방향
ax = axs[0]
frame(ax, XL, YL, "(a)")
ax.add_patch(Polygon([[0, 0], b1, b1 + b2, b2], closed=True, color=C["region"], alpha=0.3, lw=0, zorder=1))
arrow(ax, (0, 0), b1, C["tangent"]); arrow(ax, (0, 0), b2, C["tangent"])
turn(ax, b1, b2, C["main"], rad=0.45)
ax.text(*(b1 + [-0.2, -0.38]), r"$b_1$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(b2 + [-0.25, 0.1]), r"$b_2$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.text(1.0, 2.5, r"$\det(b_1, b_2) = 3$", color=C["main"], fontsize=11, bbox=WB, zorder=9, ha="center")
ax.set_xlim(*XL); ax.set_ylim(*YL)

# (b) 음의 방향
ax = axs[1]
frame(ax, XL, YL, "(b)")
ax.add_patch(Polygon([[0, 0], b1, b1 + b2, b2], closed=True, color=C["normal"], alpha=0.12, lw=0, zorder=1))
arrow(ax, (0, 0), b2, C["tangent"]); arrow(ax, (0, 0), b1, C["tangent"])
turn(ax, b2, b1, C["main"], rad=-0.45)
ax.text(*(b2 + [-0.25, 0.1]), r"$b_2$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.text(*(b1 + [-0.2, -0.38]), r"$b_1$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.text(1.0, 2.5, r"$\det(b_2, b_1) = -3$", color=C["main"], fontsize=11, bbox=WB, zorder=9, ha="center")
ax.set_xlim(*XL); ax.set_ylim(*YL)

# (c) 밀기
ax = axs[2]
frame(ax, (-1.5, 2.4), YL, "(c)")
ax.add_patch(Polygon([[0, 0], b1, b1 + b2, b2], closed=True, fill=False, ec=C["aux"], lw=0.9, ls="--", zorder=2))
c1 = b1 + b2
ax.add_patch(Polygon([[0, 0], c1, c1 + b2, b2], closed=True, color=C["region"], alpha=0.3, lw=0, zorder=1))
arrow(ax, (0, 0), c1, C["tangent"]); arrow(ax, (0, 0), b2, C["tangent"])
ax.text(*(c1 + [0.1, -0.15]), r"$b_1 + b_2$", color=C["tangent"], fontsize=11, bbox=WB, zorder=9)
ax.text(*(b2 + [-0.25, 0.1]), r"$b_2$", color=C["tangent"], fontsize=12, bbox=WB, zorder=9)
ax.set_xlim(-1.5, 2.4); ax.set_ylim(*YL)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
