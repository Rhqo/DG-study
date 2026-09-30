"""그림 1.1.1: 기저 = 생성 + 일차독립 (1.1절, 예 1.1.14와 비예 1.1.15).

(a) 기저 b_1 = (2, 1), b_2 = (-1, 1)의 비스듬한 좌표격자와 v = (1, 2) = b_1 + b_2
(b) 목록 (b_1, -b_1): 생성하는 것은 직선 하나뿐이고 v는 그 위에 없다
(c) 목록 (b_1, b_2, w), w = (0, 1): v = b_1 + b_2 = -b_2 + 3w 로 표현이 두 가지다

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-1-1-bases.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

# 본문(예 1.1.14, 비예 1.1.15)과 같은 값
b1 = np.array([2.0, 1.0])
b2 = np.array([-1.0, 1.0])
w = np.array([0.0, 1.0])
v = np.array([1.0, 2.0])

# 자기검사 (§12.1)
assert abs(np.linalg.det(np.column_stack([b1, b2])) - 3.0) < 1e-12      # 기저 (det = 3)
assert np.allclose(np.linalg.solve(np.column_stack([b1, b2]), v), [1, 1])  # v = b1 + b2
assert np.allclose(-b2 + 3 * w, v)                                        # v = -b2 + 3w
assert abs(b1[0] * v[1] - b1[1] * v[0]) > 1e-9                            # v는 span(b1) 밖


def arrow(ax, base, vec, color, lw=None, z=5):
    base = np.asarray(base, float)
    tip = base + np.asarray(vec, float)
    ax.annotate("", xy=tip, xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0),
                zorder=z)


def frame(ax, xlim, ylim, title):
    dgfig.schematic_axes(ax, xlim, ylim)
    ax.plot(list(xlim), [0, 0], color=C["aux"], lw=0.5, zorder=0)
    ax.plot([0, 0], list(ylim), color=C["aux"], lw=0.5, zorder=0)
    ax.plot(0, 0, "o", color=C["main"], ms=2.5, zorder=6)
    ax.text(0.02, 0.98, title, transform=ax.transAxes, ha="left", va="top", fontsize=11)


XL, YL = (-2.4, 3.4), (-1.6, 3.3)
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.75))

# (a) 기저와 비스듬한 격자 ------------------------------------------------------
ax = axs[0]
frame(ax, XL, YL, "(a)")
ts = np.linspace(-6, 6, 2)
for k in range(-4, 5):
    for P in (k * b1 + np.outer(ts, b2), k * b2 + np.outer(ts, b1)):
        ax.plot(P[:, 0], P[:, 1], color=C["aux"], lw=0.45, ls=(0, (3, 2.5)), zorder=1)
ax.set_xlim(*XL); ax.set_ylim(*YL)
arrow(ax, (0, 0), b1, C["tangent"])
arrow(ax, (0, 0), b2, C["tangent"])
arrow(ax, (0, 0), v, C["main"], lw=1.8)
ax.plot(*np.array([b1, v]).T, color=C["tangent"], lw=0.8, ls="--", zorder=3)
ax.plot(*np.array([b2, v]).T, color=C["tangent"], lw=0.8, ls="--", zorder=3)
ax.text(*(b1 + [0.08, -0.38]), r"$b_1$", color=C["tangent"], fontsize=12)
ax.text(*(b2 + [-0.55, -0.05]), r"$b_2$", color=C["tangent"], fontsize=12)
ax.text(*(v + [-1.35, 0.22]), r"$v = b_1 + b_2$", color=C["main"], fontsize=11)

# (b) 생성하지 못하는 목록 -------------------------------------------------------
ax = axs[1]
frame(ax, XL, YL, "(b)")
L = np.outer(np.linspace(-1.6, 1.6, 2), b1)
ax.plot(L[:, 0], L[:, 1], color=C["region"], lw=5, alpha=0.45, solid_capstyle="butt", zorder=1)
arrow(ax, (0, 0), b1, C["tangent"])
arrow(ax, (0, 0), -b1, C["tangent"])
arrow(ax, (0, 0), v, C["main"], lw=1.8)
ax.text(*(b1 + [-0.25, 0.25]), r"$b_1$", color=C["tangent"], fontsize=12)
ax.text(*(-b1 + [-0.2, 0.3]), r"$-b_1$", color=C["tangent"], fontsize=12)
ax.text(*(v + [0.1, 0.05]), r"$v$", color=C["main"], fontsize=12)
ax.text(1.35, -1.2, r"$\mathrm{span}(b_1, -b_1)$", color=C["tangent"], fontsize=10)

# (c) 일차독립이 아닌 목록: 표현이 두 가지 ---------------------------------------
ax = axs[2]
frame(ax, XL, YL, "(c)")
arrow(ax, (0, 0), b1, C["tangent"])
arrow(ax, b1, b2, C["tangent"])
arrow(ax, (0, 0), -b2, C["accent"])
for k in range(3):
    arrow(ax, -b2 + k * w, w, C["accent"])
arrow(ax, (0, 0), v, C["main"], lw=1.8)
ax.text(*(b1 * 0.75 + [0.12, -0.5]), r"$b_1$", color=C["tangent"], fontsize=12)
ax.text(*(b1 + 0.5 * b2 + [0.12, 0.1]), r"$b_2$", color=C["tangent"], fontsize=12)
ax.text(*(-0.5 * b2 + [-0.75, -0.55]), r"$-b_2$", color=C["accent"], fontsize=12)
ax.text(*(-b2 + 0.5 * w + [0.12, -0.15]), r"$3w$", color=C["accent"], fontsize=12)
ax.text(*(v + [-0.35, 0.22]), r"$v$", color=C["main"], fontsize=12)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
