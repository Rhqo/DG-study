"""그림 1.2.1: 선형사상이 격자에 하는 일 (1.2절, 예 1.2.7과 예 1.2.18).

(a) T(x) = Bx, B = [[2, -1], [1, 1]] (T e_j = b_j): 표준 격자와 단위정사각형이 기저 B의 격자와 평행사변형으로 간다.
    T(1, 1) = (1, 2) = v.
(b) P(x) = ((x^1 + x^2)/2, (x^1 + x^2)/2): 핵 ker P = span((1, -1))의 점은 모두 0으로 가고,
    x + ker P 위의 점은 모두 같은 점 Px로 간다. 상 im P = span((1, 1)).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-2-1-linear-maps.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

C = dgfig.COLORS
LW = dgfig.LW

B = np.array([[2.0, -1.0], [1.0, 1.0]])          # 예 1.2.18: 열이 b_1, b_2
P = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]])     # 예 1.2.7
e1, e2 = np.eye(2)
x0 = np.array([1.0, 1.0])
xp = np.array([2.0, -0.5])                       # (b)에서 쓰는 점
kdir = np.array([1.0, -1.0])

# 자기검사
assert np.allclose(B @ e1, [2, 1]) and np.allclose(B @ e2, [-1, 1])
assert np.allclose(B @ x0, [1, 2])
assert np.allclose(P @ kdir, 0)                      # (1,-1) ∈ ker P
assert np.allclose(P @ (xp + 1.3 * kdir), P @ xp)    # x + ker P → 같은 점
assert np.allclose(P @ P, P)
assert np.linalg.matrix_rank(P) == 1


def arrow(ax, base, vec, color, lw=None, z=5):
    base = np.asarray(base, float)
    ax.annotate("", xy=base + np.asarray(vec, float), xytext=base,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw or LW["vector"],
                                mutation_scale=11, shrinkA=0, shrinkB=0), zorder=z)


def frame(ax, xlim, ylim, tag=None):
    dgfig.schematic_axes(ax, xlim, ylim)
    ax.plot(list(xlim), [0, 0], color=C["aux"], lw=0.5, zorder=0)
    ax.plot([0, 0], list(ylim), color=C["aux"], lw=0.5, zorder=0)
    ax.plot(0, 0, "o", color=C["main"], ms=2.5, zorder=6)
    if tag:
        ax.text(0.0, 1.0, tag, transform=ax.transAxes, ha="left", va="top", fontsize=11)


def grid(ax, M, color, ls, lw=0.45, k=4):
    ts = np.array([-8.0, 8.0])
    for j in range(-k, k + 1):
        for P0, d in ((j * M[:, 0], M[:, 1]), (j * M[:, 1], M[:, 0])):
            pts = P0 + np.outer(ts, d)
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, ls=ls, zorder=1)


XL, YL = (-2.2, 3.0), (-1.7, 2.8)
fig, axs = plt.subplots(2, 2, figsize=(6.0, 5.4))

# (a) 가역인 선형사상 T ---------------------------------------------------------
ax = axs[0, 0]
frame(ax, XL, YL, "(a)")
grid(ax, np.eye(2), C["aux"], (0, (3, 2.5)))
ax.add_patch(Polygon([[0, 0], e1, e1 + e2, e2], closed=True, color=C["region"], alpha=0.3, lw=0, zorder=2))
arrow(ax, (0, 0), e1, C["tangent"])
arrow(ax, (0, 0), e2, C["tangent"])
arrow(ax, (0, 0), x0, C["main"], lw=1.6)
ax.text(1.0, -0.42, r"$e_1$", color=C["tangent"], fontsize=12)
ax.text(-0.45, 0.85, r"$e_2$", color=C["tangent"], fontsize=12)
ax.text(1.08, 1.08, r"$(1,1)$", color=C["main"], fontsize=11)
ax.set_xlim(*XL); ax.set_ylim(*YL)

ax = axs[0, 1]
frame(ax, XL, YL)
grid(ax, B, C["aux"], (0, (3, 2.5)))
ax.add_patch(Polygon([[0, 0], B @ e1, B @ (e1 + e2), B @ e2], closed=True, color=C["region"], alpha=0.3,
                     lw=0, zorder=2))
arrow(ax, (0, 0), B @ e1, C["tangent"])
arrow(ax, (0, 0), B @ e2, C["tangent"])
arrow(ax, (0, 0), B @ x0, C["main"], lw=1.6)
ax.text(1.75, 0.45, r"$Te_1 = b_1$", color=C["tangent"], fontsize=11)
ax.text(-2.1, 1.2, r"$Te_2 = b_2$", color=C["tangent"], fontsize=11)
ax.text(0.2, 2.2, r"$T(1,1) = v$", color=C["main"], fontsize=11)
ax.set_xlim(*XL); ax.set_ylim(*YL)
dgfig.map_arrow(fig, axs[0, 0], axs[0, 1], r"$T$", xy_from=(0.93, 0.8), xy_to=(0.07, 0.8), rad=-0.3)

# (b) 계수 1인 선형사상 P ---------------------------------------------------------
ax = axs[1, 0]
frame(ax, XL, YL, "(b)")
L = np.outer([-1.6, 1.6], kdir)
ax.plot(L[:, 0], L[:, 1], color=C["accent"], lw=2.2, zorder=3)
Lx = xp + np.outer([-0.9, 1.1], kdir)
ax.plot(Lx[:, 0], Lx[:, 1], color=C["accent"], lw=1.0, ls="--", zorder=3)
for s in (-0.6, 0.0, 0.7):
    ax.plot(*(xp + s * kdir), "o", color=C["main"], ms=3.5, zorder=6)
ax.text(-1.95, 1.75, r"$\ker P$", color=C["accent"], fontsize=12)
ax.text(1.05, 0.62, r"$x + \ker P$", color=C["accent"], fontsize=11, ha="left")
ax.text(xp[0] + 0.12, xp[1] + 0.15, r"$x$", color=C["main"], fontsize=12)
ax.set_xlim(*XL); ax.set_ylim(*YL)

ax = axs[1, 1]
frame(ax, XL, YL)
L = np.outer([-1.9, 2.5], [1.0, 1.0])
ax.plot(L[:, 0], L[:, 1], color=C["region"], lw=5, alpha=0.45, solid_capstyle="butt", zorder=1)
ax.plot(*(P @ xp), "o", color=C["main"], ms=4, zorder=6)
ax.plot(0, 0, "o", color=C["accent"], ms=5, zorder=7)
ax.text(*(P @ xp + [0.15, -0.35]), r"$Px$", color=C["main"], fontsize=12)
ax.text(1.55, 2.25, r"$\mathrm{im}\,P$", color=C["tangent"], fontsize=12)
ax.text(0.15, -0.45, r"$0$", color=C["accent"], fontsize=12)
ax.set_xlim(*XL); ax.set_ylim(*YL)
dgfig.map_arrow(fig, axs[1, 0], axs[1, 1], r"$P$", xy_from=(0.93, 0.8), xy_to=(0.07, 0.8), rad=-0.3)

fig.subplots_adjust(wspace=0.25, hspace=0.12, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
