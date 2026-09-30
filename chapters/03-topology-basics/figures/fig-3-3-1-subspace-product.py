"""그림 3.3.1: 부분공간 위상과 곱위상 (3.3절, 정의 3.3.1, 예 3.3.2, 정의 3.3.12).

(a) 원 S^1과 R^2의 열린 원판 U = B_{0.7}((0.85, 0.55)). 교집합 S^1 ∩ U(주황 호, 끝점 제외)는 S^1의 열린집합이다.
(b) [0, 1] ⊆ R과 열린구간 (-1/2, 1/2). 교집합 [0, 1/2)는 [0, 1]의 열린집합이지만 R의 열린집합은 아니다.
(c) R^2의 열린 원판 W = B_1(0)은 상자 U × V가 아니지만, 각 점 p를 포함하고 W 안에 들어가는 상자가 있다.
    따라서 W는 상자들의 합집합이고 곱위상에서 열려 있다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-3-1-subspace-product.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

C = dgfig.COLORS
LW = dgfig.LW

fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.0, 1.0]))

# (a) S^1 ∩ U ---------------------------------------------------------------------
ax = axs[0]
dgfig.schematic_axes(ax, (-1.3, 1.75), (-1.3, 1.45))
c, rad = np.array([0.85, 0.55]), 0.7
th = np.linspace(-np.pi, np.pi, 2001)
S1 = np.stack([np.cos(th), np.sin(th)], 1)
inside = np.linalg.norm(S1 - c, axis=1) < rad
ax.add_patch(Circle(c, rad, facecolor=C["region"], alpha=dgfig.ALPHA["region"], lw=0, zorder=1))
ax.add_patch(Circle(c, rad, fill=False, edgecolor=C["tangent"], lw=1.0, ls=(0, (4, 3)), zorder=2))
ax.plot(*S1.T, color=C["main"], lw=LW["main"] * 0.8, zorder=3)
arc = S1[inside]
# 자기검사: 교집합은 하나의 연결된 호 (인덱스가 연속)이다
idx = np.flatnonzero(inside)
assert np.all(np.diff(idx) == 1)
ax.plot(*arc.T, color=C["accent"], lw=3.6, solid_capstyle="butt", zorder=4)
for q in (arc[0], arc[-1]):
    ax.plot(*q, "o", ms=4.5, mfc="white", mec=C["accent"], mew=1.2, zorder=6)
ax.text(c[0] + 0.3, c[1] + 0.55, r"$U$", fontsize=12, color=C["tangent"])
ax.text(-1.2, 0.9, r"$S^1$", fontsize=12)
ax.text(0.28, 0.22, r"$S^1 \cap U$", fontsize=10, color=C["accent"], ha="right")
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) [0,1] ∩ (-1/2, 1/2) ----------------------------------------------------------
ax = axs[1]
ax.set_xlim(-0.9, 1.35); ax.set_ylim(-0.8, 0.8)
ax.set_axis_off()
ax.plot([-0.85, 1.3], [0.3, 0.3], color=C["aux"], lw=0.7)
ax.plot([-0.5, 0.5], [0.3, 0.3], color=C["tangent"], lw=4, alpha=0.55, solid_capstyle="butt")
for x0 in (-0.5, 0.5):
    ax.plot(x0, 0.3, "o", ms=4.5, mfc="white", mec=C["tangent"], mew=1.2, zorder=6)
ax.plot([0, 1], [-0.3, -0.3], color=C["main"], lw=LW["main"])
ax.plot([0, 0.5], [-0.3, -0.3], color=C["accent"], lw=4, solid_capstyle="butt", zorder=4)
ax.plot(0, -0.3, "o", ms=4.8, color=C["accent"], zorder=6)
ax.plot(0.5, -0.3, "o", ms=4.5, mfc="white", mec=C["accent"], mew=1.2, zorder=6)
ax.plot(1, -0.3, "o", ms=4, color=C["main"], zorder=6)
ax.text(-0.55, 0.45, r"$(-\frac{1}{2}, \frac{1}{2})$", fontsize=10, color=C["tangent"])
ax.text(1.02, -0.2, r"$[0,1]$", fontsize=10)
ax.text(0.25, -0.62, r"$[0, \frac{1}{2})$", fontsize=10, color=C["accent"], ha="center")
ax.text(0.0, -0.18, r"$0$", fontsize=10, ha="center")
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (c) 원판은 상자들의 합집합 --------------------------------------------------------
ax = axs[2]
dgfig.schematic_axes(ax, (-1.3, 1.3), (-1.3, 1.45))
ax.add_patch(Circle((0, 0), 1.0, facecolor=C["region"], alpha=0.18, lw=0, zorder=1))
ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=C["main"], lw=1.0, ls=(0, (4, 3)), zorder=2))
pts = [np.array([0.35, 0.3]), np.array([-0.55, -0.45]), np.array([0.62, -0.58]), np.array([-0.2, 0.78])]
for p in pts:
    # p를 중심으로 하는 정사각형 상자: 반변 = (1 - |p|)/sqrt2 × 0.95이면 원판 안에 들어간다
    hw = 0.95 * (1 - np.linalg.norm(p)) / np.sqrt(2)
    corners = p + hw * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    assert np.all(np.linalg.norm(corners, axis=1) < 1)          # 자기검사: 상자 ⊆ W
    ax.add_patch(Rectangle(p - hw, 2 * hw, 2 * hw, fill=False, edgecolor=C["tangent"], lw=1.1, zorder=3))
    ax.plot(*p, "o", ms=3, color=C["main"], zorder=5)
ax.text(pts[0][0] + 0.05, pts[0][1] + 0.05, r"$p$", fontsize=11)
ax.text(0.72, 0.78, r"$W$", fontsize=12)
ax.text(0.0, 1.0, "(c)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
