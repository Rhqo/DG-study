"""그림 3.6.1: 열린덮개 (3.6절, 비예 3.6.3, 정리 3.6.6).

(a) (0, 1)의 열린덮개 {(1/k, 1) : k = 2, 3, ...} 가운데 k = 2, ..., 9. 유한개를 고르면 가장 큰 k에 대해
    (0, 1/k]가 덮이지 않고 남는다(주황 틈). 그래서 유한 부분덮개가 없다.
(b) 튜브 보조정리 (개념도). 세로 조각 {x} × Y를 덮는 상자 U_i × V_i (i = 1, 2, 3) 유한개가 있으면
    W = U_1 ∩ U_2 ∩ U_3에 대해 튜브 W × Y(주황 테두리)도 그 상자들로 덮인다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-6-1-open-covers.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

C = dgfig.COLORS
LW = dgfig.LW

fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.8), gridspec_kw=dict(width_ratios=[1.25, 1.0]))

# (a) --------------------------------------------------------------------------
ax = axs[0]
ax.set_xlim(-0.08, 1.12); ax.set_ylim(-0.25, 1.05)
ax.set_axis_off()
ks = range(2, 10)
# 자기검사: 유한개 (1/k, 1), k <= K의 합집합은 (1/K, 1)이고 1/(2K)를 덮지 못한다
for K in ks:
    union = lambda x: any(1 / k < x < 1 for k in range(2, K + 1))
    assert not union(1 / (2 * K)) and union((1 / K + 1) / 2)
ax.plot([0, 1], [0, 0], color=C["main"], lw=LW["main"])
ax.plot(0, 0, "o", ms=5, mfc="white", mec=C["main"], mew=1.2, zorder=5)
ax.plot(1, 0, "o", ms=5, mfc="white", mec=C["main"], mew=1.2, zorder=5)
for j, k in enumerate(ks):
    y = 0.12 + 0.105 * j
    ax.plot([1 / k, 1], [y, y], color=C["tangent"], lw=2.2, solid_capstyle="butt")
    ax.plot(1 / k, y, "o", ms=3.5, mfc="white", mec=C["tangent"], mew=1.0, zorder=5)
    ax.text(1.03, y, r"$k=%d$" % k, fontsize=8, va="center", color=C["tangent"])
ax.plot([0, 1 / 9], [0, 0], color=C["accent"], lw=4, solid_capstyle="butt", zorder=4)
ax.text(0.0, -0.13, r"$0$", fontsize=10, ha="center")
ax.text(1.0, -0.13, r"$1$", fontsize=10, ha="center")
ax.text(1 / 9, -0.15, r"$\frac{1}{9}$", fontsize=10, ha="center", color=C["accent"])
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) 튜브 보조정리 (개념도) ----------------------------------------------------------
ax = axs[1]
dgfig.schematic_axes(ax, (-0.25, 1.2), (-0.25, 1.15))
ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=C["main"], lw=1.0))
x0 = 0.55
boxes = [((0.42, 0.66), (-0.02, 0.42)), ((0.47, 0.63), (0.33, 0.74)), ((0.38, 0.7), (0.66, 1.02))]
# 자기검사: 상자들이 조각 {x0} × [0, 1]을 덮고, W = ∩U_i는 x0을 포함한다
ys = np.linspace(0, 1, 1001)
assert all(any(a < x0 < b and c < y < d for (a, b), (c, d) in boxes) for y in ys)
W = (max(b[0][0] for b in boxes), min(b[0][1] for b in boxes))
assert W[0] < x0 < W[1]
for (a, b), (c, d) in boxes:
    ax.add_patch(Rectangle((a, max(c, 0)), b - a, min(d, 1) - max(c, 0), facecolor=C["region"], alpha=0.3,
                           edgecolor=C["tangent"], lw=1.0))
ax.add_patch(Rectangle((W[0], 0), W[1] - W[0], 1, fill=False, edgecolor=C["accent"], lw=1.8, zorder=5))
ax.plot([x0, x0], [0, 1], color=C["main"], lw=1.6, zorder=6)
ax.text(x0, -0.13, r"$x$", fontsize=11, ha="center")
ax.text((W[0] + W[1]) / 2, 1.05, r"$W \times Y$", fontsize=10, ha="center", color=C["accent"])
ax.text(1.05, 0.5, r"$Y$", fontsize=11, va="center")
ax.text(0.95, -0.13, r"$X$", fontsize=11, ha="center")
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
