"""그림 3.5.1: 하우스도르프 조건 (3.5절, 예 3.5.2, 비예 3.5.6).

(a) 거리공간의 두 점 p ≠ q는 반지름 d(p, q)/2인 두 공으로 분리된다 (p = (-0.6, -0.2), q = (0.7, 0.35)).
(b) 원점이 두 개인 직선 (개념도). 0이 아닌 점은 하나씩이고 원점만 0_1(위), 0_2(아래) 두 개다.
    0_1의 근방(파랑)과 0_2의 근방(주황)은 원점 양옆에서 반드시 겹친다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-5-1-hausdorff.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

C = dgfig.COLORS
LW = dgfig.LW

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.35]))

# (a) --------------------------------------------------------------------------
ax = axs[0]
dgfig.schematic_axes(ax, (-1.45, 1.55), (-1.05, 1.15))
p, q = np.array([-0.6, -0.2]), np.array([0.7, 0.35])
rad = np.linalg.norm(p - q) / 2
# 자기검사: 두 공이 서로소 (중심 거리 = 2 rad, 열린공이므로 서로소)
X = rng = np.random.default_rng(0).uniform(-2, 2, size=(20000, 2))
inp = np.linalg.norm(X - p, axis=1) < rad
inq = np.linalg.norm(X - q, axis=1) < rad
assert not np.any(inp & inq)
for c, col in ((p, C["tangent"]), (q, C["accent"])):
    ax.add_patch(Circle(c, rad, facecolor=C["region"], alpha=0.18, lw=0, zorder=1))
    ax.add_patch(Circle(c, rad, fill=False, edgecolor=col, lw=1.2, ls=(0, (4, 3)), zorder=2))
    ax.plot(*c, "o", ms=4, color=C["main"], zorder=5)
ax.plot(*np.array([p, q]).T, color=C["aux"], lw=0.8, ls=(0, (2, 2)), zorder=3)
ax.text(p[0] - 0.08, p[1] - 0.2, r"$p$", fontsize=12)
ax.text(q[0] + 0.05, q[1] + 0.07, r"$q$", fontsize=12)
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) 원점이 두 개인 직선 (개념도) ----------------------------------------------------
ax = axs[1]
ax.set_xlim(-2.2, 2.2); ax.set_ylim(-0.9, 0.9)
ax.set_axis_off()
gap = 0.06
ax.plot([-2.1, -gap], [0, 0], color=C["main"], lw=LW["main"])
ax.plot([gap, 2.1], [0, 0], color=C["main"], lw=LW["main"])
ax.plot(0, 0, "o", ms=5, mfc="white", mec=C["main"], mew=1.0, zorder=5)
o1, o2 = np.array([0.0, 0.32]), np.array([0.0, -0.32])
ax.plot(*o1, "o", ms=5, color=C["tangent"], zorder=6)
ax.plot(*o2, "o", ms=5, color=C["accent"], zorder=6)
ax.text(0.06, 0.42, r"$0_1$", fontsize=12, color=C["tangent"])
ax.text(0.06, -0.55, r"$0_2$", fontsize=12, color=C["accent"])
e1, e2 = 1.1, 0.75
# 0_1의 근방: 원점 양옆의 구간 (-e1, 0) ∪ (0, e1)과 0_1 자신
for sgn in (-1, 1):
    ax.plot([gap * sgn, e1 * sgn], [0.07, 0.07], color=C["tangent"], lw=3, alpha=0.8, solid_capstyle="butt")
    ax.plot([gap * sgn, e2 * sgn], [-0.07, -0.07], color=C["accent"], lw=3, alpha=0.8, solid_capstyle="butt")
ax.plot([-gap, 0, gap], [0.07, o1[1], 0.07], color=C["tangent"], lw=1.0)
ax.plot([-gap, 0, gap], [-0.07, o2[1], -0.07], color=C["accent"], lw=1.0)
x0 = min(e1, e2) / 2
ax.plot(x0, 0, "o", ms=4, color=C["main"], zorder=7)
ax.text(x0, 0.18, r"$[x]$", fontsize=11, ha="center")
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
