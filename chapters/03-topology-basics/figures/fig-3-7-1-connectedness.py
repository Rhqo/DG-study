"""그림 3.7.1: 연결과 경로연결 (3.7절, 비예 3.7.3, 정의 3.7.8).

(a) A = D_1 ∪ D_2 (서로소인 두 닫힌 원판, 중심 (-0.7, 0)과 (0.8, 0.1), 반지름 0.55와 0.45).
    D_1 = A ∩ {x^1 < 0.05}, D_2 = A ∩ {x^1 > 0.05}이므로 둘 다 A에서 열려 있고, A는 연결이 아니다.
(b) 고리 영역 {1/2 < |x| < 1}의 두 점 p = γ(0), q = γ(1)을 잇는 경로 γ(t) = ρ(t)(cos θ(t), sin θ(t)),
    ρ(t) = 0.75 + 0.1 sin(3πt), θ(t) = π/6 + (7π/6) t, t ∈ [0, 1]. 경로는 고리를 벗어나지 않는다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-7-1-connectedness.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

C = dgfig.COLORS
LW = dgfig.LW

fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.8))

# (a) 두 조각 -------------------------------------------------------------------
ax = axs[0]
dgfig.schematic_axes(ax, (-1.4, 1.45), (-0.85, 0.95))
c1, r1 = np.array([-0.7, 0.0]), 0.55
c2, r2 = np.array([0.8, 0.1]), 0.45
# 자기검사: 두 원판은 서로소이고 직선 x^1 = 0.05가 둘을 가른다
assert np.linalg.norm(c1 - c2) > r1 + r2
assert c1[0] + r1 < 0.05 < c2[0] - r2
for c, rr, col in ((c1, r1, C["tangent"]), (c2, r2, C["accent"])):
    ax.add_patch(Circle(c, rr, facecolor=C["region"], alpha=0.3, lw=0, zorder=1))
    ax.add_patch(Circle(c, rr, fill=False, edgecolor=col, lw=1.4, zorder=2))
ax.plot([0.05, 0.05], [-0.8, 0.9], color=C["aux"], lw=0.8, ls=(0, (3, 2)), zorder=0)
ax.text(c1[0], c1[1] - 0.08, r"$D_1$", fontsize=12, ha="center", color=C["tangent"])
ax.text(c2[0], c2[1] - 0.08, r"$D_2$", fontsize=12, ha="center", color=C["accent"])
ax.text(0.1, 0.78, r"$x^1 = 0.05$", fontsize=9, color=C["aux"])
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) 고리 안의 경로 ---------------------------------------------------------------
ax = axs[1]
dgfig.schematic_axes(ax, (-1.2, 1.2), (-1.15, 1.15))
th = np.linspace(0, 2 * np.pi, 400)
ax.fill(np.r_[np.cos(th), 0.5 * np.cos(th[::-1])], np.r_[np.sin(th), 0.5 * np.sin(th[::-1])],
        color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
for rr in (1.0, 0.5):
    ax.plot(rr * np.cos(th), rr * np.sin(th), color=C["main"], lw=1.0, ls=(0, (4, 3)))
t = np.linspace(0, 1, 400)
rho = 0.75 + 0.1 * np.sin(3 * np.pi * t)
ang = np.pi / 6 + (7 * np.pi / 6) * t
gam = np.stack([rho * np.cos(ang), rho * np.sin(ang)], 1)
# 자기검사: 경로가 고리 1/2 < |x| < 1 안에 있다
assert np.all((np.linalg.norm(gam, axis=1) > 0.5) & (np.linalg.norm(gam, axis=1) < 1))
ax.plot(*gam.T, color=C["accent"], lw=LW["main"], zorder=3)
p, q = gam[0], gam[-1]
for pt in (p, q):
    ax.plot(*pt, "o", ms=4, color=C["main"], zorder=5)
ax.text(p[0] + 0.05, p[1] + 0.02, r"$p = \gamma(0)$", fontsize=10)
ax.text(q[0] - 0.05, q[1] - 0.2, r"$q = \gamma(1)$", fontsize=10, ha="center")
ax.text(-0.2, 0.9, r"$\gamma$", fontsize=12, color=C["accent"])
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
