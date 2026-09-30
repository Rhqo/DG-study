"""그림 11.1.2: 십자 모양 X = {xy = 0}은 원점에서 국소 유클리드가 아니다 (11.1절, 비예 11.1.11).

(a) X와 원점 둘레의 공 B_ε(0) (ε = 0.8, 점선). X ∩ B_ε(0)에서 원점을 빼면 네 조각(색 네 개)이 남는다.
(b) 비교: R^1의 구간 (c - δ, c + δ)에서 c를 빼면 두 조각(δ = 0.8), R^2의 원판 B_δ(c)에서 c를 빼면 한 조각(연결).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-1-2-cross.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

C = dgfig.COLORS
LW = dgfig.LW
EPS = 0.8
DELTA = 0.8
COLS = [C["tangent"], C["accent"], C["third"], C["covector"]]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1.0, 1.15]})

# ---- (a) 십자
ax = axA
dgfig.schematic_axes(ax, (-1.35, 1.35), (-1.3, 1.3))
ax.plot([-1.25, 1.25], [0, 0], color=C["aux"], lw=0.9, zorder=1)
ax.plot([0, 0], [-1.25, 1.25], color=C["aux"], lw=0.9, zorder=1)
dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
for (dx, dy), col in zip(dirs, COLS):
    s = np.linspace(0.0, EPS, 50)
    ax.plot(dx * s, dy * s, color=col, lw=2.6, solid_capstyle="butt", zorder=3)
    # 자기검사: 조각의 점은 X 위에 있고 B_ε(0) 안에 있다
    pts = np.stack([dx * s, dy * s], axis=1)
    assert np.allclose(pts[:, 0] * pts[:, 1], 0) and np.all(np.hypot(*pts.T) <= EPS + 1e-12)
ax.add_patch(Circle((0, 0), EPS, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 3)), zorder=2))
ax.plot(0, 0, "o", ms=6, mfc="white", mec=C["main"], mew=1.1, zorder=5)
ax.text(0.08, 0.08, r"$0$", fontsize=11)
ax.text(0.62, 0.72, r"$B_\varepsilon(0)$", fontsize=11)
ax.text(1.02, 0.08, r"$X$", fontsize=12)
ax.text(-1.3, 1.12, r"(a)", fontsize=11)

# ---- (b) 비교
ax = axB
dgfig.schematic_axes(ax, (-1.45, 1.55), (-1.3, 1.3))
y1 = 0.72
c1 = np.array([0.0, y1])
ax.plot([c1[0] - DELTA, c1[0]], [y1, y1], color=COLS[0], lw=2.6, solid_capstyle="butt", zorder=3)
ax.plot([c1[0], c1[0] + DELTA], [y1, y1], color=COLS[1], lw=2.6, solid_capstyle="butt", zorder=3)
for xe in (c1[0] - DELTA, c1[0] + DELTA, c1[0]):
    ax.plot(xe, y1, "o", ms=6 if xe == c1[0] else 5, mfc="white", mec=C["main"], mew=1.1, zorder=5)
ax.text(c1[0] - 0.06, y1 + 0.13, r"$c$", fontsize=11)
ax.text(1.0, y1 - 0.05, r"$\mathbb{R}^1$", fontsize=12)
c2 = np.array([0.0, -0.52])
r2 = DELTA * 0.72
ax.add_patch(Circle(c2, r2, facecolor=COLS[0], alpha=0.3, lw=0, zorder=2))
ax.add_patch(Circle(c2, r2, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 3)), zorder=3))
ax.plot(*c2, "o", ms=6, mfc="white", mec=C["main"], mew=1.1, zorder=5)
ax.text(c2[0] + 0.07, c2[1] + 0.07, r"$c$", fontsize=11)
ax.text(1.0, c2[1] - 0.05, r"$\mathbb{R}^2$", fontsize=12)
ax.text(-1.4, 1.12, r"(b)", fontsize=11)

fig.subplots_adjust(wspace=0.05, left=0.01, right=0.99, top=0.97, bottom=0.02)
dgfig.save(fig, __file__)
