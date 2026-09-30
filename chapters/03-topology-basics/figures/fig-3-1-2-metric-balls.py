"""그림 3.1.2: 세 거리의 공은 서로의 안에 들어간다 (3.1절, 예 3.1.26).

평면 R^2에서 원점을 중심으로 하는 공
  B^inf_1 (d_inf, 정사각형) ⊇ B_1 (유클리드 거리 d, 원판) ⊇ B^1_1 (d_1, 마름모) ⊇ B^inf_{1/2} (작은 정사각형).
부등식 d_inf <= d <= d_1 <= 2 d_inf (n = 2)에서 나온다. 한 거리의 공 안에는 언제나 다른 거리의 공이 들어가므로
세 거리는 같은 열린집합을 준다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-1-2-metric-balls.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

C = dgfig.COLORS
LW = dgfig.LW


def d_inf(x):
    return np.max(np.abs(x), axis=-1)


def d_2(x):
    return np.linalg.norm(x, axis=-1)


def d_1(x):
    return np.sum(np.abs(x), axis=-1)


# 자기검사: 무작위 점에서 d_inf <= d <= d_1 <= 2 d_inf (n = 2)
rng = np.random.default_rng(0)
X = rng.normal(size=(20000, 2)) * rng.uniform(0.01, 3, size=(20000, 1))
assert np.all(d_inf(X) <= d_2(X) + 1e-12)
assert np.all(d_2(X) <= d_1(X) + 1e-12)
assert np.all(d_1(X) <= 2 * d_inf(X) + 1e-12)
# 따라서 공의 포함관계
th = np.linspace(0, 2 * np.pi, 721)
circle = np.stack([np.cos(th), np.sin(th)], axis=1)
assert np.all(d_inf(circle) <= 1 + 1e-12)                    # B_1 ⊆ B^inf_1
diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
assert np.all(d_2(diamond) <= 1 + 1e-12)                     # B^1_1 ⊆ B_1
small = 0.5 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
assert np.all(d_1(small) <= 1 + 1e-12)                       # B^inf_{1/2} ⊆ B^1_1

fig, ax = plt.subplots(figsize=(3.6, 3.4))
dgfig.schematic_axes(ax, (-1.35, 1.75), (-1.3, 1.3))
ax.plot([-1.3, 1.3], [0, 0], color=C["aux"], lw=0.5, zorder=0)
ax.plot([0, 0], [-1.25, 1.25], color=C["aux"], lw=0.5, zorder=0)

sq = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
shapes = [  # (패치를 만드는 함수, 테두리 색, 채우기 alpha)
    (lambda **kw: Polygon(sq, closed=True, **kw), C["main"], 0.10),
    (lambda **kw: Circle((0, 0), 1.0, **kw), C["tangent"], 0.12),
    (lambda **kw: Polygon(diamond, closed=True, **kw), C["third"], 0.16),
    (lambda **kw: Polygon(small, closed=True, **kw), C["accent"], 0.28),
]
for z, (make, edge, a) in enumerate(shapes):
    # 채우기와 테두리를 따로 그린다 (alpha가 테두리에 적용되지 않게)
    ax.add_patch(make(facecolor=C["region"], alpha=a, lw=0, zorder=1 + z))
    ax.add_patch(make(fill=False, edgecolor=edge, lw=1.3, ls=(0, (4, 2.5)), zorder=5 + z))
ax.plot(0, 0, "o", color=C["main"], ms=3, zorder=12)

ax.text(1.05, 1.05, r"$B^{\infty}_{1}$", fontsize=11, color=C["main"])
ax.text(0.74, 0.74, r"$B_{1}$", fontsize=11, color=C["tangent"], ha="left", va="bottom")
ax.text(0.5, 0.5, r"$B^{1}_{1}$", fontsize=11, color=C["third"], ha="left", va="bottom")
ax.text(0.08, 0.12, r"$B^{\infty}_{1/2}$", fontsize=10, color=C["accent"], ha="left", va="bottom")
ax.text(1.03, -0.15, r"$1$", fontsize=10)
ax.text(0.53, -0.17, r"$1/2$", fontsize=9)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
