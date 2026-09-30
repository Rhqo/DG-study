"""그림 1.6.2: R^3의 외적 (1.6절, 정리 1.6.16).

u = (1.1, 0, 0.15), v = (0.25, 0.9, 0.1)이 만드는 평행사변형(하늘색)과 외적 u × v(주홍, 실제 길이).
|u × v|는 평행사변형의 넓이와 같고, (u, v, u × v)는 오른손 기저다(u에서 v로 도는 방향을 오른손 네 손가락으로
감으면 엄지가 u × v를 가리킨다). 아래쪽 점선은 v × u = -(u × v), 회색 점선은 v에서 u까지의 높이.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/01-linear-algebra/figures/fig-1-6-2-cross-product.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

C = dgfig.COLORS

u = np.array([1.1, 0.0, 0.15])
v = np.array([0.25, 0.9, 0.1])
n = np.cross(u, v)

# 자기검사 (정리 1.6.16)
assert np.allclose(n, [-0.135, -0.0725, 0.99])
assert abs(n @ u) < 1e-12 and abs(n @ v) < 1e-12
assert np.isclose(n @ n, (u @ u) * (v @ v) - (u @ v) ** 2)
assert np.linalg.det(np.column_stack([u, v, n])) > 0
foot = (v @ u) / (u @ u) * u                     # v를 u 위로 내린 수선의 발
assert np.isclose(np.linalg.norm(u) * np.linalg.norm(v - foot), np.linalg.norm(n))

fig = plt.figure(figsize=(4.4, 3.8))
ax = dgfig.axes3d(fig, elev=24, azim=-62)

# 평행사변형과 높이
P = np.array([[0, 0, 0], u, u + v, v])
ax.add_collection3d(Poly3DCollection([P], facecolor=C["region"], alpha=0.35, edgecolor=C["tangent"],
                                     linewidth=0.6, zorder=2))
ax.plot(*np.array([v, foot]).T, color=C["aux"], lw=0.9, ls="--", zorder=4)

# u에서 v로 도는 원호 (평행사변형 평면 안)
uh = u / np.linalg.norm(u)
wh = v - (v @ uh) * uh
wh /= np.linalg.norm(wh)
ang = np.arccos(uh @ (v / np.linalg.norm(v)))
ts = np.linspace(0.15, ang - 0.15, 40)
arc = 0.32 * (np.outer(np.cos(ts), uh) + np.outer(np.sin(ts), wh))
ax.plot(*arc.T, color=C["main"], lw=1.0, zorder=6)
dgfig.arrow3d(ax, arc[-2], arc[-1] - arc[-2], role="main", lw=1.0, head=9, zorder=6)

# 벡터
dgfig.arrow3d(ax, (0, 0, 0), u, role="tangent", label=r"$u$", label_offset=(0.02, -0.12, -0.06), zorder=10)
dgfig.arrow3d(ax, (0, 0, 0), v, role="tangent", label=r"$v$", label_offset=(-0.02, 0.1, 0.04), zorder=10)
dgfig.arrow3d(ax, (0, 0, 0), n, role="normal", label=r"$u \times v$", label_offset=(0.28, 0.0, -0.05), zorder=11)
ax.plot(*np.array([[0, 0, 0], -0.8 * n]).T, color=C["normal"], lw=0.9, ls=(0, (3, 2)), zorder=3)
dgfig.arrow3d(ax, -0.8 * n, -0.2 * n, role="normal", lw=0.9, head=9, zorder=3)
ax.text(*(-0.75 * n + np.array([0.15, 0.0, 0.0])), r"$v \times u$", color=C["normal"], fontsize=10)
dgfig.point3d(ax, (0, 0, 0), size=8)

dgfig.equal_aspect(ax, np.array([[0, 0, 0], u, v, u + v, 1.08 * n, -1.12 * n]), zoom=1.35)
dgfig.save(fig, __file__)
