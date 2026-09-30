"""그림 11.4.1: ℝℙ²의 두 차트 φ_3, φ_1을 평면 x^3 = 1, x^1 = 1과의 교점으로 읽기 (11.4절, 예 11.4.2).

원점을 지나는 직선 [d], d = (0.6, 0.5, 0.9).
  평면 x^3 = 1과의 교점 d/d^3 = (2/3, 5/9, 1)  →  φ_3[d] = (2/3, 5/9)
  평면 x^1 = 1과의 교점 d/d^1 = (1, 5/6, 3/2)  →  φ_1[d] = (5/6, 3/2)
좌표변환 φ_1∘φ_3^{-1}(u) = (u^2/u^1, 1/u^1)가 (2/3, 5/9)를 (5/6, 3/2)로 보낸다.
단위구면 위의 대척점 ±d/|d|도 함께 그린다 (ℝℙ² = S²/±, 명제 3.4.14).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-4-1-rp2-charts.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

C = dgfig.COLORS
LW = dgfig.LW

d = np.array([0.6, 0.5, 0.9])
p3 = d / d[2]
p1 = d / d[0]
phi3 = np.array([p3[0], p3[1]])
phi1 = np.array([p1[1], p1[2]])

# 자기검사 (§12.1)
assert np.allclose(phi3, [2 / 3, 5 / 9]) and np.allclose(phi1, [5 / 6, 3 / 2])
assert np.allclose([phi3[1] / phi3[0], 1 / phi3[0]], phi1)      # φ_1∘φ_3^{-1}
assert np.isclose(p3[2], 1) and np.isclose(p1[0], 1)

fig = plt.figure(figsize=(6.0, 4.8))
ax = dgfig.axes3d(fig, elev=18, azim=-58)

# 단위구면 (옅게)
th, ph = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60))
X, Y, Z = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(ax, X, Y, Z, alpha=0.12, grid_every=6, zorder=1)

# 평면 x^3 = 1 (가로), x^1 = 1 (세로)
L = 1.7
sq3 = [[-0.6, -0.6, 1], [L, -0.6, 1], [L, L, 1], [-0.6, L, 1]]
sq1 = [[1, -0.6, -0.6], [1, L, -0.6], [1, L, L + 0.4], [1, -0.6, L + 0.4]]
for sq in (sq3, sq1):
    ax.add_collection3d(Poly3DCollection([sq], facecolor=C["region"], alpha=0.18, edgecolor=C["tangent"], lw=0.7, zorder=2))

# 직선 [d]
tt = np.array([-1.25, 1.85])
line = np.outer(tt, d)
ax.plot(*line.T, color=C["main"], lw=1.4, zorder=6)
u = d / np.linalg.norm(d)
for q, lab in ((u, r"$x$"), (-u, r"$-x$")):
    dgfig.point3d(ax, q, role="main", size=16, label=lab, label_offset=(0.05, 0.05, 0.05), fontsize=11)
dgfig.point3d(ax, p3, role="accent", size=26, zorder=14)
dgfig.point3d(ax, p1, role="accent", size=26, zorder=14)
ax.text(*(p3 + np.array([-0.55, -0.1, 0.12])), r"$(\varphi_3[x],\,1)$", fontsize=10, color=C["accent"])
ax.text(*(p1 + np.array([0.12, 0.0, -0.08])), r"$(1,\,\varphi_1[x])$", fontsize=10, color=C["accent"])
ax.text(-0.5, L - 0.1, 1.02, r"$x^3 = 1$", fontsize=11, color=C["tangent"])
ax.text(1.0, L - 0.2, L + 0.25, r"$x^1 = 1$", fontsize=11, color=C["tangent"])
ax.scatter([0], [0], [0], s=10, color=C["aux"])
ax.text(0.03, -0.12, -0.2, r"$0$", fontsize=10, color=C["aux"])

dgfig.equal_aspect(ax, np.array(sq3), np.array(sq1), line, np.array([[-1.1, -1.1, -1.35], [1, 1, 1]]), zoom=1.05)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
dgfig.save(fig, __file__)
