"""그림 3.5.2: 유리수 중심, 유리수 반지름의 공들은 R^2의 가산 기저다 (3.5절, 명제 3.5.9).

열린집합 U(개념도 얼룩), 점 p = (0.43, 0.27) ∈ U, B_ε(p) ⊆ U (ε = 0.8 × p에서 경계까지의 거리).
유리수 r (0 < r < ε/2)과 |q - p| < r인 유리점 q(격자 간격 1/10의 점)를 고르면 p ∈ B_r(q) ⊆ B_ε(p) ⊆ U이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-5-2-rational-balls.py``
"""

from fractions import Fraction

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

C = dgfig.COLORS
LW = dgfig.LW

fig, ax = plt.subplots(figsize=(3.6, 3.2))
dgfig.schematic_axes(ax, (-1.35, 1.4), (-1.2, 1.25))
pts = dgfig.blob(ax, center=(0, 0), radius=1.0, seed=5, fill="region", edge=None)
ax.add_patch(Polygon(pts[:-1], closed=True, fill=False, edgecolor=C["main"], lw=1.1, ls=(0, (4, 3)), zorder=3))


def dist_to_boundary(p):
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    return np.min(np.linalg.norm(a + t[:, None] * ab - p, axis=1))


p = np.array([0.43, 0.27])
eps = 0.8 * dist_to_boundary(p)
# 유리수 반지름 r < ε/2 (분모 20), 유리점 q (격자 1/10) with |q - p| < r
r = Fraction(int(np.floor(eps / 2 * 20 - 1e-9)), 20)
assert 0 < r < eps / 2
grid = [np.array([Fraction(i, 10), Fraction(j, 10)]) for i in range(-12, 13) for j in range(-12, 13)]
cands = [g for g in grid if np.hypot(float(g[0]) - p[0], float(g[1]) - p[1]) < float(r)]
cands = [g for g in cands if np.hypot(float(g[0]) - p[0], float(g[1]) - p[1]) < 0.8 * float(r)]
# 그림에서 p와 겹치지 않도록, 조건을 만족하는 유리점 가운데 p에서 가장 먼 것을 고른다
qf = max(cands, key=lambda g: np.hypot(float(g[0]) - p[0], float(g[1]) - p[1]))
q = np.array([float(qf[0]), float(qf[1])])
rf = float(r)
# 자기검사: p ∈ B_r(q), B_r(q) ⊆ B_{2r}(p) ⊆ B_ε(p)
assert np.linalg.norm(p - q) < rf
assert np.linalg.norm(p - q) + rf < eps

G = np.array([[float(g[0]), float(g[1])] for g in grid])
near = np.linalg.norm(G - p, axis=1) < 0.95
ax.plot(*G[near].T, ".", ms=2.2, color=C["aux"], zorder=2)
ax.add_patch(Circle(p, eps, fill=False, edgecolor=C["tangent"], lw=1.2, zorder=4))
ax.add_patch(Circle(q, rf, facecolor=C["accent"], alpha=0.25, lw=0, zorder=4))
ax.add_patch(Circle(q, rf, fill=False, edgecolor=C["accent"], lw=1.3, zorder=5))
ax.plot(*p, "o", ms=4, color=C["main"], zorder=7)
ax.plot(*q, "s", ms=4, color=C["accent"], zorder=7)
ax.text(p[0] + 0.03, p[1] + 0.05, r"$p$", fontsize=12, zorder=8)
ax.text(q[0] - 0.14, q[1] - 0.13, r"$q$", fontsize=12, color=C["accent"], zorder=8)
ax.text(p[0] - eps * 0.2, p[1] - eps - 0.16, r"$B_\varepsilon(p)$", fontsize=10, color=C["tangent"], ha="center")
ax.text(-1.05, 0.85, r"$U$", fontsize=13)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
print(f"eps = {eps:.4f}, r = {r}, q = ({qf[0]}, {qf[1]})")
