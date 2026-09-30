"""그림 3.1.1: 열린집합과 열린집합이 아닌 집합 (3.1절, 정의 3.1.5, 예 3.1.7, 비예 3.1.8).

(a) 평면의 열린집합 U(경계는 U에 속하지 않으므로 점선). U의 각 점 p, p'마다 U 안에 들어가는
    열린공이 있다. 경계에 가까운 점일수록 반지름이 작아야 한다.
(b) 닫힌 원판 D = {|x| <= 1}(경계 포함, 실선). 내부의 점 p에는 D 안의 공이 있지만,
    경계 위의 점 q = (1, 0)을 중심으로 하는 공은 반지름을 아무리 작게 해도 D 밖으로 삐져나간다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/03-topology-basics/figures/fig-3-1-1-open-sets.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon

C = dgfig.COLORS
LW = dgfig.LW

fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))

# (a) 열린집합 U ---------------------------------------------------------------
ax = axs[0]
dgfig.schematic_axes(ax, (-1.45, 1.45), (-1.3, 1.3))
pts = dgfig.blob(ax, center=(0, 0), radius=1.0, seed=3, fill="region", edge=None)
ax.add_patch(Polygon(pts[:-1], closed=True, fill=False, edgecolor=C["main"], lw=1.2,
                     ls=(0, (4, 3)), zorder=3))


def dist_to_boundary(p):
    """p에서 다각형 경계(점 목록)까지의 거리 (선분별 최소 거리)."""
    a, b = pts[:-1], pts[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    proj = a + t[:, None] * ab
    return np.min(np.linalg.norm(proj - p, axis=1))


def inside(p):
    """짝홀 규칙으로 p가 다각형 안에 있는지 판정한다."""
    x, y = p
    xs, ys = pts[:-1, 0], pts[:-1, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = ((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / (ys2 - ys + 1e-300) + xs)
    return np.count_nonzero(cross) % 2 == 1


p = np.array([-0.15, 0.05])
p2 = np.array([0.62, -0.52])
eps1 = 0.8 * dist_to_boundary(p)
eps2 = 0.8 * dist_to_boundary(p2)
# 자기검사: 두 점은 U 안에 있고, 그린 공은 U 안에 들어간다 (반지름 < 경계까지의 거리)
assert inside(p) and inside(p2)
assert eps1 < dist_to_boundary(p) and eps2 < dist_to_boundary(p2)
assert eps2 < eps1  # 경계에 가까운 점은 더 작은 공

for q, e in ((p, eps1), (p2, eps2)):
    ax.add_patch(Circle(q, e, fill=False, edgecolor=C["tangent"], lw=LW["vector"] * 0.8, zorder=4))
    ax.plot(*q, "o", color=C["main"], ms=3.5, zorder=6)
ax.text(p[0] + 0.05, p[1] + 0.06, r"$p$", fontsize=12, zorder=7)
ax.text(p[0] - 0.08, p[1] + eps1 + 0.06, r"$B_\varepsilon(p)$", fontsize=11, color=C["tangent"],
        ha="center", zorder=7)
ax.text(p2[0] + 0.04, p2[1] + 0.05, r"$p'$", fontsize=12, zorder=7)
ax.text(-1.05, 0.85, r"$U$", fontsize=13)
ax.text(0.0, 1.0, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

# (b) 닫힌 원판 D ---------------------------------------------------------------
ax = axs[1]
dgfig.schematic_axes(ax, (-1.45, 1.6), (-1.3, 1.3))
ax.add_patch(Circle((0, 0), 1.0, facecolor=C["region"], alpha=dgfig.ALPHA["region"], lw=0, zorder=1))
ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor=C["main"], lw=LW["main"] * 0.8, zorder=3))
pb = np.array([-0.3, 0.1])
eb = 0.5
assert np.linalg.norm(pb) + eb < 1.0          # B_eb(pb) ⊆ D
ax.add_patch(Circle(pb, eb, fill=False, edgecolor=C["tangent"], lw=LW["vector"] * 0.8, zorder=4))
ax.plot(*pb, "o", color=C["main"], ms=3.5, zorder=6)
ax.text(pb[0] + 0.05, pb[1] + 0.06, r"$p$", fontsize=12)

q = np.array([1.0, 0.0])
for e in (0.36, 0.2):
    # 공 B_e(q)의 점 (1 + e/2, 0)은 |x| > 1이므로 D 밖에 있다
    assert np.linalg.norm(q + np.array([e / 2, 0])) > 1.0
    ax.add_patch(Circle(q, e, fill=False, edgecolor=C["normal"], lw=LW["vector"] * 0.8, zorder=4))
    ax.plot(*(q + [e / 2, 0]), "o", color=C["normal"], ms=2.5, zorder=6)
ax.plot(*q, "o", color=C["main"], ms=3.5, zorder=6)
ax.text(q[0] - 0.2, q[1] + 0.07, r"$q$", fontsize=12, zorder=7)
ax.text(-0.95, 0.85, r"$D$", fontsize=13)
ax.text(0.0, 1.0, "(b)", transform=ax.transAxes, ha="left", va="top", fontsize=11)

fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
