"""그림 6.2.2: 정칙곡면이 아닌 집합에서 한 점을 지나는 곡선들의 속도 (6.2절, 비예 6.2.2, 비예 6.2.6).

(a) 두 겹 원뿔 K₂ = {x² + y² = z²}의 |z| ≤ 1 부분. 꼭짓점 0에서 곡선 t ↦ t w (w ∈ K₂)의 속도 w를
    여덟 방향으로 그린다(주황). 속도 전체는 K₂ 자신이며 평면이 아니다.
(b) 8자 기둥 x(u, v) = (sin 2u, sin u, v)의 원점 근처(|u| ≤ 0.5와 |u| ≥ π − 0.5, |v| ≤ 0.5).
    원점을 지나는 두 평면 span{(2, 1, 0), (0, 0, 1)}(파랑)과 span{(−2, 1, 0), (0, 0, 1)}(주황)과
    그 안의 속도 (2, 1, 0), (−2, 1, 0)(각각 1/2배로 그림).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-2-2-velocity-sets.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

C = dgfig.COLORS

# 자기검사: 원뿔의 생성선 방향은 K₂ 위, 8자 기둥의 두 곡선 속도
ws = [np.array([np.cos(a), np.sin(a), s]) for a in np.linspace(0, 2 * np.pi, 5)[:-1] for s in (1.0, -1.0)]
for w in ws:
    assert abs(w[0] ** 2 + w[1] ** 2 - w[2] ** 2) < 1e-12
h = 1e-6
v1 = (np.array([np.sin(2 * h), np.sin(h), 0]) - np.array([np.sin(-2 * h), np.sin(-h), 0])) / (2 * h)
v2 = (np.array([-np.sin(2 * h), np.sin(h), 0]) - np.array([np.sin(2 * h), np.sin(-h), 0])) / (2 * h)
assert np.allclose(v1, [2, 1, 0], atol=1e-6) and np.allclose(v2, [-2, 1, 0], atol=1e-6)
# (−sin 2t, sin t)는 8자 곡선 위의 점: t > 0이면 β(π − t)
t = 0.3
assert np.allclose([np.sin(2 * (np.pi - t)), np.sin(np.pi - t)], [-np.sin(2 * t), np.sin(t)])

fig = plt.figure(figsize=(6.4, 3.2))

# (a) 두 겹 원뿔 -----------------------------------------------------------------------------
ax = fig.add_axes([0.0, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=16, azim=-60)
ax.set_axis_off()
R, T = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(0, 2 * np.pi, 73))
dgfig.surface(ax, R * np.cos(T), R * np.sin(T), R, alpha=0.3, grid_every=6, zorder=1)
for w in ws:
    dgfig.arrow3d(ax, [0, 0, 0], 0.8 * w / np.linalg.norm(w), role="accent", zorder=10, head=9)
dgfig.point3d(ax, [0, 0, 0], size=14, zorder=12)
ax.text(0.12, -0.05, -0.12, r"$0$", fontsize=12, zorder=13)
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.25)
ax.text2D(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11, va="top")

# (b) 8자 기둥의 교차선 ----------------------------------------------------------------------
ax = fig.add_axes([0.5, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=42, azim=-78)
ax.set_axis_off()
H = 0.5
for (a0, a1) in ((-0.5, 0.5), (np.pi - 0.5, np.pi), (-np.pi, -np.pi + 0.5)):
    U_, V_ = np.meshgrid(np.linspace(a0, a1, 25), np.linspace(-H, H, 9))
    dgfig.surface(ax, np.sin(2 * U_), np.sin(U_), V_, alpha=0.35, grid_every=4, zorder=1)


def plane(ax, e1, e2, size, color, zorder):
    e1 = np.asarray(e1, float) / np.linalg.norm(e1)
    e2 = np.asarray(e2, float) / np.linalg.norm(e2)
    c = [size * (s1 * e1 + s2 * e2) for s1, s2 in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_collection3d(Poly3DCollection([c], facecolor=color, alpha=0.18, edgecolor=color, lw=0.8, zorder=zorder))


plane(ax, [2, 1, 0], [0, 0, 1], 0.55, C["tangent"], 3)
plane(ax, [-2, 1, 0], [0, 0, 1], 0.55, C["accent"], 3)
dgfig.arrow3d(ax, [0, 0, 0], 0.5 * np.array([2, 1, 0]), role="tangent", zorder=10, head=9)
dgfig.arrow3d(ax, [0, 0, 0], 0.5 * np.array([-2, 1, 0]), role="accent", zorder=10, head=9)
dgfig.arrow3d(ax, [0, 0, 0], [0, 0, 0.45], role="main", zorder=10, head=9)
dgfig.point3d(ax, [0, 0, 0], size=14, zorder=12)
ax.text(0.08, -0.12, -0.12, r"$p$", fontsize=12, zorder=13)
dgfig.equal_aspect(ax, np.array([[-1.1, -0.6, -H], [1.1, 0.6, H]]), zoom=1.0)
ax.text2D(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11, va="top")

dgfig.save(fig, __file__)
