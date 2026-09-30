"""그림 6.5.1: f(x, y, z) = x² + y² − z²의 등위곡면 (6.5절, 예 6.5.5, 비예 6.5.6).

(a) f = 1 (한 겹 쌍곡면, |z| ≤ 1.3)과 그 안의 f = 0 (원뿔, 주황 선화).
    점 p = (cosh s₀ cos φ₀, cosh s₀ sin φ₀, sinh s₀), s₀ = 0.5, φ₀ = −π/3에서 grad f(p)(주홍, 0.3배)와
    p에 붙인 접평면(하늘색).
(b) f = −1 (두 겹 쌍곡면, 1 ≤ |z| ≤ cosh 1.35 ≈ 2.06)과 f = 0 (원뿔, 주황 선화).
원뿔의 꼭짓점 0은 f의 유일한 임계점이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-5-1-level-sets.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS

f = lambda P: P[..., 0] ** 2 + P[..., 1] ** 2 - P[..., 2] ** 2
grad = lambda P: np.array([2 * P[0], 2 * P[1], -2 * P[2]])

S0, PH0 = 0.5, -np.pi / 3
p = np.array([np.cosh(S0) * np.cos(PH0), np.cosh(S0) * np.sin(PH0), np.sinh(S0)])
g = grad(p)
e1 = np.array([-np.sin(PH0), np.cos(PH0), 0.0])                      # 위도원 방향
e2 = np.array([np.sinh(S0) * np.cos(PH0), np.sinh(S0) * np.sin(PH0), np.cosh(S0)])  # 쌍곡선 방향

# 자기검사: p는 f = 1 위, 접벡터 두 개는 기울기에 수직, 격자점이 등위곡면 위
assert abs(f(p) - 1) < 1e-12
assert abs(g @ e1) < 1e-12 and abs(g @ e2) < 1e-12
Sg, Tg = np.meshgrid(np.linspace(-1.1, 1.1, 41), np.linspace(0, 2 * np.pi, 73))
H1 = np.stack([np.cosh(Sg) * np.cos(Tg), np.cosh(Sg) * np.sin(Tg), np.sinh(Sg)], axis=-1)
assert np.allclose(f(H1), 1)
Rg, _ = np.meshgrid(np.linspace(0.0, 1.35, 31), np.linspace(0, 2 * np.pi, 73))
Hm = np.stack([np.sinh(Rg) * np.cos(Tg[:, :31]), np.sinh(Rg) * np.sin(Tg[:, :31]), np.cosh(Rg)], axis=-1)
assert np.allclose(f(Hm), -1)

fig = plt.figure(figsize=(6.4, 3.4))


def cone(ax, zmax):
    s = np.linspace(-zmax, zmax, 2)
    for a in np.linspace(0, 2 * np.pi, 13)[:-1]:
        ax.plot(s * np.cos(a), s * np.sin(a), s, color=C["accent"], lw=0.8, zorder=3)
    t = np.linspace(0, 2 * np.pi, 200)
    for z0 in (-zmax, zmax):
        ax.plot(abs(z0) * np.cos(t), abs(z0) * np.sin(t), z0 + 0 * t, color=C["accent"], lw=0.8, zorder=3)
    dgfig.point3d(ax, [0, 0, 0], size=10, zorder=12)


# (a) 한 겹 쌍곡면 -------------------------------------------------------------------------
ax = fig.add_axes([0.0, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=12, azim=0)
ax.set_axis_off()
cone(ax, 1.3)
dgfig.surface(ax, H1[..., 0], H1[..., 1], H1[..., 2], alpha=0.3, grid_every=6, zorder=4)
dgfig.tangent_plane(ax, p, e1, e2, size=0.42, zorder=6)
dgfig.point3d(ax, p, size=14, zorder=12)
dgfig.arrow3d(ax, p, 0.3 * g, role="normal", label=r"$\mathrm{grad}\,f(p)$", zorder=14, label_offset=(0, 0.45, -0.22))
ax.text(*(p + np.array([0.0, 0.12, 0.12])), r"$p$", fontsize=12, zorder=15)
dgfig.equal_aspect(ax, np.array([[-1.7, -1.7, -1.3], [1.7, 1.7, 1.3]]), zoom=1.3)
ax.text2D(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11, va="top")

# (b) 두 겹 쌍곡면 -------------------------------------------------------------------------
ax = fig.add_axes([0.5, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=14, azim=-35)
ax.set_axis_off()
cone(ax, 2.0)
for sgn in (1, -1):
    dgfig.surface(ax, Hm[..., 0], Hm[..., 1], sgn * Hm[..., 2], alpha=0.35, grid_every=6, zorder=4)
ax.text(0.15, 0, -0.1, r"$0$", fontsize=12, zorder=15)
dgfig.equal_aspect(ax, np.array([[-2, -2, -2.1], [2, 2, 2.1]]), zoom=1.2)
ax.text2D(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11, va="top")

dgfig.save(fig, __file__)
