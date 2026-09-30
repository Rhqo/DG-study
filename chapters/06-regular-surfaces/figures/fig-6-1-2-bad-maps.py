"""그림 6.1.2: 정의 6.1.1의 조건 (1) 또는 (3)을 어기는 사상의 상 (6.1절, 비예 6.1.3, 비예 6.1.9).

(a) 원뿔 z = √(x² + y²): 사상 (u, v) ↦ (u, v, √(u² + v²))는 위상동형사상이지만 (0, 0)에서 미분가능하지 않다.
    원판 u² + v² ≤ 1 위의 부분, 꺾인 좌표선 u = 0과 v = 0(검정), 꼭짓점 O.
(b) 뾰족 기둥 {x = |y|^{2/3}}: 사상 (u, v) ↦ (u², u³, v)는 매끄럽고 위상동형사상이지만
    u = 0에서 전미분이 단사가 아니다. |u| ≤ 1, |v| ≤ 0.7 부분과 모서리 u = 0(주황).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-1-2-bad-maps.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

C = dgfig.COLORS
LW = dgfig.LW

u, v = sp.symbols("u v", real=True)
cone = sp.Matrix([u, v, sp.sqrt(u ** 2 + v ** 2)])
cusp = sp.Matrix([u ** 2, u ** 3, v])
fcone = sp.lambdify((u, v), list(cone), "numpy")
fcusp = sp.lambdify((u, v), list(cusp), "numpy")

# 자기검사: 뾰족 기둥의 전미분은 u = 0에서 계수 1, 그 밖에서는 계수 2
Dc = cusp.jacobian([u, v])
assert Dc.subs(u, 0).rank() == 1
assert Dc.subs(u, sp.Rational(1, 2)).rank() == 2
# 상은 x = |y|^{2/3} 위에 있다
uu = np.linspace(-1, 1, 41)
P = np.array(fcusp(uu, 0 * uu), dtype=float).T
assert np.allclose(P[:, 0], np.abs(P[:, 1]) ** (2 / 3))


def grid(f, U, V):
    return [np.array([np.broadcast_to(c, U.shape) for c in f(U, V)]) for _ in [0]][0]


fig = plt.figure(figsize=(6.4, 3.0))

# (a) 원뿔 ------------------------------------------------------------------------------
ax = fig.add_axes([0.0, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=22, azim=-58)
ax.set_axis_off()
# 면은 원판 u² + v² ≤ 1 위에서 극좌표로 그리고, 좌표선 u = 0, v = 0(꺾인 곡선)을 굵게 그린다
Rg, Tg = np.meshgrid(np.linspace(0, 1, 31), np.linspace(0, 2 * np.pi, 97))
Xc = grid(fcone, Rg * np.cos(Tg), Rg * np.sin(Tg))
dgfig.surface(ax, *Xc, alpha=0.45, grid_every=8, zorder=1)
s = np.linspace(-1, 1, 201)
ax.plot(*np.array(fcone(s, 0 * s)), color=C["main"], lw=1.3, zorder=5)
ax.plot(*np.array(fcone(0 * s, s)), color=C["main"], lw=1.3, zorder=5)
dgfig.point3d(ax, [0, 0, 0], size=16, zorder=10)
ax.text(0.08, -0.05, -0.18, r"$O$", fontsize=12)
dgfig.equal_aspect(ax, *Xc, zoom=1.25)
ax.text2D(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11, va="top")

# (b) 뾰족 기둥 ---------------------------------------------------------------------------
ax = fig.add_axes([0.5, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=42, azim=-160)
ax.set_axis_off()
Ug, Vg = np.meshgrid(np.linspace(-1, 1, 81), np.linspace(-0.7, 0.7, 29))
Xk = grid(fcusp, Ug, Vg)
dgfig.surface(ax, *Xk, alpha=0.45, grid_every=8, zorder=1)
s = np.linspace(-1, 1, 201)
for v0 in (-0.7, 0.7):
    ax.plot(*np.array([np.broadcast_to(c, s.shape) for c in fcusp(s, v0 + 0 * s)]), color=C["main"], lw=1.2, zorder=5)
ax.plot([0, 0], [0, 0], [-0.7, 0.7], color=C["accent"], lw=2.0, zorder=6)
ax.text(0.0, 0.0, -0.8, r"$u = 0$", fontsize=11, color=C["accent"], ha="center", va="top")
dgfig.equal_aspect(ax, *Xk, zoom=0.95)
ax.text2D(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11, va="top")

dgfig.save(fig, __file__)
