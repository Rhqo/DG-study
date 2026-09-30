"""그림 6.5.3: 회전면 — 현수면 (6.5절, 정의 6.5.9, 명제 6.5.10, 예 6.5.11).

생성곡선 α(u) = (ρ(u), z(u)) = (cosh u, u), |u| ≤ 1.2 (반평면 𝓗 = {ρ > 0} 안의 곡선).
x(u, v) = (ρ(u) cos v, ρ(u) sin v, z(u)) (dgsym.EXAMPLES["revolution"]에 ρ = cosh, z = id를 넣은 것).
(a) 반평면 𝓗와 곡선 α(검정), 점 α(u₀) (u₀ = 0.5), 회전축 ρ = 0(회색 점선).
(b) 현수면과 좌표곡선 격자, u = u₀인 원(청록)과 v = v₀인 곡선(파랑), p = x(u₀, v₀), v₀ = −π/4,
    x_u, x_v(파랑, 실제 길이; 서로 수직), x_u × x_v(주홍, 1/2배).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-5-3-revolution.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["revolution"]
u, v = ex["coords"]
rho, z = ex["functions"]
Xe = ex["expr"].subs({rho(u): sp.cosh(u), z(u): u}).doit()
X = sp.lambdify((u, v), list(Xe), "numpy")
Xu = sp.lambdify((u, v), list(Xe.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xe.diff(v)), "numpy")


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


U0, V0 = 0.5, -np.pi / 4
p = np.array(X(U0, V0), float)
xu, xv = np.array(Xu(U0, V0), float), np.array(Xv(U0, V0), float)
n = np.cross(xu, xv)

# 자기검사: x_u ⊥ x_v, |x_u × x_v| = ρ √(ρ'² + z'²) = cosh u · cosh u
assert abs(xu @ xv) < 1e-12
assert abs(np.linalg.norm(n) - np.cosh(U0) ** 2) < 1e-12
assert sp.simplify(Xe.diff(u).cross(Xe.diff(v)).norm() ** 2 - sp.cosh(u) ** 4) == 0

fig = plt.figure(figsize=(6.4, 3.3))

# (a) 생성곡선 --------------------------------------------------------------------------
ax0 = fig.add_axes([0.03, 0.1, 0.27, 0.8])
ax0.fill([0, 2.2, 2.2, 0], [-1.5, -1.5, 1.5, 1.5], color=C["region"], alpha=0.15, lw=0)
ax0.plot([0, 0], [-1.5, 1.5], color=C["aux"], lw=0.8, ls=(0, (4, 3)))
uu = np.linspace(-1.2, 1.2, 200)
ax0.plot(np.cosh(uu), uu, color=C["main"], lw=1.8)
ax0.plot([np.cosh(U0)], [U0], "o", color=C["main"], ms=4)
ax0.text(np.cosh(U0) + 0.1, U0 - 0.12, r"$\alpha(u_0)$", fontsize=11)
ax0.text(2.1, -1.35, r"$\rho$", fontsize=12, ha="right")
ax0.text(0.08, 1.3, r"$z$", fontsize=12, ha="left")
ax0.text(2.0, 0.1, r"$\mathcal{H}$", fontsize=12, ha="right")
ax0.text(np.cosh(1.2) - 0.15, 1.15, r"$C$", fontsize=12, ha="right", va="top")
dgfig.schematic_axes(ax0, (-0.3, 2.3), (-1.6, 1.9))
ax0.text(-0.3, 1.9, "(a)", fontsize=11, va="top")

# (b) 현수면 ----------------------------------------------------------------------------
ax = fig.add_axes([0.32, 0.0, 0.68, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=15, azim=-60)
ax.set_axis_off()
Ug, Vg = np.meshgrid(np.linspace(-1.2, 1.2, 41), np.linspace(0, 2 * np.pi, 97))
S = xmap(Ug, Vg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.3, grid=False, zorder=1)


def nrm(P):
    rr = np.hypot(P[:, 0], P[:, 1])
    return np.stack([P[:, 0] / rr, P[:, 1] / rr, -np.sinh(P[:, 2])], axis=1)


def draw(curve, role="#6F6F6F", lw=0.45, zorder=3):
    dgfig.curve3d(ax, curve, role=role, lw=lw, visible=dgfig.visible_mask(ax, curve, nrm(curve)), zorder=zorder)


s = np.linspace(0, 2 * np.pi, 300)
for u1 in np.linspace(-1.2, 1.2, 9):
    draw(xmap(u1 * np.ones_like(s), s))
for v1 in np.linspace(0, 2 * np.pi, 13)[:-1]:
    draw(xmap(uu, v1 * np.ones_like(uu)))
draw(xmap(U0 * np.ones_like(s), s), role="third", lw=1.5, zorder=5)
draw(xmap(uu, V0 * np.ones_like(uu)), role="tangent", lw=1.5, zorder=5)
dgfig.point3d(ax, p, size=14, zorder=12)
dgfig.arrow3d(ax, p, xu, role="tangent", label=r"$\mathbf{x}_u$", zorder=14, label_offset=(0, 0, 0.1))
dgfig.arrow3d(ax, p, xv, role="tangent", label=r"$\mathbf{x}_v$", zorder=14, label_offset=(0.1, 0.1, 0))
dgfig.arrow3d(ax, p, 0.5 * n, role="normal", label=r"$\mathbf{x}_u\times\mathbf{x}_v$", zorder=14,
              label_offset=(-0.35, -0.2, 0.12))
ax.text(*(p + np.array([-0.05, -0.1, -0.22])), r"$p$", fontsize=12, zorder=15)
ax.plot([0, 0], [0, 0], [-1.3, 1.3], color=C["aux"], lw=0.8, ls=(0, (4, 3)), zorder=0)
dgfig.equal_aspect(ax, np.array([[-1.8, -1.8, -1.2], [1.8, 1.8, 1.2]]), zoom=1.35)
ax.text2D(0.02, 0.95, "(b)", transform=ax.transAxes, fontsize=11, va="top")

dgfig.save(fig, __file__)
