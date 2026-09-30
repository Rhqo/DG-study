"""그림 6.5.2: 원환면의 기준 매개화 (6.5절, 예 6.5.8).

x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u) (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8.
왼쪽: U = (0, 2π) × (0, 2π)와 좌표격자, 변 u = 0, 2π와 v = 0, 2π (주황 점선), 점 q = (u₀, v₀) = (π/3, π/4).
오른쪽: 원환면과 격자, 매개화가 빠뜨리는 두 원 {u = 0}(바깥 적도)과 {v = 0}(주황),
        p = x(q)에서 x_u(파랑, 0.6배), x_v(파랑, 0.4배), x_u × x_v(주홍, 1/2배; 관의 중심원 쪽, 곧 안쪽을 향한다).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-5-2-torus.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
Xe = ex["expr"].subs({Rs: RV, rs: rV})
X = sp.lambdify((u, v), list(Xe), "numpy")
Xu = sp.lambdify((u, v), list(Xe.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xe.diff(v)), "numpy")


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


U0, V0 = np.pi / 3, np.pi / 4
p = np.array(X(U0, V0), float)
xu, xv = np.array(Xu(U0, V0), float), np.array(Xv(U0, V0), float)
n = np.cross(xu, xv)
core = np.array([RV * np.cos(V0), RV * np.sin(V0), 0.0])

# 자기검사: 외적 = −r(R + r cos u)(cos u cos v, cos u sin v, sin u), 안쪽(중심원 쪽)을 향함
expect = -rV * (RV + rV * np.cos(U0)) * np.array([np.cos(U0) * np.cos(V0), np.cos(U0) * np.sin(V0), np.sin(U0)])
assert np.allclose(n, expect)
assert n @ (p - core) < 0
# 격자점은 방정식 (ρ − R)² + z² = r² 위에 있다
G = xmap(*np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, 2 * np.pi, 30)))
rho = np.hypot(G[..., 0], G[..., 1])
assert np.allclose((rho - RV) ** 2 + G[..., 2] ** 2, rV ** 2)

fig = plt.figure(figsize=(6.8, 3.2))

# 왼쪽: U ------------------------------------------------------------------------------
axU = fig.add_axes([0.02, 0.12, 0.26, 0.76])
T = 2 * np.pi
axU.fill([0, T, T, 0], [0, 0, T, T], color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
for k in range(1, 12):
    axU.plot([k * T / 12] * 2, [0, T], color="#8C8C8C", lw=0.4)
    axU.plot([0, T], [k * T / 12] * 2, color="#8C8C8C", lw=0.4)
for xs, ys in (([0, 0], [0, T]), ([T, T], [0, T]), ([0, T], [0, 0]), ([0, T], [T, T])):
    axU.plot(xs, ys, color=C["accent"], lw=1.6, ls=(0, (4, 2.5)))
axU.plot([U0], [V0], "o", color=C["main"], ms=4)
axU.text(U0 + 0.15, V0 + 0.15, r"$q$", fontsize=12)
axU.text(T / 2, T + 0.3, r"$U$", fontsize=13, ha="center")
axU.text(T / 2, -0.35, r"$u$", fontsize=12, ha="center", va="top")
axU.text(-0.35, T / 2, r"$v$", fontsize=12, ha="right", va="center")
axU.text(T, -0.12, r"$2\pi$", fontsize=10, ha="center", va="top")
dgfig.schematic_axes(axU, (-0.9, T + 0.3), (-0.9, T + 0.8))

# 오른쪽: 원환면 ------------------------------------------------------------------------
ax = fig.add_axes([0.30, 0.0, 0.70, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=38, azim=20)
ax.set_axis_off()
Ug, Vg = np.meshgrid(np.linspace(0, T, 49), np.linspace(0, T, 97))
S = xmap(Ug, Vg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.28, grid=False, zorder=1)


def normal_out(P):
    rr = np.hypot(P[:, 0], P[:, 1])
    c = np.stack([RV * P[:, 0] / rr, RV * P[:, 1] / rr, 0 * rr], axis=1)
    return P - c


def draw(curve, role="#6F6F6F", lw=0.45, zorder=3):
    dgfig.curve3d(ax, curve, role=role, lw=lw, visible=dgfig.visible_mask(ax, curve, normal_out(curve)), zorder=zorder)


s = np.linspace(0, T, 300)
for k in range(1, 12):
    draw(xmap(k * T / 12 * np.ones_like(s), s))
    draw(xmap(s, k * T / 12 * np.ones_like(s)))
draw(xmap(np.zeros_like(s), s), role="accent", lw=2.0, zorder=6)
draw(xmap(s, np.zeros_like(s)), role="accent", lw=2.0, zorder=6)
dgfig.point3d(ax, p, size=14, zorder=12)
dgfig.arrow3d(ax, p, 0.6 * xu, role="tangent", label=r"$\mathbf{x}_u$", zorder=14, label_offset=(0, 0, 0.08))
dgfig.arrow3d(ax, p, 0.4 * xv, role="tangent", label=r"$\mathbf{x}_v$", zorder=14, label_offset=(0.05, 0, 0.05))
dgfig.arrow3d(ax, p, 0.5 * n, role="normal", label=r"$\mathbf{x}_u\times\mathbf{x}_v$", zorder=14,
              label_offset=(0.0, 0.0, -0.12))
ax.text(*(p + np.array([0.12, -0.15, 0.1])), r"$p$", fontsize=12, zorder=15)
dgfig.equal_aspect(ax, np.array([[-2.8, -2.8, -0.8], [2.8, 2.8, 0.8]]), zoom=1.3)

dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(0.95, 0.75), xy_to=(0.16, 0.72), rad=-0.3)

dgfig.save(fig, __file__)
