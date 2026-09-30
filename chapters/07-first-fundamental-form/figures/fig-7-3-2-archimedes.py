"""그림 7.3.2: 아르키메데스의 투영 — 넓이는 같지만 모양은 다르다 (7.3절, 예 7.3.11).

구면 x(θ, φ) (dgsym.EXAMPLES["sphere"])과 원기둥 (dgsym.EXAMPLES["cylinder"]), r = 1.
ψ(x(θ, φ)) = (r cos φ, r sin φ, r cos θ): 구면의 점을 z축에서 수평으로 바깥쪽의 외접 원기둥으로 옮긴다.
구면 조각 R = x([θ₁, θ₂] × [φ₁, φ₂])(하늘색)와 그 상 ψ(R)(주황), θ₁ = 0.5, θ₂ = 1.0, φ₁ = 0.2, φ₂ = 1.4.
왼쪽: 구면과 R, 오른쪽: 외접 원기둥과 ψ(R) (같은 축척). 두 조각의 넓이는 r²(φ₂ − φ₁)(cos θ₁ − cos θ₂) ≈ 0.405로 같다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-3-2-archimedes.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS

sph = dgsym.EXAMPLES["sphere"]
th, ph = sph["coords"]
(rs,) = sph["params"]
Xs = sph["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xs), "numpy")
cyl = dgsym.EXAMPLES["cylinder"]
cu, cv = cyl["coords"]
(cr,) = cyl["params"]
Xc = sp.lambdify((cu, cv), list(cyl["expr"].subs(cr, 1)), "numpy")
Ys = cyl["expr"].subs(cr, 1).subs({cu: ph, cv: sp.cos(th)})          # ψ∘x(θ, φ)
Es, Fs, Gs = dgsym.first_ff(Xs, th, ph, (sp.sin(th),))
Ey, Fy, Gy = dgsym.first_ff(Ys, th, ph, (sp.sin(th),))
# 자기검사: 넓이요소가 같다 (√(EG − F²) = sin θ), 그러나 계수는 다르다 (등거리가 아님)
assert sp.simplify(sp.sqrt(Ey * Gy - Fy ** 2) - sp.sin(th)) == 0 or sp.simplify(Ey * Gy - Fy ** 2 - (Es * Gs - Fs ** 2)) == 0
assert sp.simplify(Ey - Es) != 0


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


def psi(P3):
    P3 = np.asarray(P3, float)
    rr = np.hypot(P3[..., 0], P3[..., 1])
    return np.stack([P3[..., 0] / rr, P3[..., 1] / rr, P3[..., 2]], axis=-1)


T1, T2, F1, F2 = 0.5, 1.0, 0.2, 1.4
area_s = (F2 - F1) * (np.cos(T1) - np.cos(T2))
tt = np.linspace(T1, T2, 400)
pp = np.linspace(F1, F2, 400)
TT, PP = np.meshgrid(tt, pp)
num_s = np.trapezoid(np.trapezoid(np.sin(TT), tt, axis=1), pp)
assert np.isclose(num_s, area_s, rtol=1e-4)
assert np.isclose(area_s, 0.405, atol=5e-4)

fig = plt.figure(figsize=(6.4, 3.4))
edge_s = np.concatenate([xmap(tt, F1 * np.ones_like(tt)), xmap(T2 * np.ones_like(pp), pp),
                         xmap(tt[::-1], F2 * np.ones_like(tt)), xmap(T1 * np.ones_like(pp), pp[::-1])])
edge_c = psi(edge_s)
assert np.allclose(np.hypot(edge_c[:, 0], edge_c[:, 1]), 1.0)
box = np.array([[-1, -1, -1], [1, 1, 1]])
s = np.linspace(0, 2 * np.pi, 300)

# 왼쪽: 구면 --------------------------------------------------------------------------------
ax1 = fig.add_axes([0.0, 0.0, 0.46, 1.0], projection="3d", computed_zorder=False)
ax1.view_init(elev=18, azim=25)
ax1.set_axis_off()
Tg, Pg = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = xmap(Tg, Pg)
dgfig.surface(ax1, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False, zorder=1)
v1 = dgfig.view_vector(ax1)
assert np.all(edge_s @ v1 > 0)
for k in range(1, 6):
    c = xmap(k * np.pi / 6 * np.ones_like(s), s)
    dgfig.curve3d(ax1, c, role="#9A9A9A", lw=0.4, visible=c @ v1 > 0, hidden=None, zorder=3)
ax1.add_collection3d(Poly3DCollection([edge_s * 1.003], facecolor=C["region"], alpha=0.75, edgecolor=C["tangent"],
                                      linewidth=1.0, zorder=6))
ax1.text(*(np.array(X((T1 + T2) / 2, (F1 + F2) / 2), float) * 1.02), r"$R$", fontsize=12, zorder=15, ha="center", va="center")
ax1.text(0, 0, -1.45, r"$S^2(r)$", fontsize=12, ha="center")
dgfig.equal_aspect(ax1, box, zoom=1.15)

# 오른쪽: 원기둥 -----------------------------------------------------------------------------
ax2 = fig.add_axes([0.54, 0.0, 0.46, 1.0], projection="3d", computed_zorder=False)
ax2.view_init(elev=18, azim=25)
ax2.set_axis_off()
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 121), np.linspace(-1, 1, 9))
Cg = np.stack([np.broadcast_to(c, Ug.shape) for c in Xc(Ug, Vg)], axis=-1)
dgfig.surface(ax2, Cg[..., 0], Cg[..., 1], Cg[..., 2], alpha=0.25, grid=False, zorder=1)
v2 = dgfig.view_vector(ax2)
for z0 in np.cos(np.arange(1, 6) * np.pi / 6):
    ring = np.stack(Xc(s, z0 * np.ones_like(s)), axis=-1)
    vis = np.stack([ring[:, 0], ring[:, 1], 0 * s], 1) @ v2 > 0
    dgfig.curve3d(ax2, ring, role="#9A9A9A", lw=0.4, visible=vis, hidden=None, zorder=3)
assert np.all(np.stack([edge_c[:, 0], edge_c[:, 1], 0 * edge_c[:, 2]], 1) @ v2 > 0)
ax2.add_collection3d(Poly3DCollection([edge_c * np.array([1.003, 1.003, 1.0])], facecolor=C["accent"], alpha=0.55,
                                      edgecolor=C["accent"], linewidth=1.2, zorder=6))
ax2.text(*(psi(np.array(X((T1 + T2) / 2, (F1 + F2) / 2), float)) * np.array([1.05, 1.05, 1.0]) + np.array([0, 0, 0.0])),
         r"$\psi(R)$", fontsize=12, zorder=15, ha="center", va="center")
ax2.text(0, 0, -1.45, r"$C_r$", fontsize=12, ha="center")
dgfig.equal_aspect(ax2, box, zoom=1.15)

dgfig.map_arrow(fig, ax1, ax2, r"$\psi$", xy_from=(0.78, 0.72), xy_to=(0.28, 0.72), rad=-0.35)

dgfig.save(fig, __file__)
