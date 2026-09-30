"""그림 7.4.1: 입체사영은 각을 보존한다 (7.4절, 예 7.4.9).

σ⁻¹(u, v) = (2u, 2v, u² + v² − 1)/(u² + v² + 1) (dgsym.stereographic(2)의 sigma_inv), 단위구면 S²(1).
왼쪽: 평면 P = {z = 0}의 직교 격자 u = k/2, v = k/2 (|k| ≤ 4)와 작은 정사각형 Q = [0.5, 0.75] × [0.25, 0.5](하늘색),
      점선 원은 단위원 u² + v² = 1(적도의 상).
오른쪽: 구면과 격자의 상(북극 e₃ = (0, 0, 1)을 지나는 원들), Q의 상 σ⁻¹(Q)(하늘색), 적도(점선).
두 격자선이 만나는 곳의 각은 직각 그대로이고, 칸의 크기만 2/(1 + u² + v²)배로 바뀐다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-4-1-stereographic.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS

st = dgsym.stereographic(2)
U1, U2 = st["chart_coords"]
inv = st["sigma_inv"]
Sinv = sp.lambdify((U1, U2), list(inv), "numpy")
Dinv_u = sp.lambdify((U1, U2), list(inv.diff(U1)), "numpy")
Dinv_v = sp.lambdify((U1, U2), list(inv.diff(U2)), "numpy")
E, F, G = dgsym.first_ff(inv, U1, U2)
lam2 = sp.lambdify((U1, U2), E, "numpy")
assert sp.simplify(F) == 0 and sp.simplify(E - G) == 0                 # 등온 매개화


def smap(u, v):
    return np.stack([np.broadcast_to(c, np.shape(u)) for c in Sinv(u, v)], axis=-1)


# 자기검사: 격자선의 상은 구면 위에 있고, 교점에서 직교한다
for (a, b) in ((0.5, 0.5), (1.0, -0.5), (-1.5, 1.0)):
    xu, xv = np.array(Dinv_u(a, b), float), np.array(Dinv_v(a, b), float)
    assert abs(xu @ xv) < 1e-12 and np.isclose(xu @ xu, xv @ xv)
    assert np.isclose(np.linalg.norm(xu), 2 / (1 + a * a + b * b))
    assert np.isclose(np.linalg.norm(smap(a, b)), 1.0)

fig = plt.figure(figsize=(6.6, 3.3))

# 왼쪽: 평면 ---------------------------------------------------------------------------------
axP = fig.add_axes([0.02, 0.08, 0.4, 0.84])
L = 2.0
ks = np.arange(-4, 5) / 2
for k in ks:
    axP.plot([k, k], [-L, L], color=C["tangent"] if k == 0.5 else "#8A8A8A", lw=1.0 if k == 0.5 else 0.5)
    axP.plot([-L, L], [k, k], color=C["third"] if k == 0.5 else "#8A8A8A", lw=1.0 if k == 0.5 else 0.5)
s = np.linspace(0, 2 * np.pi, 200)
axP.plot(np.cos(s), np.sin(s), color=C["aux"], lw=0.9, ls=(0, (3, 2)))
Q = np.array([[0.5, 0.25], [0.75, 0.25], [0.75, 0.5], [0.5, 0.5]])
axP.fill(*Q.T, color=C["region"], alpha=0.8, lw=0)
axP.plot(*np.vstack([Q, Q[:1]]).T, color=C["tangent"], lw=0.9)
axP.text(0.625, 0.375, r"$Q$", fontsize=9, ha="center", va="center")
axP.text(0, L + 0.25, r"$P = \{z = 0\}$", fontsize=12, ha="center")
dgfig.schematic_axes(axP, (-L - 0.1, L + 0.1), (-L - 0.1, L + 0.6))

# 오른쪽: 구면 ------------------------------------------------------------------------------
ax = fig.add_axes([0.44, -0.02, 0.56, 1.04], projection="3d", computed_zorder=False)
ax.view_init(elev=20, azim=35)
ax.set_axis_off()
Tg, Pg = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = np.stack([np.sin(Tg) * np.cos(Pg), np.sin(Tg) * np.sin(Pg), np.cos(Tg)], axis=-1)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.2, grid=False, zorder=1)
view = dgfig.view_vector(ax)
tt = np.concatenate([-np.logspace(2, -2, 300), np.logspace(-2, 2, 300)])
for k in ks:
    for curve, col, w in ((smap(k * np.ones_like(tt), tt), C["tangent"] if k == 0.5 else "#8A8A8A", 1.3 if k == 0.5 else 0.5),
                          (smap(tt, k * np.ones_like(tt)), C["third"] if k == 0.5 else "#8A8A8A", 1.3 if k == 0.5 else 0.5)):
        dgfig.curve3d(ax, curve, role=col, lw=w, visible=curve @ view > 0, hidden=None, zorder=4)
eq = np.stack([np.cos(s), np.sin(s), 0 * s], axis=1)
dgfig.curve3d(ax, eq, role=C["aux"], lw=0.9, ls=(0, (3, 2)), visible=eq @ view > 0, hidden=None, zorder=4)
pp = np.linspace(0, 1, 20)
edge = np.concatenate([smap(0.5 + 0.25 * pp, 0.25 + 0 * pp), smap(0.75 + 0 * pp, 0.25 + 0.25 * pp),
                       smap(0.75 - 0.25 * pp, 0.5 + 0 * pp), smap(0.5 + 0 * pp, 0.5 - 0.25 * pp)])
assert np.all(edge @ view > 0)
ax.add_collection3d(Poly3DCollection([edge * 1.004], facecolor=C["region"], alpha=0.85, edgecolor=C["tangent"],
                                     linewidth=0.9, zorder=7))
dgfig.point3d(ax, [0, 0, 1], size=14, zorder=12)
ax.text(0.05, 0.05, 1.1, r"$e_3$", fontsize=11, zorder=15)
c = smap(0.625, 0.375) * 1.12
ax.text(*c, r"$\sigma^{-1}(Q)$", fontsize=10, zorder=15, ha="center")
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.3)
dgfig.map_arrow(fig, axP, ax, r"$\sigma^{-1}$", xy_from=(0.95, 0.85), xy_to=(0.2, 0.82), rad=-0.3)

dgfig.save(fig, __file__)
