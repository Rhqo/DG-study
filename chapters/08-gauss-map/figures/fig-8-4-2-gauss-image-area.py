"""그림 8.4.2: 가우스 사상은 넓이를 |K|배로 바꾸고, K < 0이면 방향을 뒤집는다 (8.4절, 명제 8.4.11).

원환면 (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8. 이 그림은 보이는 쪽을 위해 바깥쪽 법벡터 −N (N = N_x)을 쓴다
(K는 법벡터의 선택에 무관하다).
좌표 정사각형 Q_i = [u_i − h, u_i + h] × [v_i − h, v_i + h], h = 0.3,
p₁ = x(0.4, −π/2) (K > 0), p₂ = x(2.6, 5π/6) (K < 0).
왼쪽: 원환면과 두 조각 x(Q_i)(하늘색), 경계를 (u, v) 평면에서 반시계 방향으로 도는 화살표(검정).
오른쪽: 단위구면과 가우스 상 −N(x(Q_i))(하늘색), 같은 경계의 상과 화살표.
자기검사: (−N)_u × (−N)_v = K x_u × x_v (수치), 넓이 비율 A(−N(x(Q_i)))/A(x(Q_i))가 |K(p_i)|에 가깝다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-4-2-gauss-image-area.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
vals = {Rs: 2.0, rs: 0.8}
Xe = ex["expr"].subs(vals)
ne = -dgsym.unit_normal(ex["expr"], u, v, ex["positive"]).subs(vals)     # 바깥쪽
Ke, _ = dgsym.K_H(ex["expr"], u, v, ex["positive"])
Ke = Ke.subs(vals)
X = sp.lambdify((u, v), list(Xe), "numpy")
Nf = sp.lambdify((u, v), list(ne), "numpy")
Kf = sp.lambdify((u, v), Ke, "numpy")
cross_x = sp.lambdify((u, v), list(Xe.diff(u).cross(Xe.diff(v))), "numpy")
cross_n = sp.lambdify((u, v), list(ne.diff(u).cross(ne.diff(v))), "numpy")
for (a, b) in [(0.4, -1.0), (2.6, 2.0), (1.3, 0.2)]:
    assert np.allclose(np.array(cross_n(a, b), float), Kf(a, b) * np.array(cross_x(a, b), float))


def smap(F, U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in F(U, V)], axis=-1)


h = 0.3
centers = [(0.4, -np.pi / 2), (2.6, 5 * np.pi / 6)]
# 넓이 비율 확인 (중점 규칙)
for (a, b) in centers:
    g = np.linspace(-h, h, 61)
    gm = 0.5 * (g[1:] + g[:-1])
    Ug, Vg = np.meshgrid(a + gm, b + gm, indexing="ij")
    Ax = np.sum(np.linalg.norm(smap(cross_x, Ug, Vg), axis=-1))
    An = np.sum(np.linalg.norm(smap(cross_n, Ug, Vg), axis=-1))
    ratio = An / Ax
    assert abs(ratio - abs(Kf(a, b))) < 0.05 * abs(Kf(a, b))


def boundary(a, b, n=60):
    s = np.linspace(-h, h, n)
    edges = [(a + s, b - h + 0 * s), (a + h + 0 * s, b + s), (a - s, b + h + 0 * s), (a - h + 0 * s, b - s)]
    return edges


def draw_patch(ax, F, a, b, zorder):
    g = np.linspace(-h, h, 13)
    Ug, Vg = np.meshgrid(a + g, b + g, indexing="ij")
    P = smap(F, Ug, Vg)
    quads = [[P[i, j], P[i + 1, j], P[i + 1, j + 1], P[i, j + 1]] for i in range(12) for j in range(12)]
    ax.add_collection3d(Poly3DCollection(quads, facecolor=C["region"], alpha=0.55, edgecolor="none", zorder=zorder))
    for (U_, V_) in boundary(a, b):
        pts = smap(F, U_, V_)
        ax.plot(*pts.T, color=C["main"], lw=1.1, zorder=zorder + 1)
        k = len(U_) // 2
        dgfig.arrow3d(ax, pts[k - 3], pts[k + 3] - pts[k - 3], role="main", head=9, lw=1.1, zorder=zorder + 2)


fig = plt.figure(figsize=(6.6, 3.4))
axL = dgfig.axes3d(fig, pos=121, elev=35, azim=-62)
axR = dgfig.axes3d(fig, pos=122, elev=25, azim=-62)
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = smap(X, Ug, Vg)
dgfig.surface(axL, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False, zorder=1)
for (a, b) in centers:
    draw_patch(axL, X, a, b, zorder=5)
p1, p2 = np.array(X(*centers[0]), float), np.array(X(*centers[1]), float)
axL.text(*(p1 + np.array([0.25, -0.25, 0.25])), r"$p_1$", fontsize=11, zorder=12)
axL.text(*(p2 + np.array([-0.55, 0.1, 0.45])), r"$p_2$", fontsize=11, zorder=12)
dgfig.equal_aspect(axL, S.reshape(-1, 3), zoom=1.25)

th, ph = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
Sx, Sy, Sz = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(axR, Sx, Sy, Sz, alpha=0.16, grid=False, zorder=1)
for (a, b) in centers:
    draw_patch(axR, Nf, a, b, zorder=5)
n1, n2 = np.array(Nf(*centers[0]), float), np.array(Nf(*centers[1]), float)
axR.text(*(1.12 * n1 + np.array([0.05, -0.1, -0.22])), r"$-\mathbf{N}(p_1)$", fontsize=10, zorder=12)
axR.text(*(1.1 * n2 + np.array([0.05, 0.0, 0.12])), r"$-\mathbf{N}(p_2)$", fontsize=10, zorder=12)
dgfig.equal_aspect(axR, np.array([[-1, -1, -1], [1, 1, 1]]) * 1.0, zoom=1.3)
fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)
dgfig.map_arrow(fig, axL, axR, r"$-\mathbf{N}$", xy_from=(0.9, 0.78), xy_to=(0.12, 0.78), rad=-0.3)

dgfig.save(fig, __file__)
