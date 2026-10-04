"""그림 8.3.2: 원환면의 주방향과 곡률선 (8.3절, 예 8.3.6, 명제 8.3.12).

원환면 (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8, N = N_x (안쪽).
경선(v 고정)과 위도원(u 고정)이 곡률선이다. 경선 하나와 위도원 하나를 주황으로 강조한다.
여러 점에서 주방향 e₁ = x_u/r (파랑, κ₁ = 1/r)과 e₂ = x_v/(R + r cos u) (청록, κ₂ = cos u/(R + r cos u))를
0.45배로 그렸다.
자기검사: e₁, e₂가 W의 고유벡터이고 서로 수직인 단위벡터.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-3-2-principal-directions.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["torus"]
u, v = ex["coords"]
Rs, rs = ex["params"]
RV, rV = 2.0, 0.8
vals = {Rs: RV, rs: rV}
Xe = ex["expr"].subs(vals)
X = sp.lambdify((u, v), list(Xe), "numpy")
Xu = sp.lambdify((u, v), list(Xe.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xe.diff(v)), "numpy")
W = sp.lambdify((u, v), dgsym.shape_operator(ex["expr"], u, v, ex["positive"]).subs(vals), "numpy")


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


fig = plt.figure(figsize=(6.4, 4.4))
ax = dgfig.axes3d(fig, elev=38, azim=-62)
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 73), np.linspace(0, 2 * np.pi, 145))
S = xmap(Ug, Vg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.3, grid=False, zorder=1)
view = dgfig.view_vector(ax)
tt = np.linspace(0, 2 * np.pi, 400)


def normal_out(U, V):
    return np.stack([np.cos(U) * np.cos(V), np.cos(U) * np.sin(V), np.sin(U) * np.ones_like(V)], axis=-1)


for k in range(12):
    c = xmap(tt, k * np.pi / 6 * np.ones_like(tt))
    vis = (normal_out(tt, k * np.pi / 6) @ view) > 0
    dgfig.curve3d(ax, c, role="#9A9A9A", lw=0.4, visible=vis, hidden=None, zorder=2)
for k in range(8):
    c = xmap(k * np.pi / 4 * np.ones_like(tt), tt)
    vis = (normal_out(k * np.pi / 4, tt) @ view) > 0
    dgfig.curve3d(ax, c, role="#9A9A9A", lw=0.4, visible=vis, hidden=None, zorder=2)
# 강조: 경선 v = -π/3과 위도원 u = π/4
cm = xmap(tt, -np.pi / 3 * np.ones_like(tt))
dgfig.curve3d(ax, cm, role="accent", lw=1.5, visible=(normal_out(tt, -np.pi / 3) @ view) > 0, hidden="dashed", zorder=4)
cp = xmap(np.pi / 4 * np.ones_like(tt), tt)
dgfig.curve3d(ax, cp, role="accent", lw=1.5, visible=(normal_out(np.pi / 4, tt) @ view) > 0, hidden="dashed", zorder=4)

pts = [(0.0, -np.pi / 2), (0.0, -np.pi / 6), (np.pi / 4, -np.pi / 3), (np.pi / 2, -2 * np.pi / 3), (np.pi / 4, 0.3),
       (-np.pi / 4, -np.pi / 2), (3 * np.pi / 4, -np.pi / 2.4), (np.pi / 2, -np.pi / 4)]
for (a, b) in pts:
    p = np.array(X(a, b), float)
    if normal_out(a, b) @ view <= 0.05:
        continue
    xu, xv = np.array(Xu(a, b), float), np.array(Xv(a, b), float)
    e1, e2 = xu / np.linalg.norm(xu), xv / np.linalg.norm(xv)
    Wm = np.array(W(a, b), float)
    assert abs(e1 @ e2) < 1e-12
    assert np.allclose(Wm[:, 0], [Wm[0, 0], 0]) and np.allclose(Wm[:, 1], [0, Wm[1, 1]])   # 대각 → 고유벡터
    dgfig.arrow3d(ax, p, 0.45 * e1, role="tangent", head=9, lw=1.3, zorder=10)
    dgfig.arrow3d(ax, p, 0.45 * e2, role="third", head=9, lw=1.3, zorder=10)
    ax.scatter(*p, s=6, color=C["main"], depthshade=False, zorder=11)
a, b = np.pi / 4, -np.pi / 3
p = np.array(X(a, b), float)
xu, xv = np.array(Xu(a, b), float), np.array(Xv(a, b), float)
ax.text(*(p + 0.55 * xu / np.linalg.norm(xu) + np.array([0, 0, 0.05])), r"$e_1$", color=C["tangent"], fontsize=11, zorder=12)
ax.text(*(p + 0.55 * xv / np.linalg.norm(xv) + np.array([0, 0, 0.04])), r"$e_2$", color=C["third"], fontsize=11, zorder=12)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.2)

dgfig.save(fig, __file__)
