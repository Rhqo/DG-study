"""그림 7.5.1: 뫼비우스 띠에서 단위법벡터를 한 바퀴 옮기면 반대가 된다 (7.5절, 예 7.5.9, 명제 7.5.10).

x(u, v) = ((1 + v cos(u/2)) cos u, (1 + v cos(u/2)) sin u, v sin(u/2)), u ∈ ℝ, |v| < 1/2 (예 7.5.9).
중심원 c(u) = x(u, 0)(주황)을 따라 u = kπ/4 (k = 0, …, 8)에서 N_x(u, 0) = (x_u × x_v)/|x_u × x_v|(주홍, 0.35배).
c(0) = c(2π)인 점 p에서 u = 0의 N_x(0, 0) = (0, 0, −1)과 u = 2π의 N_x(2π, 0) = (0, 0, 1)은 반대 방향이다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-5-1-moebius.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

u, v = sp.symbols("u v", real=True)
Xe = sp.Matrix([(1 + v * sp.cos(u / 2)) * sp.cos(u), (1 + v * sp.cos(u / 2)) * sp.sin(u), v * sp.sin(u / 2)])
n_sym = Xe.diff(u).cross(Xe.diff(v))                  # x_u × x_v (정규화는 수치로: 기호 정규화는 u = 0에서 0/0 꼴이 된다)
X = sp.lambdify((u, v), list(Xe), "numpy")
nf = sp.lambdify((u, v), list(n_sym), "numpy")
EFG = [sp.lambdify((u, v), c, "numpy") for c in dgsym.first_ff(Xe, u, v)]


def Nf(uu, vv):
    n = np.array(nf(uu, vv), float)
    return n / np.linalg.norm(n)


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


# 자기검사: x(u + 2π, v) = x(u, −v), N_x(2π, 0) = −N_x(0, 0), N은 x_u, x_v에 수직인 단위벡터
assert np.allclose(xmap(np.array(0.7 + 2 * np.pi), np.array(0.2)), xmap(np.array(0.7), np.array(-0.2)))
n0, n1 = np.array(Nf(0.0, 0.0), float), np.array(Nf(2 * np.pi, 0.0), float)
assert np.allclose(n0, [0, 0, -1]) and np.allclose(n1, -n0)
Xu = sp.lambdify((u, v), list(Xe.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xe.diff(v)), "numpy")
# |x_u × x_v|² = EG − F² = (1 + v cos(u/2))² + v²/4 > 0 (dgsym.first_ff)
for uu, vv in ((0.3, 0.4), (2.0, -0.4), (5.0, 0.1)):
    assert np.isclose(np.linalg.norm(nf(uu, vv)) ** 2, EFG[0](uu, vv) * EFG[2](uu, vv) - EFG[1](uu, vv) ** 2)
    assert np.isclose(EFG[0](uu, vv) * EFG[2](uu, vv) - EFG[1](uu, vv) ** 2, (1 + vv * np.cos(uu / 2)) ** 2 + vv ** 2 / 4)
for uu in np.linspace(0, 2 * np.pi, 9):
    nn = np.array(Nf(uu, 0.0), float)
    assert np.isclose(nn @ nn, 1) and abs(nn @ np.array(Xu(uu, 0.0), float)) < 1e-12 and abs(nn @ np.array(Xv(uu, 0.0), float)) < 1e-12

fig = plt.figure(figsize=(6.0, 3.8))
ax = dgfig.axes3d(fig, elev=32, azim=-62)
Ug, Vg = np.meshgrid(np.linspace(0, 2 * np.pi, 161), np.linspace(-0.45, 0.45, 13))
S = xmap(Ug, Vg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.35, grid=False, zorder=1, shade=True)
for vv in (-0.45, 0.45):
    e = xmap(np.linspace(0, 2 * np.pi, 400), vv * np.ones(400))
    ax.plot(*e.T, color=C["main"], lw=1.0, zorder=3)
for uu in np.linspace(0, 2 * np.pi, 17)[:-1]:
    seg = xmap(uu * np.ones(20), np.linspace(-0.45, 0.45, 20))
    ax.plot(*seg.T, color="#8A8A8A", lw=0.4, zorder=2)
core = xmap(np.linspace(0, 2 * np.pi, 400), np.zeros(400))
ax.plot(*core.T, color=C["accent"], lw=1.6, zorder=5)
for k in range(9):
    uu = k * np.pi / 4
    p = np.array(X(uu, 0.0), float)
    nn = np.array(Nf(uu, 0.0), float)
    if k == 8:
        dgfig.arrow3d(ax, p, 0.35 * nn, role="normal", zorder=14, head=10, lw=1.9)
        ax.text(*(p + 0.35 * nn + np.array([0.0, 0.0, 0.06])), r"$\mathbf{N}_{\mathbf{x}}(2\pi, 0)$", color=C["normal"],
                fontsize=10, zorder=15, ha="center", va="bottom")
    elif k == 0:
        dgfig.arrow3d(ax, p, 0.35 * nn, role="normal", zorder=14, head=10, lw=1.9)
        ax.text(*(p + 0.35 * nn + np.array([0.0, 0.0, -0.06])), r"$\mathbf{N}_{\mathbf{x}}(0, 0)$", color=C["normal"],
                fontsize=10, zorder=15, ha="center", va="top")
    else:
        dgfig.arrow3d(ax, p, 0.35 * nn, role="normal", zorder=14, head=8, lw=1.1)
p0 = np.array(X(0.0, 0.0), float)
dgfig.point3d(ax, p0, size=14, zorder=16)
ax.text(*(p0 + np.array([0.12, -0.1, 0.0])), r"$p$", fontsize=11, zorder=16)
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.3)

dgfig.save(fig, __file__)
