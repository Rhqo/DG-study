"""그림 7.5.2: 구면의 방향 — 바깥쪽 단위법벡터장과 양의 기저 (7.5절, 정리 7.5.5, 예 7.5.7).

구면 x(θ, φ) (dgsym.EXAMPLES["sphere"]), r = 1. N = x/r (바깥쪽, dgsym.unit_normal).
보이는 쪽의 격자점 θ = kπ/6, φ = jπ/6에서 N(주홍, 0.3배).
점 p = x(π/3, π/4)에서 접평면(하늘색), x_θ, x_φ(파랑, 0.45배)와 x_θ에서 x_φ로 도는 호(검정 화살표).
det(x_θ, x_φ, N) = |x_θ × x_φ| > 0이므로 (x_θ, x_φ)는 N이 정하는 방향에서 양의 기저다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-5-2-sphere-orientation.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
Xe = ex["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xe), "numpy")
Xt = sp.lambdify((th, ph), list(Xe.diff(th)), "numpy")
Xp = sp.lambdify((th, ph), list(Xe.diff(ph)), "numpy")
Nsym = dgsym.unit_normal(ex["expr"], th, ph, ex["positive"]).subs(rs, 1)
assert sp.simplify(Nsym - ex["expected"]["normal"]) == sp.zeros(3, 1)       # 바깥쪽 N = x/r
Nf = sp.lambdify((th, ph), list(Nsym), "numpy")


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


T0, P0 = np.pi / 3, np.pi / 4
p = np.array(X(T0, P0), float)
a, b = np.array(Xt(T0, P0), float), np.array(Xp(T0, P0), float)
n = np.array(Nf(T0, P0), float)
assert np.linalg.det(np.stack([a, b, n], axis=1)) > 0
assert np.isclose(np.linalg.det(np.stack([a, b, n], axis=1)), np.linalg.norm(np.cross(a, b)))

fig = plt.figure(figsize=(5.0, 4.2))
ax = dgfig.axes3d(fig, elev=-6, azim=5)
Tg, Pg = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(0, 2 * np.pi, 121))
S = xmap(Tg, Pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False, zorder=1)
view = dgfig.view_vector(ax)
s = np.linspace(0, 2 * np.pi, 300)
for k in range(1, 6):
    c = xmap(k * np.pi / 6 * np.ones_like(s), s)
    dgfig.curve3d(ax, c, role="#9A9A9A", lw=0.4, visible=c @ view > 0, hidden=None, zorder=2)
s2 = np.linspace(0, np.pi, 200)
for j in range(12):
    c = xmap(s2, j * np.pi / 6 * np.ones_like(s2))
    dgfig.curve3d(ax, c, role="#9A9A9A", lw=0.4, visible=c @ view > 0, hidden=None, zorder=2)
for k in range(1, 6):
    for j in range(12):
        q = np.array(X(k * np.pi / 6, j * np.pi / 6), float)
        if q @ view > 0.25 and np.linalg.norm(q - p) > 0.3:
            dgfig.arrow3d(ax, q, 0.3 * np.array(Nf(k * np.pi / 6, j * np.pi / 6), float), role="normal", head=7, lw=1.0, zorder=10)
dgfig.tangent_plane(ax, p, a, b, size=0.42, zorder=11)
dgfig.arrow3d(ax, p, 0.45 * a, role="tangent", label=r"$\mathbf{x}_\theta$", zorder=14, label_offset=(0, 0, -0.03))
dgfig.arrow3d(ax, p, 0.45 * b, role="tangent", label=r"$\mathbf{x}_\varphi$", zorder=14, label_offset=(0.0, 0.03, 0.02))
dgfig.arrow3d(ax, p, 0.5 * n, role="normal", label=r"$\mathbf{N}$", zorder=14, lw=1.8, label_offset=(0, 0, 0.04))
e1 = a / np.linalg.norm(a)
e2 = b / np.linalg.norm(b)
ang = np.linspace(0.15, np.pi / 2 - 0.15, 40)
arc = p + 0.2 * (np.outer(np.cos(ang), e1) + np.outer(np.sin(ang), e2))
ax.plot(*arc.T, color=C["main"], lw=1.0, zorder=15)
dgfig.arrow3d(ax, arc[-3], arc[-1] - arc[-3], role="main", head=8, lw=1.0, zorder=15)
dgfig.point3d(ax, p, size=12, zorder=16)
ax.text(*(p + np.array([0.05, -0.12, -0.08])), r"$p$", fontsize=11, zorder=16)
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]) * 1.15, zoom=1.25)

dgfig.save(fig, __file__)
