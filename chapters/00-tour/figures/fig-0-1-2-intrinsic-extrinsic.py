"""Figure 0.1.2: intrinsic vs. extrinsic — 같은 종이, 다른 모양 (0.1절).

종이: 직사각형 0 ≤ x ≤ W = 3π/2 (r = 1), 0 ≤ y ≤ 2.
왼쪽: 평평한 종이(공간 안의 평면 조각). 오른쪽: 종이를 반지름 r = 1인 cylinder에 감은 것
      φ(x, y) = (r cos(x/r + c), r sin(x/r + c), y) (Section 7.2의 Example 7.2.9와 같은 map, 각을 c만큼 돌림).
주황: 선분 σ와 그 image(helix의 일부). 파랑: 삼각형 T와 그 image. 주홍: unit normal vector N.
자기검사: σ의 길이, T의 변의 길이와 꼭짓점의 각은 감아도 같다. 감은 종이 위의 점은 cylinder 위에 있다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-1-2-intrinsic-extrinsic.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

C = dgfig.COLORS
R = 1.0
W, HS = 1.5 * np.pi * R, 2.0
AZIM = -62
CSHIFT = np.deg2rad(AZIM) - W / (2 * R)     # 종이의 가운데가 보는 사람 쪽을 향하게


def flat(P):
    P = np.asarray(P, float)
    return np.stack([P[..., 0] - W / 2, -R * np.ones(P.shape[:-1]), P[..., 1]], axis=-1)


def rolled(P):
    P = np.asarray(P, float)
    a = P[..., 0] / R + CSHIFT
    return np.stack([R * np.cos(a), R * np.sin(a), P[..., 1]], axis=-1)


def normal_rolled(P):
    a = np.asarray(P, float)[..., 0] / R + CSHIFT
    return np.stack([np.cos(a), np.sin(a), 0 * a], axis=-1)


def poly_len(Q):
    return np.sum(np.linalg.norm(np.diff(Q, axis=0), axis=1))


def edge_pts(T, n=200):
    out = []
    for i in range(3):
        a, b = T[i], T[(i + 1) % 3]
        s = np.linspace(0, 1, n)[:, None]
        out.append(a + s * (b - a))
    return out


seg = np.stack([np.linspace(0.25, W - 0.25, 2001), np.linspace(0.2, HS - 0.25, 2001)], axis=1)
tri = np.array([[1.35, 0.35], [3.05, 0.55], [2.0, 1.55]])

# 자기검사 ---------------------------------------------------------------------------------
assert np.isclose(poly_len(rolled(seg)), poly_len(seg), rtol=1e-5)
assert np.isclose(poly_len(flat(seg)), poly_len(seg), rtol=1e-12)
for e in edge_pts(tri):
    assert np.isclose(poly_len(rolled(e)), poly_len(e), rtol=1e-4)
assert np.allclose(np.hypot(*rolled(seg)[:, :2].T), R)


def corner_angle(curves_at_vertex):
    a, b = (c / np.linalg.norm(c) for c in curves_at_vertex)
    return np.arccos(np.clip(a @ b, -1, 1))


for i in range(3):          # 꼭짓점의 각: 감은 뒤에는 변의 처음 접선으로 잰다
    p, q, s = tri[i], tri[(i + 1) % 3], tri[(i - 1) % 3]
    h = 1e-5
    flat_ang = corner_angle([q - p, s - p])
    rp = rolled(p)
    rol_ang = corner_angle([rolled(p + h * (q - p)) - rp, rolled(p + h * (s - p)) - rp])
    assert abs(flat_ang - rol_ang) < 1e-4

fig = plt.figure(figsize=(6.6, 3.1))
axes = [fig.add_axes([0.0, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False),
        fig.add_axes([0.5, 0.0, 0.5, 1.0], projection="3d", computed_zorder=False)]
for ax in axes:
    ax.view_init(elev=22, azim=AZIM)
    ax.set_proj_type("ortho")
    ax.set_axis_off()

Xg, Yg = np.meshgrid(np.linspace(0, W, 61), np.linspace(0, HS, 9))
grid_x = np.arange(0, W + 1e-9, W / 12)
grid_y = np.arange(0, HS + 1e-9, 0.5)
npts = [np.array([W / 2 - 1.0, 1.3]), np.array([W / 2 + 0.15, 1.3]), np.array([W / 2 + 1.15, 1.3])]

for k, (ax, F) in enumerate(zip(axes, (flat, rolled))):
    S = F(np.stack([Xg, Yg], axis=-1))
    dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.3, grid=False, zorder=1)
    view = dgfig.view_vector(ax)

    def vis(P2):
        if F is flat:
            return np.ones(len(P2), bool)
        return normal_rolled(P2) @ view > 0

    yy = np.linspace(0, HS, 50)
    xx = np.linspace(0, W, 200)
    for gx in grid_x:
        P2 = np.stack([gx * np.ones_like(yy), yy], axis=1)
        dgfig.curve3d(ax, F(P2), role="#9A9A9A", lw=0.4, visible=vis(P2), hidden=None, zorder=2)
    for gy in grid_y:
        P2 = np.stack([xx, gy * np.ones_like(xx)], axis=1)
        dgfig.curve3d(ax, F(P2), role="#9A9A9A", lw=0.4, visible=vis(P2), hidden="dashed", zorder=2)
    dgfig.curve3d(ax, F(seg), role="accent", lw=1.8, visible=vis(seg), hidden="dashed", zorder=5)
    E3 = np.concatenate([F(e) for e in edge_pts(tri, 60)])
    poly = Poly3DCollection([E3], facecolor=C["tangent"], alpha=0.25, edgecolor=C["tangent"],
                            linewidth=1.3, zorder=6)
    ax.add_collection3d(poly)
    for q in npts:
        base = F(q)
        nv = np.array([0.0, -1.0, 0.0]) if F is flat else normal_rolled(q)
        if True:
            dgfig.arrow3d(ax, base, nv, role="normal", scale=0.6, zorder=9)
    q0 = npts[2]
    lab = F(q0) + 0.6 * (np.array([0, -1, 0]) if F is flat else normal_rolled(q0))
    ax.text(*(lab + np.array([0.05, 0, 0.18])), r"$\mathbf{N}$", color=C["normal"], fontsize=12, zorder=12)
    ax.text(*(F(tri.mean(0)) + np.array([0.0, -0.05, -0.05])), r"$T$", fontsize=11, zorder=12,
            ha="center", va="center")
    tips = np.array([F(q) + 0.65 * (np.array([0.0, -1.0, 0.0]) if F is flat else normal_rolled(q)) for q in npts])
    dgfig.equal_aspect(ax, S.reshape(-1, 3), tips, zoom=1.2)

dgfig.save(fig, __file__)
