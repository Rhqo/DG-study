"""그림 8.1.1: 가우스 사상과 그 미분 — 안장면 위의 곡선과 그 가우스 상 (8.1절, 명제 8.1.4, 예 8.1.5).

안장면 x(u, v) = (u, v, u² − v²) (dgsym.EXAMPLES["saddle"]), 위쪽 N = N_x (dgsym.unit_normal).
왼쪽: |u|, |v| ≤ 0.7인 조각, 곡선 α(t) = x(t cos β, t sin β), β = π/6 (주황),
      t = 0, ±0.25, ±0.5에서의 N(α(t)) (주홍, 0.35배), p = α(0) = 0에서의 w = α'(0) (파랑, 0.5배).
오른쪽: 단위구면 S², N∘α (주황), 원점에서 N(α(t))로 가는 화살표 (주홍, 실제 길이),
        N(p) = e₃에서의 dN_p(w) = (N∘α)'(0) = (−2 cos β, 2 sin β, 0) (파랑, 0.25배).
자기검사: N은 단위벡터이고 x_u, x_v에 수직, (N∘α)'(0)을 유한차분으로 확인, dN_p(w) ⊥ N(p).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/08-gauss-map/figures/fig-8-1-1-gauss-map.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["saddle"]
u, v = ex["coords"]
Xs = ex["expr"]
Ns = dgsym.unit_normal(Xs, u, v)
X = sp.lambdify((u, v), list(Xs), "numpy")
Nf = sp.lambdify((u, v), list(Ns), "numpy")
Xu = sp.lambdify((u, v), list(Xs.diff(u)), "numpy")
Xv = sp.lambdify((u, v), list(Xs.diff(v)), "numpy")


def xmap(U, V):
    return np.stack([np.broadcast_to(c, np.shape(U)) for c in X(U, V)], axis=-1)


def nvec(a, b):
    return np.array(Nf(a, b), float)


beta = np.pi / 6
cb, sb = np.cos(beta), np.sin(beta)

# 자기검사
for (a, b) in [(0.0, 0.0), (0.3, -0.2), (-0.5, 0.4)]:
    n = nvec(a, b)
    assert abs(np.linalg.norm(n) - 1) < 1e-12
    assert abs(n @ np.array(Xu(a, b), float)) < 1e-12 and abs(n @ np.array(Xv(a, b), float)) < 1e-12
h = 1e-6
dNw_fd = (nvec(h * cb, h * sb) - nvec(-h * cb, -h * sb)) / (2 * h)
dNw = np.array([-2 * cb, 2 * sb, 0.0])
assert np.allclose(dNw_fd, dNw, atol=1e-6)
assert abs(dNw @ nvec(0, 0)) < 1e-12

fig = plt.figure(figsize=(6.6, 3.3))
axL = dgfig.axes3d(fig, pos=121, elev=24, azim=-58)
axR = dgfig.axes3d(fig, pos=122, elev=24, azim=-58)

# 왼쪽: 안장면
g = np.linspace(-0.7, 0.7, 41)
U, V = np.meshgrid(g, g)
S = xmap(U, V)
dgfig.surface(axL, S[..., 0], S[..., 1], S[..., 2], alpha=0.35, grid_every=5, zorder=1)
t = np.linspace(-0.62, 0.62, 200)
alpha = xmap(t * cb, t * sb)
dgfig.curve3d(axL, alpha, role="accent", lw=1.8, zorder=6)
ts = [-0.5, -0.25, 0.0, 0.25, 0.5]
for ti in ts:
    base = np.array(X(ti * cb, ti * sb), float)
    dgfig.arrow3d(axL, base, 0.35 * nvec(ti * cb, ti * sb), role="normal", head=8, lw=1.3, zorder=10)
w = np.array([cb, sb, 0.0])
dgfig.arrow3d(axL, [0, 0, 0], 0.5 * w, role="tangent", label=r"$w$", zorder=12, label_offset=(0.05, 0.0, -0.03))
dgfig.point3d(axL, [0, 0, 0], size=12, zorder=13)
axL.text(-0.12, 0.05, -0.12, r"$p$", fontsize=11, zorder=14)
axL.text(0.72, -0.72, 0.05, r"$S$", fontsize=12)
axL.text(*(np.array(X(0.5 * cb, 0.5 * sb), float) + 0.35 * nvec(0.5 * cb, 0.5 * sb) + np.array([0.04, 0, 0.05])),
         r"$\mathbf{N}$", color=C["normal"], fontsize=12)
dgfig.equal_aspect(axL, S[..., 0], S[..., 1], S[..., 2], zoom=1.12)

# 오른쪽: 단위구면
th, ph = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
Sx, Sy, Sz = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(axR, Sx, Sy, Sz, alpha=0.18, grid=False, zorder=1)
view = dgfig.view_vector(axR)
s = np.linspace(0, 2 * np.pi, 300)
eq = np.stack([np.cos(s), np.sin(s), 0 * s], axis=1)
dgfig.curve3d(axR, eq, role="aux", lw=0.6, visible=eq @ view > 0, hidden="dashed", zorder=2)
img = np.array([nvec(ti * cb, ti * sb) for ti in t])
dgfig.curve3d(axR, img, role="accent", lw=1.8, zorder=6)
for ti in ts:
    n = nvec(ti * cb, ti * sb)
    dgfig.arrow3d(axR, [0, 0, 0], n, role="normal", head=8, lw=1.0, zorder=8)
    axR.scatter(*n, s=10, color=C["main"], depthshade=False, zorder=9)
dgfig.arrow3d(axR, [0, 0, 1], 0.25 * dNw, role="tangent", label=r"$d\mathbf{N}_p(w)$", zorder=12,
              label_offset=(0.05, -0.05, 0.12), fontsize=11)
axR.scatter(0, 0, 0, s=8, color=C["main"], depthshade=False, zorder=9)
axR.text(0.06, 0.0, 1.1, r"$\mathbf{N}(p)$", fontsize=11, zorder=14)
axR.text(0.55, -0.9, -0.6, r"$S^2$", fontsize=12)
dgfig.equal_aspect(axR, np.array([[-1, -1, -1], [1, 1, 1]]) * 1.05, zoom=1.25)

fig.subplots_adjust(wspace=0.05, left=0.0, right=1.0, top=1.0, bottom=0.0)
dgfig.map_arrow(fig, axL, axR, r"$\mathbf{N}$", xy_from=(0.86, 0.72), xy_to=(0.18, 0.72), rad=-0.3)

dgfig.save(fig, __file__)
