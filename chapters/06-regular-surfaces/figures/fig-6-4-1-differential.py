"""그림 6.4.1: 곡면 사이 사상의 미분 (6.4절, 정의 6.4.1, 예 6.4.6).

구면의 경도 φ와 섞이지 않도록 이 그림의 사상은 ψ로 부른다(GUIDELINES §6.7).
ψ = A|_{S²}: S²(1) → E, A = diag(a, b, c) = diag(1.5, 1.0, 0.7), E = A(S²(1))는 타원면.
p = x(π/3, π/4) (구면의 기준 매개화, dgsym.EXAMPLES["sphere"]).
α(t) = x(π/3 + 0.5t, π/4 + 1.2t): p를 지나는 곡선, w = α'(0) = 0.5 x_θ + 1.2 x_φ (실제 길이, 주황).
ψ∘α (주황)와 그 속도 dψ_p(w) = A w (주황). 접평면은 p, ψ(p)에 붙여 그린 하늘색 평행사변형.
두 패널 사이의 곡선 화살표 ψ.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-4-1-differential.py``
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
(r,) = ex["params"]
Xs = ex["expr"].subs(r, 1)
X = sp.lambdify((th, ph), list(Xs), "numpy")
Xt = sp.lambdify((th, ph), list(Xs.diff(th)), "numpy")
Xp = sp.lambdify((th, ph), list(Xs.diff(ph)), "numpy")
A = np.diag([1.5, 1.0, 0.7])


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


TH0, PH0 = np.pi / 3, np.pi / 4
p = np.array(X(TH0, PH0), float)
xt, xp = np.array(Xt(TH0, PH0), float), np.array(Xp(TH0, PH0), float)
w0 = 0.5 * xt + 1.2 * xp
tt = np.linspace(-0.9, 0.9, 241)
alpha = xmap(TH0 + 0.5 * tt, PH0 + 1.2 * tt)
beta = alpha @ A.T
dphi_w = A @ w0

# 자기검사: α는 구면 위, φ∘α는 타원면 위, 속도는 수치 미분과 일치, A w는 T_{φ(p)}E에 있다
assert np.allclose(np.linalg.norm(alpha, axis=1), 1)
Ainv2 = np.diag(1 / np.diag(A) ** 2)
assert np.allclose(np.einsum("ij,jk,ik->i", beta, Ainv2, beta), 1)
i0 = 120
assert abs(tt[i0]) < 1e-12
assert np.allclose((alpha[i0 + 1] - alpha[i0 - 1]) / (tt[i0 + 1] - tt[i0 - 1]), w0, atol=1e-3)
assert np.allclose((beta[i0 + 1] - beta[i0 - 1]) / (tt[i0 + 1] - tt[i0 - 1]), dphi_w, atol=1e-3)
n_E = Ainv2 @ (A @ p)                       # 기울기 방향 (x/a², y/b², z/c²)
assert abs(n_E @ dphi_w) < 1e-12

fig = plt.figure(figsize=(6.8, 3.3))
tg, pg = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
S = xmap(tg, pg)


def panel(ax, pts, curve, base, vec, e1, e2, lab_p, lab_v, size):
    dgfig.surface(ax, pts[..., 0], pts[..., 1], pts[..., 2], alpha=0.22, grid=False, zorder=1)
    for k in range(1, 6):
        c = xmap(k * np.pi / 6 * np.ones(200), np.linspace(0, 2 * np.pi, 200))
        c = c @ (A.T if pts is not S else np.eye(3))
        nrm = c @ (Ainv2 if pts is not S else np.eye(3))
        dgfig.curve3d(ax, c, role="#8C8C8C", lw=0.4, visible=dgfig.visible_mask(ax, c, nrm), zorder=2)
    nrm = curve @ (Ainv2 if pts is not S else np.eye(3))
    dgfig.curve3d(ax, curve, role="accent", lw=1.4, visible=dgfig.visible_mask(ax, curve, nrm), zorder=5)
    dgfig.tangent_plane(ax, base, e1, e2, size=size, zorder=6)
    dgfig.point3d(ax, base, size=14, zorder=12)
    dgfig.arrow3d(ax, base, vec, role="accent", label=lab_v, zorder=14, label_offset=(0.0, 0.25, 0.0))
    ax.text(*(base + np.array([0.0, -0.1, 0.12])), lab_p, fontsize=12, zorder=15, ha="right")


ax1 = fig.add_axes([0.0, 0.0, 0.44, 1.0], projection="3d", computed_zorder=False)
ax1.view_init(elev=20, azim=58)
ax1.set_axis_off()
panel(ax1, S, alpha, p, w0, xt, xp, r"$p$", r"$w$", 0.4)
dgfig.equal_aspect(ax1, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.35)
ax1.text2D(0.5, 0.02, r"$S^2(1)$", transform=ax1.transAxes, fontsize=12, ha="center")

ax2 = fig.add_axes([0.46, 0.0, 0.54, 1.0], projection="3d", computed_zorder=False)
ax2.view_init(elev=20, azim=58)
ax2.set_axis_off()
SE = S @ A.T
panel(ax2, SE, beta, A @ p, dphi_w, A @ xt, A @ xp, r"$\psi(p)$", r"$d\psi_p(w)$", 0.45)
dgfig.equal_aspect(ax2, np.array([[-1.5, -1.5, -1.0], [1.5, 1.5, 1.0]]), zoom=1.3)
ax2.text2D(0.5, 0.02, r"$E$", transform=ax2.transAxes, fontsize=12, ha="center")

dgfig.map_arrow(fig, ax1, ax2, r"$\psi$", xy_from=(0.78, 0.8), xy_to=(0.22, 0.8), rad=-0.3)

dgfig.save(fig, __file__)
