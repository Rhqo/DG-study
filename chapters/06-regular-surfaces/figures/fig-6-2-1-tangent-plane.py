"""그림 6.2.1: 구면의 접평면 (6.2절, 정의 6.2.1, 명제 6.2.4, 예 6.2.8).

반지름 r = 1인 구면, 기준 매개화 x (dgsym.EXAMPLES["sphere"]), p = x(θ₀, φ₀), (θ₀, φ₀) = (π/3, π/4).
- p를 지나는 좌표곡선: 위도원(θ = θ₀)과 경선(φ = φ₀) (검정)
- p를 지나는 대원 하나 α(t) = cos t · p + sin t · w₀ (주황), w₀ = (x_θ + 2x_φ/sinθ₀)/|…| 정규화
- 파랑: x_θ(q), x_φ(q) (실제 길이 1, sin θ₀),  주황: α'(0) = w₀ (길이 1)
- 주홍: x_θ × x_φ = r sin θ₀ · p (실제 길이 sin θ₀)
- 하늘색 평행사변형: p에 붙여 그린 아핀 접평면 p + T_pS

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-2-1-tangent-plane.py``
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


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


TH0, PH0 = np.pi / 3, np.pi / 4
p = np.array(X(TH0, PH0), float)
xt = np.array(Xt(TH0, PH0), float)
xp = np.array(Xp(TH0, PH0), float)
n = np.cross(xt, xp)
w0 = xt + 2 * xp / np.sin(TH0)
w0 /= np.linalg.norm(w0)

# 자기검사: 접벡터는 p에 수직, 외적 = sinθ₀·p, 대원은 구면 위에 있고 α'(0) = w₀
assert abs(p @ xt) < 1e-12 and abs(p @ xp) < 1e-12 and abs(p @ w0) < 1e-12
assert np.allclose(n, np.sin(TH0) * p)
tt = np.linspace(-np.pi, np.pi, 401)
alpha = np.cos(tt)[:, None] * p + np.sin(tt)[:, None] * w0
assert np.allclose(np.linalg.norm(alpha, axis=1), 1.0)
assert np.allclose((alpha[201] - alpha[199]) / (tt[201] - tt[199]), w0, atol=1e-3)

fig = plt.figure(figsize=(5.6, 4.6))
ax = dgfig.axes3d(fig, elev=22, azim=92)
tg, pg = np.meshgrid(np.linspace(0, np.pi, 49), np.linspace(0, 2 * np.pi, 97))
S = xmap(tg, pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False, zorder=1)


def draw(curve, role="main", lw=1.0, zorder=4):
    vis = dgfig.visible_mask(ax, curve, curve)
    dgfig.curve3d(ax, curve, role=role, lw=lw, visible=vis, zorder=zorder)


s = np.linspace(0, 2 * np.pi, 300)
draw(xmap(TH0 * np.ones_like(s), s), lw=1.0)
draw(xmap(np.linspace(0, np.pi, 200), PH0 * np.ones(200)), lw=1.0)
draw(alpha, role="accent", lw=1.3)
dgfig.tangent_plane(ax, p, xt, xp / np.linalg.norm(xp), size=0.6, zorder=6)
dgfig.point3d(ax, p, size=16, zorder=12)
dgfig.arrow3d(ax, p, xt, role="tangent", label=r"$\mathbf{x}_\theta$", zorder=14, label_offset=(0.0, 0.0, -0.08))
dgfig.arrow3d(ax, p, xp, role="tangent", label=r"$\mathbf{x}_\varphi$", zorder=14, label_offset=(0.0, 0.05, 0.05))
dgfig.arrow3d(ax, p, w0, role="accent", label=r"$\alpha'(0)$", zorder=14, label_offset=(0.0, 0.0, 0.08))
dgfig.arrow3d(ax, p, n, role="normal", label=r"$\mathbf{x}_\theta\times\mathbf{x}_\varphi$", zorder=14,
              label_offset=(0.0, 0.0, 0.1))
ax.text(*(p + np.array([0.15, 0.0, 0.1])), r"$p$", fontsize=12, zorder=15)
corner = p + 0.6 * (-xt + xp / np.linalg.norm(xp))
ax.text(*(corner + np.array([0.0, 0.05, 0.05])), r"$p + T_pS$", fontsize=12, color=C["tangent"], zorder=15)
tips = np.array([p + xt, p + xp, p + w0, p + 1.25 * n])
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), tips, zoom=1.2)

dgfig.save(fig, __file__)
