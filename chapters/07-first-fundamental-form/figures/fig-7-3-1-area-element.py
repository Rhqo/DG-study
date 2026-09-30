"""그림 7.3.1: 좌표 칸의 상과 접평면의 평행사변형 (7.3절 동기, 정의 7.3.2).

x(θ, φ) = (r sin θ cos φ, r sin θ sin φ, r cos θ) (dgsym.EXAMPLES["sphere"]), r = 1.
왼쪽: U의 좌표 칸 Q = [θ₀, θ₀ + Δ] × [φ₀, φ₀ + Δ], θ₀ = π/4, φ₀ = π/6, Δ = 0.5.
오른쪽: 구면의 일부와 그 위의 칸 x(Q)(하늘색),
        꼭짓점 p = x(θ₀, φ₀)에 붙인 평행사변형 {p + s Δ x_θ + t Δ x_φ : 0 ≤ s, t ≤ 1}(주황 테두리),
        두 변 Δ x_θ, Δ x_φ(파랑). 평행사변형의 넓이 |x_θ × x_φ| Δ² = r² sin θ₀ Δ²는 x(Q)의 넓이의 근사다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/07-first-fundamental-form/figures/fig-7-3-1-area-element.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(rs,) = ex["params"]
Xe = ex["expr"].subs(rs, 1)
X = sp.lambdify((th, ph), list(Xe), "numpy")
Xt = sp.lambdify((th, ph), list(Xe.diff(th)), "numpy")
Xp = sp.lambdify((th, ph), list(Xe.diff(ph)), "numpy")
Es, Fs, Gs = dgsym.first_ff(ex["expr"], th, ph, ex["positive"])
dA = sp.lambdify(th, sp.sqrt(Es * Gs - Fs ** 2).subs(rs, 1), "numpy")


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


T0, P0, D = np.pi / 4, np.pi / 6, 0.5
p = np.array(X(T0, P0), float)
a = D * np.array(Xt(T0, P0), float)
b = D * np.array(Xp(T0, P0), float)
par_area = np.linalg.norm(np.cross(a, b))
# 칸의 실제 넓이 ∫∫ sin θ dθ dφ = Δ (cos θ₀ − cos(θ₀ + Δ))
cell_area = D * (np.cos(T0) - np.cos(T0 + D))
assert np.isclose(par_area, dA(T0) * D ** 2)
assert abs(par_area / cell_area - 1) < 0.3            # 근사 (Δ가 작아지면 비는 1로 간다)
for d in (0.1, 0.01, 0.001):
    ratio = (dA(T0) * d ** 2) / (d * (np.cos(T0) - np.cos(T0 + d)))
    assert abs(ratio - 1) < 1.2 * d                   # 오차는 Δ에 비례

fig = plt.figure(figsize=(6.4, 3.3))

# 왼쪽: U --------------------------------------------------------------------------------
axU = fig.add_axes([0.03, 0.12, 0.28, 0.76])
tl, tr, pl, pr = 0.35, 1.45, 0.05, 1.05
axU.fill([tl, tr, tr, tl], [pl, pl, pr, pr], color=C["region"], alpha=0.15, lw=0)
for k in np.arange(0.35, 1.46, 0.35 / 2):
    axU.plot([k, k], [pl, pr], color="#A0A0A0", lw=0.35)
for k in np.arange(0.0, 1.06, 0.35 / 2):
    if k >= pl:
        axU.plot([tl, tr], [k, k], color="#A0A0A0", lw=0.35)
axU.fill([T0, T0 + D, T0 + D, T0], [P0, P0, P0 + D, P0 + D], color=C["region"], alpha=0.7, lw=0)
axU.plot([T0, T0 + D, T0 + D, T0, T0], [P0, P0, P0 + D, P0 + D, P0], color=C["tangent"], lw=1.0)
axU.plot([T0], [P0], "o", color=C["main"], ms=3.5)
axU.text(T0 - 0.03, P0 - 0.04, r"$q$", fontsize=11, ha="right", va="top")
axU.text(T0 + D / 2, P0 + D / 2, r"$Q$", fontsize=12, ha="center", va="center")
axU.annotate("", xy=(T0 + D, P0 - 0.09), xytext=(T0, P0 - 0.09),
             arrowprops=dict(arrowstyle="<->", color=C["main"], lw=0.8, shrinkA=0, shrinkB=0))
axU.text(T0 + D / 2, P0 - 0.12, r"$\Delta$", fontsize=10, ha="center", va="top")
axU.text((tl + tr) / 2, pl - 0.18, r"$\theta$", fontsize=12, ha="center", va="top")
axU.text(tl - 0.12, (pl + pr) / 2, r"$\varphi$", fontsize=12, ha="right", va="center")
dgfig.schematic_axes(axU, (tl - 0.3, tr + 0.05), (pl - 0.35, pr + 0.08))

# 오른쪽: 구면의 일부 -----------------------------------------------------------------------
ax = fig.add_axes([0.32, -0.02, 0.68, 1.04], projection="3d", computed_zorder=False)
ax.view_init(elev=28, azim=52)
ax.set_axis_off()
Tg, Pg = np.meshgrid(np.linspace(0.2, 1.5, 40), np.linspace(-0.2, 1.3, 46))
S = xmap(Tg, Pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.25, grid=False, zorder=1)
s1 = np.linspace(0.2, 1.5, 80)
s2 = np.linspace(-0.2, 1.3, 80)
for k in np.arange(0.35, 1.46, 0.35 / 2):
    ax.plot(*xmap(k * np.ones_like(s2), s2).T, color="#9A9A9A", lw=0.35, zorder=2)
for k in np.arange(0.0, 1.21, 0.35 / 2):
    ax.plot(*xmap(s1, k * np.ones_like(s1)).T, color="#9A9A9A", lw=0.35, zorder=2)
tt = np.linspace(T0, T0 + D, 30)
pp = np.linspace(P0, P0 + D, 30)
edge = np.concatenate([xmap(tt, P0 * np.ones_like(tt)), xmap((T0 + D) * np.ones_like(pp), pp),
                       xmap(tt[::-1], (P0 + D) * np.ones_like(tt)), xmap(T0 * np.ones_like(pp), pp[::-1])])
ax.add_collection3d(Poly3DCollection([edge], facecolor=C["region"], alpha=0.55, edgecolor=C["tangent"],
                                     linewidth=0.9, zorder=5))
par = np.array([p, p + a, p + a + b, p + b])
ax.add_collection3d(Poly3DCollection([par], facecolor=C["accent"], alpha=0.22, edgecolor="none", zorder=7))
ax.plot(*np.vstack([par, par[:1]]).T, color=C["accent"], lw=1.4, ls=(0, (4, 2)), zorder=13)
dgfig.point3d(ax, p, size=12, zorder=12)
dgfig.arrow3d(ax, p, a, role="tangent", zorder=14, head=9)
dgfig.arrow3d(ax, p, b, role="tangent", zorder=14, head=9)
ax.text(*(p + a + np.array([0.13, -0.12, -0.03])), r"$\Delta\,\mathbf{x}_\theta$", color=C["tangent"], fontsize=11, zorder=15)
ax.text(*(p + b + np.array([-0.02, 0.03, 0.04])), r"$\Delta\,\mathbf{x}_\varphi$", color=C["tangent"], fontsize=11, zorder=15)
ax.text(*(p + np.array([-0.03, -0.05, 0.03])), r"$p$", fontsize=11, zorder=15, ha="right")
ax.text(*(np.array(X(T0 + D * 0.55, P0 + D * 0.5), float) * 1.03), r"$\mathbf{x}(Q)$", fontsize=11, zorder=15,
        ha="center", va="center")
dgfig.equal_aspect(ax, S.reshape(-1, 3), zoom=1.25)
dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(0.95, 0.8), xy_to=(0.25, 0.8), rad=-0.3)

dgfig.save(fig, __file__)
