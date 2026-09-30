"""그림 6.1.4: 정칙곡면은 국소적으로 그래프다 (6.1절, 명제 6.1.8).

반지름 r = 1인 구면의 기준 매개화 x (dgsym.EXAMPLES["sphere"]), q = (θ₀, φ₀) = (π/3, π/4).
∂(x, y)/∂(θ, φ) = r² sin θ cos θ ≠ 0이므로 P_xy(x, y, z) = (x, y)를 쓴다.
왼쪽: U의 일부와 q를 포함하는 작은 열린 직사각형 U₀ = (θ₀ ± 0.3) × (φ₀ ± 0.45).
오른쪽: 구면과 조각 x(U₀)(하늘색), 그 사영 Ω = P_xy(x(U₀))를 보기 쉽게 평면 z = −1.35에 그린 것,
        모서리에서 내린 수직 점선(사영 P_xy).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-1-4-local-graph.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(r,) = ex["params"]
Xs = ex["expr"].subs(r, 1)
X = sp.lambdify((th, ph), list(Xs), "numpy")
jac_xy = sp.simplify(Xs[:2, :].jacobian([th, ph]).det())


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


TH0, PH0 = np.pi / 3, np.pi / 4
DT, DP = 0.3, 0.45
ZP = -1.35  # 사영을 그릴 평면의 높이(보기 위해 옮김)

# 자기검사: 야코비 행렬식이 sin θ cos θ이고 U₀에서 0이 아니다; π∘x는 U₀에서 단사(표본)
assert sp.simplify(jac_xy - sp.sin(th) * sp.cos(th)) == 0
tt, pp = np.meshgrid(np.linspace(TH0 - DT, TH0 + DT, 25), np.linspace(PH0 - DP, PH0 + DP, 25))
assert np.all(np.sin(tt) * np.cos(tt) > 0.2)
P2 = xmap(tt, pp)[..., :2].reshape(-1, 2)
dists = np.linalg.norm(P2[:, None] - P2[None, :], axis=-1) + np.eye(len(P2))
assert dists.min() > 1e-4

fig = plt.figure(figsize=(6.6, 3.5))

# 왼쪽: U와 U₀ ---------------------------------------------------------------------------
axU = fig.add_axes([0.02, 0.08, 0.2, 0.84])
axU.fill([0, np.pi, np.pi, 0], [0, 0, 2 * np.pi, 2 * np.pi], color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
axU.fill([TH0 - DT, TH0 + DT, TH0 + DT, TH0 - DT], [PH0 - DP, PH0 - DP, PH0 + DP, PH0 + DP],
         color=C["region"], alpha=0.7, lw=0)
axU.plot([TH0 - DT, TH0 + DT, TH0 + DT, TH0 - DT, TH0 - DT], [PH0 - DP, PH0 - DP, PH0 + DP, PH0 + DP, PH0 - DP],
         color=C["tangent"], lw=1.0)
axU.plot([TH0], [PH0], "o", color=C["main"], ms=4)
axU.text(TH0 + 0.07, PH0 + 0.07, r"$q$", fontsize=12)
axU.text(TH0, PH0 + DP + 0.12, r"$U_0$", fontsize=12, ha="center", color=C["tangent"])
axU.text(np.pi / 2, -0.28, r"$\theta$", fontsize=12, ha="center", va="top")
axU.text(-0.18, np.pi, r"$\varphi$", fontsize=12, ha="right", va="center")
axU.text(np.pi / 2, 2 * np.pi + 0.15, r"$U$", fontsize=12, ha="center", va="bottom")
dgfig.schematic_axes(axU, (-0.6, np.pi + 0.1), (-0.8, 2 * np.pi + 0.7))

# 오른쪽: 구면, 조각, 사영 -------------------------------------------------------------------
ax = fig.add_axes([0.26, 0.0, 0.74, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=22, azim=20)
ax.set_axis_off()
tg, pg = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
S = xmap(tg, pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.16, grid=False, zorder=1)
Pt = xmap(tt, pp)
ax.plot_surface(Pt[..., 0], Pt[..., 1], Pt[..., 2], color=C["region"], alpha=0.6, lw=0, zorder=3,
                shade=False, rasterized=True)
# 조각의 테두리와 그 사영
edge = np.concatenate([
    xmap(np.linspace(TH0 - DT, TH0 + DT, 40), np.full(40, PH0 - DP)),
    xmap(np.full(40, TH0 + DT), np.linspace(PH0 - DP, PH0 + DP, 40)),
    xmap(np.linspace(TH0 + DT, TH0 - DT, 40), np.full(40, PH0 + DP)),
    xmap(np.full(40, TH0 - DT), np.linspace(PH0 + DP, PH0 - DP, 40)),
])
ax.plot(*edge.T, color=C["tangent"], lw=1.0, zorder=4)
shadow = edge.copy()
shadow[:, 2] = ZP
# 사영 평면(옮긴 xy 평면)의 사각형과 D
sq = np.array([[-1.0, -1.0, ZP], [1.0, -1.0, ZP], [1.0, 1.0, ZP], [-1.0, 1.0, ZP]])
ax.add_collection3d(Poly3DCollection([sq], facecolor="#F2F2F2", edgecolor=C["aux"], lw=0.6, alpha=0.6, zorder=0))
ax.add_collection3d(Poly3DCollection([shadow], facecolor=C["region"], edgecolor=C["tangent"], lw=1.0,
                                     alpha=0.55, zorder=2))
for k in (0, 40, 80, 120):
    ax.plot([edge[k, 0]] * 2, [edge[k, 1]] * 2, [edge[k, 2], ZP], color=C["aux"], lw=0.8, ls=(0, (3, 2)), zorder=2)
p = xmap(TH0, PH0)
dgfig.point3d(ax, p, size=14, zorder=10)
ax.text(*(p + np.array([0.0, 0.08, 0.1])), r"$p$", fontsize=12, zorder=11)
dgfig.point3d(ax, [p[0], p[1], ZP], size=10, zorder=10)
ax.text(p[0] + 0.05, p[1] + 0.35, ZP, r"$\Omega$", fontsize=12, color=C["tangent"], zorder=11)
mid = edge[80]
ax.text(mid[0], mid[1] + 0.12, (mid[2] + ZP) / 2, r"$P_{xy}$", fontsize=12, zorder=11)
ax.text(*(xmap(TH0 - DT - 0.05, PH0 + DP) + np.array([-0.05, 0.15, 0.12])), r"$\mathbf{x}(U_0)$",
        fontsize=12, color=C["tangent"], zorder=11)
dgfig.equal_aspect(ax, np.array([[-1.0, -1.0, ZP], [1.0, 1.0, 1.0]]), zoom=1.05)

dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(0.9, 0.62), xy_to=(0.40, 0.76), rad=-0.3)

dgfig.save(fig, __file__)
