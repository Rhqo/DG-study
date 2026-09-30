"""그림 6.1.1: 구면의 기준 매개화 x: U → S²(r) (6.1절, 예 6.1.7).

왼쪽: U = (0, π) × (0, 2π)를 하늘색으로 채우고 좌표격자 θ = kπ/6, φ = kπ/6를 그린다.
      강조한 좌표선 θ = θ₀(파랑)과 φ = φ₀(청록), 점 q = (θ₀, φ₀). 변 φ = 0, φ = 2π는 주황 점선.
오른쪽: 반지름 r = 1인 구면과 격자의 상(위도원과 경선), 점 p = x(q),
      매개화가 빠뜨리는 반원 C = {y = 0, x ≥ 0} ∩ S²(주황).
매개화는 dgsym.EXAMPLES["sphere"]에서 가져온다. θ₀ = π/3, φ₀ = π/4.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-1-1-sphere-parametrization.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(r,) = ex["params"]
X = sp.lambdify((th, ph), list(ex["expr"].subs(r, 1)), "numpy")


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


TH0, PH0 = np.pi / 3, np.pi / 4
p = xmap(TH0, PH0)

# 자기검사: 격자의 상은 구면 위에 있고, x(q)는 식대로이며, C의 점은 y = 0, x ≥ 0이다
tt, pp = np.meshgrid(np.linspace(0.01, np.pi - 0.01, 40), np.linspace(0.01, 2 * np.pi - 0.01, 40))
assert np.allclose(np.linalg.norm(xmap(tt, pp), axis=-1), 1.0)
assert np.allclose(p, [np.sin(TH0) * np.cos(PH0), np.sin(TH0) * np.sin(PH0), np.cos(TH0)])
Cpts = xmap(np.linspace(0, np.pi, 50), 0.0 * np.ones(50))
assert np.allclose(Cpts[:, 1], 0) and np.all(Cpts[:, 0] >= -1e-12)
assert np.allclose(xmap(np.linspace(0, np.pi, 50), 2 * np.pi * np.ones(50)), Cpts)

fig = plt.figure(figsize=(6.6, 3.4))

# (왼쪽) 매개변수 영역 U ------------------------------------------------------------
axU = fig.add_axes([0.03, 0.10, 0.25, 0.80])
axU.fill([0, np.pi, np.pi, 0], [0, 0, 2 * np.pi, 2 * np.pi], color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
axU.plot([0, np.pi, np.pi, 0, 0], [0, 0, 2 * np.pi, 2 * np.pi, 0], color=C["aux"], lw=LW["aux"], ls=(0, (4, 3)))
for k in range(1, 6):
    axU.plot([k * np.pi / 6] * 2, [0, 2 * np.pi], color="#8C8C8C", lw=0.4)
for k in range(1, 12):
    axU.plot([0, np.pi], [k * np.pi / 6] * 2, color="#8C8C8C", lw=0.4)
axU.plot([0, np.pi], [0, 0], color=C["accent"], lw=1.6, ls=(0, (4, 2.5)))
axU.plot([0, np.pi], [2 * np.pi, 2 * np.pi], color=C["accent"], lw=1.6, ls=(0, (4, 2.5)))
axU.plot([TH0, TH0], [0, 2 * np.pi], color=C["tangent"], lw=1.6)
axU.plot([0, np.pi], [PH0, PH0], color=C["third"], lw=1.6)
axU.plot([TH0], [PH0], "o", color=C["main"], ms=4, zorder=5)
axU.text(TH0 + 0.12, PH0 + 0.28, r"$q$", fontsize=12)
axU.text(np.pi / 2, 2 * np.pi + 0.35, r"$U$", fontsize=13, ha="center")
axU.text(np.pi / 2, -0.55, r"$\theta$", fontsize=12, ha="center")
axU.text(-0.55, np.pi, r"$\varphi$", fontsize=12, va="center")
axU.text(0, -0.2, r"$0$", fontsize=10, ha="center", va="top")
axU.text(np.pi, -0.2, r"$\pi$", fontsize=10, ha="center", va="top")
axU.text(-0.12, 2 * np.pi, r"$2\pi$", fontsize=10, ha="right", va="center")
dgfig.schematic_axes(axU, (-0.9, np.pi + 0.4), (-0.9, 2 * np.pi + 0.8))

# (오른쪽) 구면 ------------------------------------------------------------------------
ax = fig.add_axes([0.34, 0.0, 0.66, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=18, azim=28)
ax.set_axis_off()
tg, pg = np.meshgrid(np.linspace(0, np.pi, 49), np.linspace(0, 2 * np.pi, 97))
S = xmap(tg, pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.22, grid=False, zorder=1, shade=True)


def draw(curve, role="main", lw=0.5, ls="-", zorder=4):
    vis = dgfig.visible_mask(ax, curve, curve)
    dgfig.curve3d(ax, curve, role=role, lw=lw, ls=ls, visible=vis, zorder=zorder)


s = np.linspace(0, 2 * np.pi, 300)
for k in range(1, 6):
    draw(xmap(k * np.pi / 6 * np.ones_like(s), s), role="#6F6F6F", lw=0.5)
tline = np.linspace(0, np.pi, 200)
for k in range(1, 12):
    draw(xmap(tline, k * np.pi / 6 * np.ones_like(tline)), role="#6F6F6F", lw=0.5)
draw(xmap(TH0 * np.ones_like(s), s), role="tangent", lw=1.6, zorder=6)
draw(xmap(tline, PH0 * np.ones_like(tline)), role="third", lw=1.6, zorder=6)
# 매개화가 빠뜨리는 반원 C (φ = 0): 주황
Cc = xmap(tline, np.zeros_like(tline))
draw(Cc, role="accent", lw=2.0, zorder=7)
dgfig.point3d(ax, p, size=16, zorder=10)
ax.text(*(p + np.array([0.02, 0.05, 0.12])), r"$p = \mathbf{x}(q)$", fontsize=12, zorder=11)
ax.text(*(Cc[70] + np.array([0.18, -0.08, 0.0])), r"$C$", fontsize=12, color=C["accent"], zorder=11)
ax.text(0.0, -0.2, -1.32, r"$S^2(1)$", fontsize=12, ha="center")
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.45)

dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(0.98, 0.72), xy_to=(0.20, 0.72), rad=-0.3)

dgfig.save(fig, __file__)
