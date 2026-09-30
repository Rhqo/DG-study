"""그림 6.3.1: 구면의 두 매개화 사이의 매개변수 변환 (6.3절, 정리 6.3.1, 예 6.3.2).

r = 1. x = 기준 매개화 (dgsym.EXAMPLES["sphere"]), y(ū, v̄) = (ū, v̄, √(1 − ū² − v̄²)) (위쪽 반구의 그래프 매개화).
겹치는 곳 W = x(U) ∩ y(Ū) = (위쪽 열린 반구) ∖ C.
(a) 구면을 위에서 비스듬히 본 것. 위쪽 반구(하늘색)와 반원 C(주황), 위도원 θ = π/4(파랑), 경선 φ = 5π/3(청록).
(b) 왼쪽: y⁻¹(W) = 원판에서 선분 {v̄ = 0, ū ≥ 0}을 뺀 것과 극좌표 격자.
    오른쪽: x⁻¹(W) = (0, π/2) × (0, 2π)와 직교 격자. 곡선 화살표 h = x⁻¹ ∘ y.
    대응하는 원 ρ̄ = sin(π/4) ↔ 선 θ = π/4 (파랑), 반직선 φ = 5π/3 ↔ 선 φ = 5π/3 (청록).

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-3-1-change-of-parameters.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import ConnectionPatch

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["sphere"]
th, ph = ex["coords"]
(r,) = ex["params"]
X = sp.lambdify((th, ph), list(ex["expr"].subs(r, 1)), "numpy")


def xmap(T, P):
    return np.stack([np.broadcast_to(c, np.shape(T)) for c in X(T, P)], axis=-1)


TH1, PH1 = np.pi / 4, 5 * np.pi / 3

# 자기검사: y⁻¹ ∘ x (θ, φ) = (sin θ cos φ, sin θ sin φ)이고, 그 원이 θ = 상수에 대응
T, P = np.meshgrid(np.linspace(0.05, np.pi / 2 - 0.05, 20), np.linspace(0.05, 2 * np.pi - 0.05, 20))
Pts = xmap(T, P)
assert np.all(Pts[..., 2] > 0)
k = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P)], axis=-1)
assert np.allclose(Pts[..., :2], k)
hemi = np.stack([k[..., 0], k[..., 1], np.sqrt(1 - (k ** 2).sum(-1))], axis=-1)
assert np.allclose(hemi, Pts)

fig = plt.figure(figsize=(6.8, 3.0))

# (a) 구면 ------------------------------------------------------------------------------
ax = fig.add_axes([0.0, 0.0, 0.36, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=35, azim=-50)
ax.set_axis_off()
tg, pg = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81))
S = xmap(tg, pg)
dgfig.surface(ax, S[..., 0], S[..., 1], S[..., 2], alpha=0.2, grid=False, zorder=1)
tu, pu = np.meshgrid(np.linspace(0, np.pi / 2, 21), np.linspace(0, 2 * np.pi, 81))
Hs = xmap(tu, pu)
ax.plot_surface(Hs[..., 0], Hs[..., 1], Hs[..., 2], color=C["region"], alpha=0.45, lw=0, zorder=2,
                shade=False, rasterized=True)


def draw(curve, role="main", lw=1.0, zorder=5):
    vis = dgfig.visible_mask(ax, curve, curve)
    dgfig.curve3d(ax, curve, role=role, lw=lw, visible=vis, zorder=zorder)


s = np.linspace(0, 2 * np.pi, 300)
draw(xmap(np.pi / 2 * np.ones_like(s), s), role="aux", lw=0.8)
draw(xmap(TH1 * np.ones_like(s), s), role="tangent", lw=1.5)
tl = np.linspace(0, np.pi, 200)
draw(xmap(tl, PH1 * np.ones_like(tl)), role="third", lw=1.5)
draw(xmap(tl, np.zeros_like(tl)), role="accent", lw=2.0, zorder=6)
ax.text(*(xmap(1.2, 0.0) + np.array([0.15, -0.1, 0.0])), r"$C$", color=C["accent"], fontsize=12)
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.45)
ax.text2D(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11, va="top")

# (b) 두 매개변수 영역 ---------------------------------------------------------------------
axd = fig.add_axes([0.40, 0.14, 0.26, 0.72])
axd.fill(np.cos(s), np.sin(s), color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
axd.plot(np.cos(s), np.sin(s), color=C["aux"], lw=LW["aux"], ls=(0, (4, 3)))
for rho in np.sin(np.linspace(np.pi / 12, 5 * np.pi / 12, 5)):
    axd.plot(rho * np.cos(s), rho * np.sin(s), color="#8C8C8C", lw=0.4)
for ang in np.arange(1, 12) * np.pi / 6:
    axd.plot([0, np.cos(ang)], [0, np.sin(ang)], color="#8C8C8C", lw=0.4)
axd.plot([0, 1], [0, 0], color=C["accent"], lw=1.8)
axd.plot(np.sin(TH1) * np.cos(s), np.sin(TH1) * np.sin(s), color=C["tangent"], lw=1.5)
axd.plot([0, np.cos(PH1)], [0, np.sin(PH1)], color=C["third"], lw=1.5)
axd.text(0, 1.12, r"$\mathbf{y}^{-1}(W)$", fontsize=11, ha="center")
axd.text(1.05, -0.08, r"$\bar u$", fontsize=11, va="top")
dgfig.schematic_axes(axd, (-1.15, 1.25), (-1.2, 1.3))
axd.text(-1.15, 1.3, "(b)", fontsize=11, va="top")

axr = fig.add_axes([0.76, 0.1, 0.2, 0.8])
axr.fill([0, np.pi / 2, np.pi / 2, 0], [0, 0, 2 * np.pi, 2 * np.pi], color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
axr.plot([0, np.pi / 2, np.pi / 2, 0, 0], [0, 0, 2 * np.pi, 2 * np.pi, 0], color=C["aux"], lw=LW["aux"], ls=(0, (4, 3)))
for tt in np.linspace(np.pi / 12, 5 * np.pi / 12, 5):
    axr.plot([tt, tt], [0, 2 * np.pi], color="#8C8C8C", lw=0.4)
for pp in np.arange(1, 12) * np.pi / 6:
    axr.plot([0, np.pi / 2], [pp, pp], color="#8C8C8C", lw=0.4)
axr.plot([0, np.pi / 2], [0, 0], color=C["accent"], lw=1.8)
axr.plot([0, np.pi / 2], [2 * np.pi, 2 * np.pi], color=C["accent"], lw=1.8)
axr.plot([TH1, TH1], [0, 2 * np.pi], color=C["tangent"], lw=1.5)
axr.plot([0, np.pi / 2], [PH1, PH1], color=C["third"], lw=1.5)
axr.text(np.pi / 4, 2 * np.pi + 0.3, r"$\mathbf{x}^{-1}(W)$", fontsize=11, ha="center")
axr.text(np.pi / 4, -0.35, r"$\theta$", fontsize=11, ha="center", va="top")
axr.text(np.pi / 2 + 0.12, np.pi, r"$\varphi$", fontsize=11, va="center")
axr.text(np.pi / 2, -0.1, r"$\pi/2$", fontsize=9, ha="center", va="top")
dgfig.schematic_axes(axr, (-0.1, np.pi / 2 + 0.4), (-0.9, 2 * np.pi + 0.8))

dgfig.map_arrow(fig, axd, axr, r"$h$", xy_from=(0.92, 0.75), xy_to=(0.0, 0.75), rad=-0.35)

dgfig.save(fig, __file__)
