"""그림 6.1.3: 8자 기둥 — 조건 (2)를 어기는 사상 (6.1절, 비예 6.1.3(c), 비예 6.1.9(d)).

x(u, v) = (sin 2u, sin u, v),  U = (−π, π) × ℝ  (8자 곡선 β(u) = (sin 2u, sin u)는 3장 비예 3.3.11).
(a) 상 x(U)의 |v| ≤ 0.6 부분. 파랑: x((−δ, δ) × (−0.6, 0.6)), 주황: x(((−π, −π + δ) ∪ (π − δ, π)) × (−0.6, 0.6)), δ = 0.3.
    두 조각이 모두 원점 p = x(0, 0) 근처를 지난다.
(b) 매개변수 영역 U의 |v| ≤ 0.6 부분과 위의 세 직사각형. 파랑은 q = (0, 0) 근처, 주황은 U의 양 끝.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/06-regular-surfaces/figures/fig-6-1-3-figure-eight-cylinder.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

C = dgfig.COLORS
LW = dgfig.LW

u, v = sp.symbols("u v", real=True)
X8 = sp.Matrix([sp.sin(2 * u), sp.sin(u), v])
f8 = sp.lambdify((u, v), list(X8), "numpy")
D8 = X8.jacobian([u, v])

# 자기검사: 모든 점에서 전미분의 계수 2, 단사(격자 표본), 끝 조각은 원점으로 다가간다
for uu in np.linspace(-np.pi + 0.01, np.pi - 0.01, 57):
    assert np.linalg.matrix_rank(np.array(D8.subs(u, uu), dtype=float)) == 2
us = np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 4001)
B = np.stack([np.sin(2 * us), np.sin(us)], axis=1)
d = np.linalg.norm(B[:, None, :] - B[None, ::40, :], axis=-1)
close_pairs = (d < 1e-9) & (np.abs(us[:, None] - us[None, ::40]) > 1e-6)
assert not close_pairs.any()
assert np.linalg.norm(np.array(f8(np.pi - 1e-4, 0.0))) < 1e-3

DEL = 0.3
H = 0.6


def patch(u0, u1, n=30):
    U_, V_ = np.meshgrid(np.linspace(u0, u1, n), np.linspace(-H, H, 12))
    return [np.broadcast_to(c, U_.shape) for c in f8(U_, V_)]


fig = plt.figure(figsize=(6.6, 3.1))

# (a) 3D ------------------------------------------------------------------------------
ax = fig.add_axes([0.0, 0.0, 0.52, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=38, azim=-62)
ax.set_axis_off()
Xall = patch(-np.pi, np.pi, 241)
dgfig.surface(ax, *Xall, alpha=0.28, grid_every=10, zorder=1, shade=False)
Xb = patch(-DEL, DEL)
ax.plot_surface(*Xb, color=C["tangent"], alpha=0.45, lw=0, zorder=2, rasterized=True)
for (a0, a1) in ((-np.pi, -np.pi + DEL), (np.pi - DEL, np.pi)):
    Xo = patch(a0, a1)
    ax.plot_surface(*Xo, color=C["accent"], alpha=0.55, lw=0, zorder=2, rasterized=True)
s = np.linspace(-np.pi, np.pi, 400)
for v0 in (-H, H):
    ax.plot(np.sin(2 * s), np.sin(s), v0 + 0 * s, color=C["main"], lw=1.0, zorder=4)
ax.plot([0, 0], [0, 0], [-H, H], color=C["aux"], lw=0.8, ls=(0, (3, 2)), zorder=5)
dgfig.point3d(ax, [0, 0, 0], size=16, zorder=10)
ax.text(0.12, 0.0, 0.08, r"$p$", fontsize=12, zorder=11)
dgfig.equal_aspect(ax, np.array([[-1, -1, -H], [1, 1, H]]), zoom=1.1)
ax.text2D(0.02, 0.95, "(a)", transform=ax.transAxes, fontsize=11, va="top")

# (b) U -------------------------------------------------------------------------------
axU = fig.add_axes([0.56, 0.22, 0.42, 0.56])
axU.fill([-np.pi, np.pi, np.pi, -np.pi], [-H, -H, H, H], color=C["region"], alpha=dgfig.ALPHA["region"], lw=0)
axU.plot([-np.pi, -np.pi], [-H, H], color=C["aux"], lw=LW["aux"], ls=(0, (4, 3)))
axU.plot([np.pi, np.pi], [-H, H], color=C["aux"], lw=LW["aux"], ls=(0, (4, 3)))
axU.fill([-DEL, DEL, DEL, -DEL], [-H, -H, H, H], color=C["tangent"], alpha=0.45, lw=0)
for (a0, a1) in ((-np.pi, -np.pi + DEL), (np.pi - DEL, np.pi)):
    axU.fill([a0, a1, a1, a0], [-H, -H, H, H], color=C["accent"], alpha=0.55, lw=0)
axU.plot([0], [0], "o", color=C["main"], ms=4)
axU.text(0.08, 0.1, r"$q$", fontsize=12)
axU.text(0, H + 0.15, r"$U$", fontsize=13, ha="center")
axU.text(-np.pi, -H - 0.12, r"$-\pi$", fontsize=11, ha="center", va="top")
axU.text(np.pi, -H - 0.12, r"$\pi$", fontsize=11, ha="center", va="top")
axU.text(0, -H - 0.12, r"$u$", fontsize=12, ha="center", va="top")
axU.text(np.pi + 0.35, 0, r"$v$", fontsize=12, va="center")
dgfig.schematic_axes(axU, (-np.pi - 0.3, np.pi + 0.7), (-H - 0.6, H + 0.5))
axU.text(-np.pi - 0.3, H + 0.5, "(b)", fontsize=11, va="top")

dgfig.map_arrow(fig, axU, ax, r"$\mathbf{x}$", xy_from=(0.45, 1.02), xy_to=(0.70, 0.66), rad=0.35)

dgfig.save(fig, __file__)
