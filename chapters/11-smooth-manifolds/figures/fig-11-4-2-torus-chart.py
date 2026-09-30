"""그림 11.4.2: 토러스 T²의 각 차트 α × α (11.4절, 예 11.4.12).

왼쪽: (α × α)의 치역 (-π, π)² 과 좌표격자 (간격 π/6).
오른쪽: 그 격자를 (α × α)^{-1}로 T² = S¹ × S¹에 되돌린 뒤, 위상동형사상 Φ (3.3.17)로 원환면 S ⊆ ℝ³에 옮긴 것.
Φ∘(α × α)^{-1}(u, v) = x(u, v)는 §7의 원환면 매개화이다 (dgsym.EXAMPLES["torus"]), R = 2, r = 0.8.
차트가 덮지 못하는 두 원 {u = π}(안쪽 적도)와 {v = π}(경선 하나)는 주황으로 그린다.

실행: 프로젝트 루트에서 ``PYTHONPATH=tools python3 chapters/11-smooth-manifolds/figures/fig-11-4-2-torus-chart.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.patches import Rectangle

from dgsym import EXAMPLES

C = dgfig.COLORS
LW = dgfig.LW

ex = EXAMPLES["torus"]
Rs, rs = ex["params"]
us, vs = ex["coords"]
Rv, rv = 2.0, 0.8
xfun = sp.lambdify((us, vs), ex["expr"].subs({Rs: Rv, rs: rv}), "numpy")


def X(u, v):
    out = np.array(xfun(u, v), dtype=float).reshape(3, *np.shape(u))
    return out


# 자기검사: Φ(a, b) = ((R + r a1) b1, (R + r a1) b2, r a2)와 x(u, v)가 일치 (3.3.17)
for u0, v0 in ((0.3, -1.2), (2.5, 0.7), (-3.0, 3.0)):
    a = np.array([np.cos(u0), np.sin(u0)]); b = np.array([np.cos(v0), np.sin(v0)])
    Phi = np.array([(Rv + rv * a[0]) * b[0], (Rv + rv * a[0]) * b[1], rv * a[1]])
    assert np.allclose(Phi, X(np.array(u0), np.array(v0)).ravel())

fig = plt.figure(figsize=(6.4, 3.2))
axL = fig.add_axes([0.02, 0.1, 0.36, 0.8])
axR = dgfig.axes3d(fig, pos=None or 111, elev=32, azim=-60)
axR.set_position([0.36, 0.0, 0.64, 1.0])

# 왼쪽 정사각형
ax = axL
dgfig.schematic_axes(ax, (-3.9, 3.7), (-3.9, 3.7))
ax.add_patch(Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi, facecolor=C["region"], alpha=0.3, lw=0))
ax.add_patch(Rectangle((-np.pi, -np.pi), 2 * np.pi, 2 * np.pi, fill=False, edgecolor=C["main"], lw=0.9, ls=(0, (4, 2.5))))
grid = np.arange(-5, 6) * np.pi / 6
for g in grid:
    ax.plot([g, g], [-np.pi, np.pi], color=C["tangent"], lw=0.45)
    ax.plot([-np.pi, np.pi], [g, g], color=C["tangent"], lw=0.45)
ax.plot([np.pi, np.pi], [-np.pi, np.pi], color=C["accent"], lw=2.0)
ax.plot([-np.pi, np.pi], [np.pi, np.pi], color=C["accent"], lw=2.0)
ax.text(0, -3.75, r"$u$", fontsize=12, ha="center")
ax.text(-3.75, 0, r"$v$", fontsize=12, va="center")
ax.text(np.pi, -3.75, r"$\pi$", fontsize=10, ha="center")
ax.text(-np.pi, -3.75, r"$-\pi$", fontsize=10, ha="center")

# 오른쪽: 원환면 S ⊆ ℝ³
ax = axR
uu, vv = np.meshgrid(np.linspace(-np.pi, np.pi, 73), np.linspace(-np.pi, np.pi, 145), indexing="ij")
P = X(uu, vv)
dgfig.surface(ax, *P, alpha=0.22, grid=False, zorder=1)
fine = np.linspace(-np.pi, np.pi, 400)


def normal(u, v):
    """바깥쪽 단위법벡터 (cos u cos v, cos u sin v, sin u): 숨은선 판정에만 쓴다."""
    return np.stack([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)], axis=-1)


def draw(u, v, role, lw, z):
    pts = X(u, v).T
    vis = dgfig.visible_mask(ax, pts, normal(u, v))
    dgfig.curve3d(ax, pts, role=role, lw=lw, visible=vis, zorder=z)


for g in grid:
    draw(np.full_like(fine, g), fine, "tangent", 0.5, 3)
    draw(fine, np.full_like(fine, g), "tangent", 0.5, 3)
draw(np.full_like(fine, np.pi), fine, "accent", 2.0, 5)
draw(fine, np.full_like(fine, np.pi), "accent", 2.0, 5)
dgfig.equal_aspect(ax, *P, zoom=1.3)

dgfig.map_arrow(fig, axL, axR, r"$\Phi\circ(\alpha\times\alpha)^{-1}$", xy_from=(0.92, 0.7), xy_to=(0.2, 0.72), rad=-0.3, fontsize=11)
dgfig.save(fig, __file__)
