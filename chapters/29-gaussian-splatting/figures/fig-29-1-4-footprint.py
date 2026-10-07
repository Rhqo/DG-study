"""Figure 29.1.4: screen footprint of a 2D Gaussian, its 3-sigma circle, and the 16 x 16 tiles it is assigned to (Section 29.1).

Image region 272 x 208 px with 16-px tiles. A 2D Gaussian centered at (130, 100) px with eigenvalues
lambda_max = 20^2, lambda_min = (10/3)^2 px^2 and major axis at 30 degrees (so 3 sqrt(lambda) = 60 px and 10 px).
Blue: 3-sigma ellipse. Orange dashed: circle of radius r = ceil(3 sqrt(lambda_max)) = 60 px. Gray: its square bounding box.
Light orange tiles: the 64 tiles inside the square (the tiles this Gaussian is assigned to).
Blue-shaded tiles: the 19 tiles that actually meet the 3-sigma ellipse.
Self-checks: lambda_max from mid +- sqrt(mid^2 - det) is 400; tile counts 64 and 19 (same as the verify script).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-1-4-footprint.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

C = dgfig.COLORS
T = 16
W, H = 17 * T, 13 * T
center = np.array([130.0, 100.0])
a = np.radians(30.0)
R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
l1, l2 = 20.0 ** 2, (10.0 / 3) ** 2
Sig = R @ np.diag([l1, l2]) @ R.T
mid = 0.5 * np.trace(Sig)
lmax = mid + np.sqrt(mid * mid - np.linalg.det(Sig))
assert abs(lmax - 400.0) < 1e-9
r = int(np.ceil(3 * np.sqrt(lmax) - 1e-9))
assert r == 60
Sinv = np.linalg.inv(Sig)

xmin, xmax = int((center[0] - r) // T), int((center[0] + r - 1) // T)
ymin, ymax = int((center[1] - r) // T), int((center[1] + r - 1) // T)
sq_tiles = [(tx, ty) for tx in range(xmin, xmax + 1) for ty in range(ymin, ymax + 1)]
el_tiles = []
for tx, ty in sq_tiles:
    gx, gy = np.meshgrid(np.linspace(tx * T, (tx + 1) * T, 33), np.linspace(ty * T, (ty + 1) * T, 33))
    d = np.stack([gx - center[0], gy - center[1]], -1)
    if np.min(np.einsum("...i,ij,...j->...", d, Sinv, d)) <= 9.0:
        el_tiles.append((tx, ty))
assert len(sq_tiles) == 64 and len(el_tiles) == 19

fig, ax = plt.subplots(figsize=(6.0, 4.6))
for tx, ty in sq_tiles:
    ax.add_patch(Rectangle((tx * T, ty * T), T, T, fc=C["accent"], alpha=0.18, lw=0, zorder=0))
for tx, ty in el_tiles:
    ax.add_patch(Rectangle((tx * T, ty * T), T, T, fc=C["region"], alpha=0.55, lw=0, zorder=0.5))
for x in range(0, W + 1, T):
    ax.plot([x, x], [0, H], color="#BBBBBB", lw=0.5, zorder=1)
for y in range(0, H + 1, T):
    ax.plot([0, W], [y, y], color="#BBBBBB", lw=0.5, zorder=1)
t = np.linspace(0, 2 * np.pi, 400)
circ = np.stack([np.cos(t), np.sin(t)], 1)
ell = center + 3 * (circ * np.sqrt([l1, l2])) @ R.T
ax.plot(*ell.T, color=C["tangent"], lw=1.8, zorder=3)
ax.plot(*(center + r * circ).T, color=C["accent"], lw=1.5, ls=(0, (5, 3)), zorder=3)
ax.add_patch(Rectangle(center - r, 2 * r, 2 * r, fill=False, ec=C["aux"], lw=1.0, zorder=3))
ax.plot(*center, "o", color=C["main"], ms=4, zorder=5)
arrow_kw = dict(arrowstyle="-|>", lw=1.3, mutation_scale=10, shrinkA=0, shrinkB=0)
tip = center + 3 * np.sqrt(l1) * R[:, 0]
ax.annotate("", xy=tip, xytext=center, arrowprops=dict(color=C["tangent"], **arrow_kw), zorder=6)
ax.text(*(center + 0.55 * (tip - center) + np.array([9, 9])), r"$3\sqrt{\lambda_{\max}} = 60$",
        color=C["tangent"], fontsize=10.5, ha="center", va="top", rotation=-30, rotation_mode="anchor",
        bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
ax.text(center[0] + r + 4, center[1] - 4, r"circle $r = 60$ px", color="#B07700", fontsize=10.5, ha="left")
ax.text(center[0] - r, center[1] + r + 6, "square bbox: 64 tiles", color="#555555", fontsize=10.5, va="top")
ax.text(center[0] - r, center[1] - r - 6, "3-sigma ellipse meets 19 tiles", color=C["tangent"], fontsize=10.5,
        va="bottom")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)          # image convention: y grows downward
ax.set_aspect("equal")
ax.set_xlabel("pixel $u$")
ax.set_ylabel("pixel $v$")
ax.set_xticks(range(0, W + 1, 4 * T))
ax.set_yticks(range(0, H + 1, 4 * T))
ax.tick_params(labelsize=9)

dgfig.save(fig, __file__)
