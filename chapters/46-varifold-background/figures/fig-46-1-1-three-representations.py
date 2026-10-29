"""Figure 46.1.1: one circle, three data types, one kind of object.

(a) the unit circle (r = 1) represented on three arcs by an inscribed polygon (mesh), by a nonuniform point cloud
with unoriented tangent lines and weights, and by flat 2D Gaussians.
(b) the same data after forgetting the type: short tangent segments at positions, with thickness proportional to
the mass per unit length (all are lists of (position, unoriented tangent line, mass)).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-1-1-three-representations.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
r = 1.0

# --- mesh: inscribed polygon on t in [0, 2pi/3], vertices every pi/9
tm = np.linspace(0.0, 2 * np.pi / 3, 7)
Pm = v.circle(tm, r)
# --- point cloud on t in [2pi/3, 4pi/3]: nonuniform, Voronoi weights (arc length of the cell)
u = (np.arange(9) + 0.5) / 9
tp = 2 * np.pi / 3 + (2 * np.pi / 3) * (u + 0.08 * np.sin(2 * np.pi * u))
mids = 0.5 * (tp[1:] + tp[:-1])
edges = np.concatenate([[2 * np.pi / 3], mids, [4 * np.pi / 3]])
wp = r * np.diff(edges)
Xp = v.circle(tp, r)
Tp = v.circle_tangent(tp)
# --- Gaussians on t in [4pi/3, 2pi]: 5 cells, s1 = half cell length / 1.2, s2 = 0.035
cells = np.linspace(4 * np.pi / 3, 2 * np.pi, 6)
tg = 0.5 * (cells[1:] + cells[:-1])
Xg = v.circle(tg, r)
Tg = v.circle_tangent(tg)
cell_len = r * np.diff(cells)
s1 = cell_len / 2 / 1.2
s2 = np.full(5, 0.05)

# self-checks: points on the circle, tangent orthogonal to radius, weights add up to the arc length
assert np.allclose(np.linalg.norm(Xp, axis=1), r) and np.allclose(np.linalg.norm(Xg, axis=1), r)
assert np.allclose(np.sum(Tp * Xp, axis=1), 0) and abs(wp.sum() - 2 * np.pi / 3 * r) < 1e-12
# covariance of each Gaussian: its largest-eigenvalue eigenvector is the tangent
for k in range(5):
    R = np.column_stack([Tg[k], Xg[k] / r])
    S = R @ np.diag([s1[k] ** 2, s2[k] ** 2]) @ R.T
    w, Q = np.linalg.eigh(S)
    assert abs(abs(Q[:, 1] @ Tg[k]) - 1) < 1e-12

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3))
tt = np.linspace(0, 2 * np.pi, 400)

ax = axes[0]
ax.plot(*v.circle(tt, r).T, color=C["aux"], lw=0.8, ls=":", zorder=1)
ax.plot(Pm[:, 0], Pm[:, 1], color=C["main"], lw=1.6, zorder=3)
ax.plot(Pm[:, 0], Pm[:, 1], "o", color=C["main"], ms=3, zorder=4)
for x, t_, w in zip(Xp, Tp, wp):
    ax.plot([x[0] - 0.13 * t_[0], x[0] + 0.13 * t_[0]], [x[1] - 0.13 * t_[1], x[1] + 0.13 * t_[1]],
            color=C["tangent"], lw=1.4, zorder=3)
ax.scatter(Xp[:, 0], Xp[:, 1], s=110 * wp, color=C["main"], zorder=4)
for x, t_, a, b in zip(Xg, Tg, s1, s2):
    ang = np.degrees(np.arctan2(t_[1], t_[0]))
    ax.add_patch(Ellipse(x, 2 * a, 2 * b, angle=ang, facecolor=C["region"], alpha=0.45,
                         edgecolor=C["tangent"], lw=0.8, zorder=3))
    ax.plot(*x, ".", color=C["main"], ms=3, zorder=4)
ax.text(0.95, 0.98, "mesh", fontsize=11, ha="left", va="bottom")
ax.text(-1.55, 0.05, "points", fontsize=11, ha="right", va="center")
ax.text(0.35, -1.12, "Gaussians", fontsize=11, ha="left", va="top")
ax.set_title("(a) three data types", fontsize=12)

ax = axes[1]
ax.plot(*v.circle(tt, r).T, color=C["aux"], lw=0.8, ls=":", zorder=1)
# mesh edges: mass per unit length 1 -> width 1.6
for a, b in zip(Pm[:-1], Pm[1:]):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=C["tangent"], lw=1.6, solid_capstyle="butt", zorder=3)
# atoms: segment of length 0.84 m along the tangent (mass of its cell; the gap shows the atoms are separate)
for X, T, M in ((Xp, Tp, wp), (Xg, Tg, cell_len)):
    for x, t_, m in zip(X, T, M):
        ax.plot([x[0] - 0.42 * m * t_[0], x[0] + 0.42 * m * t_[0]],
                [x[1] - 0.42 * m * t_[1], x[1] + 0.42 * m * t_[1]],
                color=C["tangent"], lw=1.6, solid_capstyle="butt", zorder=3)
        ax.plot(*x, "o", color=C["main"], ms=2.5, zorder=4)
ax.text(0.0, 0.0, r"$(x,\ P,\ m)$", fontsize=12, ha="center", va="center")
ax.set_title("(b) positions, lines, masses", fontsize=12)

for ax in axes:
    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.45)
    ax.set_ylim(-1.35, 1.3)
    ax.set_axis_off()

dgfig.save(fig, __file__)
