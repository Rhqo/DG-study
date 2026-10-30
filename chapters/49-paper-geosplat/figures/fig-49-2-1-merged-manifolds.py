"""Figure 49.2.1: a cross-section of a scene in the sense of Definition 2.1 of [GeoSplat25] (merged local manifolds).

The cross-section (a plane x-z slice of a room) is a union of three curves:
  floor  z = 0, 0 <= x <= 3;  wall  x = 0, 0 <= z <= 2;  a ball of radius 0.5 resting on the floor at (2, 0.5).
Each curve is smooth except at the singular points: the corner (0, 0), where floor and wall meet, and the contact point
(2, 0), where the ball touches the floor. Gaussian centres (dots) are placed on the curves at arc-length spacing 0.25,
each drawn as a flat 1-sigma ellipse (half-axes 0.11 along the tangent, 0.025 along the normal). Three "floaters"
(grey crosses) are off every curve: Definition 2.1 assumes that no centre is like them.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-2-1-merged-manifolds.py
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

C = dgfig.COLORS
h = 0.25
pts, tans = [], []
for x in np.arange(h / 2, 3.0, h):                      # floor
    pts.append((x, 0.0)); tans.append((1.0, 0.0))
for z in np.arange(h / 2, 2.0, h):                      # wall
    pts.append((0.0, z)); tans.append((0.0, 1.0))
c, r = np.array([2.0, 0.5]), 0.5
for a in np.arange(0, 2 * np.pi, h / r) + 0.5 * h / r:   # ball
    p = c + r * np.array([np.cos(a), np.sin(a)])
    pts.append(tuple(p)); tans.append((-np.sin(a), np.cos(a)))
pts, tans = np.array(pts), np.array(tans)
floaters = np.array([[1.0, 1.4], [2.7, 1.6], [0.6, 0.9]])

# self-checks: every centre is on one of the three curves; floaters are at least 0.35 from all of them
def dist_to_scene(p):
    d_floor = abs(p[1]) if 0 <= p[0] <= 3 else np.hypot(min(abs(p[0]), abs(p[0] - 3)), p[1])
    d_wall = abs(p[0]) if 0 <= p[1] <= 2 else np.hypot(p[0], min(abs(p[1]), abs(p[1] - 2)))
    d_ball = abs(np.linalg.norm(p - c) - r)
    return min(d_floor, d_wall, d_ball)


assert max(dist_to_scene(p) for p in pts) < 1e-12
assert min(dist_to_scene(p) for p in floaters) > 0.35
assert np.allclose(np.linalg.norm(tans, axis=1), 1)

fig, ax = plt.subplots(figsize=(5.6, 3.9))
ax.plot([0, 3], [0, 0], color=C["main"], lw=1.8)
ax.plot([0, 0], [0, 2], color=C["main"], lw=1.8)
t = np.linspace(0, 2 * np.pi, 300)
ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=C["main"], lw=1.8)
for p, tv in zip(pts, tans):
    ang = np.degrees(np.arctan2(tv[1], tv[0]))
    ax.add_patch(Ellipse(p, 2 * 0.11, 2 * 0.025, angle=ang, facecolor=C["region"], alpha=0.55,
                         edgecolor=C["tangent"], lw=0.6))
ax.plot(pts[:, 0], pts[:, 1], "o", color=C["tangent"], ms=2.2)
for p in floaters:
    ax.add_patch(Ellipse(p, 0.16, 0.10, angle=30, facecolor=C["surface"], alpha=0.7, edgecolor=C["aux"], lw=0.6))
    ax.plot(*p, "x", color=C["aux"], ms=6, mew=1.4)
for p, lab, off in (((0, 0), "corner", (0.12, 0.12)), ((2, 0), "contact", (0.1, -0.22))):
    ax.plot(*p, "o", mfc="white", mec=C["normal"], ms=8, mew=1.6, zorder=6)
    ax.text(p[0] + off[0], p[1] + off[1], lab, color=C["normal"], fontsize=10)
ax.text(2.75, 1.85, "floaters", color=C["aux"], fontsize=10, ha="center")
ax.text(1.0, -0.22, r"$\mathcal{M}_1$ (floor)", fontsize=10.5, ha="center", va="top")
ax.text(-0.12, 1.55, r"$\mathcal{M}_2$", fontsize=10.5, ha="right")
ax.text(2.0, 0.5, r"$\mathcal{M}_3$", fontsize=10.5, ha="center", va="center")
ax.set_aspect("equal")
ax.set_xlim(-0.55, 3.15)
ax.set_ylim(-0.55, 2.15)
ax.set_axis_off()
dgfig.save(fig, __file__)
