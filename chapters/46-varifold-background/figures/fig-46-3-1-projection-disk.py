"""Figure 46.3.1: lines through the origin of R^2 and their projection matrices.

(a) lines with angles 0, 30, 60, 90, 120, 150 degrees; the lines at 0 and 30 degrees are highlighted.
(b) each projection P(theta) drawn at (P11 - P22, 2 P12) = (cos 2 theta, sin 2 theta): G_{1,2} is the unit circle,
the angle is doubled (theta and theta + pi give the same point). The chord between the points of 0 and 30 degrees
has length 1, and ||P - Q||_op = sin 30 = 1/2 is half of it. The centre is I/2 = (P_h + P_v)/2, the average of the
horizontal and vertical lines (not a projection).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-3-1-projection-disk.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
angles = np.array([0, 30, 60, 90, 120, 150])
th = np.radians(angles)
P = v.line_proj(th)
xy = v.proj_coords(P)

# self-checks: coordinates are (cos 2 theta, sin 2 theta); half chord = operator distance = |sin|
assert np.allclose(xy, np.stack([np.cos(2 * th), np.sin(2 * th)], 1))
chord = np.linalg.norm(xy[0] - xy[1])
assert abs(chord - 1.0) < 1e-12 and abs(v.op_dist(P[0], P[1]) - 0.5) < 1e-12
assert abs(v.fro_dist(P[0], P[1]) - np.sqrt(2) / 2) < 1e-12
assert np.allclose(0.5 * (v.P_H + v.P_V), np.eye(2) / 2) and np.allclose(v.proj_coords(np.eye(2) / 2), 0)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
ax = axes[0]
for a, t in zip(angles, th):
    hi = a in (0, 30)
    u = np.array([np.cos(t), np.sin(t)])
    ax.plot([-u[0], u[0]], [-u[1], u[1]], color=C["tangent"] if hi else C["aux"], lw=1.8 if hi else 0.9)
    ax.text(1.13 * u[0], 1.13 * u[1], f"{a}°", fontsize=10, ha="center", va="center",
            color=C["main"] if hi else C["aux"])
ax.plot(0, 0, "o", color=C["main"], ms=3)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.set_axis_off()
ax.set_title(r"(a) lines through $0$", fontsize=12)

ax = axes[1]
tt = np.linspace(0, 2 * np.pi, 400)
ax.fill(np.cos(tt), np.sin(tt), color=C["region"], alpha=0.15, lw=0)
ax.plot(np.cos(tt), np.sin(tt), color=C["main"], lw=1.4)
ax.plot([xy[0, 0], xy[1, 0]], [xy[0, 1], xy[1, 1]], color=C["normal"], lw=1.6)
for a, p in zip(angles, xy):
    hi = a in (0, 30)
    ax.plot(*p, "o", color=C["tangent"] if hi else C["aux"], ms=6 if hi else 4, zorder=3)
    ax.text(1.2 * p[0], 1.2 * p[1], f"{a}°", fontsize=10, ha="center", va="center",
            color=C["main"] if hi else C["aux"])
mid = 0.5 * (xy[0] + xy[1])
ax.text(mid[0] - 0.12, mid[1] - 0.05, r"$2\,\|P-Q\|$", fontsize=10, ha="right", va="center", color=C["normal"])
ax.plot(0, 0, "s", color=C["accent"], ms=6, zorder=3)
ax.text(0.0, -0.13, r"$I/2$", fontsize=11, ha="center", va="top")
ax.set_aspect("equal")
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel(r"$P_{11} - P_{22}$")
ax.set_ylabel(r"$2P_{12}$")
ax.set_xticks([-1, 1])
ax.set_yticks([-1, 1])
ax.set_title(r"(b) $G_{1,2}$ as a circle", fontsize=12)

dgfig.save(fig, __file__)
