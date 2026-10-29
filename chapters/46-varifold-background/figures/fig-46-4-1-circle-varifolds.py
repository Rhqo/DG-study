"""Figure 46.4.1: three varifolds of the unit circle, drawn in R^2 and in the (position, line) chart.

(a) the inscribed regular 12-gon (black) and a 12-point cloud with tangent lines (blue ticks, weights 2 pi / 12)
placed half-way between the polygon vertices.
(b) the same objects in the chart (t, theta): t = polar angle of the position, theta in [0, 180) = angle of the
unoriented line. The circle varifold v(C, 1) is the sawtooth theta = t + 90 (mod 180) (grey); each polygon edge is a
horizontal segment (its line is constant while its points sweep an arc of angles); each atom is a dot on the
sawtooth.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-4-1-circle-varifolds.py
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
N = 12
tv = 2 * np.pi * np.arange(N + 1) / N            # polygon vertices
tp = 2 * np.pi * (np.arange(N) + 0.5) / N        # atoms between the vertices
Xv = v.circle(tv)
Xp = v.circle(tp)
Tp = v.circle_tangent(tp)


def line_angle_deg(u):
    return np.degrees(np.arctan2(u[..., 1], u[..., 0])) % 180.0


# self-checks: each edge's line angle is the tangent angle at the edge midpoint; the offset from the sawtooth is
# at most 180/N degrees / 1 (= half the turning angle pi/N)
edge = np.diff(Xv, axis=0)
ang_e = line_angle_deg(edge)
assert np.allclose(ang_e, (np.degrees(tp) + 90) % 180)
assert np.allclose(line_angle_deg(Tp), (np.degrees(tp) + 90) % 180)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.3})
ax = axes[0]
tt = np.linspace(0, 2 * np.pi, 400)
ax.plot(*v.circle(tt).T, color=C["aux"], lw=0.8, ls=":")
ax.plot(Xv[:, 0], Xv[:, 1], color=C["main"], lw=1.4)
for x, u in zip(Xp, Tp):
    ax.plot([x[0] - 0.18 * u[0], x[0] + 0.18 * u[0]], [x[1] - 0.18 * u[1], x[1] + 0.18 * u[1]],
            color=C["tangent"], lw=1.6)
    ax.plot(*x, "o", color=C["tangent"], ms=3)
ax.set_aspect("equal")
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-1.25, 1.25)
ax.set_axis_off()
ax.set_title(r"(a) in $\mathbb{R}^2$", fontsize=12)

ax = axes[1]
# sawtooth theta = t + 90 mod 180, drawn piecewise to avoid the vertical jumps
td = np.linspace(0, 360, 2001)
th = (td + 90) % 180
jumps = np.where(np.diff(th) < 0)[0]
start = 0
for j in list(jumps) + [len(td) - 1]:
    ax.plot(td[start:j + 1], th[start:j + 1], color=C["aux"], lw=1.0)
    start = j + 1
for k in range(N):
    ax.plot(np.degrees([tv[k], tv[k + 1]]), [ang_e[k], ang_e[k]], color=C["main"], lw=2.0, solid_capstyle="butt")
ax.plot(np.degrees(tp), line_angle_deg(Tp), "o", color=C["tangent"], ms=4.5, zorder=3)
ax.set_xlim(0, 360)
ax.set_ylim(0, 180)
ax.set_xticks([0, 90, 180, 270, 360])
ax.set_yticks([0, 45, 90, 135, 180])
ax.set_xlabel(r"position angle $t$ (degrees)")
ax.set_ylabel(r"line angle $\theta$ (degrees)")
ax.set_title(r"(b) in the chart $(t, \theta)$", fontsize=12)

dgfig.save(fig, __file__)
