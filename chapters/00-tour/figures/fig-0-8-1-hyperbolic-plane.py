"""Figure 0.8.1: the hyperbolic plane U² = {y > 0} with g = (dx² + dy²)/y² (Section 0.8).

Grey circles: the g-unit circles {v : |v|_g = 1} at several points, drawn at 0.15 times their size. At (x₀, y₀) the
g-unit circle is the Euclidean circle of radius y₀, so the circles shrink towards the boundary y = 0.
Orange: geodesics — the vertical line x = −2.2 and the upper semicircles x² + y² = 4 and (x − 1.4)² + y² = 0.81
(centred on the x-axis). Blue: the geodesic arc from A = (−1, √3) to B = (1, √3) on x² + y² = 4, of g-length ln 3 ≈ 1.099;
the dashed horizontal segment from A to B has g-length 2/√3 ≈ 1.155. The geodesic bends upwards, where lengths are shorter.
The geodesic is checked by integrating the geodesic equation with dgnum (Christoffel symbols from dgsym).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-8-1-hyperbolic-plane.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgnum
import dgsym

C = dgfig.COLORS
LW = dgfig.LW

ex = dgsym.EXAMPLES["hyperbolic_plane"]
Gamma = dgnum.numeric_christoffel(ex["expr"], ex["coords"])

A = np.array([-1.0, np.sqrt(3)])
B = np.array([1.0, np.sqrt(3)])

# self-checks (§12.1): the geodesic from A with velocity tangent to x² + y² = 4 stays on that circle and reaches B
t_end = np.log(3)                               # unit g-speed, so the parameter is g-arc length
ts = np.linspace(0, t_end, 4001)
v0 = np.array([np.sqrt(3), 1.0]) / 2 * A[1]     # Euclidean unit tangent (pointing right) times y, so |v0|_g = 1
xs, vs = dgnum.integrate_geodesic(Gamma, A, v0, ts)
assert np.max(np.abs(np.hypot(xs[:, 0], xs[:, 1]) - 2)) < 1e-7
assert np.linalg.norm(xs[-1] - B) < 1e-6
speed = np.hypot(vs[:, 0], vs[:, 1]) / xs[:, 1]
assert np.max(np.abs(speed - 1)) < 1e-7
assert abs(2 / np.sqrt(3) - 1.1547005) < 1e-6 and 2 / np.sqrt(3) > np.log(3)

fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.set_xlim(-2.6, 2.6)
ax.set_ylim(0, 3.05)
ax.set_aspect("equal")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_linewidth(1.4)
ax.set_yticks([])
ax.set_xticks([])
ax.text(2.55, 0.07, r"$y = 0$", fontsize=11, ha="right", va="bottom", color=C["aux"])

t = np.linspace(0, 2 * np.pi, 120)
for y0, n in ((0.3, 9), (0.6, 7), (1.2, 5), (2.55, 4)):
    for x0 in np.linspace(-2.1, 2.1, n):
        r = 0.15 * y0
        ax.plot(x0 + r * np.cos(t), y0 + r * np.sin(t), color=C["aux"], lw=0.8, zorder=2)

ax.plot([-2.2, -2.2], [0, 3.05], color=C["accent"], lw=LW["main"], zorder=3)
for c, r in ((0.0, 2.0), (1.4, 0.9)):
    tt = np.linspace(0, np.pi, 300)
    ax.plot(c + r * np.cos(tt), r * np.sin(tt), color=C["accent"], lw=LW["main"], zorder=3)
ax.plot(xs[:, 0], xs[:, 1], color=C["tangent"], lw=2.6, zorder=4)
ax.plot([A[0], B[0]], [A[1], B[1]], color=C["main"], lw=1.0, ls=(0, (4, 3)), zorder=4)
for P, lab, off in ((A, r"$A$", (-0.18, 0.05)), (B, r"$B$", (0.08, 0.05))):
    ax.plot(*P, "o", color=C["main"], ms=4.5, zorder=6)
    ax.text(P[0] + off[0], P[1] + off[1], lab, fontsize=12, zorder=6)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03)
dgfig.save(fig, __file__)
