"""Figure 38.1.2: local versus global failure.

(a) the figure-eight cylinder x(u, v) = (sin u, sin u cos u, v), u in [0, 2 pi], v in [-0.9, 0.9]: an immersion
(|x_u x x_v| = |x_u| >= sqrt 7 / 4 > 0) whose image crosses itself along the z-axis (u = 0 and u = pi). At the point
(0, 0, 0.5) the two sheets have the unit normals (1, -1, 0)/sqrt 2 and (1, 1, 0)/sqrt 2.
(b) the double cone x(rho, t) = (rho cos t, rho sin t, rho), rho in [-1, 1]: x_rho x x_t = (-rho cos t, -rho sin t, rho)
vanishes on rho = 0, where the whole line {0} x R is mapped to the apex; the rulings x_rho(0, t) = (cos t, sin t, 1)
fill a cone of directions, not a plane.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-1-2-immersed-embedded.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg38  # noqa: E402

C = dgfig.COLORS

# self-checks (a)
u = np.linspace(0, 2 * np.pi, 2001)
xu, xv = ddg38.partials(ddg38.figure_eight_cylinder, u, 0 * u)
cr = np.linalg.norm(np.cross(xu, xv, axis=0), axis=0)
assert abs(cr.min() - np.sqrt(7) / 4) < 1e-5     # immersion: |x_u|^2 = 4c^4 - 3c^2 + 1 >= 7/16 (c = cos u)
assert np.allclose(ddg38.figure_eight_cylinder(0.0, 0.5), ddg38.figure_eight_cylinder(np.pi, 0.5))
n0 = np.cross(*ddg38.partials(ddg38.figure_eight_cylinder, np.array(0.0), np.array(0.5)))
n1 = np.cross(*ddg38.partials(ddg38.figure_eight_cylinder, np.array(np.pi), np.array(0.5)))
n0, n1 = n0 / np.linalg.norm(n0), n1 / np.linalg.norm(n1)
assert np.allclose(n0, [1 / np.sqrt(2), -1 / np.sqrt(2), 0], atol=1e-7)
assert np.allclose(n1, [1 / np.sqrt(2), 1 / np.sqrt(2), 0], atol=1e-7)
# self-checks (b)
for rho in (-0.7, 0.0, 0.4):
    for t in (0.3, 2.0):
        a, b = ddg38.partials(ddg38.double_cone, np.array(rho), np.array(t))
        assert np.allclose(np.cross(a, b), [-rho * np.cos(t), -rho * np.sin(t), rho], atol=1e-6)

fig = plt.figure(figsize=(7.0, 3.7))
ax = dgfig.axes3d(fig, 121, elev=38, azim=-70)
U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 121), np.linspace(-0.9, 0.9, 13), indexing="ij")
X, Y, Z = ddg38.figure_eight_cylinder(U, V)
dgfig.surface(ax, X, Y, Z, alpha=0.35, grid_every=6)
ax.plot([0, 0], [0, 0], [-0.9, 0.9], color=C["accent"], lw=2.2, zorder=6)
top = ddg38.figure_eight_cylinder(np.linspace(0, 2 * np.pi, 400), 0.9 + 0 * np.linspace(0, 1, 400))
ax.plot(*top, color=C["main"], lw=1.2, zorder=5)
p = np.array([0.0, 0.0, 0.5])
dgfig.arrow3d(ax, p, 0.55 * n0, role="normal", zorder=9, head=9)
dgfig.arrow3d(ax, p, 0.55 * n1, role="normal", zorder=9, head=9)
ax.text(*(p + 0.62 * n0 + np.array([0.05, 0, 0.05])), r"$\mathbf{N}(0, v)$", color=C["normal"], fontsize=11)
ax.text(*(p + 0.62 * n1 + np.array([0.0, 0.05, 0.08])), r"$\mathbf{N}(\pi, v)$", color=C["normal"], fontsize=11)
ax.text(0.06, 0.0, -1.12, r"$u = 0,\ \pi$", color=C["accent"], fontsize=11)
dgfig.equal_aspect(ax, np.stack([X, Y, Z], -1), zoom=1.15)
ax.set_title("(a)", loc="left", fontsize=12)

ax2 = dgfig.axes3d(fig, 122, elev=16, azim=-50)
Rh, T = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(0, 2 * np.pi, 73), indexing="ij")
X, Y, Z = ddg38.double_cone(Rh, T)
dgfig.surface(ax2, X, Y, Z, alpha=0.35, grid_every=4)
for t in np.linspace(0, 2 * np.pi, 9)[:-1]:
    d = np.array([np.cos(t), np.sin(t), 1.0])
    dgfig.arrow3d(ax2, np.zeros(3), 0.55 * d, role="tangent", zorder=9, head=7, lw=1.2)
dgfig.point3d(ax2, np.zeros(3), None, size=14)
ax2.text(0.35, -0.2, -0.32, r"$\mathbf{x}(0, t)$", fontsize=11)
ax2.text(0.55, 0.0, 0.95, r"$\mathbf{x}_\rho(0, t)$", color=C["tangent"], fontsize=11)
dgfig.equal_aspect(ax2, np.stack([X, Y, Z], -1), zoom=1.15)
ax2.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95, wspace=0.0)
dgfig.save(fig, __file__)
