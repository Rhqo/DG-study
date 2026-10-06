"""Figure 0.7.4: the closed 1-form ω = (−y dx + x dy)/(x² + y²) on R² ∖ {0} as a stack of rays (Section 0.7).

Locally ω = dθ (θ = polar angle), so its "stack" is the family of rays θ = 2πk/16 (pink), one ray per 2π/16.
Blue: the circle γ₁ of radius 1.5 around the origin, counterclockwise; it crosses all 16 rays in the same direction,
∮_{γ₁} ω = 2π. Orange: the circle γ₂ of radius 0.6 centred at (1.9, 1.1), which does not enclose the origin; it
crosses each ray it meets once forwards and once backwards, ∮_{γ₂} ω = 0. The origin (white disk) is not in the domain.
Both integrals are checked numerically.

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-7-4-dtheta.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

C = dgfig.COLORS
LW = dgfig.LW


def omega(x, y, dx, dy):
    return (-y * dx + x * dy) / (x ** 2 + y ** 2)


def loop_integral(c, r, n=20000):
    t = np.linspace(0, 2 * np.pi, n + 1)
    x, y = c[0] + r * np.cos(t), c[1] + r * np.sin(t)
    dx, dy = -r * np.sin(t), r * np.cos(t)
    vals = omega(x, y, dx, dy)
    return np.sum((vals[1:] + vals[:-1]) / 2 * np.diff(t))


g1 = ((0.0, 0.0), 1.5)
g2 = ((1.9, 1.1), 0.6)
# self-checks (§12.1)
assert abs(loop_integral(*g1) - 2 * np.pi) < 1e-8
assert abs(loop_integral(*g2)) < 1e-8
assert np.hypot(*g2[0]) > g2[1]           # γ₂ does not enclose the origin

fig, ax = plt.subplots(figsize=(4.8, 4.2))
R = 2.8
dgfig.schematic_axes(ax, (-R, R), (-R * 0.86, R * 0.86))
for k in range(16):
    a = 2 * np.pi * k / 16
    ax.plot([0.12 * np.cos(a), 4 * np.cos(a)], [0.12 * np.sin(a), 4 * np.sin(a)], color=C["covector"], lw=1.0,
            zorder=1)
ax.add_patch(Circle((0, 0), 0.12, facecolor="white", edgecolor=C["main"], lw=1.0, zorder=3))
t = np.linspace(0, 2 * np.pi, 400)
for (c, r), col, lab, lpos in ((g1, C["tangent"], r"$\gamma_1$", (-1.25, -1.25)),
                               (g2, C["accent"], r"$\gamma_2$", (2.45, 1.75))):
    ax.plot(c[0] + r * np.cos(t), c[1] + r * np.sin(t), color=col, lw=LW["main"], zorder=4)
    for a0 in (0.25 * np.pi, 1.25 * np.pi):
        p0 = (c[0] + r * np.cos(a0), c[1] + r * np.sin(a0))
        p1 = (c[0] + r * np.cos(a0 + 0.12), c[1] + r * np.sin(a0 + 0.12))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, color=col, lw=LW["main"],
                                     shrinkA=0, shrinkB=0, zorder=5))
    ax.text(*lpos, lab, fontsize=13, color=col, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.9), zorder=6)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
dgfig.save(fig, __file__)
