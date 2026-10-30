"""Figure 49.1.1: four surfaces with the same tangent plane and normal at the origin, and their normal curvatures.

All four pass through the origin with tangent plane z = 0 and unit normal N = e_3 (upward):
  plane z = 0;  sphere of radius 1 centred at (0, 0, 1);  cylinder of radius 1 with axis {x = 0, z = 1} (along y);
  saddle z = x^2 - y^2 (GUIDELINES §7).
(a) Profiles in the x-direction (solid) and the y-direction (dashed) over |l| <= 0.8, with a flat Gaussian of 1-sigma
    half-width 0.3 in the tangent plane (grey band).
(b) Normal curvature kappa_n(theta) = kappa_1 cos^2 theta + kappa_2 sin^2 theta (Euler's formula) with N = e_3:
    plane (0, 0), sphere (1, 1), cylinder (1, 0), saddle (2, -2).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-1-1-same-normal.py
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
l = np.linspace(-0.8, 0.8, 321)
circ = 1 - np.sqrt(1 - l ** 2)
profiles = {  # name: (x-profile, y-profile, colour, (kappa_1, kappa_2) w.r.t. N = e3 along x, y)
    "plane": (0 * l, 0 * l, C["aux"], (0.0, 0.0)),
    "sphere": (circ, circ, C["main"], (1.0, 1.0)),
    "cylinder": (circ, 0 * l, C["tangent"], (1.0, 0.0)),
    "saddle": (l ** 2, -l ** 2, C["normal"], (2.0, -2.0)),
}
# self-checks: second derivative at 0 of each profile = normal curvature in that direction
h = l[1] - l[0]
i0 = len(l) // 2
for name, (px, py, _, (kx, ky)) in profiles.items():
    for p, k in ((px, kx), (py, ky)):
        assert abs((p[i0 + 1] - 2 * p[i0] + p[i0 - 1]) / h ** 2 - k) < 1e-3, name
# the deviation at l = 0.1 quoted in Example 49.1.1
assert abs((1 - np.sqrt(1 - 0.01)) - 0.0050126) < 1e-6

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.38})
ax = axes[0]
from matplotlib.patches import Rectangle  # noqa: E402
ax.add_patch(Rectangle((-0.3, -0.035), 0.6, 0.07, color=C["region"], alpha=0.45, lw=0, zorder=1))
ax.text(0.0, -0.11, "flat Gaussian", ha="center", va="top", fontsize=8.5, color=C["tangent"])
widths = {"plane": 1.4, "sphere": 1.3, "cylinder": 3.4, "saddle": 1.6}
alphas = {"plane": 1.0, "sphere": 1.0, "cylinder": 0.55, "saddle": 1.0}
for name in ("plane", "cylinder", "saddle", "sphere"):
    px, py, col, _ = profiles[name]
    ax.plot(l, px, color=col, lw=widths[name], alpha=alphas[name], label=name, zorder=3)
    if name in ("saddle",):
        ax.plot(l, py, color=col, lw=1.3, ls="--", zorder=3)
ax.annotate("", xy=(0, 0.25), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.4))
ax.text(0.05, 0.16, r"$\mathbf{N}$", color=C["normal"], fontsize=12)
ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.7, 0.95)
ax.set_xlabel(r"$\ell$ (x: solid, y: dashed)")
ax.set_ylabel(r"$z$")
ax.set_title("(a) profiles", fontsize=11)
ax.legend(fontsize=8.5, loc="upper center", ncol=2, frameon=False, handlelength=1.4, columnspacing=0.8)

ax = axes[1]
th = np.linspace(0, np.pi, 181)
for name, (_, _, col, (k1, k2)) in profiles.items():
    ax.plot(np.degrees(th), k1 * np.cos(th) ** 2 + k2 * np.sin(th) ** 2, color=col, lw=1.6)
ax.axhline(0, color=C["aux"], lw=0.6)
ax.set_xticks([0, 45, 90, 135, 180])
ax.set_xlabel(r"direction $\theta$ (deg) from $x$")
ax.set_ylabel(r"$\kappa_n(\theta)$")
ax.set_title("(b) normal curvature", fontsize=11)
ax.text(100, 1.08, "sphere", fontsize=9, color=C["main"])
ax.text(118, 0.38, "cylinder", fontsize=9, color=C["tangent"])
ax.text(60, -1.75, "saddle", fontsize=9, color=C["normal"])
ax.text(140, 0.08, "plane", fontsize=9, color=C["aux"])
ax.set_ylim(-2.2, 2.2)
dgfig.save(fig, __file__)
