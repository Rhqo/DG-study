"""Figure 39.5.1: the second fundamental form is the rate of change of the first under normal offsets.

Cross-sections perpendicular to the rulings. (a) the plane (y = 0 line), unit normal N = e_y, offset by t = 0.6:
a piece of length 1 stays of length 1 (II = 0). (b) the cylinder of radius r = 1 (unit circle), outward N,
offset by t = 0.6 (circle of radius 1.6): the piece with du = 1 has length r du = 1 before and (r + t) du = 1.6
after, so I_t(d_u, d_u) = (r + t)^2 = r^2 - 2 t II + t^2 III with II = -r, III = 1 (Proposition 39.5.1).
Both surfaces have the same first fundamental form r^2 du^2 + dv^2 in the coordinates (u, v) (Example 39.5.2).

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-5-1-offset-forms.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
T = 0.6
r = 1.0
# self-check of I_t = I - 2 t II + t^2 III on the cylinder: (r + t)^2 = r^2 - 2 t (-r) + t^2 * 1
assert abs((r + T) ** 2 - (r ** 2 - 2 * T * (-r) + T ** 2)) < 1e-15

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
ax = axes[0]
ax.plot([-1.6, 1.6], [0, 0], color=C["main"], lw=dgfig.LW["main"])
ax.plot([-1.6, 1.6], [T, T], color=C["accent"], lw=1.4, ls=(0, (5, 3)))
for x in np.linspace(-1.4, 1.4, 8):
    ax.annotate("", xy=(x, 0.32), xytext=(x, 0), arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.1,
                                                              mutation_scale=9, shrinkA=0, shrinkB=0))
for y, lab in ((0.0, r"$1$"), (T, r"$1$")):
    ax.plot([-0.5, 0.5], [y, y], color=C["tangent"], lw=3.2, solid_capstyle="butt", zorder=5)
    ax.text(0, y - 0.17, lab, ha="center", fontsize=12, color=C["tangent"])
ax.text(1.0, 0.36, r"$\mathbf{N}$", color=C["normal"], fontsize=12)
ax.text(1.62, T + 0.03, r"$t$", fontsize=12, color=C["accent"])
ax.text(-1.6, -0.4, r"$\mathrm{II} = 0$", fontsize=12)
dgfig.schematic_axes(ax, (-1.75, 1.85), (-0.5, 1.75))
ax.set_title("(a) plane", loc="left", fontsize=12)

ax = axes[1]
ph = np.linspace(np.radians(15), np.radians(165), 300)
ax.plot(r * np.cos(ph), r * np.sin(ph), color=C["main"], lw=dgfig.LW["main"])
ax.plot((r + T) * np.cos(ph), (r + T) * np.sin(ph), color=C["accent"], lw=1.4, ls=(0, (5, 3)))
for a in np.linspace(np.radians(25), np.radians(155), 8):
    p = np.array([np.cos(a), np.sin(a)])
    ax.annotate("", xy=p * (r + 0.32), xytext=p * r, arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.1,
                                                                    mutation_scale=9, shrinkA=0, shrinkB=0))
pc = np.linspace(np.pi / 2 - 0.5, np.pi / 2 + 0.5, 50)
for R_, lab in ((r, r"$r\,du = 1$"), (r + T, r"$(r + t)\,du = 1.6$")):
    ax.plot(R_ * np.cos(pc), R_ * np.sin(pc), color=C["tangent"], lw=3.2, solid_capstyle="butt", zorder=5)
    ax.text(0, R_ - 0.2, lab, ha="center", fontsize=11, color=C["tangent"])
ax.plot([0, 0.5 * np.cos(np.pi / 2 - 0.5)], [0, 0.5 * np.sin(np.pi / 2 - 0.5)], color=C["aux"], lw=0.7)
ax.plot([0, 0.5 * np.cos(np.pi / 2 + 0.5)], [0, 0.5 * np.sin(np.pi / 2 + 0.5)], color=C["aux"], lw=0.7)
ax.plot(0, 0, "+", color=C["aux"], ms=7)
ax.text(0.08, 0.12, r"$du$", fontsize=11, color=C["aux"])
ax.text(1.08, 0.62, r"$\mathbf{N}$", color=C["normal"], fontsize=12)
ax.text(-1.75, -0.4, r"$\mathrm{II} = -r\,du^2$", fontsize=12)
dgfig.schematic_axes(ax, (-1.8, 1.8), (-0.5, 1.75))
ax.set_title("(b) cylinder", loc="left", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.92, wspace=0.05)
dgfig.save(fig, __file__)
