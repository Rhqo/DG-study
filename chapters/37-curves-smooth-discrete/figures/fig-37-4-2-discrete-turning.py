"""Figure 37.4.2: discrete turning numbers.

(a) the pentagram {5/2} (vertices of a regular pentagon visited with step 2): every turning angle is 4 pi/5, k = 2;
(b) the figure-eight (sin t, sin t cos t) sampled at 12 parameter values: turning angles of both signs, k = 0.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-4-2-discrete-turning.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402

C = dgfig.COLORS
a = np.pi / 2 + 2 * np.pi * 2 * np.arange(5) / 5
P5 = np.stack([np.cos(a), np.sin(a)], 1)
th5 = dd.turning_angles(P5, True)
assert np.allclose(th5, 4 * np.pi / 5) and abs(th5.sum() / (2 * np.pi) - 2) < 1e-12
P8, _ = ddg37.sample("figure8", 12, phase=np.pi / 12)
th8 = dd.turning_angles(P8, True)
assert abs(th8.sum()) < 1e-12 and np.abs(th8).max() < np.pi - 0.1


def draw(ax, P, th, r):
    n = len(P)
    E = np.roll(P, -1, 0) - P
    for i in range(n):
        ax.annotate("", xy=P[(i + 1) % n], xytext=P[i],
                    arrowprops=dict(arrowstyle="-|>", color=C["main"], lw=1.4, mutation_scale=9, shrinkA=0,
                                    shrinkB=2))
    ax.plot(P[:, 0], P[:, 1], "o", color=C["main"], ms=3.2, zorder=5)
    for i in range(n):
        a_in = np.degrees(np.arctan2(E[i - 1, 1], E[i - 1, 0]))
        col = C["tangent"] if th[i] > 0 else C["normal"]
        t1, t2 = (a_in, a_in + np.degrees(th[i])) if th[i] > 0 else (a_in + np.degrees(th[i]), a_in)
        ext = P[i] + 0.9 * r * E[i - 1] / np.linalg.norm(E[i - 1])
        ax.plot([P[i, 0], ext[0]], [P[i, 1], ext[1]], color=C["aux"], lw=0.7, ls=":")
        ax.add_patch(Arc(P[i], 2 * r, 2 * r, theta1=t1, theta2=t2, color=col, lw=1.4))


fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
draw(ax, P5, th5, 0.2)
ax.text(0, -1.3, r"$\theta_i = 4\pi/5,\ \ k = 2$", ha="center", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.45, 1.25)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12, loc="left")
draw(bx, P8, th8, 0.12)
bx.text(0, -0.72, r"$\sum_i \theta_i = 0,\ \ k = 0$", ha="center", fontsize=11)
bx.text(-1.25, 0.55, r"$\theta_i > 0$", color=C["tangent"], fontsize=10.5)
bx.text(0.75, 0.55, r"$\theta_i < 0$", color=C["normal"], fontsize=10.5)
bx.set_aspect("equal")
bx.set_xlim(-1.35, 1.35)
bx.set_ylim(-0.85, 0.7)
bx.set_axis_off()
bx.set_title("(b)", fontsize=12, loc="left")
fig.tight_layout()
dgfig.save(fig, __file__)
