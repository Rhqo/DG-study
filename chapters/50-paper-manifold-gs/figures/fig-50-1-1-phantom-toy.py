"""Figure 50.1.1: why a watertight closure creates phantom collision surface (a 2D toy of our construction).

The true scene is a thin wall: the arc of the unit circle with angles in [200, 340] degrees, seen from cameras below.
(a) a watertight candidate closes the unseen side (the full circle, solid inside); the certified candidate keeps only the
open arc [215, 325] degrees. (b) floater = share of the candidate farther than tol = 0.02 from the wall,
coverage = share of the wall within tol of the candidate, phantom = share of free-space probes (uniform in [-1.5, 1.5]^2,
farther than tol from the wall) that the candidate blocks.

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-1-1-phantom-toy.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import mgs50 as g  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

C = dgfig.COLORS
m = g.toy_metrics()
fw, cw, pw = m["watertight"]
fc, cc, pc = m["certified"]
# self-checks: the closure covers the whole wall, its floater is about the unseen share 220/360, the certified arc has
# no floater and no phantom, and covers about 110/140 of the wall
assert cw == 100.0 and abs(fw - 100 * 220 / 360) < 1.5
assert fc == 0.0 and pc == 0.0 and abs(cc - 100 * 110 / 140) < 2.5
assert abs(pw - 100 * np.pi / (9 - 0.0)) < 2.0

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 3.3), gridspec_kw={"width_ratios": [1.05, 1]})
ax.add_patch(Circle((0, 0), 1.0, facecolor=C["accent"], alpha=0.18, edgecolor="none"))
full = g.arc_points(0, 360, 400)
ax.plot(full[:, 0], full[:, 1], color=C["accent"], lw=1.4, ls="--", label="watertight")
gt = g.arc_points(*g.TOY_GT, 200)
ax.plot(gt[:, 0], gt[:, 1], color=C["main"], lw=3.2, label="true wall")
ce = g.arc_points(*g.TOY_CERT, 200)
ax.plot(ce[:, 0], ce[:, 1], color=C["tangent"], lw=1.6, label="certified")
for cx in (-0.9, 0.0, 0.9):
    for ang in (225.0, 270.0, 315.0):
        hit = g.arc_points(ang, ang, 1)[0]
        ax.plot([cx, hit[0]], [-2.1, hit[1]], color=C["aux"], lw=0.5, ls=":")
    ax.plot([cx], [-2.1], marker="^", color=C["main"], ms=6)
ax.annotate("unseen side", xy=(0.55, 0.84), xytext=(1.0, 1.32), fontsize=9.5, ha="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C["aux"]))
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-2.25, 1.6)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc="upper left", fontsize=8.5, frameon=False, handlelength=1.6, bbox_to_anchor=(-0.03, 1.02))
ax.set_title("(a) toy scene", fontsize=11)

x = np.arange(3)
w = 0.36
bx.bar(x - w / 2, [fw, cw, pw], w, color=C["accent"], label="watertight")
bx.bar(x + w / 2, [fc, cc, pc], w, color=C["tangent"], label="certified")
for xi, a, b in zip(x, (fw, cw, pw), (fc, cc, pc)):
    bx.text(xi - w / 2, a + 2, f"{a:.1f}", ha="center", fontsize=8.5)
    bx.text(xi + w / 2, b + 2, f"{b:.1f}", ha="center", fontsize=8.5)
bx.set_xticks(x)
bx.set_xticklabels(["floater", "coverage", "phantom"])
bx.set_ylabel("percent")
bx.set_ylim(0, 115)
bx.legend(loc="upper right", fontsize=8.5, frameon=False)
bx.set_title("(b) three rates", fontsize=11)
fig.tight_layout()
dgfig.save(fig, __file__)
print("watertight floater %.2f coverage %.2f phantom %.2f | certified %.2f %.2f %.2f" % (fw, cw, pw, fc, cc, pc))
