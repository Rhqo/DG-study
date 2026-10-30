"""Figure 50.5.1: the normal-angle column of [ManifoldGS26] Table 1 against the angle of a random unoriented normal.

Grey: histogram of the angle between 400000 uniformly random unoriented normals and a fixed axis (seed 5); black: the
exact density sin(theta) (per radian, drawn per degree). Its mean is 1 rad = 57.30 degrees and its median 60 degrees.
Vertical lines: the four values of the column "Norm." of Table 1 (RGB 64.91, RGB + comp. 55.18, + adapt. comp. 55.95,
+ depth oracle 8.84), read as mean angles in degrees (the paper does not state the unit or the statistic).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-5-1-random-normals.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import mgs50 as g  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
ang = g.random_normal_angles()
# self-checks: mean 1 rad and median 60 degrees, as the density sin(theta) predicts
assert abs(ang.mean() - math.degrees(1.0)) < 0.15 and abs(np.median(ang) - 60.0) < 0.2
fig, ax = plt.subplots(figsize=(5.0, 3.1))
ax.hist(ang, bins=45, range=(0, 90), density=True, color=C["surface"], edgecolor="white", lw=0.4)
t = np.linspace(0, 90, 200)
ax.plot(t, np.sin(np.radians(t)) * math.pi / 180, color=C["main"], lw=1.4, label=r"density $\sin\theta$")
ax.axvline(math.degrees(1.0), color=C["aux"], ls=":", lw=1.0)
ax.text(math.degrees(1.0) + 1, 0.0010, "mean 57.3", fontsize=8.5, color=C["aux"], ha="left")
vals = [(name, row[1]) for name, row in g.TABLE1.items()]
cols = (C["main"], C["tangent"], C["third"], C["normal"])
place = {"RGB": (1.0, 0.0198, "left"), "RGB + comp.": (-1.0, 0.0196, "right"),
         "+ adapt. comp.": (-1.8, 0.0158, "right"), "+ depth oracle": (1.0, 0.0150, "left")}
for (name, a), c in zip(vals, cols):
    dx, y, ha = place[name]
    ax.axvline(a, color=c, lw=1.5)
    ax.text(a + dx, y, f"{name}\n{a:.2f}", fontsize=8, color=c, ha=ha)
ax.set_xlim(0, 90)
ax.set_ylim(0, 0.023)
ax.set_xlabel(r"angle to the true normal $\theta$ (degrees)")
ax.set_ylabel("density per degree")
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)
print("random normals: mean %.3f, median %.3f; P(angle <= v) for the table:" % (ang.mean(), np.median(ang)),
      {name: round(1 - math.cos(math.radians(a)), 4) for name, a in vals})
