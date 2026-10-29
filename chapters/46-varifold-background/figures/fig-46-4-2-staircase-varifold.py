"""Figure 46.4.2: the staircase as a varifold, in the chart (position along the diagonal, line angle).

(a) the staircase S_4 (black) with the diagonal D (vermilion), with short ticks showing the line of each piece.
(b) chart: horizontal axis s = (x + y) / sqrt2 in [0, sqrt2] (position along D), vertical axis theta in [0, 180).
The pieces of S_4 lie on theta = 0 (horizontal pieces) and theta = 90 (vertical pieces), alternating in s; the limit
V_infinity of v(S_n, 1) covers both lines with density sqrt2 / 2 each (blue bands); v(D, 1) is the single line
theta = 45 with density 1 (vermilion).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-4-2-staircase-varifold.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
n = 4
S = v.staircase(n)
segs = list(zip(S[:-1], S[1:]))


def s_of(p):
    return (p[0] + p[1]) / math.sqrt(2)


# self-checks: every piece projects onto an s-interval of length 1/(n sqrt2); the two families tile [0, sqrt2] twice
lens = [abs(s_of(b) - s_of(a)) for a, b in segs]
assert np.allclose(lens, 1 / (n * math.sqrt(2)))
assert abs(sum(lens) - math.sqrt(2)) < 1e-12
# mass of S_n per unit s is (length per piece)/(s-length per piece) = sqrt2 in total, sqrt2/2 per family
assert abs((1 / n) / (1 / (n * math.sqrt(2))) / 2 - math.sqrt(2) / 2) < 1e-12

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), gridspec_kw={"width_ratios": [1, 1.5], "wspace": 0.35})
ax = axes[0]
ax.plot([0, 1], [0, 1], color=C["normal"], lw=1.6)
ax.plot(S[:, 0], S[:, 1], color=C["main"], lw=1.6)
for a, b in segs:
    m = 0.5 * (a + b)
    horiz = abs(b[1] - a[1]) < 1e-12
    col = C["tangent"] if horiz else C["third"]
    ax.plot(m[0], m[1], "o", color=col, ms=4)
ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_title(r"(a) $S_4$ and $D$", fontsize=12)

ax = axes[1]
L = math.sqrt(2)
# limit bands
ax.fill_between([0, L], [-6, -6], [6, 6], color=C["tangent"], alpha=0.18, lw=0)
ax.fill_between([0, L], [84, 84], [96, 96], color=C["third"], alpha=0.18, lw=0)
# pieces of S_4
for a, b in segs:
    horiz = abs(b[1] - a[1]) < 1e-12
    th = 0.0 if horiz else 90.0
    ax.plot([s_of(a), s_of(b)], [th, th], color=C["tangent"] if horiz else C["third"], lw=3.0, solid_capstyle="butt")
ax.plot([0, L], [45, 45], color=C["normal"], lw=2.0)
ax.text(L + 0.03, 0, r"$\frac{\sqrt{2}}{2}$", fontsize=12, va="center", ha="left")
ax.text(L + 0.03, 90, r"$\frac{\sqrt{2}}{2}$", fontsize=12, va="center", ha="left")
ax.text(L + 0.03, 45, r"$1$", fontsize=12, va="center", ha="left", color=C["normal"])
ax.set_xlim(0, L + 0.25)
ax.set_ylim(-15, 180)
ax.set_xticks([0, L / 2, L])
ax.set_xticklabels([r"$0$", r"$\frac{\sqrt{2}}{2}$", r"$\sqrt{2}$"])
ax.set_yticks([0, 45, 90, 135, 180])
ax.set_xlabel(r"position along $D$:  $s = (x+y)/\sqrt{2}$")
ax.set_ylabel(r"line angle $\theta$ (degrees)")
ax.set_title(r"(b) in the chart $(s, \theta)$", fontsize=12)

dgfig.save(fig, __file__)
