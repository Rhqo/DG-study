"""Figure 46.1.2: staircases approach the diagonal, but their length and tangents do not.

(a) staircases S_n from (0, 0) to (1, 1) with n = 2, 4, 16 steps and the diagonal D.
(b) Hausdorff distance d_H(S_n, D) = 1 / (sqrt2 n) against n (log-log); the length of S_n stays 2 while the
length of D is sqrt2 = 1.414.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-1-2-staircase.py
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
ns = np.array([1, 2, 4, 8, 16, 32, 64])
dH = []
for n in ns:
    S = v.staircase(n)
    assert abs(v.polyline_length(S, closed=False) - 2.0) < 1e-12
    if n <= 16:
        d = v.hausdorff_polyline_to_segment(S, (0, 0), (1, 1))
        assert abs(d - 1 / (np.sqrt(2) * n)) < 1e-6, (n, d)
    dH.append(1 / (np.sqrt(2) * n))
dH = np.array(dH)

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
ax = axes[0]
styles = {2: (C["aux"], 1.0, "--"), 4: (C["tangent"], 1.3, "-"), 16: (C["main"], 1.5, "-")}
for n, (col, lw, ls) in styles.items():
    S = v.staircase(n)
    ax.plot(S[:, 0], S[:, 1], color=col, lw=lw, ls=ls, label=fr"$S_{{{n}}}$")
ax.plot([0, 1], [0, 1], color=C["normal"], lw=1.6, label=r"$D$")
ax.set_aspect("equal")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.6)
ax.set_title("(a) staircases", fontsize=12)

ax = axes[1]
ax.loglog(ns, dH, "o-", color=C["tangent"], lw=1.4, ms=4, label=r"$d_H(S_n, D)$")
ax.set_xlabel(r"$n$")
ax.set_xticks(ns)
ax.set_xticklabels([str(n) for n in ns])
ax.minorticks_off()
ax.set_ylim(0.008, 1.5)
ax.text(1.1, 0.03, r"length$(S_n) = 2$" + "\n" + r"length$(D) = \sqrt{2}$", fontsize=11, va="bottom")
ax.legend(loc="upper right", frameon=False, fontsize=10)
ax.set_title("(b) distance vs. length", fontsize=12)

dgfig.save(fig, __file__)
