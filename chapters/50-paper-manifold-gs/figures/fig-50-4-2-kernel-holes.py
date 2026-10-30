"""Figure 50.4.2: a fixed-bandwidth kernel varifold distance hides holes (the converse of (23) fails, [ManifoldGS26] App. B).

V = the area measure of the flat square [-1, 1]^2 (all planes equal), W = the same square minus a disk of radius r at
the centre; both normalised to mass 1. Solid: MMD_k(V, W) for the kernel (22) with sigma = 0.05 and sigma = 0.2 (the
plane part of the kernel is 1 because the planes agree). Dashed: MMD_k between V and the same square moved by 0.01 along
its normal (a 0.5 percent shift of its width). The marked radius r* is where the two are equal: holes smaller than r*
look closer to V than this harmless shift.

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/figures/fig-50-4-2-kernel-holes.py
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
rs = np.geomspace(0.01, 0.5, 28)
fig, ax = plt.subplots(figsize=(4.8, 3.4))
out = {}
for sig, c in ((0.05, C["tangent"]), (0.2, C["normal"])):
    vals = np.array([g.mmd_hole(r, sig) for r in rs])
    shift = g.mmd_shift(0.01, sig)
    rstar = g.crossing_radius(sig, shift)
    out[sig] = (shift, rstar, g.mmd_hole(0.1, sig), g.mmd_hole(0.2, sig))
    # self-checks: small holes cost about their area fraction pi r^2 / 4; the curve is increasing
    assert abs(vals[0] / (math.pi * rs[0] ** 2 / 4) - 1) < 0.05 and np.all(np.diff(vals) > 0)
    ax.loglog(rs, vals, "-", color=c, lw=1.6, label=rf"hole, $\sigma = {sig}$")
    ax.axhline(shift, color=c, ls="--", lw=0.9)
    ax.plot([rstar], [shift], "o", color=c, ms=5)
    ax.annotate(rf"$r^* = {rstar:.2f}$", xy=(rstar, shift), xytext=(rstar * (0.33 if sig == 0.05 else 1.25), shift * (2.2 if sig == 0.05 else 0.22)),
                fontsize=9.5, color=c, arrowprops=dict(arrowstyle="-", lw=0.6, color=c))
ax.loglog(rs, math.pi * rs ** 2 / 4, ":", color=C["aux"], lw=0.9, label=r"area share $\pi r^2/4$")
ax.text(0.011, 0.0125 * 1.35, "shift by 0.01", fontsize=9, color=C["aux"])
ax.set_xlabel(r"hole radius $r$")
ax.set_ylabel(r"$\mathrm{MMD}_k$")
ax.set_ylim(5e-5, 0.3)
ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
ax.set_xticklabels(["0.01", "0.02", "0.05", "0.1", "0.2", "0.5"])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.legend(loc="upper left", fontsize=8.5, frameon=False)
fig.tight_layout()
dgfig.save(fig, __file__)
for sig, (shift, rstar, m1, m2) in out.items():
    print(f"sigma={sig}: shift 0.01 -> {shift:.5f}, r* = {rstar:.4f}, hole r=0.1 -> {m1:.5f}, r=0.2 -> {m2:.5f}")
