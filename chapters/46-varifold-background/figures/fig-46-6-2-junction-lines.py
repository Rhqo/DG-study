"""Figure 46.6.2: which junctions of half-lines are curvature varifolds (Proposition 46.6.5).

Four junctions of unit-multiplicity half-lines from the origin, each drawn to length 1:
(a) X: directions 30, 150, 210, 330 degrees (two full lines);
(b) Y: 90, 210, 330 degrees (the 120-degree junction);
(c) T: 0, 90, 180 degrees;
(d) nine half-lines through the vertices of a regular 9-gon (the example of BLM22, Section 4).
A half-line whose opposite half-line is also present is drawn in blue (it belongs to a full line), an unpaired one in
vermilion. The text under each panel says whether the junction is stationary (sum of u_l = 0, Example 46.5.8) and whether
it is a curvature varifold in Hutchinson's sense (every half-line paired, Proposition 46.6.5). The G-linear variation
-sum u_l (x) u_l (x) u_l delta_0 of BLM22 (4.4) vanishes for (a) and (d) only.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-6-2-junction-lines.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
CASES = [
    ("(a) X", [30, 150, 210, 330]),
    ("(b) Y", [90, 210, 330]),
    ("(c) T", [0, 90, 180]),
    ("(d) nine half-lines", [40 * l for l in range(9)]),
]


def units(deg):
    a = np.radians(np.asarray(deg, float))
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def paired(U):
    """for each half-line: is the opposite direction also present?"""
    return np.array([np.any(np.all(np.isclose(U, -u, atol=1e-12), axis=1)) for u in U])


def stationary(U):
    return bool(np.allclose(U.sum(axis=0), 0, atol=1e-12))


def third_moment(U):
    return np.einsum("li,lj,lk->ijk", U, U, U)


# self-checks against Proposition 46.6.5 and BLM22 (4.4)
expect = {"(a) X": (True, True, True), "(b) Y": (True, False, False), "(c) T": (False, False, False),
          "(d) nine half-lines": (True, False, True)}
for name, deg in CASES:
    U = units(deg)
    st, cv, glin = expect[name]
    assert stationary(U) == st, name
    assert bool(paired(U).all()) == cv, name
    assert bool(np.allclose(third_moment(U), 0, atol=1e-12)) == glin, name

fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.6), gridspec_kw={"hspace": 0.55, "wspace": 0.15})
for ax, (name, deg) in zip(axes.ravel(), CASES):
    U = units(deg)
    pr = paired(U)
    for u, p in zip(U, pr):
        col = C["tangent"] if p else C["normal"]
        ax.plot([0, u[0]], [0, u[1]], color=col, lw=1.8 if p else 2.0, solid_capstyle="butt")
    ax.plot(0, 0, "o", color=C["main"], ms=4, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()
    ax.set_title(name, fontsize=12)
    st = "yes" if stationary(U) else "no"
    cv = "yes" if pr.all() else "no"
    ax.text(0.5, -0.06, f"stationary: {st}\ncurvature varifold: {cv}", transform=ax.transAxes,
            ha="center", va="top", fontsize=9.5, linespacing=1.3)

dgfig.save(fig, __file__)
