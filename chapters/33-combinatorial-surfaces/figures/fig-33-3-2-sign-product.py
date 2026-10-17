"""Figure 33.3.2: relative orientations around a closed chain of triangles (Section 33.3).

The face lists of the untwisted band (lib/ch33.py cylinder_band(6)) and of the Moebius band (dgmesh.mobius_mesh(6))
are drawn unrolled: bottom-row vertex k at (k, 0), top-row vertex 6 + k at (k, 1); the right-hand column x = 6 is
glued to the left-hand column x = 0. Every triangle carries the orientation of its row of F (circular arrow).
The number on each shared edge is the relative orientation o(t, t') of the two triangles that share it
(Definition 33.3.6); the product around the closed chain t_0, t_1, ..., t_11, t_0 is +1 for the band, -1 for the
Moebius band (Proposition 33.3.16).
Self-check: the signs are computed with dgdec.relative_orientation; exactly one sign is -1 (the seam) for the
Moebius band and none for the band.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-3-2-sign-product.py``
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
BLUE, RED, GREEN = C["tangent"], C["normal"], C["third"]
n = 6


def positions(F, twisted):
    """Drawn position of each corner of each triangle (the last quad uses the glued column x = n)."""
    out = []
    for t, f in enumerate(F):
        k = t // 2
        pts = []
        for v in f:
            if k == n - 1 and v in ((n, 0) if twisted else (0, n)):
                y = {True: {n: 0.0, 0: 1.0}, False: {0: 0.0, n: 1.0}}[twisted][v]
                pts.append((float(n), y))
            else:
                pts.append((float(v % n), 0.0 if v < n else 1.0))
        out.append(np.array(pts))
    return out


def draw(ax, F, twisted, title):
    F = [tuple(int(v) for v in f) for f in F]
    P = positions(F, twisted)
    signs = [dd.relative_orientation(F[t], F[(t + 1) % len(F)]) for t in range(len(F))]
    for t, T in enumerate(P):
        ax.add_patch(Polygon(T, closed=True, facecolor=C["region"], alpha=0.22, edgecolor="#555555", lw=0.9, zorder=1))
        area = 0.5 * ((T[1, 0] - T[0, 0]) * (T[2, 1] - T[0, 1]) - (T[1, 1] - T[0, 1]) * (T[2, 0] - T[0, 0]))
        assert area > 0                                        # every row is counterclockwise in the drawing
        c = T.mean(0)
        th = np.linspace(np.deg2rad(-50), np.deg2rad(220), 40)
        arc = c + 0.13 * np.c_[np.cos(th), np.sin(th)]
        ax.plot(*arc.T, color=BLUE, lw=0.9, zorder=3)
        ax.add_patch(FancyArrowPatch(arc[-2], arc[-1], arrowstyle="-|>", mutation_scale=7, color=BLUE, lw=0.9,
                                     zorder=3))
        ax.text(*(c + np.array([0.0, -0.005])), rf"$t_{{{t}}}$", fontsize=10, ha="center", va="center",
                color="#333333", zorder=4)
    # signs on the shared edges inside the strip
    for t in range(len(F) - 1):
        shared = set(F[t]) & set(F[t + 1])
        pa = [p for v, p in zip(F[t], P[t]) if v in shared]
        mid = (pa[0] + pa[1]) / 2
        s = signs[t]
        vertical = abs(pa[0][0] - pa[1][0]) < 1e-9
        lo = pa[0] if pa[0][1] < pa[1][1] else pa[1]
        hi = pa[1] if pa[0][1] < pa[1][1] else pa[0]
        spot = mid if vertical else lo + 0.25 * (hi - lo)      # a quarter up the diagonal, away from the arcs
        ax.text(*spot, "+" if s > 0 else "\u2212", color=GREEN if s > 0 else RED, fontsize=12,
                ha="center", va="center", zorder=5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.9))
    # the seam: the column x = 0 and the column x = n are the same edge
    s = signs[-1]
    col = GREEN if s > 0 else RED
    for x in (0.0, float(n)):
        ax.plot([x, x], [0, 1], color=col, lw=3.0, zorder=2, solid_capstyle="butt")
        ax.text(x + (-0.22 if x == 0 else 0.22), 0.5, "+" if s > 0 else "\u2212", color=col, fontsize=17,
                ha="center", va="center", fontweight="bold")
    # direction of the seam edge as traversed by t_0 (left) and by t_{2n-1} (right)
    for t, x in ((0, 0.0), (len(F) - 1, float(n))):
        f, T = F[t], P[t]
        for a in range(3):
            i, j = f[a], f[(a + 1) % 3]
            if {i, j} == ({0, n}):
                p, q = T[a], T[(a + 1) % 3]
                ax.add_patch(FancyArrowPatch(p + 0.3 * (q - p), p + 0.72 * (q - p), arrowstyle="-|>",
                                             mutation_scale=12, color="black", lw=1.4, zorder=6))
    for k in range(n):
        ax.text(k, -0.13, rf"${k}$", ha="center", va="top", fontsize=11)
        ax.text(k, 1.11, rf"${n + k}$", ha="center", va="bottom", fontsize=11)
    bot, topl = (n, 0) if twisted else (0, n)
    ax.text(n, -0.13, rf"${bot}$", ha="center", va="top", fontsize=11, color=col)
    ax.text(n, 1.11, rf"${topl}$", ha="center", va="bottom", fontsize=11, color=col)
    prod = int(np.prod(signs))
    ax.set_title(title + rf"$\quad$product $= {'+1' if prod > 0 else '-1'}$", fontsize=12)
    dgfig.schematic_axes(ax, xlim=(-0.45, n + 0.45), ylim=(-0.35, 1.35))
    return signs


fig, (axa, axb) = plt.subplots(2, 1, figsize=(6.4, 4.0))
sa = draw(axa, ch33.cylinder_band(n)[1], False, "(a) band")
sb = draw(axb, dm.mobius_mesh(n)[1], True, "(b) Möbius band")
assert sa.count(-1) == 0 and sb.count(-1) == 1 and sb[-1] == -1
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, hspace=0.35)
dgfig.save(fig, __file__)
