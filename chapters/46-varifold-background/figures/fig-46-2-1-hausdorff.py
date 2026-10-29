"""Figure 46.2.1: covering the ellipse (2 cos t, sin t) by small balls, the idea behind the Hausdorff measure.

(a) k = ceil(L / delta) balls of diameter delta = 1 centred on the ellipse at equal arc-length spacing L / k
(L = 9.6884 is the length); (b) the same with delta = 0.25 (39 balls).
(c) the sum  k * omega_d (delta / 2)^d  of such a cover against delta for d = 0, 1, 2 (log-log): it blows up for
d = 0, tends to the length L for d = 1 and tends to 0 for d = 2.

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/figures/fig-46-2-1-hausdorff.py
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

import var46 as v  # noqa: E402

C = dgfig.COLORS
L = v.ellipse_length_total()
assert abs(L - 9.688448) < 1e-6, L

# arc-length parametrisation by inverting the cumulative length on a fine grid
tg = np.linspace(0, 2 * np.pi, 20001)
sg = np.concatenate([[0.0], np.cumsum(0.5 * (v.ellipse_speed(tg[1:]) + v.ellipse_speed(tg[:-1])) * np.diff(tg))])
sg *= L / sg[-1]


def cover(delta):
    k = math.ceil(L / delta)
    s = (np.arange(k) + 0.5) * L / k
    t = np.interp(s, sg, tg)
    return k, v.ellipse(t)


def check_cover(delta):
    """every point of the ellipse lies in some ball of radius delta / 2."""
    k, X = cover(delta)
    P = v.ellipse(np.linspace(0, 2 * np.pi, 4000))
    d = np.min(np.linalg.norm(P[:, None, :] - X[None, :, :], axis=2), axis=1)
    return d.max() <= delta / 2 + 1e-12


for dl, kk in ((1.0, 10), (0.25, 39)):
    k, _ = cover(dl)
    assert k == kk and check_cover(dl), (dl, k)

fig = plt.figure(figsize=(6.4, 4.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.35], hspace=0.45, wspace=0.12)
tt = np.linspace(0, 2 * np.pi, 400)
for j, (dl, title) in enumerate(((1.0, r"(a) $\delta = 1$: 10 balls"), (0.25, r"(b) $\delta = 0.25$: 39 balls"))):
    ax = fig.add_subplot(gs[0, j])
    k, X = cover(dl)
    for x in X:
        ax.add_patch(Circle(x, dl / 2, facecolor=C["region"], alpha=0.25, edgecolor=C["tangent"], lw=0.7))
    ax.plot(*v.ellipse(tt).T, color=C["main"], lw=1.4)
    ax.plot(X[:, 0], X[:, 1], ".", color=C["tangent"], ms=3)
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.55, 1.55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)

ax = fig.add_subplot(gs[1, :])
deltas = np.geomspace(0.01, 2.0, 60)
styles = {0: (C["normal"], r"$d = 0$"), 1: (C["main"], r"$d = 1$"), 2: (C["tangent"], r"$d = 2$")}
sums = {}
for d, (col, lab) in styles.items():
    ks = np.array([math.ceil(L / dl) for dl in deltas])
    sums[d] = ks * v.OMEGA[d] * (deltas / 2) ** d
    ax.loglog(deltas, sums[d], color=col, lw=1.5, label=lab)
ax.axhline(L, color=C["aux"], lw=0.8, ls="--")
ax.text(0.0115, L * 1.25, r"$L = 9.688$", fontsize=10, ha="right", va="bottom")
ax.plot([1.0, 0.25], [10.0, 9.75], "o", color=C["main"], ms=4)
ax.invert_xaxis()
ax.set_xlabel(r"$\delta$ (ball diameter)")
ax.set_ylabel(r"$\sum_j\,\omega_d\,(\mathrm{diam}\,C_j/2)^d$")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.set_title(r"(c) the sum for $d = 0, 1, 2$", fontsize=12)

# self-checks on the three curves
assert sums[0][0] > 900 and abs(sums[1][0] - L) < 0.02 and sums[2][0] < 0.1
dgfig.save(fig, __file__)
print(f"L = {L:.4f}; d=1 sums at delta=1, 0.25: 10, 9.75; d=2 at 0.01: {sums[2][0]:.4f}; d=0 at 0.01: {sums[0][0]:.0f}")
