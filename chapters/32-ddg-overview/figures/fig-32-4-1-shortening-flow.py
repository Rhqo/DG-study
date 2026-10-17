"""Figure 32.4.1: curve shortening flow of the flower curve r(t) = 1 + 0.3 cos 5t (Definition 32.4.1), computed with a
fine discretization (400 points, ddg32.smooth_flow). (a) snapshots; (b) enclosed area against time, which decreases
exactly like A(0) - 2 pi t for the smooth flow (Exercise 32.4.2).

Run: PYTHONPATH=tools python3 chapters/32-ddg-overview/figures/fig-32-4-1-shortening-flow.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgdec as dd  # noqa: E402
import ddg32  # noqa: E402

C = dgfig.COLORS
X0 = ddg32.sample("flower", 4000)[0]
A0 = np.pi * (1 + 0.3 ** 2 / 2)                       # area of r = 1 + 0.3 cos 5t: (1/2) int r^2 dt
assert abs(ddg32.polygon_area(X0) - A0) < 1e-5
times = [0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.48]
curves = ddg32.smooth_flow(X0, times, n=400)
areas = np.array([ddg32.polygon_area(X) for X in curves])
assert np.max(np.abs(areas - (A0 - 2 * np.pi * np.array(times)))) < 1e-3      # dA/dt = -2 pi
th_last = dd.turning_angles(curves[-1], True)
assert np.all(th_last > 0)                                                   # convex at t = 0.48
first_convex = None
for X, t in zip(curves, times):
    if np.all(dd.turning_angles(X, True) > 0):
        first_convex = t
        break
print("first snapshot that is convex: t =", first_convex)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.8, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
grays = np.linspace(0.0, 0.62, len(times))[::-1]
for X, t, g in zip(curves, times, grays):
    Xc = np.vstack([X, X[:1]])
    if t == 0:
        ax.plot(Xc[:, 0], Xc[:, 1], color=C["main"], lw=dgfig.LW["main"])
    else:
        ax.plot(Xc[:, 0], Xc[:, 1], color=C["tangent"], lw=1.0, alpha=0.35 + 0.65 * (1 - g / 0.62))
ax.plot(0, 0, "+", color=C["aux"], ms=6)
ax.text(1.12, 0.62, r"$t = 0$", fontsize=11)
ax.annotate(r"$t = 0.48$", xy=(curves[-1][0, 0], 0.0), xytext=(0.62, -1.15), fontsize=11,
            arrowprops=dict(arrowstyle="-", lw=0.6, color=C["tangent"]), color=C["tangent"])
ax.set_aspect("equal")
ax.set_xlim(-1.4, 1.5)
ax.set_ylim(-1.35, 1.35)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12)

tt = np.linspace(0, 0.52, 100)
bx.plot(tt, A0 - 2 * np.pi * tt, color=C["aux"], lw=1.0, ls="--", label=r"$A(0) - 2\pi t$")
bx.plot(times, areas, "o", color=C["tangent"], ms=4.5, label="computed")
bx.set_xlabel(r"time $t$")
bx.set_ylabel("enclosed area")
bx.legend(frameon=False, fontsize=10)
bx.set_title("(b)", fontsize=12)
for sp_ in ("top", "right"):
    bx.spines[sp_].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
print("areas", np.round(areas, 4), "A0", A0, "extinction", A0 / (2 * np.pi))
