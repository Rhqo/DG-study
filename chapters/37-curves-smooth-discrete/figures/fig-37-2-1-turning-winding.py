"""Figure 37.2.1: turning number versus winding number.

(a) the figure-eight (sin t, sin t cos t) with the winding numbers of the three regions of its complement;
(b) the tangent angle phi(s) / 2 pi against s / L for five running closed curves: the net rise is the turning number.

Run: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/figures/fig-37-2-1-turning-winding.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import ddg37  # noqa: E402

C = dgfig.COLORS
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.05, 1]})

# (a) the figure-eight and the winding numbers of the regions
tt = np.linspace(0, 2 * np.pi, 600)
F = ddg37.point("figure8", tt)
P400 = ddg37.sample("figure8", 400)[0]
ax.plot(F[:, 0], F[:, 1], color=C["main"], lw=dgfig.LW["main"])
for t0 in (0.35, 2.4, 3.6, 5.6):
    p = ddg37.point("figure8", np.array([t0]))[0]
    v = ddg37.velocity("figure8", np.array([t0]))[0]
    v = 0.16 * v / np.linalg.norm(v)
    ax.annotate("", xy=p + v, xytext=p - v, arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.4,
                                                            mutation_scale=11))
labels = {(0.5, 0.0): -1, (-0.5, 0.0): 1, (0.0, 0.75): 0}
for (x, y), w in labels.items():
    wn = ddg37.winding_number_polygon(P400, np.array([x, y]))
    assert abs(wn - w) < 1e-9
    ax.text(x, y, r"$%+d$" % w if w else r"$0$", ha="center", va="center", fontsize=13,
            color=C["normal"] if w else C["aux"])
ax.plot([0], [0], "o", ms=5, mfc="white", mec=C["main"], zorder=5)
ax.text(0.0, -0.78, r"$\mathrm{rot} = 0$", ha="center", fontsize=11)
ax.set_aspect("equal")
ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-0.9, 0.95)
ax.set_axis_off()
ax.set_title("(a)", fontsize=12, loc="left")

# (b) tangent angle lifts
styles = [("circle", r"circle", C["aux"], "-", 1), ("flower", r"flower", C["tangent"], "-", 1),
          ("limacon", r"limaçon $a=\frac{1}{2}$", C["normal"], "-", 2), ("figure8", r"figure-eight", C["main"], "-", 0)]
m = 4000
for name, lab, col, ls, k in styles:
    L = ddg37.smooth_length(name)
    s = np.linspace(0, L, m + 1)
    t = ddg37.t_of_s(name, s, 200000)
    phi = ddg37.angle_lift(ddg37.velocity(name, t))
    y = (phi - phi[0]) / (2 * np.pi)
    assert abs(y[-1] - k) < 1e-6, (name, y[-1])
    bx.plot(s / L, y, color=col, ls=ls, lw=1.6, label=lab + r" ($%d$)" % k)
# the clockwise circle (rotation index -1)
s = np.linspace(0, 1, 50)
bx.plot(s, -s, color=C["aux"], ls="--", lw=1.4, label=r"circle, clockwise ($-1$)")
bx.axhline(0, color=C["aux"], lw=0.6)
bx.set_xlabel(r"$s/L$")
bx.set_ylabel(r"$(\varphi(s) - \varphi(0))/2\pi$")
bx.set_yticks([-1, 0, 1, 2])
bx.set_xlim(0, 1)
bx.set_ylim(-1.25, 2.6)
bx.legend(fontsize=9.5, frameon=False, loc="upper left", labelspacing=0.3, borderaxespad=0.2)
bx.set_title("(b)", fontsize=12, loc="left")
for sp2 in ("top", "right"):
    bx.spines[sp2].set_visible(False)

fig.tight_layout()
dgfig.save(fig, __file__)
