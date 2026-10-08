"""Figure 30.1.3: functions that satisfy the eikonal equation almost everywhere and vanish on S, but are not the SDF.

(a) 1D, S = {-1, 1}, the "object" is the interval [-1, 1] (shaded):
    black  = SDF f = |x| - 1 (negative inside),
    orange = zigzag: |x| - 1 for |x| >= 0.6, 0.2 - |x| for |x| < 0.6 (extra zeros at +-0.2),
    blue dashed = unsigned distance ||x| - 1| (no sign; not differentiable on S).
    All three have |f'| = 1 except at finitely many kinks and vanish at +-1: their eikonal loss is 0.
(b) 2D radial version on [-1.6, 1.6]^2: f = r - 1 (r >= 0.7), 0.4 - r (r < 0.7). |grad f| = 1 except on the kink circle
    r = 0.7 (orange dashed) and at the origin; the zero set is the circle r = 1 AND a spurious circle r = 0.4.
    Vermillion arrows: grad f on the two zero circles (outward on r = 1, inward on r = 0.4).
Self-checks: |f'| = 1 a.e.; the zero sets are as stated.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-1-3-eikonal-not-sdf.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS

x = np.linspace(-2, 2, 4001) + 1.234567e-7
sdf = np.abs(x) - 1
zig = np.where(np.abs(x) >= 0.6, np.abs(x) - 1, 0.2 - np.abs(x))
uns = np.abs(np.abs(x) - 1)
for fz in (sdf, zig, uns):
    d = np.diff(fz) / np.diff(x)
    assert np.mean(np.abs(np.abs(d) - 1) < 1e-6) > 0.998
zz = x[np.where(np.diff(np.sign(zig)) != 0)[0]]
assert np.allclose(np.sort(np.abs(zz)), [0.2, 0.2, 1.0, 1.0], atol=2e-3)


def frad(r):
    return np.where(r >= 0.7, r - 1, 0.4 - r)


rr = np.linspace(0, 1.6, 16001) + 1.234567e-7
zr = rr[np.where(np.diff(np.sign(frad(rr))) != 0)[0]]
assert np.allclose(zr, [0.4, 1.0], atol=2e-4)

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.6, 8.0), gridspec_kw=dict(height_ratios=[1.0, 1.35], hspace=0.3))

# (a) 1D
axa.axvspan(-1, 1, color=C["region"], alpha=0.18, lw=0)
axa.axhline(0, color=C["aux"], lw=0.8)
axa.plot(x, uns, color=C["tangent"], lw=2.0, ls=(0, (5, 3)), label="unsigned", zorder=3)
axa.plot(x, zig, color=C["accent"], lw=2.4, label="zigzag", zorder=4)
axa.plot(x, sdf, color=C["main"], lw=1.8, label="SDF", zorder=5)
axa.plot([-1, 1], [0, 0], "o", color=C["main"], ms=8, zorder=6)
axa.plot([-0.2, 0.2], [0, 0], "o", color=C["accent"], ms=8, mec=C["main"], mew=0.8, zorder=6)
axa.annotate(r"spurious zeros $\pm 0.2$", xy=(0.22, -0.03), xytext=(1.45, -0.62), fontsize=11.5, ha="center",
             va="center", arrowprops=dict(arrowstyle="->", lw=0.9, color=C["main"]),
             bbox=dict(fc="white", ec="none", pad=1.0))
axa.text(-1.0, -0.35, r"$S$", fontsize=13, ha="center")
axa.text(1.0, -0.35, r"$S$", fontsize=13, ha="center")
axa.set_xlim(-2, 2)
axa.set_ylim(-1.1, 1.1)
axa.set_xlabel(r"$x$")
axa.set_ylabel(r"$f(x)$")
axa.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=3, fontsize=11.5, handlelength=2.2,
           borderaxespad=0.1, columnspacing=1.2)
axa.set_title(r"(a) 1D: $|f'| = 1$ a.e. and $f(\pm 1) = 0$", fontsize=13.5, pad=26)

# (b) 2D radial
h = 0.008
gx = np.arange(-1.6, 1.6 + h / 2, h)
X, Y = np.meshgrid(gx, gx)
R = np.hypot(X, Y)
F = frad(R)
pm = axb.pcolormesh(X, Y, F, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=1.27), shading="auto",
                    rasterized=True)
tt = np.linspace(0, 2 * np.pi, 400)
axb.plot(np.cos(tt), np.sin(tt), color=C["main"], lw=2.2)
axb.plot(0.4 * np.cos(tt), 0.4 * np.sin(tt), color=C["main"], lw=2.2)
axb.plot(0.7 * np.cos(tt), 0.7 * np.sin(tt), color=C["accent"], lw=1.6, ls=(0, (4, 3)))
arrow_kw = dict(arrowstyle="-|>", lw=1.4, mutation_scale=11, shrinkA=0, shrinkB=0)
for k in range(8):
    a = 2 * np.pi * k / 8 + np.pi / 8
    u = np.array([np.cos(a), np.sin(a)])
    axb.annotate("", xy=1.0 * u + 0.32 * u, xytext=1.0 * u, arrowprops=dict(color=C["normal"], **arrow_kw))
    if k % 2 == 0:
        axb.annotate("", xy=0.4 * u - 0.22 * u, xytext=0.4 * u, arrowprops=dict(color=C["normal"], **arrow_kw))
axb.text(0.0, 0.0, r"$f>0$", fontsize=10, ha="center", va="center")      # the bubble: "outside" sign inside the object
axb.text(0.0, 1.45, r"$S$: $r = 1$", fontsize=12.5, ha="center", bbox=dict(fc="white", ec="none", pad=1.0))
axb.annotate("spurious zero set\n" + r"$r = 0.4$", xy=(-0.28, -0.28), xytext=(-0.8, -1.25), fontsize=11.5,
             ha="center", arrowprops=dict(arrowstyle="->", lw=0.9, color=C["main"]),
             bbox=dict(fc="white", ec="none", pad=1.0))
axb.annotate("kink " + r"$r = 0.7$", xy=(0.495, -0.495), xytext=(1.1, -1.35), fontsize=11.5, ha="center",
             arrowprops=dict(arrowstyle="->", lw=0.9, color=C["accent"]), bbox=dict(fc="white", ec="none", pad=1.0))
axb.set_aspect("equal")
axb.set_xlim(-1.6, 1.6)
axb.set_ylim(-1.6, 1.6)
axb.set_xlabel(r"$x$")
axb.set_ylabel(r"$y$")
axb.set_title(r"(b) 2D: $|\nabla f| = 1$ a.e., $f = 0$ on $S$, not the SDF", fontsize=13.5)
cb = fig.colorbar(pm, ax=axb, fraction=0.045, pad=0.03, ticks=[-0.4, 0, 0.4, 0.8, 1.2])
cb.set_label(r"$f$")

dgfig.save(fig, __file__)
