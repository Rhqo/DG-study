"""Figure 27.1.3: radial lens distortion is a diffeomorphism only up to the fold radius r* (Section 27.1).

Model (27.1.6) without tangential terms: r_d = r g(r), g = 1 + k1 r^2 + k2 r^4, in normalized image coordinates.
(a) r_d as a function of r for k1 = -0.3 and k2 = 0, 0.03, 0.1. Dots: the fold radius r* where (r g)' = 0
    (r* = 1.054 and 1.214; k2 = 0.1 has no fold). Top axis: the ray angle theta = atan r. Gray dotted line: the
    corner of a 90 deg x 4:3 pinhole image (r = 1.25). The pair r = 0.533 and r = 1.5 (k2 = 0) have the same r_d.
(b) The image of the grid |x| <= 1.2, |y| <= 0.9 (spacing 0.15; a 4:3 image of about 100 deg horizontal FOV) under k1 = -0.3, k2 = 0: grid points with
    r < r* in blue, r > r* in orange. Dashed circle: r_d = r* g(r*) = 0.703, the image of the fold circle.
Self-checks: r* and r_d* values, the two preimages of r_d = 0.4875, det DD = g (r g)' changes sign at r*.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-1-3-distortion-fold.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
K1 = -0.3


def rd(r, k1, k2):
    return r * (1 + k1 * r**2 + k2 * r**4)


def rstar(k1, k2):
    if k2 == 0:
        return np.sqrt(-1 / (3 * k1))
    disc = (3 * k1) ** 2 - 20 * k2
    if disc < 0:
        return None
    return np.sqrt((-3 * k1 - np.sqrt(disc)) / (10 * k2))


rs0 = rstar(K1, 0.0)
assert abs(rs0 - 1.0541) < 1e-4 and abs(rd(rs0, K1, 0) - 0.7027) < 1e-4
rs1 = rstar(K1, 0.03)
assert abs(rs1 - 1.2137) < 1e-3
assert rstar(K1, 0.1) is None
assert abs(rd(1.5, K1, 0) - 0.4875) < 1e-12 and abs(rd(0.53290036, K1, 0) - 0.4875) < 1e-7
# det DD = g (r g)' changes sign at r*
g = lambda r: 1 + K1 * r**2
dg = lambda r: 1 + 3 * K1 * r**2
assert g(rs0 - 1e-3) * dg(rs0 - 1e-3) > 0 > g(rs0 + 1e-3) * dg(rs0 + 1e-3)

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.4, 4.3), gridspec_kw=dict(width_ratios=[1.15, 1.0]))
r = np.linspace(0, 2.0, 801)
styles = [(0.0, C["normal"], "-"), (0.03, C["accent"], "-"), (0.1, C["tangent"], "-")]
for k2, col, ls in styles:
    ax.plot(r, rd(r, K1, k2), color=col, lw=1.8, ls=ls, label=rf"$k_2={k2:g}$")
    rs = rstar(K1, k2)
    if rs is not None:
        ax.plot([rs], [rd(rs, K1, k2)], "o", color=col, ms=6, zorder=6)
        ax.plot([rs, rs], [0, rd(rs, K1, k2)], color=col, lw=0.8, ls=":")
ax.plot(r, r, color=C["aux"], lw=0.8, ls="--")
ax.text(1.55, 1.38, r"$r_d=r$", fontsize=11, color=C["aux"], rotation=33)
ax.axvline(1.25, color=C["aux"], lw=0.8, ls=":")
ax.text(1.27, 0.06, "corner of a\n90$^\\circ$, 4:3 image", fontsize=10, color=C["aux"])
ax.annotate(r"fold $r^*=1.054$ ($46.5^\circ$)", xy=(rs0, rd(rs0, K1, 0)), xytext=(0.12, 0.93), fontsize=11,
            color=C["normal"], arrowprops=dict(arrowstyle="->", color=C["normal"], lw=0.9))
ax.text(rs1 + 0.04, rd(rs1, K1, 0.03) + 0.06, r"$r^*=1.214$", fontsize=11, color=C["accent"])
# the two preimages
ax.plot([0.5329, 1.5], [0.4875, 0.4875], color=C["normal"], lw=0.9, ls="--")
ax.plot([0.5329, 1.5], [0.4875, 0.4875], "s", color=C["normal"], ms=4)
ax.text(0.52, 0.255, r"$r=0.533$ and $r=1.5$" + "\n" + r"give the same $r_d=0.49$", fontsize=10, color=C["normal"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
ax.set_xlim(0, 2.0)
ax.set_ylim(0, 1.8)
ax.set_xlabel(r"undistorted radius $r=|\bar u|$")
ax.set_ylabel(r"distorted radius $r_d$")
ax.legend(loc="upper left", frameon=False, title=r"$k_1=-0.3$", fontsize=10, title_fontsize=10)
top = ax.secondary_xaxis("top", functions=(lambda x: np.degrees(np.arctan(x)), lambda d: np.tan(np.radians(d))))
top.set_ticks([0, 20, 40, 50, 60])
top.set_xlabel(r"ray angle $\theta$ (deg)", fontsize=10)
ax.set_title("(a) radial map $r\\mapsto r_d$", fontsize=12, pad=30)

# ---------------------------------------------------------------- (b)
XM, YM = 1.2, 0.9
glx = np.arange(-XM, XM + 1e-9, 0.15)
gly = np.arange(-YM, YM + 1e-9, 0.15)
sx = np.linspace(-XM, XM, 601)
sy = np.linspace(-YM, YM, 451)


def D(x, y):
    gg = 1 + K1 * (x**2 + y**2)
    return x * gg, y * gg


lines = [(np.full_like(sy, a), sy) for a in glx] + [(sx, np.full_like(sx, b)) for b in gly]
for (x, y) in lines:
    if True:
        rr = np.hypot(x, y)
        xd, yd = D(x, y)
        inside = rr < rs0
        for mask, col, lw in ((inside, C["tangent"], 0.7), (~inside, C["accent"], 1.3)):
            xm, ym = np.where(mask, xd, np.nan), np.where(mask, yd, np.nan)
            bx.plot(xm, ym, color=col, lw=lw)
th = np.linspace(0, 2 * np.pi, 361)
R0 = rd(rs0, K1, 0)
bx.plot(R0 * np.cos(th), R0 * np.sin(th), color="k", lw=1.2, ls="--")
bx.text(0.3, 0.72, r"$r_d=0.703$", fontsize=11)
cx_, cy_ = D(np.array([1.2]), np.array([0.9]))
bx.plot(cx_, cy_, "*", color=C["normal"], ms=11, zorder=8)
bx.annotate("grid corner\n$(1.2,\ 0.9)$, $r=1.5$,\nlands inside", xy=(cx_[0], cy_[0]), xytext=(-0.85, 0.42),
            fontsize=10, color=C["normal"], arrowprops=dict(arrowstyle="->", color=C["normal"], lw=0.9),
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
bx.text(-0.9, -0.95, r"blue: $r<r^*$, orange: $r>r^*$", fontsize=10)
bx.set_xlim(-0.92, 0.92)
bx.set_ylim(-0.98, 0.85)
bx.set_aspect("equal")
bx.set_xlabel(r"distorted $x_d$")
bx.set_ylabel(r"distorted $y_d$")
bx.set_title("(b) image of the grid $|x|\\leq 1.2$, $|y|\\leq 0.9$, $k_2=0$", fontsize=12, pad=30)
fig.tight_layout()
dgfig.save(fig, __file__)
