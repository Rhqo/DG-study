"""Figure 27.3.4: a short baseline makes depth unstable (Section 27.3).

Fronto-parallel stereo in the top view (x, z): camera 1 at (0, 0), camera 2 at (b, 0), both looking along +z, point X at
(0, 10).
(a) Each camera's ray to X with an angular uncertainty of +-0.5 deg (exaggerated; about 7 px at f = 800); the intersection
    of the two wedges is the region of possible positions. Blue: b = 2 (triangulation angle 11.3 deg); orange: b = 0.5
    (2.9 deg). Both drawn on the same axes; the x axis is stretched about 8 times relative to z (aspect 0.12).
(b) Relative depth error sigma_z / z as a function of the triangulation angle gamma (log-log): Monte Carlo (dots, 20000
    samples each, pixel noise sigma_u = 0.5 px in both images, f = 800 px, z = 10) and the first-order formula
    sqrt(2) sigma_u / (f tan gamma) (line). Gray lines: COLMAP's default min triangulation angle 1.5 deg and the
    minimum angle for the initial pair, 16 deg.
Self-checks: the region for b = 0.5 is more than 3.5 times longer along z than for b = 2 (ratio 4.4); Monte Carlo
agrees with the formula within 5%.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-3-4-baseline-depth.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

C = dgfig.COLORS
rng = np.random.default_rng(2733)
Z = 10.0
DELTA = np.radians(0.5)


def line_through(c, ang):
    # direction from camera center c (on z = 0) making angle ang with +z, measured toward +x
    return c, np.array([np.sin(ang), np.cos(ang)])


def intersect(c1, d1, c2, d2):
    A = np.column_stack([d1, -d2])
    s = np.linalg.solve(A, c2 - c1)
    return c1 + s[0] * d1


def region(b):
    c1, c2 = np.array([0.0, 0]), np.array([b, 0.0])
    a1 = 0.0
    a2 = np.arctan2(-b, Z)
    pts = []
    for e1, e2 in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        _, d1 = line_through(c1, a1 + e1 * DELTA)
        _, d2 = line_through(c2, a2 + e2 * DELTA)
        pts.append(intersect(c1, d1, c2, d2))
    return np.array(pts), c1, c2, a1, a2


fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.8, 4.6), gridspec_kw=dict(width_ratios=[0.9, 1.2]))
ext = {}
# orange (b = 0.5) first, blue (b = 2) on top so that the small blue region stays visible
for b, col, lab, alpha, z in ((0.5, C["accent"], r"$b=0.5$ ($\gamma=2.9^\circ$)", 0.35, 2),
                              (2.0, C["tangent"], r"$b=2$ ($\gamma=11.3^\circ$)", 0.55, 3)):
    poly, c1, c2, a1, a2 = region(b)
    ext[b] = np.ptp(poly[:, 1])
    ax.add_patch(Polygon(poly, closed=True, facecolor=col, alpha=alpha, edgecolor=col, lw=1.4, label=lab, zorder=z))
    for e in (-1, 1):                     # camera 2's two rays (+-0.5 deg), in the color of its baseline
        _, d = line_through(c2, a2 + e * DELTA)
        p0, p1 = c2 + 6.5 * d / d[1], c2 + 16.5 * d / d[1]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=0.8, zorder=4)
for e in (-1, 1):                         # camera 1's two rays: the same in both cases (black)
    _, d = line_through(np.zeros(2), e * DELTA)
    p0, p1 = 6.5 * d / d[1], 16.5 * d / d[1]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="k", lw=0.8, zorder=4)
assert ext[0.5] / ext[2.0] > 3.5
ax.plot(0, Z, "k+", ms=10, mew=1.5, zorder=6)
ax.text(0.1, Z - 0.1, r"$X$", fontsize=12, zorder=6)
ax.annotate("camera 1 rays", xy=(0.12, 15.6), xytext=(0.17, 15.2), fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
ax.text(-0.39, 12.35, "camera 2 rays,\n$b=2$", fontsize=10, color=C["tangent"], va="bottom")
ax.text(0.2, 7.3, "camera 2 rays,\n$b=0.5$", fontsize=10, color=C["accent"], va="bottom")
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(7.0, 16.0)
ax.set_aspect(0.12)
ax.set_xlabel(r"$x$ (stretched 8$\times$)")
ax.set_ylabel(r"depth $z$")
ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.95, edgecolor="none")
ax.set_title("(a) possible positions\n(rays $\\pm 0.5^\\circ$)", fontsize=12)

f_, su = 800.0, 0.5
gam = np.radians(np.array([0.5, 1, 1.5, 2, 3, 5, 8, 12, 16, 25, 35]))
mc = []
for g in gam:
    b = Z * np.tan(g)
    d = f_ * b / Z
    dn = d + rng.normal(scale=su, size=20000) - rng.normal(scale=su, size=20000)
    mc.append(np.std(f_ * b / dn) / Z)
mc = np.array(mc)
form = np.sqrt(2) * su / (f_ * np.tan(gam))
assert np.all(np.abs(mc / form - 1) < 0.05), mc / form
gg = np.radians(np.logspace(np.log10(0.4), np.log10(40), 200))
bx.loglog(np.degrees(gg), np.sqrt(2) * su / (f_ * np.tan(gg)), color=C["tangent"], lw=1.8,
          label=r"$\sqrt{2}\,\sigma_u/(f\tan\gamma)$")
bx.loglog(np.degrees(gam), mc, "o", color=C["normal"], ms=5, label="Monte Carlo")
for gdeg, txt in ((1.5, "COLMAP min angle 1.5$^\\circ$:\n3.4%"), (16, "COLMAP initial pair 16$^\\circ$:\n0.31%")):
    bx.axvline(gdeg, color=C["aux"], lw=0.9, ls="--")
    yv = np.sqrt(2) * su / (f_ * np.tan(np.radians(gdeg)))
    bx.text(gdeg * 1.08, yv * (1.6 if gdeg < 5 else 1.8), txt, fontsize=10, color=C["aux"])
bx.set_xlabel(r"triangulation angle $\gamma$ (deg)")
bx.set_ylabel(r"relative depth error $\sigma_z/z$")
bx.legend(loc="lower left", fontsize=10, frameon=False)
bx.set_title(r"(b) $\sigma_u=0.5$ px, $f=800$ px", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
print("extent ratio", ext[0.5] / ext[2.0], "mc/form", np.round(mc / form, 3))
