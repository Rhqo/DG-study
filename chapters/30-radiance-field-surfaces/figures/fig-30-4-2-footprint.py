"""Figure 30.4.2: the footprint of one pixel and the spread of depth inside it, as a function of the tilt (Section 30.4).

Flatland, camera at O = (0, 0) looking along +z. A plane (line) through (0, 3) whose normal makes the angle theta with
the viewing ray (theta = 0: fronto-parallel).
(a) near the surface the two rays bounding one pixel (u = -1/2 and u = +1/2) are almost parallel (their angle is
    1/f rad), so locally one pixel is a strip of width w = d/f (drawn with w = 1); the camera is far below. Planes through the strip's center
    with theta = 0, 60, 80 deg cut it in the colored segments (the footprint, w/cos(theta)); the gray bracket is the
    depth range inside the pixel for 80 deg (w tan(theta)).
(b) exact footprint length and depth range inside one pixel at the image center, in units of d/f, for f = 500 px
    (dots, ray-plane intersections) and the first-order formulas 1/cos(theta) and tan(theta) (lines), log scale.
Self-checks: dots match the formulas to 0.2 %.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-4-2-footprint.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
D0 = 3.0


def hit(u, th, f):
    """intersection of the ray through pixel u (focal f) with the plane through (0, D0) tilted by th."""
    d = D0 * f * np.cos(th) / (f * np.cos(th) - u * np.sin(th))
    return np.array([u / f * d, d])


fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 9.0), gridspec_kw=dict(height_ratios=[1.45, 1.0], hspace=0.42))

# (a) near the surface the two rays of one pixel are almost parallel: a strip of width w = d/f (here w = 1)
w = 1.0
cols = {0: C["tangent"], 60: C["third"], 80: C["accent"]}
for xs_ in (-w / 2, w / 2):
    axa.plot([xs_, xs_], [-3.4, 3.4], color=C["main"], lw=1.4)
axa.fill_betweenx([-3.4, 3.4], -w / 2, w / 2, color=C["region"], alpha=0.18, lw=0)
for th_deg, col in cols.items():
    th = np.radians(th_deg)
    tdir = np.array([np.cos(th), np.sin(th)])
    s_ext = 1.6
    axa.plot([-s_ext * tdir[0], s_ext * tdir[0]], [-s_ext * tdir[1], s_ext * tdir[1]], color=col, lw=1.0, alpha=0.7)
    s_half = (w / 2) / np.cos(th)
    axa.plot([-s_half * tdir[0], s_half * tdir[0]], [-s_half * tdir[1], s_half * tdir[1]], color=col, lw=5,
             solid_capstyle="butt", label=rf"$\theta = {th_deg}^\circ$: footprint $= {1 / np.cos(th):.2f}\,w$")
th80 = np.radians(80)
zr = (w / 2) * np.tan(th80)
xb = 0.95
axa.plot([xb, xb], [-zr, zr], color=C["aux"], lw=2.2)
for zz in (-zr, zr):
    axa.plot([xb - 0.06, xb + 0.06], [zz, zz], color=C["aux"], lw=2.2)
    axa.plot([w / 2, xb], [zz, zz], color=C["aux"], lw=0.7, ls=":")
axa.text(xb + 0.1, 0.0, "depth range\nat " + r"$80^\circ$:" + "\n" + rf"$\tan 80^\circ\,w = {np.tan(th80):.1f}\,w$",
         va="center", fontsize=11)
axa.annotate("", xy=(w / 2, -3.0), xytext=(-w / 2, -3.0), arrowprops=dict(arrowstyle="<->", lw=1.0))
axa.text(0.0, -3.35, r"$w = d/f$", ha="center", va="top", fontsize=12)
axa.text(-0.62, 3.0, "rays of\none pixel", fontsize=11, ha="right", va="top")
axa.annotate("", xy=(-2.35, -3.1), xytext=(-2.35, -1.1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color=C["main"]))
axa.text(-2.22, -2.3, "to the\ncamera", fontsize=11, ha="left", va="center")
axa.set_aspect("equal")
axa.set_xlim(-2.6, 3.0)
axa.set_ylim(-3.9, 3.6)
axa.set_xticks([])
axa.set_yticks([])
axa.legend(loc="upper center", frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01), ncol=1)
axa.set_title("(a) one pixel near the surface: footprint and depth range", fontsize=13)

f = 500.0
ths = np.linspace(0, 88, 300)
td = np.array([0, 20, 40, 60, 70, 80, 85, 88])
foot = []
spread = []
for t in td:
    th = np.radians(t)
    a, b = hit(-0.5, th, f), hit(0.5, th, f)
    foot.append(np.linalg.norm(b - a) / (D0 / f))
    spread.append(abs(b[1] - a[1]) / (D0 / f))
foot, spread = np.array(foot), np.array(spread)
assert np.allclose(foot, 1 / np.cos(np.radians(td)), rtol=2e-3)
assert np.allclose(spread[1:], np.tan(np.radians(td[1:])), rtol=2e-3) and spread[0] < 1e-12
axb.semilogy(ths, 1 / np.cos(np.radians(ths)), color=C["main"], lw=2.2, label=r"footprint $1/\cos\theta$")
axb.semilogy(ths[1:], np.tan(np.radians(ths[1:])), color=C["accent"], lw=2.2, label=r"depth range $\tan\theta$")
axb.semilogy(td, foot, "o", color=C["main"], mfc="white", ms=6)
axb.semilogy(td[1:], spread[1:], "o", color=C["accent"], mfc="white", ms=6)
axb.set_xlim(0, 90)
axb.set_ylim(0.1, 60)
axb.set_xlabel(r"tilt $\theta$ (deg)")
axb.set_ylabel(r"in units of $d/f$")
axb.legend(loc="upper left", frameon=False)
axb.text(84, 11.5, r"$85^\circ$: $\times 11.4$", ha="right", fontsize=11)
axb.set_title(r"(b) one pixel at $f = 500$ px: exact (dots) and formulas", fontsize=13)
dgfig.save(fig, __file__)
