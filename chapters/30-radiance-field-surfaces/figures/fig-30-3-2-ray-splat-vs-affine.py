"""Figure 30.3.2: ray-splat intersection (exact) versus the affine (EWA-style) footprint, in a 2D cross-section.

Flatland: camera center O = (0, 0) looking along +z, image line z = 1 with coordinate xi = x/z (focal length 1).
Splat: a 1D Gaussian on a line through c = (0.8, 2.0) with scale s = 0.5, tilted so that its normal makes 60 degrees
with the direction to the camera. G(u) = exp(-u^2/2) on P(u) = c + s u t.
(a) geometry: the splat segment |u| <= 2.5 shaded by G, its points P(+-1), P(+-2) (dots), the exact rays from O through
    them (orange) and where they meet the image line (orange ticks); blue ticks = the affine footprint xi_0 + k xi'(0),
    k = +-1, +-2, which is what a linearization of the projection at the center gives.
(b) the footprint on the image line: exact G(u(xi)) (orange, what 2DGS evaluates by ray-splat intersection) and the
    affine Gaussian exp(-(xi - xi_0)^2 / (2 xi'(0)^2)) (blue dashed).
Self-checks: numbers against verify/v-30-3-surfels-2dgs.py.

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-3-2-ray-splat-vs-affine.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
S = 0.5
c = np.array([0.8, 2.0])
to_cam = -c / np.linalg.norm(c)
ph = np.radians(60.0)
n = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]]) @ to_cam
t = np.array([n[1], -n[0]])


def Pt(u):
    return c[None] + S * np.atleast_1d(u)[:, None] * t[None]


def xi_of_u(u):
    p = Pt(u)
    return p[:, 0] / p[:, 1]


def u_of_xi(xi):
    return -(c[0] - c[1] * xi) / (S * (t[0] - t[1] * xi))


xi0 = c[0] / c[1]
dxi = (xi_of_u(1e-6) - xi_of_u(-1e-6))[0] / 2e-6
ks = np.array([-2.0, -1.0, 1.0, 2.0])
ex = xi_of_u(ks)
af = xi0 + dxi * ks
assert np.allclose(ex, [0.6057, 0.5166, 0.2408, 0.0102], atol=1e-4)
assert abs(abs(dxi) - 0.13463) < 1e-5

fig, (axa, axb) = plt.subplots(2, 1, figsize=(5.8, 8.2), gridspec_kw=dict(height_ratios=[1.25, 1.0], hspace=0.32))

# (a) geometry
uu = np.linspace(-2.5, 2.5, 200)
pts = Pt(uu)
segs = np.stack([pts[:-1], pts[1:]], 1)
lc = LineCollection(segs, cmap="Blues", norm=plt.Normalize(-0.3, 1.0), linewidths=7, zorder=3)
lc.set_array(np.exp(-0.5 * ((uu[:-1] + uu[1:]) / 2) ** 2))
axa.add_collection(lc)
axa.plot([-0.15, 1.0], [1, 1], color=C["main"], lw=1.6, zorder=2)
axa.text(-0.17, 0.72, "image\nline", fontsize=11)
axa.plot([0], [0], "o", color=C["main"], ms=7)
axa.text(0.04, -0.02, r"$O$", fontsize=13, va="top")
for k, xe, xa in zip(ks, ex, af):
    p = Pt(k)[0]
    axa.plot([0, p[0]], [0, p[1]], color=C["accent"], lw=0.9, zorder=1)
    axa.plot([p[0]], [p[1]], "o", color=C["accent"], ms=5, mec=C["main"], mew=0.6, zorder=5)
    axa.plot([xe, xe], [1.0, 1.07], color=C["accent"], lw=2.4, zorder=4)
    axa.plot([xa, xa], [0.93, 1.0], color=C["tangent"], lw=2.4, zorder=4)
    axa.text(p[0] + 0.05, p[1] + 0.02, rf"$u = {int(k):+d}$", fontsize=11, zorder=6)
axa.plot([xi0, xi0], [0.9, 1.1], color=C["main"], lw=1.0, ls=":")
axa.plot([0, c[0]], [0, c[1]], color=C["aux"], lw=0.8, ls=(0, (4, 3)))
axa.plot([c[0]], [c[1]], "o", color=C["main"], ms=5)
axa.annotate("", xy=c + 0.35 * n, xytext=c, arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=1.5, mutation_scale=12))
axa.text(*(c + 0.38 * n + np.array([-0.08, -0.08])), r"$t_w$", color=C["normal"], fontsize=13)
axa.text(0.70, 1.07, "exact", fontsize=11, color=C["accent"], ha="left", va="bottom")
axa.text(0.70, 0.93, "affine", fontsize=11, color=C["tangent"], ha="left", va="top")
axa.set_aspect("equal")
axa.set_xlim(-0.2, 1.75)
axa.set_ylim(-0.12, 2.75)
axa.set_xlabel(r"$x$")
axa.set_ylabel(r"$z$ (depth)")
axa.set_title(r"(a) a tilted splat ($60^\circ$, $s = 0.5$, depth 2) and its rays", fontsize=13)

# (b) footprints
xi = np.linspace(-0.05, 0.8, 4001)
Gex = np.exp(-0.5 * u_of_xi(xi) ** 2)
Gaf = np.exp(-0.5 * ((xi - xi0) / dxi) ** 2)
axb.plot(xi, Gex, color=C["accent"], lw=2.4, label="exact (ray-splat intersection)")
axb.plot(xi, Gaf, color=C["tangent"], lw=2.0, ls=(0, (5, 3)), label="affine (linearized at the center)")
axb.axvline(xi0, color=C["main"], lw=0.9, ls=":")
for xe in ex:
    axb.plot([xe, xe], [0, 0.06], color=C["accent"], lw=2.4)
for xa in af:
    axb.plot([xa, xa], [-0.06, 0.0], color=C["tangent"], lw=2.4)
axb.text(xi0 + 0.012, 0.05, r"$\xi_0$", fontsize=13)
axb.set_xlim(-0.05, 0.8)
axb.set_ylim(-0.08, 1.12)
axb.set_xlabel(r"image coordinate $\xi = x/z$")
axb.set_ylabel(r"footprint $G$")
axb.legend(loc="upper left", frameon=False, fontsize=10.5, bbox_to_anchor=(0.0, 1.0))
axb.set_title(r"(b) same peak, different shape: max difference $0.18$", fontsize=13)
dgfig.save(fig, __file__)
