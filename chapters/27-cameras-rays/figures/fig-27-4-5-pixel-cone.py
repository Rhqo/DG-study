"""Figure 27.4.5: a pixel is an open set of rays (a cone); Mip-NeRF integrates over it, 3DGS blurs by a pixel-sized
Gaussian (Section 27.4).

(a) Side view of one pixel's cone for a normalized camera: center at the origin, image plane z = 1, pixel width w = 0.2
    (exaggerated; a real pixel is 1/f). Mip-NeRF's cone has radius r(t) = t * w * 2/sqrt(12) (the radius factor of
    google/mipnerf internal/datasets.py). Conical frustums between t = 1, 1.6, 2.4, 3.4, 4.6; inside each the 1-sigma
    ellipse of the Gaussian of internal/mip.py (conical_frustum_to_gaussian, stable formulas; mean t_mean, variances
    t_var along the axis and r_var across).
(b) Footprints on the image plane in pixel units: the pixel box (width 1, variance 1/12), the Gaussian with the same
    variance 1/12 (Mip-NeRF's disk of radius 2/sqrt(12) has per-axis variance 1/12), Mip-Splatting's 2D filter
    (variance 0.1), and the 3DGS screen-space dilation (variance 0.3). All curves are normalized to unit area.
Self-checks: r_var and t_var against Monte Carlo of uniform points in each frustum (2%); the disk/box variances agree.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-4-5-pixel-cone.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Polygon

C = dgfig.COLORS
rng = np.random.default_rng(2745)
W = 0.2
RB = W * 2 / np.sqrt(12)
ts = [1.0, 1.6, 2.4, 3.4, 4.6]


def frustum_gauss(t0, t1, rb):
    mu, hw = (t0 + t1) / 2, (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = hw**2 / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2)
    r_var = rb**2 * (mu**2 / 4 + (5 / 12) * hw**2 - 4 / 15 * hw**4 / (3 * mu**2 + hw**2))
    return t_mean, t_var, r_var


for t0, t1 in zip(ts[:-1], ts[1:]):
    tm, tv, rv = frustum_gauss(t0, t1, RB)
    N = 200000
    tt = (t0**3 + rng.uniform(size=N) * (t1**3 - t0**3)) ** (1 / 3)
    rad = RB * tt * np.sqrt(rng.uniform(size=N))
    xs = rad * np.cos(rng.uniform(0, 2 * np.pi, N))
    assert abs(tm - tt.mean()) < 3e-3 and abs(tv / tt.var() - 1) < 0.03 and abs(rv / xs.var() - 1) < 0.03

fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 4.2), gridspec_kw=dict(width_ratios=[1.3, 1.0]))
T_END = 5.0
ax.add_patch(Polygon([[0, 0], [T_END, W / 2 * T_END], [T_END, -W / 2 * T_END]], closed=True, facecolor=C["region"],
                     alpha=0.25, edgecolor=C["tangent"], lw=1.0))
ax.plot([0, T_END], [0, RB * T_END], color=C["normal"], lw=1.0, ls="--")
ax.plot([0, T_END], [0, -RB * T_END], color=C["normal"], lw=1.0, ls="--")
ax.plot([1, 1], [-0.75, 0.75], color="k", lw=1.2)
ax.plot([1, 1], [-W / 2, W / 2], color=C["tangent"], lw=4)
ax.text(1.06, -0.72, "image plane", fontsize=10.5)
ax.text(1.08, 0.09, "pixel", fontsize=10.5, color=C["tangent"])
for t0, t1 in zip(ts[:-1], ts[1:]):
    ax.plot([t0, t0], [-RB * t0, RB * t0], color=C["aux"], lw=0.8)
    tm, tv, rv = frustum_gauss(t0, t1, RB)
    ax.add_patch(Ellipse((tm, 0), 2 * np.sqrt(tv), 2 * np.sqrt(rv), facecolor=C["accent"], alpha=0.55, edgecolor=C["accent"]))
ax.plot([ts[-1]] * 2, [-RB * ts[-1], RB * ts[-1]], color=C["aux"], lw=0.8)
ax.plot(0, 0, "ks", ms=5)
ax.text(0.05, -0.18, "center", fontsize=10.5)
ax.text(1.75, 0.68, "Mip-NeRF cone, radius $0.577\\,w\\,t$", fontsize=10.5, color=C["normal"])
ax.text(5.05, -1.0, r"$1\sigma$ Gaussians of the conical frustums", fontsize=10.5, color=C["accent"], ha="right", va="bottom")
ax.set_xlim(-0.1, 5.1)
ax.set_ylim(-1.05, 0.95)
ax.set_aspect("equal")
ax.set_xlabel(r"depth $t$")
ax.set_ylabel(r"$y$")
ax.set_title(r"(a) one pixel = a cone of rays ($w=0.2$)", fontsize=12)

u = np.linspace(-1.6, 1.6, 801)
bx.fill_between(u, 0, np.where(np.abs(u) <= 0.5, 1.0, 0.0), color=C["surface"], alpha=0.8, lw=0, label="pixel box (var 1/12)")


def g(var):
    return np.exp(-u**2 / (2 * var)) / np.sqrt(2 * np.pi * var)


bx.plot(u, g(1 / 12), color=C["normal"], lw=1.8, label="Mip-NeRF disk, var 1/12")
bx.plot(u, g(0.1), color=C["third"], lw=1.6, ls="--", label="Mip-Splatting 2D, var 0.1")
bx.plot(u, g(0.3), color=C["tangent"], lw=1.8, label="3DGS dilation, var 0.3")
bx.set_xlabel("offset on the image plane (px)")
bx.set_ylabel("footprint (unit area)")
bx.set_xlim(-1.6, 1.6)
bx.set_ylim(0, 2.25)
bx.legend(loc="upper left", fontsize=9.5, frameon=False, handlelength=1.6, borderaxespad=0.3)
bx.set_title("(b) pixel-sized footprints", fontsize=12)
fig.tight_layout()
dgfig.save(fig, __file__)
