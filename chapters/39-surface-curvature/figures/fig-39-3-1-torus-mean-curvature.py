"""Figure 39.3.1: K and the mean curvature vector H N on the torus of revolution.

Torus (R, r) = (2, 0.8). (a) the meridian v = 0 in the xz-plane (circle of radius r about (R, 0)), colored by
K = cos u/(r (R + r cos u)) (RdBu_r, centered at 0), with the mean curvature vector H N at 24 points (orange, drawn at
half length). H N is the same for both choices of N; with N = x_u x x_v/|...| (inward) H > 0 everywhere because
R > 2r. (b) kappa1 = 1/r, kappa2 = cos u/(R + r cos u), H = (kappa1 + kappa2)/2 (inward N) and K against u.

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-3-1-torus-mean-curvature.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

import ddg39  # noqa: E402

C = dgfig.COLORS
R, r = ddg39.R_TORUS, ddg39.r_TORUS
u = np.linspace(-np.pi, np.pi, 721)
k1 = np.full_like(u, 1 / r)
k2 = np.cos(u) / (R + r * np.cos(u))
H = (k1 + k2) / 2
K = k1 * k2
# self-checks against the numeric Weingarten map, and H N independent of the normal
for uu in (0.0, 1.0, 2.5, np.pi):
    Kn, Hn = ddg39.KH(ddg39.torus, uu, 0.0)
    assert abs(Kn - np.cos(uu) / (r * (R + r * np.cos(uu)))) < 1e-6
    assert abs(Hn - (R + 2 * r * np.cos(uu)) / (2 * r * (R + r * np.cos(uu)))) < 1e-6
    Nin = ddg39.unit_normal(ddg39.torus, uu, 0.0)
    # outward normal: -N, H changes sign, H N unchanged
    Kout, Hout = ddg39.KH(lambda a, b: ddg39.torus(a, -b), uu, 0.0)   # reparametrization reversing orientation
    assert abs(Kout - Kn) < 1e-6 and abs(Hout + Hn) < 1e-6
    Nout = ddg39.unit_normal(lambda a, b: ddg39.torus(a, -b), uu, 0.0)
    assert np.allclose(Nout, -Nin, atol=1e-8) and np.allclose(Hout * Nout, Hn * Nin, atol=1e-6)
assert H.min() > 0 and abs(H.max() - 3.6 / 4.48) < 1e-9 and abs(H.min() - 0.4 / 1.92) < 1e-6

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
pts = np.stack([R + r * np.cos(u), r * np.sin(u)], 1)
segs = np.stack([pts[:-1], pts[1:]], 1)
Kmid = 0.5 * (K[:-1] + K[1:])
norm = TwoSlopeNorm(vmin=K.min(), vcenter=0.0, vmax=K.max())
lc = LineCollection(segs, cmap="RdBu_r", norm=norm, lw=4.0)
lc.set_array(Kmid)
ax.add_collection(lc)
uk = np.linspace(-np.pi, np.pi, 25)[:-1]
for a in uk:
    p = np.array([R + r * np.cos(a), r * np.sin(a)])
    Nin = -np.array([np.cos(a), np.sin(a)])
    Ha = (R + 2 * r * np.cos(a)) / (2 * r * (R + r * np.cos(a)))
    ax.annotate("", xy=p + 0.5 * Ha * Nin, xytext=p, arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.3,
                                                                     mutation_scale=9, shrinkA=0, shrinkB=0))
ax.plot(R, 0, "+", color=C["aux"], ms=7)
ax.annotate("", xy=(0.95, -0.95), xytext=(1.35, -0.95), arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.9))
ax.text(1.0, -0.88, r"axis", fontsize=10, color=C["aux"])
ax.text(R + r + 0.1, -0.03, r"$u = 0$", fontsize=10)
ax.text(R - r - 0.45, -0.03, r"$u = \pi$", fontsize=10)
ax.text(R + 0.05, r + 0.1, r"$u = \pi/2$", fontsize=10)
ax.text(R - 0.12, -0.2, r"$H\mathbf{N}$", color=C["accent"], fontsize=12)
cb = fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8)
cb.set_label(r"$K$", fontsize=11)
cb.ax.tick_params(labelsize=9)
cb.set_ticks([-1.0, -0.5, 0.0, 0.4])
dgfig.schematic_axes(ax, (0.9, 3.25), (-1.05, 1.05))
ax.set_title("(a)", loc="left", fontsize=12)

ax = axes[1]
ax.axhline(0, color=C["aux"], lw=0.7)
ax.plot(u, k1, color=C["tangent"], lw=1.6, label=r"$\kappa_1$")
ax.plot(u, k2, color=C["third"], lw=1.6, label=r"$\kappa_2$")
ax.plot(u, H, color=C["accent"], lw=1.8, label=r"$H$")
ax.plot(u, K, color=C["main"], lw=1.8, label=r"$K$")
for a in (-np.pi / 2, np.pi / 2):
    ax.axvline(a, color=C["aux"], lw=0.7, ls=(0, (3, 2)))
ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
ax.set_xlabel(r"$u$")
ax.set_xlim(-np.pi, np.pi)
ax.legend(fontsize=10, loc="lower center", ncol=4, frameon=False, handlelength=1.6)
ax.set_ylim(-1.35, 1.45)
ax.set_title("(b)", loc="left", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.98, bottom=0.14, top=0.92, wspace=0.12)
dgfig.save(fig, __file__)
