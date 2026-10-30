"""Figure 49.3.3: where split puts new Gaussians -- paper eq. (10) as printed versus the 3DGS split (Section 49.3).

Gaussian at q = (2.8, 0, 0) on the torus T_{2,0.8}: normal u3 = e_x, w1 = e_z (tau_1 = -1.25), w2 = e_y (tau_2 = -1/2.8),
xi_min = 0.001, and the warm-up shape of eq. (5) with s_nbr = 0.2: scales (0.1871, 0.0535, 0.001) along (w2, w1, u3).
  eq. (10):    mu' = mu + (rho_2/tau_2) w2 + (rho_1/tau_1) w1 + rho_3 xi_min u3,  rho ~ N(0, I)
  3DGS split:  mu' ~ N(mu, Sigma)  (official implementation: samples = normal(0, scales), rotated, added to mu)
(a) 300 samples of each in 3D next to the torus.  (b) histograms of the distance from 20000 samples to the torus.

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/figures/fig-49-3-3-split.py
"""
import sys
from pathlib import Path

import numpy as np

import dgfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import geo49  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

C = dgfig.COLORS
q = np.array([2.8, 0.0, 0.0])
u3, w1, w2 = np.array([1.0, 0, 0]), np.array([0.0, 0, 1]), np.array([0.0, 1, 0])
t1, t2 = -1.25, -1 / 2.8
S = geo49.warmup_scales(t1, t2, 0.2)                      # (s1 along w2, s2 along w1, s3 along u3)
Rm = np.stack([w2, w1, u3], axis=1)                      # columns r1, r2, r3 as in eq. (5)
n_big = 20000
A = geo49.split_samples_eq10(q, w1, w2, u3, t1, t2, n_big, seed=3)
rng = np.random.default_rng(4)
B = q + (rng.normal(size=(n_big, 3)) * S) @ Rm.T
dA, dB = geo49.torus_distance(A), geo49.torus_distance(B)
mA, mB = np.median(dA), np.median(dB)
dispA, dispB = np.median(np.linalg.norm(A - q, axis=1)), np.median(np.linalg.norm(B - q, axis=1))
print(f"median distance to torus: eq.(10) {mA:.3f}, 3DGS split {mB:.4f}; median displacement {dispA:.2f}, {dispB:.3f}")
assert abs(mA - 0.83) < 0.01 and abs(mB - 0.0050) < 0.0005
assert abs(dispA - 2.06) < 0.01 and abs(dispB - 0.14) < 0.01

fig = plt.figure(figsize=(7.8, 3.6))
ax = dgfig.axes3d(fig, pos=121, elev=30, azim=-20)
uu, vv = np.meshgrid(np.linspace(-np.pi / 2, np.pi / 2, 40), np.linspace(-np.pi, np.pi, 40))
T = geo49.torus_xyz(vv, uu)
dgfig.surface(ax, T[..., 0], T[..., 1], T[..., 2], grid_every=5, alpha=0.35)
k = 300
ax.scatter(*A[:k].T, s=4, color=C["accent"], depthshade=False, zorder=8, label="eq. (10)")
ax.scatter(*B[:k].T, s=4, color=C["tangent"], depthshade=False, zorder=9, label="3DGS split")
dgfig.point3d(ax, q, size=14)
box = np.array([[-1, -5.5, -2.6], [3.2, 5.5, 2.6]])
dgfig.equal_aspect(ax, box, zoom=1.55)
ax.set_title("(a) 300 new centres of each kind", fontsize=10.5)
ax.legend(loc="lower left", fontsize=8, frameon=False, markerscale=3)

ax2 = fig.add_subplot(122)
bins = np.logspace(-6, 1, 50)
ax2.hist(dB, bins=bins, color=C["tangent"], alpha=0.75, label="3DGS split")
ax2.hist(dA, bins=bins, color=C["accent"], alpha=0.75, label="eq. (10)")
ax2.axvline(mB, color=C["tangent"], lw=1.2, ls="--")
ax2.axvline(mA, color=C["accent"], lw=1.2, ls="--")
ax2.text(mB * 0.8, 2450, f"median\n{mB:.4f}", color=C["tangent"], fontsize=8.5, ha="right", va="top")
ax2.text(14, 2450, f"median\n{mA:.2f}", color=C["accent"], fontsize=8.5, ha="left", va="top")
ax2.set_xscale("log")
ax2.set_xlim(3e-7, 4e2)
ax2.set_xlabel("distance of a new centre to the torus")
ax2.set_ylabel("count (of 20000)")
ax2.set_ylim(0, 2600)
ax2.legend(loc="center left", fontsize=8, frameon=False)
ax2.set_title("(b) distance to the surface", fontsize=10.5)
fig.subplots_adjust(left=0.0, right=0.98, wspace=0.22, bottom=0.15)
dgfig.save(fig, __file__)
