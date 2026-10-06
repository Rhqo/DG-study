"""Figure 0.8.4: circumference of a geodesic circle of radius ρ in the three model spaces (Section 0.8).

Unit sphere (K = 1): 2π sin ρ (0 ≤ ρ ≤ π); Euclidean plane (K = 0): 2πρ; hyperbolic plane (K = −1): 2π sinh ρ.
Self-checks: in U² the geodesic circle of radius ρ about (0, 1) is the Euclidean circle with centre (0, cosh ρ) and radius
sinh ρ; its points are at hyperbolic distance ρ from (0, 1) and its g-length, computed numerically, is 2π sinh ρ.
On the unit sphere the circle of latitude at colatitude ρ has length 2π sin ρ.

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-8-4-circle-growth.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW


def d_hyp(p, q):
    """distance in U² = {y > 0}, g = (dx² + dy²)/y²"""
    return np.arccosh(1 + ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) / (2 * p[1] * q[1]))


# self-checks (§12.1)
t = np.linspace(0, 2 * np.pi, 20001)
for rho in (0.5, 1.0, 2.0):
    cx, cy, R = 0.0, np.cosh(rho), np.sinh(rho)
    pts = np.stack([cx + R * np.cos(t), cy + R * np.sin(t)], 1)
    assert np.max(np.abs(d_hyp(np.array([0.0, 1.0]), pts.T) - rho)) < 1e-10
    mid = (pts[1:] + pts[:-1]) / 2
    L = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1) / mid[:, 1])
    assert abs(L - 2 * np.pi * np.sinh(rho)) < 1e-5
    lat = np.stack([np.sin(rho) * np.cos(t), np.sin(rho) * np.sin(t)], 1)
    assert abs(np.sum(np.linalg.norm(np.diff(lat, axis=0), axis=1)) - 2 * np.pi * np.sin(rho)) < 1e-6

fig, ax = plt.subplots(figsize=(5.2, 3.3))
r = np.linspace(0, 3.0, 400)
rs = np.linspace(0, np.pi, 400)
ax.plot(r, 2 * np.pi * np.sinh(r), color=C["third"], lw=LW["main"])
ax.plot(r, 2 * np.pi * r, color=C["main"], lw=LW["main"])
ax.plot(rs, 2 * np.pi * np.sin(rs), color=C["accent"], lw=LW["main"])
ax.set_xlim(0, 3.2)
ax.set_ylim(0, 45)
ax.set_xlabel(r"$\rho$", fontsize=12)
ax.set_ylabel(r"$L(\rho)$", fontsize=12)
ax.set_xticks([0, 1, 2, 3, np.pi])
ax.set_xticklabels([r"$0$", r"$1$", r"$2$", r"$3$", r"$\pi$"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(2.25, 36, r"$2\pi\sinh\rho$  $(K = -1)$", fontsize=11, color=C["third"], ha="right")
ax.text(3.0, 21.5, r"$2\pi\rho$  $(K = 0)$", fontsize=11, color=C["main"], ha="right")
ax.text(2.6, 6.6, r"$2\pi\sin\rho$  $(K = 1)$", fontsize=11, color=C["accent"], ha="center")
fig.subplots_adjust(left=0.13, right=0.98, top=0.97, bottom=0.16)
dgfig.save(fig, __file__)
