"""Figure 41.2.1: two normal variations of the torus of revolution T_{R,r}, (R, r) = (2, 0.8), seen in the meridian
plane v = 0 (the xz-plane; the torus is this circle revolved about the z-axis).

x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u), outward unit normal N = (cos u cos v, cos u sin v, sin u)
(GUIDELINES 7: x_u x x_v points inward, so N = -x_u x x_v / |x_u x x_v|).
(a) moving R: dx/dR = (cos v, sin v, 0), normal component phi = <N, dx/dR> = cos u (orange arrows phi N, drawn at
    0.25 times their length); the circle moves to the right (gray dashed: R -> R + 0.25), gaining the light blue crescent
    and losing the light vermilion one.
(b) moving r: dx/dr = N, phi = 1; the circle grows (gray dashed: r -> r + 0.25).
Numbers (Example 41.2.2): dV/dR = 2 pi^2 r^2 = int phi dA, dA/dR = 4 pi^2 r = -int 2 H phi dA, d(int H dA)/dR = -2 pi^2
= -int K phi dA, and dV/dr = 4 pi^2 R r, dA/dr = 4 pi^2 R, d(int H dA)/dr = 0.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-2-1-torus-variations.py
"""
import numpy as np

import dgfig
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

C = dgfig.COLORS
R, r, s = 2.0, 0.8, 0.25
t = np.linspace(0, 2 * np.pi, 721)
circ = lambda c, rad: np.c_[c + rad * np.cos(t), rad * np.sin(t)]  # noqa: E731


def integral(f):
    """int over T_{R,r} of f(u) dA, dA = r (R + r cos u) du dv (midpoint rule in u, exact in v)."""
    u = (np.arange(20000) + 0.5) * 2 * np.pi / 20000
    return 2 * np.pi * np.sum(f(u) * r * (R + r * np.cos(u))) * 2 * np.pi / 20000


H = lambda u: -(R + 2 * r * np.cos(u)) / (2 * r * (R + r * np.cos(u)))  # noqa: E731
K = lambda u: np.cos(u) / (r * (R + r * np.cos(u)))  # noqa: E731
assert abs(integral(np.cos) - 2 * np.pi ** 2 * r ** 2) < 1e-9
assert abs(integral(lambda u: -2 * H(u) * np.cos(u)) - 4 * np.pi ** 2 * r) < 1e-9
assert abs(integral(lambda u: -K(u) * np.cos(u)) + 2 * np.pi ** 2) < 1e-9
assert abs(integral(lambda u: 0 * u + 1) - 4 * np.pi ** 2 * R * r) < 1e-9
assert abs(integral(lambda u: -2 * H(u)) - 4 * np.pi ** 2 * R) < 1e-9 and abs(integral(K)) < 1e-9

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, (c1, r1), title in [(axes[0], (R + s, r), "(a)"), (axes[1], (R, r + s), "(b)")]:
    old, new = circ(R, r), circ(c1, r1)
    pts = np.stack(np.meshgrid(np.linspace(0.8, 3.5, 500), np.linspace(-1.3, 1.3, 400)), -1).reshape(-1, 2)
    in_old = MPath(old).contains_points(pts)
    in_new = MPath(new).contains_points(pts)
    ax.scatter(*pts[in_new & ~in_old].T, s=1.2, color=C["region"], alpha=0.35, lw=0, rasterized=True)
    ax.scatter(*pts[in_old & ~in_new].T, s=1.2, color=C["normal"], alpha=0.12, lw=0, rasterized=True)
    ax.plot(*old.T, color=C["main"], lw=1.6)
    ax.plot(*new.T, color=C["aux"], lw=1.0, ls="--")
    ax.plot(R, 0, "o", color=C["aux"], ms=3)
    for u in np.arange(16) * 2 * np.pi / 16:
        p = np.array([R + r * np.cos(u), r * np.sin(u)])
        n = np.array([np.cos(u), np.sin(u)])
        phi = np.cos(u) if title == "(a)" else 1.0
        if abs(phi) > 1e-9:
            ax.annotate("", p + 0.25 * phi * n, p, arrowprops=dict(arrowstyle="-|>", color=C["accent"], lw=1.3,
                                                                  mutation_scale=8))
    ax.set_aspect("equal")
    ax.set_xlim(0.85, 3.45)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel(r"$x$")
    ax.set_title(title, loc="left", fontsize=12)
axes[0].set_ylabel(r"$z$")
axes[0].text(0.95, 1.08, r"$\varphi=\cos u$", color=C["accent"], fontsize=11)
axes[1].text(0.95, 1.08, r"$\varphi=1$", color=C["accent"], fontsize=11)
axes[0].text(2.9, -1.2, r"$R\to R+\varepsilon$", color=C["aux"], fontsize=10)
axes[1].text(2.9, -1.2, r"$r\to r+\varepsilon$", color=C["aux"], fontsize=10)
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.14, top=0.92, wspace=0.12)
dgfig.save(fig, __file__)
