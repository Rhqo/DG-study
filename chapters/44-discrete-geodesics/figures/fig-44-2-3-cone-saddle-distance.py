"""Figure 44.2.3: distance from a point on a polyhedral cone and on a polyhedral saddle (Section 44.2).

The surfaces are infinite polyhedral cones with one vertex v and total angle Theta = 5 pi/3 (cone, delta = pi/3) and
Theta = 8 pi/3 (saddle, delta = -2 pi/3), the same angle sums as the stars of Figure 44.2.2. Points are given in
polar coordinates (r, alpha) at v, alpha in [0, Theta); the source p is at (1, 0). Exact distance: ch44.cone_distance
(unfold the shorter way around; if both angular separations exceed pi, the shortest path runs through v).
Drawing: the angle alpha is mapped to 2 pi alpha / Theta (schematic angles; radii to scale). Isolines of d(p, .)
every 0.25; red ray: points with two shortest paths (cone: alpha = Theta/2); orange wedge: points whose shortest path
passes through v (saddle: pi < alpha < Theta - pi), where the isolines are circles about v.
Self-checks: d is continuous across alpha = pi on the saddle; on the cone ray the two one-sided unfoldings agree.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-2-3-cone-saddle-distance.py
"""

import pathlib
import sys

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3))
Rmax = 2.4
for ax, Theta, name, dl in zip(axes, (5 * np.pi / 3, 8 * np.pi / 3), ("cone", "saddle"),
                               (r"\pi/3", r"-2\pi/3")):
    r = np.linspace(1e-3, Rmax, 260)
    a = np.linspace(0, Theta, 721)
    Rg, Ag = np.meshgrid(r, a, indexing="ij")
    D, sep = ch44.cone_distance(1.0, 0.0, Rg, Ag, Theta)
    # self-checks
    if Theta > 2 * np.pi:
        d1, _ = ch44.cone_distance(1.0, 0.0, 1.3, np.pi - 1e-9, Theta)
        d2, _ = ch44.cone_distance(1.0, 0.0, 1.3, np.pi + 1e-9, Theta)
        assert abs(d1 - d2) < 1e-6
    else:
        rr = 1.7
        left = np.sqrt(1 + rr ** 2 - 2 * rr * np.cos(Theta / 2))
        assert abs(ch44.cone_distance(1.0, 0.0, rr, Theta / 2, Theta)[0] - left) < 1e-12
    ph = 2 * np.pi * Ag / Theta
    X, Y = Rg * np.cos(ph), Rg * np.sin(ph)
    ax.pcolormesh(X, Y, D, cmap="Blues_r", vmin=0, vmax=3.4, shading="gouraud", rasterized=True, zorder=0)
    ax.contour(X, Y, D, levels=np.arange(0.25, 3.4, 0.25), colors=col["main"], linewidths=0.4, zorder=1)
    if Theta < 2 * np.pi:
        phc = np.pi
        ax.plot([0, Rmax * np.cos(phc)], [0, Rmax * np.sin(phc)], color="#C00000", lw=2.0, zorder=3)
        ax.text(-2.35, 0.15, "two shortest\npaths", fontsize=9.5, color="#C00000")
    else:
        lo, hi = 2 * np.pi * np.pi / Theta, 2 * np.pi * (Theta - np.pi) / Theta
        aa = np.linspace(lo, hi, 60)
        ax.fill(np.concatenate([[0], Rmax * np.cos(aa), [0]]), np.concatenate([[0], Rmax * np.sin(aa), [0]]),
                color=col["accent"], alpha=0.35, lw=0, zorder=2)
        ax.text(-2.3, -0.5, "through $v$", fontsize=10, color=col["main"])
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(Rmax * np.cos(tt), Rmax * np.sin(tt), color=col["aux"], lw=0.6)
    ax.plot(0, 0, "o", color=col["main"], ms=4, zorder=4)
    ax.text(0.08, 0.1, r"$v$", fontsize=12)
    ax.plot(1.0, 0.0, "o", color=col["accent"], ms=5, zorder=4)
    ax.text(1.05, 0.12, r"$p$", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_axis_off()
    th = r"5\pi/3" if Theta < 2 * np.pi else r"8\pi/3"
    ax.set_title(f"({'a' if Theta < 2 * np.pi else 'b'}) {name}: $\\Theta = {th}$, $\\delta = {dl}$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.05)
dgfig.save(fig, __file__)
