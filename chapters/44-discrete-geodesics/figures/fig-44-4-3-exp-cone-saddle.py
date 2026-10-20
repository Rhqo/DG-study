"""Figure 44.4.3: the discrete exponential map near a cone vertex and near a saddle vertex (Section 44.4, Propositions 44.4.12 and 44.4.13).

Infinite polyhedral cones with one vertex v: (a) Theta = 5 pi/3 (delta = pi/3), (b) Theta = 8 pi/3 (delta = -2 pi/3);
p at distance 1 from v. Straightest geodesics from p (ch44.cone_ray) leave at angles psi = pi + k pi/12 (k = -11..12,
measured in a planar unfolding around p; psi = pi aims at v) and have length 2.6. Drawing: polar coordinates (r, alpha)
at v with the angle alpha mapped to 2 pi alpha / Theta (schematic angles, as in Figure 44.2.3). Thin gray: rays;
orange: the ray through v (it leaves with equal angles Theta/2); red dots in (a): the meeting points of the pairs
psi = pi -+ beta, beta = 0.1, 0.2, 0.35, 0.5 (ch44.cone_meeting_point), all on the ray alpha = Theta/2; orange wedge in
(b): the sector pi < alpha < Theta - pi that no ray except the orange one enters.
Self-checks: in (a) each pair meets at the same point of the cone with the same length; in (b) no ray other than the
one through v has |alpha| >= pi.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-4-3-exp-cone-saddle.py
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
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.4))
L = 2.6
for ax, Theta, lab in zip(axes, (5 * np.pi / 3, 8 * np.pi / 3), ("(a) cone", "(b) saddle")):
    delta = 2 * np.pi - Theta
    draw = lambda r, a: (r * np.cos(2 * np.pi * a / Theta), r * np.sin(2 * np.pi * a / Theta))  # noqa: E731
    if Theta > 2 * np.pi:
        aa = np.linspace(np.pi, Theta - np.pi, 50)
        X, Y = draw(2.7 + 0 * aa, aa)
        ax.fill(np.concatenate([[0], X, [0]]), np.concatenate([[0], Y, [0]]), color=col["accent"], alpha=0.25, lw=0)
    for k in range(-11, 13):
        psi = np.pi + k * np.pi / 12
        r, a = ch44.cone_ray(psi, Theta, L)
        if k == 0:
            continue
        if Theta > 2 * np.pi:
            assert np.abs(a).max() < np.pi
        ax.plot(*draw(r, a), color=col["aux"], lw=0.7)
    r, a = ch44.cone_ray(np.pi, Theta, L)
    ax.plot(*draw(r, a), color=col["accent"], lw=2.0)
    if Theta < 2 * np.pi:
        for beta in (0.1, 0.2, 0.35, 0.5):
            t, s = ch44.cone_meeting_point(beta, delta)
            p1 = np.array([1 + t * np.cos(np.pi - beta), t * np.sin(np.pi - beta)])
            p2 = np.array([1 + t * np.cos(np.pi + beta), t * np.sin(np.pi + beta)])
            a1, a2 = np.arctan2(p1[1], p1[0]), np.arctan2(p2[1], p2[0])
            assert abs(np.linalg.norm(p1) - s) < 1e-12 and abs(np.linalg.norm(p2) - s) < 1e-12
            assert abs(((a1 - a2) % Theta)) < 1e-9 or abs(((a1 - a2) % Theta) - Theta) < 1e-9
            for psi in (np.pi - beta, np.pi + beta):
                rr, al = ch44.cone_ray(psi, Theta, t)
                ax.plot(*draw(rr, al), color=col["tangent"], lw=1.0)
            ax.plot(*draw(s, Theta / 2), "o", color="#C00000", ms=4, zorder=5)
        ax.text(-2.6, 0.25, "rays meet", fontsize=10, color="#C00000")
    else:
        ax.text(-2.6, -0.7, "never\nreached", fontsize=10, color=col["main"])
    ax.plot(0, 0, "o", color=col["main"], ms=4, zorder=6)
    ax.text(0.08, 0.12, r"$v$", fontsize=12)
    ax.plot(1, 0, "o", color=col["accent"], ms=5, zorder=6)
    ax.text(1.05, 0.12, r"$p$", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-2.8, 2.8)
    ax.set_axis_off()
    th = r"5\pi/3" if Theta < 2 * np.pi else r"8\pi/3"
    ax.set_title(f"{lab}: $\\Theta = {th}$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01, wspace=0.04)
dgfig.save(fig, __file__)
