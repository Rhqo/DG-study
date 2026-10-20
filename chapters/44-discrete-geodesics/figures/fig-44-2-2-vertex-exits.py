"""Figure 44.2.2: which continuations through a vertex are locally shortest (Section 44.2, Proposition 44.2.5).

Three vertex stars of m equilateral triangles: m = 5 (Theta = 5 pi/3, delta = pi/3, cone), m = 6 (flat), m = 8
(Theta = 8 pi/3, delta = -2 pi/3, saddle). Angle diagram: the directions at the vertex are drawn on a circle with the
intrinsic angle alpha in [0, Theta) mapped linearly to the drawn angle 2 pi alpha / Theta (so angles are to scale
relative to Theta, not Euclidean: schematic of the exact angle bookkeeping). Gray spokes: the edges of the star.
Black arrow: the incoming path, arriving along alpha_in = 0. Orange: the outgoing directions alpha_out with
theta_l = alpha_out - alpha_in >= pi and theta_r = Theta - theta_l >= pi, i.e. alpha_out in [pi, Theta - pi]
(empty, a single ray, an arc of width Theta - 2 pi = -delta).
Self-checks: the angle sums of ch44.equilateral_star(m) equal m pi / 3; the admissible arc has width max(0, -delta).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-2-2-vertex-exits.py
"""

import pathlib
import sys

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
for ax, m, name in zip(axes, (5, 6, 8), ("cone", "flat", "saddle")):
    V, F = ch44.equilateral_star(m)
    Theta = dm.angle_sums(V, F)[0]
    assert abs(Theta - m * np.pi / 3) < 1e-9
    delta = 2 * np.pi - Theta
    to_draw = lambda a: 2 * np.pi * a / Theta + np.pi  # noqa: E731  (incoming from the left: alpha = 0 at angle pi)
    tt = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(tt), np.sin(tt), color=col["aux"], lw=0.6)
    for k in range(m):
        a = to_draw(k * np.pi / 3 + np.pi / 6)
        ax.plot([0, np.cos(a)], [0, np.sin(a)], color=col["aux"], lw=0.6, ls="--")
    # incoming path along alpha_in = 0
    a0 = to_draw(0.0)
    ax.annotate("", xy=(0, 0), xytext=(1.25 * np.cos(a0), 1.25 * np.sin(a0)),
                arrowprops=dict(arrowstyle="-|>", color=col["main"], lw=1.6))
    lo, hi = np.pi, Theta - np.pi
    width = max(0.0, hi - lo)
    assert abs(width - max(0.0, -delta)) < 1e-9
    if hi > lo + 1e-9:
        aa = np.linspace(to_draw(lo), to_draw(hi), 60)
        ax.fill(np.concatenate([[0], 0.95 * np.cos(aa), [0]]), np.concatenate([[0], 0.95 * np.sin(aa), [0]]),
                color=col["accent"], alpha=0.35, lw=0)
        for a in (lo, hi):
            ax.plot([0, 1.2 * np.cos(to_draw(a))], [0, 1.2 * np.sin(to_draw(a))], color=col["accent"], lw=2.0)
    elif abs(hi - lo) < 1e-9:
        a = to_draw(lo)
        ax.plot([0, 1.25 * np.cos(a)], [0, 1.25 * np.sin(a)], color=col["accent"], lw=2.2)
    else:
        ax.text(0.15, -0.35, "no exit", fontsize=11, color=col["normal"])
    ax.plot(0, 0, "o", color=col["main"], ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_axis_off()
    frac = {5: r"5\pi/3", 6: r"2\pi", 8: r"8\pi/3"}[m]
    dlab = {5: r"\pi/3", 6: r"0", 8: r"-2\pi/3"}[m]
    ax.set_title(f"{name}: $\\Theta = {frac}$\n$\\delta = {dlab}$", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)
