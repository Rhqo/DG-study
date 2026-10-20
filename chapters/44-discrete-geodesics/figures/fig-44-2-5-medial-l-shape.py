"""Figure 44.2.5: the medial axis of an L-shaped polygon (Section 44.2, Example 44.2.15).

Region: [0, 2]^2 minus (1, 2] x (1, 2]; reflex vertex (1, 1). Red dots: medial axis sampled on a grid of spacing
0.0025 (ch44.polygon_medial_samples). Orange curves: the two parabolic arcs (1 - x)^2 = 2y - 1 and (1 - y)^2 = 2x - 1,
2 - sqrt 2 <= x, y <= 1 (equidistant from the reflex vertex and the bottom, resp. left, edge). Black dashed: the
diagonal y = x from (0, 0) to (2 - sqrt 2, 2 - sqrt 2). Blue circle: the medial ball centred at the meeting point
(2 - sqrt 2, 2 - sqrt 2) with radius 2 - sqrt 2; it touches the bottom edge, the left edge and the reflex vertex.
Self-checks: the sampled points near the arcs satisfy the parabola equations to within 0.003; the ball at the meeting
point touches three boundary features at the same distance.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-2-5-medial-l-shape.py
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
L = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
P, r = ch44.polygon_medial_samples(L, 0.0025)
sel = (P[:, 0] > 0.62) & (P[:, 0] < 0.97) & (P[:, 1] < 0.8) & (P[:, 1] > 0.4)
assert np.abs((1 - P[sel, 0]) ** 2 - (2 * P[sel, 1] - 1)).max() < 3e-3
x0 = 2 - np.sqrt(2)
c = np.array([x0, x0])
dmin, _, D = ch44.polygon_closest(c[None], L)
assert abs(dmin[0] - x0) < 1e-12 and abs(np.linalg.norm(c - [1, 1]) - x0) < 1e-12
print(f"meeting point ({x0:.4f}, {x0:.4f}), radius {x0:.4f}; samples {len(P)}")

fig, ax = plt.subplots(figsize=(3.6, 3.4))
ax.fill(L[:, 0], L[:, 1], color=col["surface"], alpha=0.35, lw=0)
ax.plot(*np.vstack([L, L[:1]]).T, color=col["main"], lw=1.2)
ax.plot(P[:, 0], P[:, 1], ".", color="#C00000", ms=0.8, rasterized=True)
xx = np.linspace(x0, 1, 100)
ax.plot(xx, 0.5 * (1 + (1 - xx) ** 2), color=col["accent"], lw=1.6)
ax.plot(0.5 * (1 + (1 - xx) ** 2), xx, color=col["accent"], lw=1.6)
ax.plot([0, x0], [0, x0], color=col["main"], lw=0.9, ls="--")
tt = np.linspace(0, 2 * np.pi, 200)
ax.plot(x0 + x0 * np.cos(tt), x0 + x0 * np.sin(tt), color=col["tangent"], lw=0.9)
ax.plot(1, 1, "o", color=col["main"], ms=3.5)
ax.text(1.04, 1.06, "reflex vertex", fontsize=9)
ax.text(1.0, 0.08, r"$(1-x)^2 = 2y - 1$", fontsize=9, color=col["accent"])
ax.set_aspect("equal")
ax.set_xlim(-0.1, 2.15)
ax.set_ylim(-0.1, 2.1)
ax.set_axis_off()
dgfig.save(fig, __file__)
