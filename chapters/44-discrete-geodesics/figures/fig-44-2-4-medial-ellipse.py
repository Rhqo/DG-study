"""Figure 44.2.4: the medial axis of the ellipse and of an inscribed polygon (Section 44.2, Proposition 44.2.13).

(a) the ellipse (2 cos t, sin t) (GUIDELINES 2.5 running example); its medial axis, the open segment
    {(x, 0) : |x| < 3/2} (red); medial balls (blue circles) centred at x = 0, +-0.75, +-1.5; the ball at the endpoint
    (3/2, 0) has radius 1/2 = b^2/a, the osculating circle at the vertex (2, 0).
(b) the inscribed 16-gon with vertices (2 cos t_k, sin t_k), t_k = 2 pi k / 16; its medial axis sampled on a grid of
    spacing 0.004 (red dots: points whose two nearest boundary points on different edges are at distances within one
    grid spacing). Branches run to all 16 vertices.
Self-checks: ch44.ellipse_closest gives two closest points on the segment and one off it (sample points); the
radius at (x, 0) for |x| <= 3/2 equals sqrt(1 - x^2/3) (Proposition 44.2.13); every polygon vertex has sampled
medial-axis points within 0.03.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-2-4-medial-ellipse.py
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
a, b = 2.0, 1.0
xs = np.array([0.0, 0.75, 1.5, -0.75, -1.5])
d, k = ch44.ellipse_closest(np.stack([xs, 0 * xs], 1))
assert (k[np.abs(xs) < 1.5 - 1e-9] == 2).all()
assert np.allclose(d, np.sqrt(1 - xs ** 2 / 3), atol=1e-7)
d2, k2 = ch44.ellipse_closest(np.array([[1.7, 0.0], [0.3, 0.25], [-1.0, -0.4]]))
assert (k2 == 1).all()
print("radius at the endpoint (3/2, 0):", round(float(d[2]), 6))

t = 2 * np.pi * np.arange(16) / 16
poly = np.stack([a * np.cos(t), b * np.sin(t)], 1)
Pm, rm = ch44.polygon_medial_samples(poly, 0.004)
for v in poly:
    assert np.linalg.norm(Pm - v, axis=1).min() < 0.03
print(f"16-gon: {len(Pm)} medial-axis samples")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.4))
tt = np.linspace(0, 2 * np.pi, 400)
for ax in (axa, axb):
    ax.set_aspect("equal")
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_axis_off()
axa.plot(a * np.cos(tt), b * np.sin(tt), color=col["main"], lw=1.6)
for x0, r0 in zip(xs, d):
    axa.plot(x0 + r0 * np.cos(tt), r0 * np.sin(tt), color=col["tangent"], lw=0.8)
    axa.plot(x0, 0, "o", color=col["tangent"], ms=2.5)
axa.plot([-1.5, 1.5], [0, 0], color="#C00000", lw=2.2)
axa.annotate(r"$(\frac{3}{2}, 0)$", xy=(1.5, 0), xytext=(1.0, -0.85), fontsize=10,
             arrowprops=dict(arrowstyle="-", color=col["main"], lw=0.6))
axa.set_title("(a) ellipse $(2\\cos t, \\sin t)$", fontsize=11)
axb.fill(poly[:, 0], poly[:, 1], color=col["surface"], alpha=0.35, lw=0)
axb.plot(*np.vstack([poly, poly[:1]]).T, color=col["main"], lw=1.2)
axb.plot(Pm[:, 0], Pm[:, 1], ".", color="#C00000", ms=0.6, rasterized=True)
axb.plot(poly[:, 0], poly[:, 1], "o", color=col["main"], ms=2.5)
axb.set_title("(b) inscribed 16-gon", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)
