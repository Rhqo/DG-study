"""Figure 36.5.2: the signed distance from an edge to the circumcenter is (|e|/2) cot(theta) (Section 36.5,
Lemma 36.5.9).

Edge e from x_i = (-1, 0) to x_j = (1, 0) (|e| = 2, midpoint m_e = 0) and the opposite vertex p above the edge.
(a) p = (0.35, 1.25): theta = 74.7 deg (acute), the circumcenter c = (0, h) lies on the same side as p, h = cot(theta)
    = 0.274 > 0.
(b) p = (0.3, 0.4): theta = 133.2 deg (obtuse), the circumcenter lies on the other side of e, h = cot(theta) =
    -0.9375 < 0.
Gray dashed: the circumcircle; orange: the segment m_e c on the perpendicular bisector; the right triangle
(m_e, x_j, c) has the angle theta at c when theta is acute (panel a).
Self-check: c is equidistant from x_i, x_j, p; c_y = (p_x^2 + p_y^2 - 1) / (2 p_y) = cot(theta) (Lemma 36.5.9);
the angle at c in the triangle (m_e, x_j, c) equals theta in (a); dgdec.circumcenters agrees.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-5-2-cotan-geometry.py``
"""

import dgdec as dd

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, Polygon  # noqa: E402

xi, xj = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
cases = [np.array([0.35, 1.25]), np.array([0.3, 0.4])]
info = []
for p in cases:
    u, w = xi - p, xj - p
    theta = np.arccos(u @ w / np.linalg.norm(u) / np.linalg.norm(w))
    c = dd.circumcenters(np.array([xi, xj, p]), np.array([[0, 1, 2]]))[0]
    R = np.linalg.norm(c - xi)
    assert abs(np.linalg.norm(c - xj) - R) < 1e-12 and abs(np.linalg.norm(c - p) - R) < 1e-12
    assert abs(c[0]) < 1e-12 and abs(c[1] - (p @ p - 1) / (2 * p[1])) < 1e-12
    assert abs(c[1] - 1 / np.tan(theta)) < 1e-12
    info.append((theta, c, R))
th_a, c_a, _ = info[0]
ang_c = np.arctan2(1.0, c_a[1])                  # angle at c in the right triangle (m_e, x_j, c)
assert abs(ang_c - th_a) < 1e-12
assert np.degrees(info[1][0]) > 90 and info[1][1][1] < 0

col = dgfig.COLORS
fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.3))
for ax, p, (theta, c, R), name in zip(axes, cases, info, ("(a) acute angle: $h > 0$", "(b) obtuse angle: $h < 0$")):
    ax.add_patch(Circle(c, R, fill=False, ls=(0, (4, 3)), color="#9A9A9A", lw=1.0))
    ax.add_patch(Polygon(np.array([xi, xj, p]), closed=True, facecolor=col["region"], alpha=0.3,
                         edgecolor="black", lw=1.4))
    ax.plot(*np.array([xi, xj]).T, color=col["tangent"], lw=2.4)
    ax.plot([0, 0], [0, c[1]], color=col["accent"], lw=2.4)
    ax.plot(*c, "o", color=col["accent"], ms=6)
    ax.plot(0, 0, "o", color="black", ms=4)
    ax.plot(*np.array([c, xj]).T, color="#777777", lw=1.0, ls=":")
    # right angle mark at m_e
    s = 0.09 * np.sign(c[1])
    ax.plot([0.09, 0.09, 0.0], [0.0, s, s], color="#555555", lw=0.8)
    # angle theta at p
    a1 = np.degrees(np.arctan2(*(xi - p)[::-1]))
    a2 = np.degrees(np.arctan2(*(xj - p)[::-1]))
    lo, hi = sorted((a1 % 360, a2 % 360))
    if hi - lo > 180:
        lo, hi = hi, lo + 360
    ax.add_patch(Arc(p, 0.36, 0.36, theta1=lo, theta2=hi, color=col["normal"], lw=1.4))
    toff = np.array([0.0, -0.33]) if c[1] > 0 else np.array([0.36, -0.02])
    ax.text(*(p + toff), r"$\theta$", fontsize=12, color=col["normal"], ha="center")
    ax.text(*(p + np.array([0.06, 0.08])), r"$p$", fontsize=11)
    ax.text(*(xi + np.array([-0.18, -0.05])), r"$x_i$", fontsize=11, ha="center")
    ax.text(*(xj + np.array([0.18, -0.05])), r"$x_j$", fontsize=11, ha="center")
    ax.text(0.05, -0.17 if c[1] > 0 else 0.07, r"$m_e$", fontsize=10.5)
    ax.text(*(c + np.array([0.08, 0.02])), r"$c$", fontsize=12, color="#8a5a00")
    ax.text(-0.42, 0.5 * c[1], rf"$h = {c[1]:.4g}$", fontsize=10.5, color="#8a5a00", ha="center")
    ax.text(0.0, -1.7 if c[1] > 0 else 1.25, rf"$\theta = {np.degrees(theta):.1f}^\circ$", fontsize=11,
            ha="center")
    ax.set_title(name, fontsize=11)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-2.45, 1.75)
    ax.set_aspect("equal")
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)
