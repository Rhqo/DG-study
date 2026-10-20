"""Figure 40.4.1: the Minkowski sum P + eps B of a plane polygon, convex and not convex (eps = 0.3).

(a) the convex quadrilateral (0, 0), (2, 0), (2.4, 1.2), (0.6, 1.6): P (gray), the rectangles of width eps on the
edges (sky blue), the circular sectors at the vertices (orange) whose angles are the exterior angles (sum 2 pi);
area(P + eps B) = A + L eps + pi eps^2 (Exercise 40.4.3).
(b) the L-shaped polygon (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2) (A = 3, L = 8): sectors only at the five convex
corners; at the reflex corner (1, 1) the two rectangles overlap in an eps x eps square (dark blue, counted twice), so
area(P + eps B) = 3 + 8 eps + (5 pi / 4 - 1) eps^2, not 3 + 8 eps + pi eps^2 (Non-example 40.4.6). The black curve
is the boundary {d = eps}. Areas checked by counting a grid of spacing 0.002 (ddg40.polygon_minkowski_area_grid).

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-4-1-minkowski-polygons.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Wedge  # noqa: E402

import ddg40  # noqa: E402

C = dgfig.COLORS
EPS = 0.3
Q = np.array([[0, 0], [2, 0], [2.4, 1.2], [0.6, 1.6]], float)
Lp = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)


def area_perimeter(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y), np.sum(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1))


AQ, LQ = area_perimeter(Q)
AL, LL = area_perimeter(Lp)
assert abs(AL - 3) < 1e-12 and abs(LL - 8) < 1e-12
gQ = ddg40.polygon_minkowski_area_grid(Q, EPS, 0.002)
gL = ddg40.polygon_minkowski_area_grid(Lp, EPS, 0.002)
fQ = AQ + LQ * EPS + np.pi * EPS ** 2
fL = 3 + 8 * EPS + (5 * np.pi / 4 - 1) * EPS ** 2
assert abs(gQ - fQ) < 5e-3 and abs(gL - fL) < 5e-3 and abs(gL - (3 + 8 * EPS + np.pi * EPS ** 2)) > 0.015


def draw(ax, P, title):
    n = len(P)
    ax.add_patch(Polygon(P, closed=True, facecolor=C["surface"], edgecolor=C["main"], lw=1.2, zorder=3))
    turns = []
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        t = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([t[1], -t[0]])                               # outward normal of a counterclockwise polygon
        ax.add_patch(Polygon([a, b, b + EPS * nrm, a + EPS * nrm], closed=True, facecolor=C["region"], alpha=0.55,
                             edgecolor="none", zorder=1))
        c = P[(k + 2) % n]
        t2 = (c - b) / np.linalg.norm(c - b)
        turn = np.arctan2(t[0] * t2[1] - t[1] * t2[0], t @ t2)
        turns.append(turn)
        if turn > 0:                                                # convex vertex: a sector of angle = exterior angle
            a0 = np.degrees(np.arctan2(nrm[1], nrm[0]))
            ax.add_patch(Wedge(b, EPS, a0, a0 + np.degrees(turn), facecolor=C["accent"], alpha=0.75, edgecolor="none",
                               zorder=2))
        else:                                                       # reflex vertex: the two rectangles overlap
            n2 = np.array([t2[1], -t2[0]])
            ax.add_patch(Polygon([b, b + EPS * nrm, b + EPS * (nrm + n2), b + EPS * n2], closed=True,
                                 facecolor=C["tangent"], alpha=0.8, edgecolor="none", zorder=2))
    # boundary {d = eps}: contour of the distance function
    xs = np.linspace(P[:, 0].min() - 0.5, P[:, 0].max() + 0.5, 500)
    ys = np.linspace(P[:, 1].min() - 0.5, P[:, 1].max() + 0.5, 500)
    X, Y = np.meshgrid(xs, ys)
    Z = np.c_[X.ravel(), Y.ravel()]
    d = np.full(len(Z), np.inf)
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        ab = b - a
        tt = np.clip(((Z - a) @ ab) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.linalg.norm(Z - (a + tt[:, None] * ab), axis=1))
    from matplotlib.path import Path as MPath
    d[MPath(P).contains_points(Z)] = 0.0                           # distance to P (0 inside)
    ax.contour(X, Y, d.reshape(X.shape), levels=[EPS], colors=[C["main"]], linewidths=0.9, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(P[:, 0].min() - 0.45, P[:, 0].max() + 0.45)
    ax.set_ylim(P[:, 1].min() - 0.65, P[:, 1].max() + 0.4)
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=12)
    return np.array(turns)


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.3))
tq = draw(ax, Q, "(a)")
tl = draw(ax2, Lp, "(b)")
assert abs(tq.sum() - 2 * np.pi) < 1e-12 and abs(tl.sum() - 2 * np.pi) < 1e-12 and np.sum(tl < 0) == 1
ax.text(0.0, -0.56, r"$A + L\varepsilon + \pi\varepsilon^2 = %.3f$" % fQ, fontsize=10)
ax2.text(-0.4, -0.56, r"$3 + 8\varepsilon + (\frac{5\pi}{4} - 1)\varepsilon^2 = %.3f$" % fL, fontsize=10)
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.92, wspace=0.08)
dgfig.save(fig, __file__)
print(f"(a) A {AQ:.4f}, L {LQ:.4f}, formula {fQ:.4f}, grid {gQ:.4f}; (b) formula {fL:.4f}, grid {gL:.4f}, "
      f"naive {3 + 8 * EPS + np.pi * EPS ** 2:.4f}")
