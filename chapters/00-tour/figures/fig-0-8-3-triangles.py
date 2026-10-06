"""Figure 0.8.3: geodesic triangles and curvature (Section 0.8).

Left: on the unit sphere (K = 1) the triangle with vertices e₁, e₂, e₃ bounded by three great-circle arcs has three right
angles: the angle sum is 3π/2 = π + π/2, and its area is π/2 (one eighth of 4π).
Right: in the hyperbolic plane U² (K = −1) the triangle with vertices A = (−1, 1), B = (1, 1), C = (0, 3); its sides lie on
the semicircles centred at (0, 0), (3.5, 0) and (−3.5, 0). The angles are about 32.47°, 32.47°, 81.20° (sum ≈ 146.15° < 180°),
and its hyperbolic area π − (angle sum) ≈ 0.591 is checked independently by Stokes's theorem: area = ∮ dx / y.

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-8-3-triangles.py``
"""

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

# ---------------------------------------------------------------- hyperbolic triangle data
A, B, Cv = np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 3.0])
sides = [(A, B, np.array([0.0, 0.0])), (B, Cv, np.array([-3.5, 0.0])), (Cv, A, np.array([3.5, 0.0]))]


def arc(P, Q, c, n=400):
    """points on the circle centred at c (on the x-axis) from P to Q along the short arc"""
    a0, a1 = np.arctan2(*(P - c)[::-1]), np.arctan2(*(Q - c)[::-1])
    r = np.linalg.norm(P - c)
    tt = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(tt), c[1] + r * np.sin(tt)], 1)


def tangent(P, c, Q):
    rv = P - c
    tv = np.array([-rv[1], rv[0]]) / np.linalg.norm(rv)
    return tv if tv @ (Q - P) > 0 else -tv


def angle(P, c1, Q1, c2, Q2):
    return np.arccos(np.clip(tangent(P, c1, Q1) @ tangent(P, c2, Q2), -1, 1))


al = angle(A, sides[0][2], B, sides[2][2], Cv)
be = angle(B, sides[0][2], A, sides[1][2], Cv)
ga = angle(Cv, sides[2][2], A, sides[1][2], B)
# area by Stokes: d(dx/y) = dx∧dy / y², the hyperbolic area form; boundary A → B → C → A is counterclockwise
area = 0.0
for P, Q, c in sides:
    pts = arc(P, Q, c, 20001)
    mid = (pts[1:] + pts[:-1]) / 2
    area += np.sum(np.diff(pts[:, 0]) / mid[:, 1])

# self-checks (§12.1)
for P, Q, c in sides:
    assert abs(np.linalg.norm(P - c) - np.linalg.norm(Q - c)) < 1e-12      # both ends on the same semicircle
assert np.allclose(np.degrees([al, be, ga]), [32.4712, 32.4712, 81.2026], atol=1e-3)
assert abs(area - (np.pi - (al + be + ga))) < 1e-6                         # Gauss–Bonnet with K = −1
assert abs(area - 0.5909) < 1e-3

fig = plt.figure(figsize=(6.6, 3.2))

# ---------------------------------------------------------------- left: sphere octant
ax = dgfig.axes3d(fig, pos=121, elev=24, azim=30)
th, ph = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81), indexing="ij")
Xs, Ys, Zs = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(ax, Xs, Ys, Zs, alpha=0.25, grid=False, zorder=1)
tq, pq = np.meshgrid(np.linspace(0, np.pi / 2, 20), np.linspace(0, np.pi / 2, 20), indexing="ij")
ax.plot_surface(np.sin(tq) * np.cos(pq) * 1.002, np.sin(tq) * np.sin(pq) * 1.002, np.cos(tq) * 1.002,
                color=C["region"], alpha=0.5, linewidth=0, shade=False, zorder=2, rasterized=True)
e = np.eye(3)
s = np.linspace(0, np.pi / 2, 100)
for i, j in ((0, 1), (1, 2), (2, 0)):
    seg = np.outer(np.cos(s), e[i]) + np.outer(np.sin(s), e[j])
    ax.plot(*seg.T, color=C["accent"], lw=LW["main"] + 0.3, zorder=5)
for i in range(3):
    dgfig.point3d(ax, e[i], None, size=14, zorder=8)
    # right-angle marks
    a, b = e[(i + 1) % 3], e[(i + 2) % 3]
    m = 0.14
    corner = [e[i] * np.cos(m) + a * np.sin(m), (e[i] + (a + b) * np.tan(m)) / np.linalg.norm(e[i] + (a + b) * np.tan(m)),
              e[i] * np.cos(m) + b * np.sin(m)]
    ax.plot(*np.array(corner).T, color=C["main"], lw=0.8, zorder=7)
dgfig.equal_aspect(ax, Xs, Ys, Zs, zoom=1.2)

from mpl_toolkits.mplot3d import proj3d  # noqa: E402

box = dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85)
for pt, lab, d in ((e[0], r"$e_1$", (-6, -12)), (e[1], r"$e_2$", (12, -6)), (e[2], r"$e_3$", (14, 4))):
    x2, y2, _ = proj3d.proj_transform(*pt, ax.get_proj())
    ax.annotate(lab, xy=(x2, y2), xytext=d, textcoords="offset points", fontsize=12, ha="center", va="center",
                bbox=box, zorder=20)
ax.set_title(r"$K = 1$:  $\frac{\pi}{2} \cdot 3 = \pi + \frac{\pi}{2}$", fontsize=12, y=1.04)

# ---------------------------------------------------------------- right: hyperbolic triangle
ax2 = fig.add_subplot(122)
ax2.set_xlim(-2.0, 2.0)
ax2.set_ylim(0, 3.4)
ax2.set_aspect("equal")
for sp_ in ("top", "right", "left"):
    ax2.spines[sp_].set_visible(False)
ax2.spines["bottom"].set_linewidth(1.4)
ax2.set_xticks([])
ax2.set_yticks([])
bnd = np.concatenate([arc(P, Q, c) for P, Q, c in sides])
ax2.fill(*bnd.T, color=C["region"], alpha=0.35, lw=0)
for P, Q, c in sides:
    r = np.linalg.norm(P - c)
    tt = np.linspace(0, np.pi, 400)
    full = np.stack([c[0] + r * np.cos(tt), r * np.sin(tt)], 1)
    ax2.plot(*full.T, color=C["aux"], lw=0.6, ls=(0, (3, 3)), zorder=2)
    ax2.plot(*arc(P, Q, c).T, color=C["accent"], lw=LW["main"] + 0.3, zorder=4)
for P, lab, d in ((A, r"$A$", (-0.3, -0.05)), (B, r"$B$", (0.12, -0.05)), (Cv, r"$C$", (0.12, 0.05))):
    ax2.plot(*P, "o", color=C["main"], ms=4, zorder=6)
    ax2.text(P[0] + d[0], P[1] + d[1], lab, fontsize=12, zorder=6)
for P, lab, d in ((A, r"$\alpha$", (0.3, 0.17)), (B, r"$\beta$", (-0.36, 0.17)), (Cv, r"$\gamma$", (-0.06, -0.55))):
    ax2.text(P[0] + d[0], P[1] + d[1], lab, fontsize=12, zorder=6)
ax2.set_title(r"$K = -1$:  $\alpha + \beta + \gamma \approx 146.2^\circ$", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.03, wspace=0.08)
dgfig.save(fig, __file__)
