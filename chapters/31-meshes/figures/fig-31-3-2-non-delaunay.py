"""Figure 31.3.2: negative cotan weights and the maximum principle (Section 31.3).

(a), (b) Two quadrilaterals x_i, x_l, x_j, x_m cut by the diagonal ij. Left: alpha + beta < pi (x_m outside the
    circumcircle of x_i x_j x_l, the edge is Delaunay), w_ij > 0. Right: alpha + beta > pi (x_m inside the circle),
    w_ij < 0. w_ij = (cot alpha + cot beta)/2 = sin(alpha + beta) / (2 sin alpha sin beta).
(c) A planar mesh with one interior vertex x_0 = (0, 0) and boundary vertices (1.6, 0), (0.8, 0.25), (0, 1),
    (-1, 0.5), (-1, -0.5), (0, -1), (0.8, -0.25). Boundary values: u = 0 at (1.6, 0), u = 1 at the others.
    The discrete harmonic extension ((Lu)_0 = 0) gives u_0 = 1.29 > 1 because the weight of the edge from x_0 to
    (1.6, 0) is negative (its opposite angles are 145.3 degrees each). Colors: the piecewise-linear interpolant
    (diverging map centred at u = 1: red above 1, blue below), drawn exactly: every triangle is split into small
    flat-coloured pieces carrying the linear interpolant (matplotlib's SVG gouraud shading mixes the colours and
    showed a purple that is not on the colour scale).
Layout: (a) and (b) on the top row, (c) below at full width, so that the labels stay readable at phone width.
Self-check: the weights, the harmonic value, the Delaunay test, and the flip of the bad edge (all weights at x_0
positive, u_0 = 1).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-3-2-non-delaunay.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Arc, Circle, Polygon  # noqa: E402
from matplotlib.tri import LinearTriInterpolator, Triangulation, UniformTriRefiner  # noqa: E402

C = dgfig.COLORS


def angle_at(p, a, b):
    u, v = a - p, b - p
    return np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))


def circumcircle(a, b, c):
    A = np.array([[b[0] - a[0], b[1] - a[1]], [c[0] - a[0], c[1] - a[1]]]) * 2
    rhs = np.array([b @ b - a @ a, c @ c - a @ a])
    o = np.linalg.solve(A, rhs)
    return o, np.linalg.norm(a - o)


fig = plt.figure(figsize=(6.6, 7.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.12, wspace=0.05)

# ---------------------------------------------------------------- (a) two quads
xi, xj = np.array([0.0, 0.0]), np.array([2.0, 0.0])
cases = [(np.array([1.0, 1.0]), np.array([1.0, -1.3]), "Delaunay"), (np.array([1.0, 0.45]), np.array([1.0, -0.35]),
                                                                       "not Delaunay")]
for col, (xl, xm, title) in enumerate(cases):
    ax = fig.add_subplot(gs[0, col])
    al, be = angle_at(xl, xi, xj), angle_at(xm, xi, xj)
    w = 0.5 * (1 / np.tan(al) + 1 / np.tan(be))
    assert abs(w - np.sin(al + be) / (2 * np.sin(al) * np.sin(be))) < 1e-12
    o, R = circumcircle(xi, xj, xl)
    inside = np.linalg.norm(xm - o) < R
    assert (w > 0) == (al + be < np.pi) == (not inside)
    for T in ([xi, xj, xl], [xi, xm, xj]):
        ax.add_patch(Polygon(T, closed=True, fc=C["surface"], ec="#444444", lw=0.9, alpha=0.7))
    ax.add_patch(Circle(o, R, fill=False, ec=C["aux"], lw=0.9, ls=(0, (4, 3))))
    ax.plot(*np.array([xi, xj]).T, color=C["normal"] if w < 0 else C["tangent"], lw=2.6)
    for p, ang0, lab in ((xl, al, r"$\alpha$"), (xm, be, r"$\beta$")):
        d1, d2 = xi - p, xj - p
        a1, a2 = np.degrees(np.arctan2(d1[1], d1[0])), np.degrees(np.arctan2(d2[1], d2[0]))
        lo, hi = (a1, a2) if (a2 - a1) % 360 < 180 else (a2, a1)
        ax.add_patch(Arc(p, 0.36, 0.36, theta1=lo, theta2=hi, color=C["accent"], lw=1.6))
        sgn = 1 if p[1] > 0 else -1
        ax.text(p[0] - 0.34, p[1] + sgn * 0.14, lab, ha="center", va="center", fontsize=13, color=C["accent"])
    for p, lab, off in ((xi, r"$x_i$", (-0.3, -0.05)), (xj, r"$x_j$", (0.06, -0.05)), (xl, r"$x_l$", (0.12, 0.05)),
                        (xm, r"$x_m$", (0.12, -0.12))):
        ax.plot(*p, "o", color=C["main"], ms=3.5)
        ax.text(*(p + np.array(off)), lab, fontsize=12.5, ha="left")
    ax.text(1.0, -2.2, (r"$x_m$ inside the circle" if inside else r"$x_m$ outside the circle") + "\n"
            + rf"$\alpha + \beta = {np.degrees(al + be):.0f}^\circ$" + "\n" + rf"$w_{{ij}} = {w:+.2f}$",
            ha="center", va="center", fontsize=12.5, color=C["normal"] if w < 0 else C["main"],
            bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-2.85, 1.55)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"({'a' if col == 0 else 'b'}) {title}", fontsize=13)

# ---------------------------------------------------------------- (c) maximum principle fails
ax = fig.add_subplot(gs[1, :])
B = np.array([[1.6, 0], [0.8, 0.25], [0, 1], [-1, 0.5], [-1, -0.5], [0, -1], [0.8, -0.25]])
V = np.vstack([[0, 0], B])
F = np.array([[0, 1 + k, 1 + (k + 1) % 7] for k in range(7)])
V3 = np.c_[V, np.zeros(len(V))]
L = dm.cotan_laplacian(V3, F)
u = np.ones(len(V))
u[1] = 0.0
u[0] = -(L[0, 1:] @ u[1:]) / L[0, 0]          # (L u)_0 = 0
w01 = L[0, 1]
assert w01 < 0 and u[0] > 1
assert abs(u[0] - 1.288) < 0.001
th = np.degrees(angle_at(V[2], V[0], V[1]))
assert abs(th - 145.3) < 0.05
# the Delaunay flip of the bad edge: replace edge (x_0, (1.6, 0)) by ((0.8, 0.25), (0.8, -0.25))
Ff = np.array([t for t in F.tolist() if not (0 in t and 1 in t)] + [[0, 7, 2], [7, 1, 2]])
Lf = dm.cotan_laplacian(V3, Ff)
w0 = Lf[0, 1:][Lf[0, 1:] != 0]
assert (w0 > 0).all() and abs(-(Lf[0, 1:] @ u[1:]) / Lf[0, 0] - 1.0) < 1e-12
tri = Triangulation(V[:, 0], V[:, 1], F)
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=1.3)
# exact piecewise-linear colouring: refine every triangle and colour the small flat pieces by the linear interpolant
fine, ufine = UniformTriRefiner(tri).refine_field(u, triinterpolator=LinearTriInterpolator(tri, u), subdiv=5)
cs = ax.tripcolor(fine, ufine, shading="flat", cmap="RdBu_r", norm=norm, edgecolors="face", lw=0.2,
                  rasterized=True)
ax.triplot(tri, color="#555555", lw=0.7)
ax.plot(*np.array([V[0], V[1]]).T, color=C["normal"], lw=2.6)
for k in range(1, len(V)):
    ax.text(*(V[k] * 1.13 + np.array([0, 0.0])), f"{u[k]:.0f}", ha="center", va="center", fontsize=12.5)
ax.plot(*V[0], "o", color="white", mec="k", ms=7)
ax.annotate(rf"$u_0 = {u[0]:.2f} > 1$" + "\n(above every\nboundary value)", xy=V[0], xytext=(-2.4, 0.0),
            fontsize=12, va="center", arrowprops=dict(arrowstyle="-|>", color="k", lw=0.9))
ax.annotate(rf"$w = {w01:.2f} < 0$", xy=(0.95, 0.0), xytext=(1.05, 0.62), fontsize=12, color=C["normal"],
            arrowprops=dict(arrowstyle="-|>", color=C["normal"], lw=0.9))
ax.text(1.95, -0.75, "boundary values:\n0 at the right tip,\n1 elsewhere", fontsize=11, ha="left", va="center")
cb = fig.colorbar(cs, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
cb.set_label(r"$u$ (white = 1)", fontsize=12)
cb.ax.tick_params(labelsize=11)
ax.set_xlim(-2.4, 3.3)
ax.set_ylim(-1.45, 1.3)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(c) discrete harmonic, but above the maximum", fontsize=13)

fig.subplots_adjust(left=0.01, right=0.97, top=0.95, bottom=0.02)
dgfig.save(fig, __file__)
