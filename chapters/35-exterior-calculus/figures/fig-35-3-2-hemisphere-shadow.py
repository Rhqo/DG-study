"""Figure 35.3.2: the 2-form dx ^ dy integrated over the upper hemisphere of the unit sphere is the area of its
shadow, the unit disk (Section 35.3, Example 35.3.10(b)).

Grey surface: the upper hemisphere x(theta, phi) = (sin theta cos phi, sin theta sin phi, cos theta),
0 <= theta <= pi/2, with its theta, phi grid (step pi/12). Orientation: x_theta x x_phi = sin(theta) N points
outward (N = x, vermillion arrows at the centers of three highlighted cells).
Light blue disk: the plane z = -0.95 under the hemisphere, carrying the vertical shadow of the hemisphere (the unit
disk). Three orange cells, at theta in [pi/12, 2pi/12], [3pi/12, 4pi/12], [5pi/12, 6pi/12] and
phi in [16pi/12, 18pi/12], and their orange shadows below (grey dashed lines join the corners).
The value of dx ^ dy on a cell is its shadow area: int_cell sin(theta) cos(theta) dtheta dphi; the three cells give
0.0479, 0.0654, 0.0175 (computed below) and all cells of the upper hemisphere together give pi.
Self-checks: (dx ^ dy)(x_theta, x_phi) = sin theta cos theta; the sum of the exact cell values over the whole grid
is pi; the cell value equals the area of the cell's vertical shadow (polygon area of the projected boundary,
fine sampling); the integral over the whole sphere is 0.
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
h = np.pi / 12


def X(th, ph):
    th, ph = np.broadcast_arrays(np.asarray(th, float), np.asarray(ph, float))
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


def cell_value(t0, t1, p0, p1):
    """int_{t0}^{t1} int_{p0}^{p1} sin t cos t dphi dt (exact)."""
    return (p1 - p0) * 0.5 * (np.sin(t1) ** 2 - np.sin(t0) ** 2)


def shadow_area(t0, t1, p0, p1, m=400):
    """signed area of the vertical projection of the cell boundary, traversed in the parameter order
    (t0, p0) -> (t1, p0) -> (t1, p1) -> (t0, p1) -> back."""
    s = np.linspace(0, 1, m)
    pts = np.concatenate([
        [X(t0 + (t1 - t0) * a, p0) for a in s], [X(t1, p0 + (p1 - p0) * a) for a in s],
        [X(t1 - (t1 - t0) * a, p1) for a in s], [X(t0, p1 - (p1 - p0) * a) for a in s]])
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


# (dx ^ dy)(x_theta, x_phi)
th0, ph0 = 0.7, 1.1
xt = np.array([np.cos(th0) * np.cos(ph0), np.cos(th0) * np.sin(ph0), -np.sin(th0)])
xp = np.array([-np.sin(th0) * np.sin(ph0), np.sin(th0) * np.cos(ph0), 0.0])
assert abs((xt[0] * xp[1] - xt[1] * xp[0]) - np.sin(th0) * np.cos(th0)) < 1e-14
assert np.allclose(np.cross(xt, xp), np.sin(th0) * X(th0, ph0))
tot = sum(cell_value(i * h, (i + 1) * h, j * h, (j + 1) * h) for i in range(6) for j in range(24))
assert abs(tot - np.pi) < 1e-12
whole = sum(cell_value(i * h, (i + 1) * h, j * h, (j + 1) * h) for i in range(12) for j in range(24))
assert abs(whole) < 1e-12
cells = [(1, 16), (3, 16), (5, 16)]
vals = []
for i, j in cells:
    v = cell_value(i * h, (i + 1) * h, j * h, (j + 2) * h)
    a = shadow_area(i * h, (i + 1) * h, j * h, (j + 2) * h)
    assert abs(v - a) < 2e-5 and v > 0
    vals.append(v)

fig = plt.figure(figsize=(5.6, 4.6))
ax = dgfig.axes3d(fig, elev=30, azim=-25)
zp = -0.95
# shadow plane: disk
tt = np.linspace(0, 2 * np.pi, 200)
disk = [list(zip(np.cos(tt), np.sin(tt), np.full_like(tt, zp)))]
ax.add_collection3d(Poly3DCollection(disk, facecolor=C["region"], alpha=0.25, edgecolor=C["tangent"], lw=0.8,
                                     zorder=1))
# hemisphere surface
T, Pp = np.meshgrid(np.linspace(0, np.pi / 2, 31), np.linspace(0, 2 * np.pi, 121), indexing="ij")
XS, YS, ZS = np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T)
ax.plot_surface(XS, YS, ZS, color=C["surface"], alpha=0.35, linewidth=0, shade=False, zorder=2)
for i in range(1, 7):
    pp = np.linspace(0, 2 * np.pi, 121)
    c = X(i * h, pp)
    ax.plot(c[0], c[1], c[2], color="#8C8C8C", lw=0.4, zorder=3)
for j in range(24):
    tq = np.linspace(0, np.pi / 2, 31)
    c = X(tq, j * h)
    ax.plot(c[0], c[1], c[2], color="#8C8C8C", lw=0.4, zorder=3)
# highlighted cells, their shadows, normals
for (i, j), v in zip(cells, vals):
    t0, t1, p0, p1 = i * h, (i + 1) * h, j * h, (j + 2) * h
    s = np.linspace(0, 1, 12)
    bd = np.concatenate([[X(t0 + (t1 - t0) * a, p0) for a in s], [X(t1, p0 + (p1 - p0) * a) for a in s],
                         [X(t1 - (t1 - t0) * a, p1) for a in s], [X(t0, p1 - (p1 - p0) * a) for a in s]])
    ax.add_collection3d(Poly3DCollection([bd], facecolor=C["accent"], alpha=0.85, edgecolor=C["main"], lw=0.5,
                                         zorder=6))
    sh = bd.copy()
    sh[:, 2] = zp
    ax.add_collection3d(Poly3DCollection([sh], facecolor=C["accent"], alpha=0.85, edgecolor=C["main"], lw=0.5,
                                         zorder=4))
    for corner in (X(t0, p0), X(t1, p0), X(t1, p1), X(t0, p1)):
        ax.plot([corner[0]] * 2, [corner[1]] * 2, [corner[2], zp], color=C["aux"], lw=0.6, ls="--", zorder=5)
    cen = X(0.5 * (t0 + t1), 0.5 * (p0 + p1))
    dgfig.arrow3d(ax, cen, 0.3 * cen, role="normal", zorder=8)
fig.text(0.5, 0.08, r"$\int_{S^+} dx\wedge dy = \pi$ = area of the shadow disk", ha="center", fontsize=11)
c0 = X(1.5 * h, 17 * h)
ax.text(*(1.42 * c0 + np.array([0.0, 0.0, 0.05])), r"$\mathbf{N}$", color=C["normal"], fontsize=12, zorder=10)
dgfig.equal_aspect(ax, np.array([[-1, -1, zp], [1, 1, 1.05]]), zoom=1.05)
dgfig.save(fig, __file__)
print("cell values:", [round(v, 4) for v in vals])
