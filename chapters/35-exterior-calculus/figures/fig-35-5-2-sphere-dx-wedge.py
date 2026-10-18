"""Figure 35.5.2: the differential dx of the sphere parametrization as an R^3-valued 1-form, and its self-wedge
(Section 35.5, Example 35.5.11 and Proposition 35.5.14).

(a) Parameter plane (theta, phi): the point q = (1.0, 0.6) with the coordinate vectors d_theta (blue) and d_phi
    (green) and the unit square they span (light blue).
(b) The unit sphere x(theta, phi) = (sin theta cos phi, sin theta sin phi, cos theta) (r = 1, grey) and, at x(q):
    dx(d_theta) = x_theta (blue, length 1), dx(d_phi) = x_phi (green, length sin 1.0 = 0.841), the parallelogram they
    span (orange), and the vector (dx ^ dx)(d_theta, d_phi) = 2 x_theta x x_phi = 2 sin(theta) N (vermillion, drawn
    at half length; true length 1.683 = twice the area 0.841 of the orange parallelogram), along the outward N.
Self-checks: (dx ^ dx)(u, v) = dx(u) x dx(v) - dx(v) x dx(u) = 2 dx(u) x dx(v); x_theta x x_phi = sin(theta) x;
the inner product wedge <dx ^ dx>(d_theta, d_phi) = 0.
"""
import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
th0, ph0 = 1.0, 0.6


def X(th, ph):
    th, ph = np.broadcast_arrays(np.asarray(th, float), np.asarray(ph, float))
    return np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])


p = X(th0, ph0)
xt = np.array([np.cos(th0) * np.cos(ph0), np.cos(th0) * np.sin(ph0), -np.sin(th0)])
xp = np.array([-np.sin(th0) * np.sin(ph0), np.sin(th0) * np.cos(ph0), 0.0])
wedge = np.cross(xt, xp) - np.cross(xp, xt)
assert np.allclose(wedge, 2 * np.cross(xt, xp))
assert np.allclose(np.cross(xt, xp), np.sin(th0) * p)
assert abs(np.linalg.norm(wedge) - 2 * np.sin(th0)) < 1e-14
assert abs((xt @ xp) - (xp @ xt)) < 1e-15            # <dx ^ dx>(d_theta, d_phi) = 0
assert abs(np.linalg.norm(xp) - 0.841) < 5e-4 and abs(np.linalg.norm(wedge) - 1.683) < 5e-4

fig = plt.figure(figsize=(6.6, 3.5))
axA = fig.add_axes([0.02, 0.12, 0.36, 0.76])
q = np.array([th0, ph0])
axA.add_patch(Polygon([q, q + [1, 0], q + [1, 1], q + [0, 1]], closed=True, facecolor=C["region"], alpha=0.35,
                      edgecolor="none", zorder=0))
for vv, col, lab, off in [((1, 0), C["tangent"], r"$\partial_\theta$", (0.5, -0.16)),
                          ((0, 1), C["third"], r"$\partial_\varphi$", (-0.16, 0.5))]:
    axA.add_patch(FancyArrowPatch(tuple(q), tuple(q + vv), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.6,
                                  shrinkA=0, shrinkB=0, zorder=3))
    axA.text(q[0] + off[0], q[1] + off[1], lab, color=col, fontsize=12, ha="center")
axA.plot(*q, "o", color=C["main"], ms=3)
axA.text(q[0] - 0.05, q[1] - 0.1, r"$q$", fontsize=12, ha="right")
axA.set_xlim(0.5, 2.3)
axA.set_ylim(0.2, 1.9)
axA.set_aspect("equal")
axA.set_xlabel(r"$\theta$")
axA.set_ylabel(r"$\varphi$")
for sp_ in ("top", "right"):
    axA.spines[sp_].set_visible(False)
axA.set_xticks([1.0, 2.0])
axA.set_yticks([0.6, 1.6])
axA.set_title("(a) parameters", fontsize=12)

ax = fig.add_axes([0.38, 0.0, 0.62, 1.0], projection="3d", computed_zorder=False)
ax.view_init(elev=20, azim=72)
ax.set_axis_off()
T, Pp = np.meshgrid(np.linspace(0, np.pi, 31), np.linspace(0, 2 * np.pi, 61), indexing="ij")
ax.plot_surface(np.sin(T) * np.cos(Pp), np.sin(T) * np.sin(Pp), np.cos(T), color=C["surface"], alpha=0.3, linewidth=0,
                shade=False, zorder=1)
for tt in np.linspace(np.pi / 6, 5 * np.pi / 6, 5):
    c = X(tt, np.linspace(0, 2 * np.pi, 121))
    ax.plot(*c, color="#8C8C8C", lw=0.4, zorder=2)
for pp in np.linspace(0, 2 * np.pi, 13)[:-1]:
    c = X(np.linspace(0, np.pi, 61), pp)
    ax.plot(*c, color="#8C8C8C", lw=0.4, zorder=2)
ax.add_collection3d(Poly3DCollection([[p, p + xt, p + xt + xp, p + xp]], facecolor=C["accent"], alpha=0.55,
                                     edgecolor=C["main"], lw=0.5, zorder=4))
dgfig.arrow3d(ax, p, xt, role="tangent", label=r"$d\mathbf{x}(\partial_\theta)$", label_offset=(0.05, 0.0, -0.12),
              zorder=6, fontsize=11)
dgfig.arrow3d(ax, p, xp, role="third", label=r"$d\mathbf{x}(\partial_\varphi)$", label_offset=(0.0, 0.08, 0.12),
              zorder=6, fontsize=11, ha="left")
dgfig.arrow3d(ax, p, 0.5 * wedge, role="normal", label=r"$\frac{1}{2}(d\mathbf{x}\wedge d\mathbf{x})(\partial_\theta, \partial_\varphi)$",
              label_offset=(0.0, 0.0, 0.16), zorder=7, fontsize=11)
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1.6, 1.4, 1.3]]), zoom=1.15)
dgfig.map_arrow(fig, axA, ax, r"$d\mathbf{x}$", xy_from=(1.0, 0.45), xy_to=(0.2, 0.42), rad=0.3)
ax.set_title("(b) on the sphere", fontsize=12, y=0.93)
dgfig.save(fig, __file__)
