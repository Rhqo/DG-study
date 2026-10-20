"""Figure 43.3.1: the complex structure J = N x of the unit sphere and the holomorphic Mercator map
(Definition 43.3.1, Definition 43.3.4, Example 43.3.7).

Point p = x(theta, phi) with theta = 1.0, phi = 0.6 on the unit sphere, outward normal N = p.
(a) the sphere, the tangent plane at p (light blue), X = unit east vector (blue), JX = N x X (green, points north),
    N (vermilion, half length), and the circles of latitude and longitude through p (grey).
(b) the Mercator plane (phi, t), t = log cot(theta/2), with the images of the same two circles (grey lines), and
    dz(X), dz(JX) for z = phi + i t (blue, green, drawn at the scale cosh t = 1/sin theta).
Self-check: N x (N x X) = -X, |JX| = |X|, <X, JX> = 0, det[X, JX, N] = 1; dz(X) = 1/sin(theta) (real) and
dz(JX) = i dz(X), i.e. z is holomorphic at p.

Run from the project root:
``PYTHONPATH=tools python3 chapters/43-conformal-geometry/figures/fig-43-3-1-complex-structure.py``
"""

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
th0, ph0 = 1.0, 0.6
P = np.array([np.sin(th0) * np.cos(ph0), np.sin(th0) * np.sin(ph0), np.cos(th0)])
N = P.copy()
east = np.array([-np.sin(ph0), np.cos(ph0), 0.0])
X = east
JX = np.cross(N, X)
assert np.allclose(np.cross(N, JX), -X) and np.isclose(np.linalg.norm(JX), 1) and abs(X @ JX) < 1e-15
assert np.isclose(np.linalg.det(np.c_[X, JX, N]), 1)
north = np.array([-np.cos(th0) * np.cos(ph0), -np.cos(th0) * np.sin(ph0), np.sin(th0)])
assert np.allclose(JX, north)
# Mercator: phi east, t north; d(phi)(east) = 1/sin(theta), dt(north) = 1/sin(theta)
h = 1e-6
merc = lambda th, ph: ph + 1j * np.log(1 / np.tan(th / 2))  # noqa: E731
dz_X = (merc(th0, ph0 + h / np.sin(th0)) - merc(th0, ph0 - h / np.sin(th0))) / (2 * h)
dz_JX = (merc(th0 - h, ph0) - merc(th0 + h, ph0)) / (2 * h)
assert abs(dz_X - 1 / np.sin(th0)) < 1e-6 and abs(dz_JX - 1j * dz_X) < 1e-6
print("JX = north:", np.round(JX, 4), " dz(X) = %.4f, dz(JX) = %s" % (dz_X.real, np.round(dz_JX, 4)))

fig = plt.figure(figsize=(6.6, 3.3))
ax = dgfig.axes3d(fig, 121, elev=22, azim=72)
u = np.linspace(0, np.pi, 40)
v = np.linspace(0, 2 * np.pi, 60)
U, Vv = np.meshgrid(u, v)
dgfig.surface(ax, np.sin(U) * np.cos(Vv), np.sin(U) * np.sin(Vv), np.cos(U), alpha=0.35, grid=False)
tt = np.linspace(0, 2 * np.pi, 200)
lat = np.c_[np.sin(th0) * np.cos(tt), np.sin(th0) * np.sin(tt), np.cos(th0) * np.ones_like(tt)]
lon = np.c_[np.sin(tt) * np.cos(ph0), np.sin(tt) * np.sin(ph0), np.cos(tt)]
front = lambda Q: Q @ dgfig.view_vector(ax) > 0  # noqa: E731
for Q in (lat, lon):
    dgfig.curve3d(ax, Q, role="aux", lw=0.9, visible=front(Q))
dgfig.tangent_plane(ax, P, X, JX, size=0.55)
dgfig.arrow3d(ax, P, 0.6 * X, role="tangent", label=r"$X$", label_offset=(0.02, 0.05, -0.08))
dgfig.arrow3d(ax, P, 0.6 * JX, role="third", label=r"$JX$", label_offset=(0.0, 0.0, 0.05))
dgfig.arrow3d(ax, P, 0.5 * N, role="normal", label=r"$\mathbf{N}$", label_offset=(0.03, 0.03, 0.03))
dgfig.point3d(ax, P)
dgfig.equal_aspect(ax, np.array([[-1, -1, -1], [1, 1, 1]]), zoom=1.25)
ax.set_title(r"(a) $JX = \mathbf{N}\times X$", fontsize=11)

ax2 = fig.add_subplot(122)
t0 = np.log(1 / np.tan(th0 / 2))
ax2.axhline(t0, color=col["aux"], lw=0.9)
ax2.axvline(ph0, color=col["aux"], lw=0.9)
s = 0.6 / np.sin(th0) * 0.75
ax2.annotate("", xy=(ph0 + s, t0), xytext=(ph0, t0), arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=1.5))
ax2.annotate("", xy=(ph0, t0 + s), xytext=(ph0, t0), arrowprops=dict(arrowstyle="-|>", color=col["third"], lw=1.5))
ax2.text(ph0 + s + 0.03, t0 - 0.12, r"$dz(X)$", color=col["tangent"], fontsize=11)
ax2.text(ph0 + 0.04, t0 + s + 0.03, r"$dz(JX) = \mathrm{i}\,dz(X)$", color=col["third"], fontsize=11)
ax2.plot([ph0], [t0], "o", color=col["main"], ms=4)
ax2.text(ph0 - 0.32, t0 - 0.18, r"$z(p)$", fontsize=11)
ax2.set_xlim(-0.2, 2.0)
ax2.set_ylim(-0.2, 1.9)
ax2.set_aspect("equal")
ax2.set_xlabel(r"$\varphi = \mathrm{Re}\,z$")
ax2.set_ylabel(r"$t = \mathrm{Im}\,z$")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title(r"(b) Mercator $z = \varphi + \mathrm{i}t$", fontsize=11)
fig.subplots_adjust(left=0.0, right=0.98, top=0.9, bottom=0.13, wspace=0.12)
dgfig.save(fig, __file__)
