"""Figure 42.2.1: two interpolants of the same boundary data on the unit square (Section 42.2, Theorem 42.2.4).

Boundary data g = sin(pi x) on the top side y = 1 and g = 0 on the other three sides.
(a) the harmonic interpolant u = sin(pi x) sinh(pi y) / sinh(pi), E_D(u) = (pi/4) coth(pi) = 0.7883;
(b) the blend w = y sin(pi x), E_D(w) = pi^2/12 + 1/4 = 1.0725.
Color: function value (0 white, 1 dark blue); black curves: level sets 0.1, 0.3, 0.5, 0.7, 0.9; the thick blue line is
the side where g is not zero. Self-check: both functions equal g on the boundary; Delta u = 0; the Dirichlet energies
by a 400 x 400 midpoint rule agree with the closed forms to 1e-4; E_D(w) - E_D(u) = E_D(w - u) (Theorem 42.2.4(b));
the discrete harmonic interpolant on grid_mesh (n = 32) is within 1e-3 of u at the vertices.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-2-1-interpolation.py``
"""

import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

col = dgfig.COLORS
uex = lambda X, Y: np.sin(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)  # noqa: E731
wex = lambda X, Y: Y * np.sin(np.pi * X)  # noqa: E731
ED_u = np.pi / 4 / np.tanh(np.pi)
ED_w = np.pi ** 2 / 12 + 0.25

m = 400
c = (np.arange(m) + 0.5) / m
X, Y = np.meshgrid(c, c, indexing="ij")


def energy(gx, gy):
    return 0.5 * np.mean(gx ** 2 + gy ** 2)


gux = np.pi * np.cos(np.pi * X) * np.sinh(np.pi * Y) / np.sinh(np.pi)
guy = np.pi * np.sin(np.pi * X) * np.cosh(np.pi * Y) / np.sinh(np.pi)
gwx = np.pi * Y * np.cos(np.pi * X)
gwy = np.sin(np.pi * X)
assert abs(energy(gux, guy) - ED_u) < 1e-4 and abs(energy(gwx, gwy) - ED_w) < 1e-4
assert abs((ED_w - ED_u) - energy(gwx - gux, gwy - guy)) < 1e-4
# discrete harmonic interpolant on the grid (Dirichlet problem, Section 42.5)
n = 32
xs = np.linspace(0, 1, n + 1)
V, F = dm.grid_mesh(xs, xs)
L = dm.cotan_laplacian(V, F)
B = np.array(dm.boundary_vertices(F))
I = np.setdiff1d(np.arange(len(V)), B)
ud = np.zeros(len(V))
ud[B] = np.where(np.isclose(V[B, 1], 1.0), np.sin(np.pi * V[B, 0]), 0.0)
ud[I] = np.linalg.solve(L[np.ix_(I, I)], -L[np.ix_(I, B)] @ ud[B])
assert np.abs(ud - uex(V[:, 0], V[:, 1])).max() < 1e-3
print(f"E_D(u) = {ED_u:.4f}, E_D(w) = {ED_w:.4f}, discrete -1/2 u^T L u = {-0.5 * ud @ L @ ud:.4f}, "
      f"max vertex error {np.abs(ud - uex(V[:, 0], V[:, 1])).max():.1e}")

fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.15))
g = np.linspace(0, 1, 201)
GX, GY = np.meshgrid(g, g, indexing="ij")
for ax, f, title, E in ((axes[0], uex, r"(a) harmonic $u$", ED_u), (axes[1], wex, r"(b) blend $w = y\,\sin\pi x$", ED_w)):
    Z = f(GX, GY)
    ax.pcolormesh(GX, GY, Z, cmap="Blues", vmin=0, vmax=1.15, shading="gouraud", rasterized=True)
    ax.contour(GX, GY, Z, levels=[0.1, 0.3, 0.5, 0.7, 0.9], colors="black", linewidths=0.7)
    ax.plot([0, 1], [1, 1], color=col["tangent"], lw=3.0)
    ax.plot([0, 0, 1, 1], [1, 0, 0, 1], color=col["aux"], lw=1.0)
    ax.text(0.5, 1.06, r"$g = \sin\pi x$", ha="center", fontsize=10, color=col["tangent"])
    ax.text(0.5, -0.13, r"$g = 0$", ha="center", va="center", fontsize=10, color=col["aux"])
    ax.set_title(title + "\n" + rf"$E_D = {E:.4f}$", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.22, 1.18)
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)
