"""Figure 42.5.3: Dirichlet and Neumann problems on meshes with boundary (Section 42.5, Examples 42.5.11 and 42.5.12).

(a) Dirichlet: Laplace equation on the unit square, boundary values of the harmonic u = x^2 - y^2.
    On grid_mesh (n x n squares) the discrete solution equals u at every vertex (to rounding); on the same grid with the
    interior vertices moved by up to 0.2/n in x and y (seed 7; negative cotan weights appear) the largest vertex error
    (blue) and the area-weighted RMS error (orange) decrease like h^2.
(b) Neumann: on ch42.disk_mesh(n) (grid faces inside the unit disk), Delta u = 2 with <grad u, n> = <p, n> on the
    polygonal boundary, exact u = r^2/2 + c: largest error (blue) and RMS error (orange). For Delta u = 0 with
    <grad u, n> = n_x (exact u = x + c) the discrete solution is exact (not drawn).
Lumped (barycentric) mass, boundary integrals with 4-point Gauss rules on each boundary edge (ch42.boundary_load).
Self-check: the regular-grid Dirichlet solution and the Neumann solution for u = x are exact to 1e-12; fitted slopes of
the plotted errors lie between 1.5 and 2.4.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-5-3-boundary-convergence.py``
"""

import pathlib
import sys

import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch42  # noqa: E402

col = dgfig.COLORS


def dirichlet(V, F, g):
    L = dm.cotan_laplacian(V, F)
    B = np.array(dm.boundary_vertices(F))
    I = np.setdiff1d(np.arange(len(V)), B)
    u = np.zeros(len(V))
    u[B] = g(V[B])
    u[I] = np.linalg.solve(L[np.ix_(I, I)], -L[np.ix_(I, B)] @ u[B])
    return u


def neumann(V, F, f, grad):
    L, m = dm.cotan_laplacian(V, F), dm.mass_barycentric(V, F)
    b = m * f(V) - ch42.boundary_load(V, F, lambda p, nn: grad(p) @ nn)
    assert abs(b.sum()) < 1e-12
    return np.linalg.solve(-L + np.outer(m, m), -b), m


g = lambda P: P[:, 0] ** 2 - P[:, 1] ** 2  # noqa: E731
ns = [8, 16, 32, 64]
dmax, drms, nmax, nrms = [], [], [], []
for n in ns:
    xs = np.linspace(0, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs)
    assert np.abs(dirichlet(V, F, g) - g(V)).max() < 1e-12
    rng = np.random.default_rng(7)
    I = np.setdiff1d(np.arange(len(V)), dm.boundary_vertices(F))
    W = V.copy()
    W[I, :2] += 0.2 / n * rng.uniform(-1, 1, (len(I), 2))
    e = dirichlet(W, F, g) - g(W)
    m = dm.mass_barycentric(W, F)
    dmax.append(np.abs(e).max())
    drms.append(np.sqrt((e ** 2 * m).sum() / m.sum()))
    if n <= 32:
        Vd, Fd = ch42.disk_mesh(n)
        ux, mx = neumann(Vd, Fd, lambda P: 0 * P[:, 0], lambda p: np.array([1.0, 0, 0]))
        ex = Vd[:, 0] - (mx @ Vd[:, 0]) / mx.sum()
        assert np.abs(ux - ex).max() < 1e-12
        ur, mr = neumann(Vd, Fd, lambda P: 2 + 0 * P[:, 0], lambda p: np.array([p[0], p[1], 0.0]))
        er = 0.5 * (Vd[:, 0] ** 2 + Vd[:, 1] ** 2)
        e = ur - (er - (mr @ er) / mr.sum())
        nmax.append(np.abs(e).max())
        nrms.append(np.sqrt((e ** 2 * mr).sum() / mr.sum()))
hs = 1.0 / np.array(ns)
hd = 2.0 / np.array(ns[:3])
for arr, hh in ((dmax, hs), (drms, hs), (nmax, hd), (nrms, hd)):
    sl = np.polyfit(np.log(hh), np.log(arr), 1)[0]
    assert 1.5 < sl < 2.4, (arr, sl)
print("Dirichlet (jittered) max", [f"{v:.2e}" for v in dmax], "rms", [f"{v:.2e}" for v in drms])
print("Neumann (disk) max", [f"{v:.2e}" for v in nmax], "rms", [f"{v:.2e}" for v in nrms])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))
axa.loglog(hs, dmax, "o-", color=col["tangent"], ms=3.5, lw=1.4, label="max")
axa.loglog(hs, drms, "s-", color=col["accent"], ms=3.5, lw=1.4, label="RMS")
axa.loglog(hs, 0.5 * drms[-1] * (hs / hs[-1]) ** 2, ":", color=col["aux"], lw=1.0)
axa.set_xlabel(r"grid spacing $h = 1/n$")
axa.set_title(r"(a) Dirichlet, jittered square", fontsize=10.5)
axb.loglog(hd, nmax, "o-", color=col["tangent"], ms=3.5, lw=1.4, label="max")
axb.loglog(hd, nrms, "s-", color=col["accent"], ms=3.5, lw=1.4, label="RMS")
axb.loglog(hd, 0.5 * nrms[-1] * (hd / hd[-1]) ** 2, ":", color=col["aux"], lw=1.0)
axb.set_xlabel(r"grid spacing $h = 2/n$")
axb.set_title(r"(b) Neumann, disk, $u = r^2/2$", fontsize=10.5)
for ax in (axa, axb):
    ax.set_ylabel("error")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.spines[["top", "right"]].set_visible(False)
axa.set_xticks(hs, [f"1/{n}" for n in ns])
axb.set_xticks(hd, [f"2/{n}" for n in ns[:3]])
axa.text(hs[1], 0.5 * drms[-1] * (hs[1] / hs[-1]) ** 2 * 0.45, "slope 2", fontsize=9, color=col["aux"])
fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.18, wspace=0.38)
dgfig.save(fig, __file__)
