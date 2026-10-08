"""Figure 31.4.5: functional maps between deformed copies of a plate (Section 31.4).

Source M: the planar rectangle [0, 2] x [0, 0.7], 41 x 15 vertices, alternating diagonals (its Neumann eigenvalues
pi^2 (m^2/4 + n^2/0.49) are all distinct). Targets N have the same connectivity, so the vertex correspondence T is the
identity; only the positions change:
  (a) isometric bend: the x-direction is rolled onto the wave z = 0.25 sin(pi s), keeping arc length;
  (b) uniform stretch x -> 1.1 x;
  (c) non-uniform stretch x -> x (1 + 0.3 x / 2).
Each panel shows |C_ij| for i, j < 20, C = Phi_N^T M_N Phi_M (C_ij = <phi_j^M o T^{-1}, phi_i^N>_{M_N}),
mixed Voronoi mass. White = 0, black = 1.1.
Self-check: (a) |C| = identity to 1e-3; (b) a permutation-like matrix with entries ~ 1.05 = sqrt(1.1);
(c) 56 entries above 0.1.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-4-5-functional-map.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
a, b = 2.0, 0.7
V, F = dm.grid_mesh(np.linspace(0, a, 41), np.linspace(0, b, 15), pattern="alternate")


def eig(W, k=25):
    m = dm.mass_voronoi(W, F)
    lam, Phi = dm.generalized_eigs(dm.cotan_laplacian(W, F), m, k)
    return lam, Phi, m


lamM, PhiM, _ = eig(V)
exact = np.sort([np.pi ** 2 * (i * i / a ** 2 + j * j / b ** 2) for i in range(20) for j in range(10)])[:25]
assert np.all(np.diff(exact) > 0.1)                      # distinct eigenvalues on the smooth rectangle
assert np.allclose(lamM[:6], exact[:6], rtol=0.01, atol=1e-6)

tt = np.linspace(0, 3.0, 30001)
zz = 0.25 * np.sin(np.pi * tt)
sarc = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(tt) ** 2 + np.diff(zz) ** 2))])
xa = np.interp(V[:, 0], sarc, tt)
targets = [("(a) isometric bend", np.stack([xa, V[:, 1], 0.25 * np.sin(np.pi * xa)], 1)),
           ("(b) stretch $x \\to 1.1x$", np.c_[1.1 * V[:, 0], V[:, 1], V[:, 2]]),
           ("(c) non-uniform stretch", np.c_[V[:, 0] * (1 + 0.3 * V[:, 0] / a), V[:, 1], V[:, 2]])]
mats = []
for title, W in targets:
    lam, Phi, m = eig(W)
    Cm = Phi.T @ (m[:, None] * PhiM)
    mats.append(np.abs(Cm[:20, :20]))
assert np.abs(mats[0] - np.eye(20)).max() < 1e-3
B = mats[1]
assert ((B > 0.5).sum(0) == 1).all() and ((B > 0.5).sum(1) == 1).all() and np.allclose(B[B > 0.5], np.sqrt(1.1), atol=0.01)
assert not np.allclose(np.diag(B), np.sqrt(1.1), atol=0.05)    # not diagonal: the order of eigenvalues changed
assert (mats[2] > 0.1).sum() == 56

fig = plt.figure(figsize=(7.6, 3.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.15)
for k, ((title, W), Mx) in enumerate(zip(targets, mats)):
    ax = fig.add_subplot(gs[0, k])
    im = ax.imshow(Mx, cmap="Greys", vmin=0, vmax=1.1, interpolation="nearest")
    ax.set_title(title, fontsize=12.5)
    ax.set_xticks([0, 5, 10, 15, 19])
    ax.set_yticks([0, 5, 10, 15, 19])
    ax.tick_params(labelsize=10.5)
    ax.set_xlabel(r"$j$ (source $\varphi^M_j$)", fontsize=12)
    if k == 0:
        ax.set_ylabel(r"$i$ (target $\varphi^N_i$)", fontsize=12)
    else:
        ax.set_yticklabels([])
cax = fig.add_subplot(gs[0, 3])
cb = fig.colorbar(im, cax=cax)
cb.set_label(r"$|C_{ij}|$", fontsize=12)
cb.ax.tick_params(labelsize=10.5)
fig.subplots_adjust(left=0.085, right=0.92, top=0.88, bottom=0.2)
dgfig.save(fig, __file__)
