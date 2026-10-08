"""Figure 31.4.3: low-pass filtering a shape with its own eigenfunctions (Section 31.4).

Shape: the level-3 icosphere (642 vertices) with radius r = 1 + 0.18 x z + 0.12 (y^2 - z^2) + 0.1 x plus 12 Gaussian
bumps of height 0.22 and width 0.22 at random directions (seed 7). Eigenfunctions of -L phi = lambda M phi of this
shape (mixed Voronoi mass). Reconstruction from the first K eigenfunctions:
    x^(K) = sum_{k < K} phi_k (phi_k^T M x)      (each coordinate separately).
(a) x^(K) for K = 4, 25, 100 and the original (K = 642), same camera and scale.
(b) Relative error ||x - x^(K)||_M / ||x - c||_M (c = centroid) against K, log-log.
Self-check: K = 1 gives the centroid; errors 0.073 (K = 4), 0.040 (25), 0.0059 (100), 0.0014 (200), 0 (642).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-4-3-lowpass.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def bumpy_blob():
    V0, F = dm.icosphere(3)
    rng = np.random.default_rng(7)
    x, y, z = V0.T
    r = 1 + 0.18 * (x * z) + 0.12 * (y * y - z * z) + 0.1 * x
    cent = rng.normal(size=(12, 3))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    for c in cent:
        r += 0.22 * np.exp(-np.sum((V0 - c) ** 2, 1) / (2 * 0.22 ** 2))
    return V0 * r[:, None], F


C = dgfig.COLORS
V, F = bumpy_blob()
m = dm.mass_voronoi(V, F)
lam, Phi = dm.generalized_eigs(dm.cotan_laplacian(V, F), m)
coef = Phi.T @ (m[:, None] * V)
c0 = (m[:, None] * V).sum(0) / m.sum()
den = np.sqrt((m[:, None] * (V - c0) ** 2).sum())


def recon(K):
    return Phi[:, :K] @ coef[:K]


assert np.allclose(recon(1), c0[None, :], atol=1e-10)
errs = {}
for K in (4, 25, 100, 200, 642):
    errs[K] = np.sqrt((m[:, None] * (V - recon(K)) ** 2).sum()) / den
assert [round(errs[K], 4) for K in (4, 25, 100, 200)] == [0.0728, 0.0399, 0.0059, 0.0014] and errs[642] < 1e-10

fig = plt.figure(figsize=(7.6, 5.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.35, wspace=0.02)
cam = dm.Ortho(elev=20, azim=-50)
base = dm.hex_to_rgb(C["surface"])
for col, K in enumerate((4, 25, 100, 642)):
    ax = fig.add_subplot(gs[0, col])
    X = recon(K)
    cols = dm.shade(X, F, base, light=cam.view + np.array([0.0, 0.0, 0.6]) - 0.4 * cam.right, ambient=0.15)
    dm.draw_mesh_2d(ax, X - c0, F, cam, cols, edgecolor="none", lw=0, cull=True, rasterized=True)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(f"K = {K}" + (" (all)" if K == 642 else ""), fontsize=11.5)
    if K < 642:
        ax.text(0.5, -0.04, f"error {errs[K]:.3f}", transform=ax.transAxes, ha="center", fontsize=10,
                color=C["tangent"])
fig.text(0.5, 0.975, r"(a) reconstructions $x^{(K)} = \sum_{k<K}\varphi_k\,(\varphi_k^{\mathsf{T}} M x)$", ha="center",
         fontsize=11.5)

axb = fig.add_subplot(gs[1, 1:3])
Ks = np.unique(np.round(np.logspace(np.log10(2), np.log10(641), 40)).astype(int))
e = [np.sqrt((m[:, None] * (V - recon(K)) ** 2).sum()) / den for K in Ks]
axb.loglog(Ks, e, "o-", color=C["tangent"], ms=3, lw=1.4)
for K in (4, 25, 100):
    axb.plot(K, errs[K], "o", color=C["accent"], ms=7, zorder=5)
axb.set_xlabel(r"number of eigenfunctions $K$")
axb.set_ylabel("relative error")
axb.grid(True, which="major", lw=0.3, alpha=0.5)
axb.set_title("(b) error against K", fontsize=11.5)

fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.1)
dgfig.save(fig, __file__)
