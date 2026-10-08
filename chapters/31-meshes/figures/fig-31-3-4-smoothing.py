"""Figure 31.3.4: uniform smoothing slides vertices along the surface, cotan smoothing does not (Section 31.3).

A planar 17 x 13 grid mesh on [0, 2] x [0, 1.4] whose x-spacing grows to the right (x_k = 2 (k/16)^1.7) and whose
squares are cut by alternating diagonals. Interior vertices get z-noise N(0, 0.04^2) (seed 3); boundary vertices
are fixed. Smoothing by implicit steps with the boundary fixed:
  uniform:  (I - tau L_u) x_new = x,            tau = 0.5   (L_u: uniform Laplacian, rows (1/deg) - 1 on the diagonal)
  cotan:    (M - tau L) x_new = M x,             tau = 0.005 (L, M recomputed from the current mesh at each step)
(a), (b) Top views after 20 steps: gray = initial (x, y) of the vertices and the grid lines, arrows = horizontal
    displacement magnified 10 times.
(c) Horizontal drift (mean over interior vertices) against the remaining z-noise (rms), one point per step.
Self-check: on the noise-free plane cotan smoothing moves no vertex (linear precision); uniform smoothing keeps
drifting while cotan drift saturates near 0.006.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-3-4-smoothing.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
nx, ny = 17, 13
xs = (np.linspace(0, 1, nx) ** 1.7) * 2.0
ys = np.linspace(0, 1.4, ny)
V, F = dm.grid_mesh(xs, ys, pattern="alternate")
bnd = np.array([i in (0, nx - 1) or j in (0, ny - 1) for i in range(nx) for j in range(ny)])
I = ~bnd
rng = np.random.default_rng(3)
V0 = V.copy()
V0[I, 2] += 0.04 * rng.normal(size=I.sum())
Lu = dm.uniform_laplacian(F)


def step(X, kind, tau):
    if kind == "uniform":
        A, rhs = np.eye(len(X)) - tau * Lu, X
    else:
        m = dm.mass_barycentric(X, F)
        A, rhs = np.diag(m) - tau * dm.cotan_laplacian(X, F), m[:, None] * X
    Xn = X.copy()
    Xn[I] = np.linalg.solve(A[np.ix_(I, I)], rhs[I] - A[np.ix_(I, ~I)] @ X[~I])
    return Xn


runs = {}
for kind, tau in (("uniform", 0.5), ("cotan", 0.005)):
    X = V0.copy()
    hist = []
    for s in range(40):
        X = step(X, kind, tau)
        hist.append((np.sqrt((X[I, 2] ** 2).mean()), np.linalg.norm((X - V0)[I, :2], axis=1).mean()))
        if s == 19:
            X20 = X.copy()
    runs[kind] = (np.array(hist), X20)
# linear precision: noise-free plane
Xc = V.copy()
for s in range(5):
    Xc = step(Xc, "cotan", 0.005)
assert np.abs(Xc - V).max() < 1e-12
assert runs["uniform"][0][-1, 1] > 0.04 and runs["cotan"][0][-1, 1] < 0.007
assert abs(runs["cotan"][0][-1, 1] - runs["cotan"][0][10, 1]) < 1e-4    # saturated

fig = plt.figure(figsize=(7.6, 6.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9], hspace=0.35, wspace=0.08)
MAG = 10.0
for col, (kind, title, color) in enumerate((("uniform", "(a) uniform, 20 steps", C["tangent"]),
                                            ("cotan", "(b) cotan, 20 steps", C["accent"]))):
    ax = fig.add_subplot(gs[0, col])
    for (i, j) in dm.edges(F):
        ax.plot(V0[[i, j], 0], V0[[i, j], 1], color="#BBBBBB", lw=0.5, zorder=1)
    X20 = runs[kind][1]
    for p, q in zip(V0[I], X20[I]):
        d = (q - p)[:2]
        if np.linalg.norm(d) > 1e-4:
            ax.add_patch(FancyArrowPatch(p[:2], p[:2] + MAG * d, arrowstyle="-|>", mutation_scale=6, color=color,
                                         lw=0.9, zorder=3))
    ax.plot(V0[I, 0], V0[I, 1], "o", color=C["main"], ms=1.6, zorder=4)
    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 1.45)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11.5)
fig.text(0.5, 0.505, r"arrows: horizontal displacement $\times 10$;  gray: initial grid (dense on the left)",
         ha="center", fontsize=10)

axc = fig.add_subplot(gs[1, :])
for kind, label, color, mk in (("uniform", "uniform", C["tangent"], "o"), ("cotan", "cotan", C["accent"], "s")):
    h = runs[kind][0]
    axc.plot(h[:, 0], h[:, 1], mk + "-", color=color, ms=3.5, lw=1.4, label=label)
axc.set_xscale("log")
axc.invert_xaxis()
axc.set_xlabel(r"remaining $z$-noise (rms)  $\rightarrow$ smoother")
axc.set_ylabel("mean horizontal drift")
axc.legend(fontsize=10, frameon=False, loc="upper left")
axc.grid(True, which="major", lw=0.3, alpha=0.5)
axc.set_title("(c) drift against denoising (one point per step)", fontsize=11.5)

fig.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.09)
dgfig.save(fig, __file__)
