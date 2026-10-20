"""Figure 42.6.4: semi-implicit mean curvature flow degenerates on a peanut; fixing the initial Laplacian does not
(Section 42.6, Example 42.6.9, [Kazhdan12]).

Mesh: dgmesh.icosphere(3) mapped to a peanut, (x, y, z) -> (x s(z), y s(z), 1.5 z) with s(z) = 1 - 0.5 exp(-(z/0.4)^2).
Steps with h = 0.02: MCF (M - h L) x' = M x with L, M (mixed Voronoi) of the current mesh, and the flow with the cotan
Laplacian L_0 of the initial mesh, (M - h L_0) x' = M x (M still of the current mesh). After every step the mesh is
rescaled about its centroid to the initial area (both flows shrink; only the shape is compared).
(a) MCF after 0, 20, 35 steps; (b) fixed-L_0 flow after 0, 40, 120 steps. Each curve is the silhouette +-rho(z):
rho = largest distance from the axis among the vertices in 40 bins of z (relative to the centroid).
Self-check: MCF gets longer and thinner (z extent 3.0 -> more than 5), its smallest angle falls below 1 degree by step
40 and the step cannot be continued (degenerate triangles) before step 60; the fixed-L_0 flow keeps every angle above
10 degrees and ends with radius std / mean below 0.06. The interactive version shows the start, MCF step 35 and the
fixed-L_0 flow at step 120.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-6-4-cmcf.py``
"""

import warnings

import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

warnings.simplefilter("ignore")
col = dgfig.COLORS
V0, F = dm.icosphere(3)
s = 1 - 0.5 * np.exp(-(V0[:, 2] / 0.4) ** 2)
P0 = np.c_[V0[:, 0] * s, V0[:, 1] * s, 1.5 * V0[:, 2]]
A0 = dm.face_areas(P0, F).sum()
L0 = dm.cotan_laplacian(P0, F)
h = 0.02


def flow(fixed, steps):
    X, out = P0.copy(), {0: P0.copy()}
    for k in range(1, steps + 1):
        L = L0 if fixed else dm.cotan_laplacian(X, F)
        m = dm.mass_voronoi(X, F)
        if not (np.all(np.isfinite(L)) and np.all(np.isfinite(m)) and m.min() > 0):
            return out, k
        X = np.linalg.solve(np.diag(m) - h * L, m[:, None] * X)
        c = X.mean(0)
        X = c + (X - c) * np.sqrt(A0 / dm.face_areas(X, F).sum())
        out[k] = X.copy()
    return out, None


mcf, broke = flow(False, 60)
cm, _ = flow(True, 120)
minang = lambda X: np.degrees(dm.corner_angles(X, F).min())  # noqa: E731
assert np.ptp(mcf[35][:, 2]) > 5 and minang(mcf[40]) < 1 and broke is not None and broke < 60
assert min(minang(cm[k]) for k in cm) > 10
rc = np.linalg.norm(cm[120] - cm[120].mean(0), axis=1)
assert rc.std() / rc.mean() < 0.06
print("MCF z extent", [round(float(np.ptp(mcf[k][:, 2])), 2) for k in (0, 20, 35)], "smallest angle",
      [round(float(minang(mcf[k])), 1) for k in (0, 20, 35, 40)], "cannot continue at step", broke)
print("fixed L0: z extent", [round(float(np.ptp(cm[k][:, 2])), 2) for k in (0, 40, 120)], "smallest angle",
      round(float(min(minang(cm[k]) for k in cm)), 1), "final std/mean", round(float(rc.std() / rc.mean()), 3))


def silhouette(X, nb=40):
    c = X.mean(0)
    z = X[:, 2] - c[2]
    rho = np.hypot(X[:, 0] - c[0], X[:, 1] - c[1])
    edges = np.linspace(z.min(), z.max(), nb + 1)
    zz, rr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (z >= a) & (z <= b)
        if sel.any():
            zz.append(0.5 * (a + b))
            rr.append(rho[sel].max())
    zz = np.r_[z.min(), zz, z.max()]
    rr = np.r_[0.0, rr, 0.0]
    return zz, rr


fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5))
for ax, dat, ks, title in ((axes[0], mcf, (0, 20, 35), "(a) mean curvature flow"),
                           (axes[1], cm, (0, 40, 120), r"(b) fixed initial $L^0$")):
    for k, c in zip(ks, (col["aux"], col["tangent"], col["accent"])):
        zz, rr = silhouette(dat[k])
        ax.plot(zz, rr, color=c, lw=1.6, label=f"step {k}")
        ax.plot(zz, -rr, color=c, lw=1.6)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$z$")
    ax.legend(fontsize=8, frameon=False, loc="upper center", ncol=3, handlelength=1.5, columnspacing=0.8)
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-1.25, 1.85)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel(r"$\pm\rho$")
fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.17, wspace=0.12)
dgfig.save(fig, __file__)

traces = []
for X, dx, c in ((P0, -3.5, "#C8C8C8"), (mcf[35], 0.0, "#D55E00"), (cm[120], 3.5, "#0072B2")):
    Y = X - X.mean(0) + np.array([dx, 0, 0])
    traces += [dm.plotly_mesh(Y, F, color=c, name="mesh"), dm.plotly_edges(Y, F, color="#444444", width=0.6)]
dm.plotly_save(traces, __file__, "peanut: start (left), MCF step 35 (middle), fixed-L0 flow step 120 (right), h = 0.02", "fig-42-6-4")
