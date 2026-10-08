"""Figure 31.4.4: the heat method on the unit sphere (Section 31.4).

Unit icosphere level 3 (642 vertices), barycentric mass, source = the vertex nearest the north pole,
t = h^2 (h = mean edge length 0.151). Steps [Crane13]:
  I.   (M - t L) u = delta_source              (one backward Euler step of the heat equation)
  II.  X = -grad u / |grad u| on every face
  III. solve the least-squares Poisson problem  min sum_f A_f |grad phi - X_f|^2, i.e. -L phi = sum_f A_f G_f^T X_f,
       with phi(source) = 0.
(a) The mesh seen from above-front: colour = log10 u, arrows = X on a subset of front faces.
(b) The recovered phi_i (orange) and the magnitude-only estimate sqrt(-4 t log(u_i / u_source)) (blue) against the
    exact geodesic distance theta_i = arccos(z_i) (black line).
Self-check: max |phi - theta| = 0.050, mean 0.021; u spans more than 8 orders of magnitude.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-4-4-heat-method.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

C = dgfig.COLORS
V, F = dm.icosphere(3)
L = dm.cotan_laplacian(V, F)
m = dm.mass_barycentric(V, F)
h = np.mean([np.linalg.norm(V[a] - V[b]) for a, b in dm.edges(F)])
t = h * h
s = int(np.argmax(V[:, 2]))
delta = np.zeros(len(V))
delta[s] = 1.0
u = np.linalg.solve(np.diag(m) - t * L, delta)
N = dm.face_normals(V, F)
A = dm.face_areas(V, F)


def face_grad(f):
    g = np.zeros((len(F), 3))
    for c in range(3):
        e = V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]]
        g += f[F[:, c], None] * np.cross(N, e) / (2 * A[:, None])
    return g


G = face_grad(u)
X = -G / np.linalg.norm(G, axis=1, keepdims=True)
b = np.zeros(len(V))
for c in range(3):
    e = V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]]
    np.add.at(b, F[:, c], 0.5 * (np.cross(N, e) * X).sum(1))   # A_f <grad phi_c, X_f>
idx = np.array([i for i in range(len(V)) if i != s])
phi = np.zeros(len(V))
phi[idx] = np.linalg.solve(-L[np.ix_(idx, idx)], b[idx])
theta = np.arccos(np.clip(V[:, 2], -1, 1))
err = np.abs(phi - theta)
assert round(err.max(), 3) == 0.050 and round(err.mean(), 3) == 0.021
assert 1e8 < u.max() / u.min() < 1e9   # u spans more than 8 orders of magnitude
naive = np.sqrt(np.maximum(-4 * t * np.log(u / u[s]), 0))

fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw=dict(width_ratios=[1.0, 1.1], wspace=0.3))
cam = dm.Ortho(elev=35, azim=-60)
lu = np.log10(u)
norm = Normalize(lu.min(), lu.max())
rgba = cm.magma(norm(lu[F].mean(1)))
dm.draw_mesh_2d(ax, V, F, cam, rgba, edgecolor="none", lw=0, cull=True, rasterized=True)
cen = V[F].mean(1)
front = (N @ cam.view) > 0.25
rng = np.random.default_rng(0)
pick = np.where(front)[0]
pick = pick[rng.permutation(len(pick))[:150]]
for f in pick:
    p0 = cam(cen[f])
    p1 = cam(cen[f] + 0.11 * X[f])
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, color="white", lw=0.8, zorder=5))
q = cam(V[s])
ax.plot(*q, "*", color=C["accent"], ms=12, mec="k", mew=0.5, zorder=6)
ax.text(q[0] + 0.08, q[1] + 0.06, "source", fontsize=10.5, color="k", zorder=7)
ax.set_xlim(-1.15, 1.15)
ax.set_ylim(-1.15, 1.2)
ax.set_aspect("equal")
ax.set_axis_off()
sm = plt.cm.ScalarMappable(norm=norm, cmap="magma")
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.8)
cb.set_label(r"$\log_{10} u$ (heat after time $t = h^2$)", fontsize=10)
ax.set_title(r"(a) heat $u$ and $X = -\nabla u / |\nabla u|$", fontsize=11.5)

axb.plot([0, np.pi], [0, np.pi], color=C["main"], lw=1.2, label="exact distance", zorder=1)
axb.plot(theta, naive, "o", ms=2.2, color=C["tangent"], label=r"$\sqrt{-4t\,\log(u/u_{\mathrm{source}})}$", zorder=2)
axb.plot(theta, phi, "o", ms=2.2, color=C["accent"], label=r"heat method $\varphi$", zorder=3)
axb.set_xlabel(r"geodesic distance $\theta$ from the source")
axb.set_ylabel("estimated distance")
axb.set_xticks([0, np.pi / 2, np.pi])
axb.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
axb.set_yticks([0, np.pi / 2, np.pi])
axb.set_yticklabels(["0", r"$\pi/2$", r"$\pi$"])
axb.set_ylim(-0.1, np.pi + 0.35)
axb.legend(fontsize=9.5, frameon=False, loc="upper left", markerscale=2.5)
axb.grid(True, lw=0.3, alpha=0.5)
axb.set_title("(b) recovered distance", fontsize=11.5)

fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12)
dgfig.save(fig, __file__)
