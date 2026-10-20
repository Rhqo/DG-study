"""Figure 42.6.1: the potential of a point source on the unit sphere (Section 42.6, Example 42.6.2).

Mesh dgmesh.icosphere(4) (2562 vertices), L = cotan Laplacian, M = mixed Voronoi areas. Source: unit mass at the
vertex at the north pole, rho = e_s / A_s; the Poisson problem L u = M (rho - mean rho) is solved with
(-L + a a^T) u = -b (Proposition 42.6.1), a = M 1, so that the area-weighted mean of u is 0.
(a) the front half of the mesh (view from elev 20) colored by u (blue low, white high), the source marked by a black dot.
(b) u_i against the polar angle theta_i of every vertex (small black dots, drawn over the curve) and the exact Green's function
    G(theta) = (1/4pi) log(1 - cos theta) + c (orange), with c fitted on theta > 0.3.
Self-check: the residual of L u = b is below 1e-12, m^T u = 0, and |u - G - c| < 6e-4 for theta > 0.3.
The interactive version shows icosphere(3) colored by u.

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/figures/fig-42-6-1-point-source.py``
"""

import pathlib
import sys

import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch42  # noqa: E402

col = dgfig.COLORS


def potential(level):
    V, F = dm.icosphere(level)
    L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
    s = int(np.argmax(V[:, 2]))
    rho = np.zeros(len(V))
    rho[s] = 1.0 / m[s]
    u, rho_bar = ch42.solve_closed_poisson(L, m, rho)
    b = m * (rho - rho_bar)
    assert np.abs(L @ u - b).max() < 1e-12 and abs(m @ u) < 1e-12
    th = np.arccos(np.clip(V[:, 2], -1, 1))
    return V, F, u, th, s


V, F, u, th, s = potential(4)
far = th > 0.3
G = np.log(1 - np.cos(th[far])) / (4 * np.pi)
c = np.mean(u[far] - G)
err = np.abs(u[far] - G - c).max()
assert err < 6e-4
print(f"icosphere(4): max |u - G - c| for theta > 0.3 = {err:.2e}, u at the source {u[s]:.3f}, min u {u.min():.3f}")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.1), gridspec_kw={"width_ratios": [1.0, 1.15]})
cam = dm.Ortho(elev=20, azim=-60)
nrm = dm.face_normals(V, F)
vis = nrm @ cam.view > 0
order = np.argsort(cam.depth(V[F].mean(1)))
order = order[vis[order]]
cmap = plt.get_cmap("Blues_r")
uf = u[F].mean(1)
lo, hi = np.percentile(u, 1), u.max()
pc = PolyCollection(cam(V[F[order]]), facecolors=cmap(np.clip((uf[order] - lo) / (hi - lo), 0, 1) * 0.85),
                    edgecolors="none")
axa.add_collection(pc)
axa.plot(*cam(V[s][None])[0], "o", color=col["main"], ms=5)
axa.text(*(cam(V[s][None])[0] + [0.06, 0.06]), "source", fontsize=9.5)
axa.set_aspect("equal")
axa.autoscale_view()
axa.set_axis_off()
axa.set_title(r"(a) potential $u$ on icosphere(4)", fontsize=11)
tt = np.linspace(0.05, np.pi, 400)
axb.plot(tt, np.log(1 - np.cos(tt)) / (4 * np.pi) + c, color=col["accent"], lw=3.0, alpha=0.7, zorder=1)
axb.plot(th, u, ".", color=col["main"], ms=1.2, rasterized=True, zorder=2)
axb.text(1.35, -0.09, r"$\frac{1}{4\pi}\log(1 - \cos\theta) + c$", fontsize=10, color=col["accent"])
axb.set_xlabel(r"polar angle $\theta$ (distance from the source)")
axb.set_ylabel(r"$u$")
axb.set_xlim(0, np.pi)
axb.set_title(r"(b) $u_i$ against $\theta_i$", fontsize=11)
axb.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.17, wspace=0.2)
dgfig.save(fig, __file__)

V3, F3, u3, th3, s3 = potential(3)
mesh = dm.plotly_mesh(V3, F3, intensity=u3, colorscale="Blues", cmin=float(np.percentile(u3, 1)), cmax=float(u3.max()),
                      name="potential", showscale=True, colorbar_title="u",
                      hovertemplate="u = %{intensity:.4f}<extra></extra>")
dm.plotly_save([mesh, dm.plotly_edges(V3, F3, color="#555555", width=0.8)], __file__,
               "icosphere(3): potential of a unit point source at the north pole (mean removed)", "fig-42-6-1")
