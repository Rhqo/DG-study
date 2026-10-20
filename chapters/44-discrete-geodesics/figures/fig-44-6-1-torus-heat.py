"""Figure 44.6.1: the three steps of the heat method on the torus mesh (Section 44.6, Example 44.6.9).

Mesh dgmesh.torus_mesh(2, 0.8, 32, 80) (2560 vertices); source: vertex 0 = x(0, 0) on the outer equator.
ch44.HeatMethod with mixed Voronoi mass, t = (mean edge length)^2 = 0.0322, zero Neumann data (closed mesh).
(a) step I and II: faces coloured by log10 u (u from (M - t L) u = delta), and the unit vectors X = -grad u/|grad u|
    on a subset of the faces facing the viewer (blue arrows, length 0.12).
(b) step III: faces coloured by phi with alternating light and dark bands of width 0.5 (isolines of phi).
View: orthographic, elev 38 deg, azim -25 deg, painter's algorithm over all faces.
Interactive version: the same mesh coloured by phi.
Self-checks: phi(source) = 0 = min phi; the integrated divergence sums to 0; phi along the outer and inner equators
agrees with (R + r)|v| and (R - r)|v| to within 0.13 and 0.03 for |v| <= pi/4.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-6-1-torus-heat.py
"""

import pathlib
import sys

import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch44  # noqa: E402

col = dgfig.COLORS
nu, nv = 32, 80
V, F = dm.torus_mesh(2, 0.8, nu, nv)
H = ch44.HeatMethod(V, F)
phi, u, X, b = H.distance(0, return_all=True)
assert abs(phi[0]) < 1e-12 and abs(b.sum()) < 1e-10
js = np.arange(1, nv // 8 + 1)
vv = 2 * np.pi * js / nv
assert np.abs(phi[js] - 2.8 * vv).max() < 0.13
phi_in = H.distance((nu // 2) * nv)
assert np.abs(phi_in[(nu // 2) * nv + js] - 1.2 * vv).max() < 0.03
print(f"t = {H.t:.4f}, max phi = {phi.max():.3f}, outer-equator error {np.abs(phi[js] - 2.8 * vv).max():.4f}")

cam = dm.Ortho(elev=38, azim=-25)
order = np.argsort(cam.depth(V[F].mean(1)))
P2 = cam(V[F[order]])
Nf = dm.face_normals(V, F)
light = np.clip(0.55 + 0.45 * (Nf @ ch44.unit(np.array([0.3, -0.4, 0.9]))), 0.25, 1.0)
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))
lu = np.log10(np.maximum(u[F].mean(1), 1e-300))
lo, hi = np.percentile(lu, 2), lu.max()
cm = plt.get_cmap("magma")
ca = cm(np.clip((lu - lo) / (hi - lo), 0, 1) * 0.92 + 0.04)
ca[:, :3] *= light[:, None]
axa.add_collection(PolyCollection(P2, facecolors=ca[order], edgecolors=ca[order], linewidths=0.2, rasterized=True))
cen = V[F].mean(1)
front = (Nf @ cam.view > 0.45) & (cam.depth(cen) > np.percentile(cam.depth(cen), 55))
sel = np.where(front)[0][::5]
for f in sel:
    a0, a1 = cam(cen[f]), cam(cen[f] + 0.12 * X[f])
    axa.annotate("", xy=a1, xytext=a0, arrowprops=dict(arrowstyle="-|>", color="#56B4E9", lw=0.7, mutation_scale=6))
src = cam(V[0])
axa.plot(*src, "*", color=col["accent"], ms=10, mec=col["main"], mew=0.5, zorder=5)
axa.set_title(r"(a) $\log_{10}u$ and $X = -\nabla u/|\nabla u|$", fontsize=11)
pm = phi[F].mean(1)
cb = plt.get_cmap("Blues_r")(np.clip(pm / phi.max(), 0, 1) * 0.85)
band = (np.floor(pm / 0.5) % 2 == 0)
cb[:, :3] *= np.where(band, 1.0, 0.82)[:, None] * light[:, None]
axb.add_collection(PolyCollection(P2, facecolors=cb[order], edgecolors=cb[order], linewidths=0.2, rasterized=True))
axb.plot(*src, "*", color=col["accent"], ms=10, mec=col["main"], mew=0.5, zorder=5)
axb.set_title(r"(b) $\varphi$, bands of width $0.5$", fontsize=11)
for ax in (axa, axb):
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, wspace=0.04)
dgfig.save(fig, __file__)

mesh = dm.plotly_mesh(V, F, intensity=phi, colorscale="Blues_r", cmin=0, cmax=float(phi.max()), name="phi",
                      showscale=True, colorbar_title="phi", hovertemplate="phi = %{intensity:.3f}<extra></extra>")
dm.plotly_save([mesh], __file__, "torus_mesh(2, 0.8, 32, 80): heat-method distance from a vertex on the outer equator",
               "fig-44-6-1")
