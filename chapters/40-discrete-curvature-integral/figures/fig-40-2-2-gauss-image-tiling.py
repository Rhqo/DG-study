"""Figure 40.2.2: the Gauss images of the vertices of a convex polyhedron tile the unit sphere (Descartes, Remark 40.2.12).

The polyhedron: dgmesh.icosahedron() with vertex k scaled by r_k = 1 + 0.18 u_k, u_k uniform in [-1, 1] (seed 3)
(ddg40.perturbed_icosahedron); every bending angle is positive (min 0.466), so every vertex is convex.
(a) the polyhedron with each vertex colored by its angle defect delta_i (RdBu_r, 0 white).
(b) the unit sphere: each point u is colored by the delta of the vertex that maximizes <u, x_i> (the vertex whose normal
cone contains u); the black arcs join the normals of faces sharing an edge, i.e. they are the sides of the spherical
polygons P_i of Proposition 40.1.7. The twelve areas are the twelve angle defects (0.648 to 1.509) and add up to 4 pi.
Also writes an interactive version.

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-2-2-gauss-image-tiling.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

C = dgfig.COLORS
V, F = ddg40.perturbed_icosahedron()
d = dm.angle_defect(V, F)
Nf = dm.face_normals(V, F)
theta = np.array([t for t, ell in dm.edge_bending_angles(V, F).values()])
assert theta.min() > 0.46 and all(ddg40.is_convex_vertex(V, F, i) for i in range(len(V)))
areas = np.array([ddg40.spherical_polygon_area(Nf[ddg40.faces_around(F, i)]) for i in range(len(V))])
assert np.allclose(areas, d, atol=1e-12) and abs(d.sum() - 4 * np.pi) < 1e-12
# sphere cells by the maximizing vertex
Vs, Fs = dm.icosphere(5)
cent = Vs[Fs].mean(1)
owner = np.argmax(cent @ V.T, axis=1)
frac = np.array([dm.face_areas(Vs, Fs)[owner == i].sum() for i in range(len(V))]) / dm.face_areas(Vs, Fs).sum() * 4 * np.pi
assert np.abs(frac - d).max() < 0.03                      # the cells have the angle defects as areas (sphere mesh resolution)

cmap = plt.get_cmap("RdBu_r")
norm = Normalize(vmin=-1.6, vmax=1.6)
cam = dm.Ortho(elev=20, azim=-50)

fig = plt.figure(figsize=(7.0, 3.4))
ax = fig.add_axes([0.0, 0.05, 0.45, 0.88])
col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
dm.draw_mesh_2d(ax, V, F, cam, col, edgecolor="#555555", lw=0.6, cull=True)
P = cam(V)
front = (V @ cam.view) > -0.05
sc = ax.scatter(*P[front].T, c=d[front], cmap=cmap, norm=norm, s=70, edgecolors="black", linewidths=0.6, zorder=5)
for i in np.where(front)[0]:
    ax.text(*(P[i] + np.array([0.06, 0.05])), f"{d[i]:.2f}", fontsize=8.5, zorder=6)
ax.set_aspect("equal")
lim = 1.25
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_axis_off()
ax.set_title("(a)", loc="left", fontsize=12)

ax2 = fig.add_axes([0.47, 0.05, 0.45, 0.88])
fc = cmap(norm(d[owner]))
dm.draw_mesh_2d(ax2, Vs, Fs, cam, fc, edgecolor="none", lw=0.0, cull=True, rasterized=True)
for (i, j), fs in dm.edge_faces(F).items():
    a, b = Nf[fs[0]], Nf[fs[1]]
    t = np.linspace(0, 1, 30)
    arc = (1 - t)[:, None] * a + t[:, None] * b
    arc /= np.linalg.norm(arc, axis=1, keepdims=True)
    seen = (arc @ cam.view) > 0
    Q = cam(arc * 1.001)
    Q[~seen] = np.nan
    ax2.plot(*Q.T, color=C["main"], lw=0.9, zorder=4)
seenN = (Nf @ cam.view) > 0
ax2.scatter(*cam(Nf[seenN]).T, s=8, color=C["main"], zorder=5)
ax2.set_aspect("equal")
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(-1.05, 1.05)
ax2.set_axis_off()
ax2.set_title("(b)", loc="left", fontsize=12)
cb = fig.colorbar(sc, cax=fig.add_axes([0.93, 0.2, 0.015, 0.6]))
cb.set_label(r"$\delta_i$", fontsize=11)
cb.ax.tick_params(labelsize=8)
dgfig.save(fig, __file__)

# interactive version: the polyhedron and, shifted to the right, the tiled sphere
off = np.array([3.0, 0.0, 0.0])
tr = [dm.plotly_mesh(V, F, intensity=d, colorscale="RdBu_r", cmin=-1.6, cmax=1.6, name="polyhedron (vertex color = delta)",
                     showscale=True, colorbar_title="delta"),
      dm.plotly_edges(V, F, color="#333333", width=2),
      dm.plotly_mesh(Vs + off, Fs, intensity=d[owner], colorscale="RdBu_r", cmin=-1.6, cmax=1.6, name="Gauss images")]
for (i, j), fs in dm.edge_faces(F).items():
    a, b = Nf[fs[0]], Nf[fs[1]]
    t = np.linspace(0, 1, 20)
    arc = (1 - t)[:, None] * a + t[:, None] * b
    arc = arc / np.linalg.norm(arc, axis=1, keepdims=True) * 1.005 + off
    tr.append(go.Scatter3d(x=arc[:, 0], y=arc[:, 1], z=arc[:, 2], mode="lines", line=dict(color="#000000", width=3),
                           hoverinfo="skip", showlegend=False))
dm.plotly_save(tr, __file__, "A convex polyhedron and the tiling of the sphere by the Gauss images of its vertices", "fig-40-2-2")
print("defects", np.round(d, 3), "sum", d.sum(), "min bending", theta.min(), "max |cell area - delta|", np.abs(frac - d).max())
