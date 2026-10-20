"""Figure 45.4.1: the smoothest direction fields on the icosphere with two prescribed singularities (Section 45.4,
Example 45.4.14).

Mesh: dgmesh.icosphere(3) (642 vertices). Indices k_i = +1 at (a) vertex 0 and its antipode vertex 3, (b) vertex 0
and the vertex j at 90 degrees from it (the vertex whose position is closest to orthogonal to x_0); k_i = 0 elsewhere,
so sum k = 2 = chi. xi = ch45.trivial_connection (Theorem 45.4.10, no generators on the sphere), psi = ch45.parallel_field
(psi = 0 in face 0), unit vectors X_f drawn at the centroids of the center children of the icosphere(2) faces (face
index = 3 mod 4) that face the viewer, as blue arrows of length 0.12 centered at the centroid. Same view in both panels: from the direction of
x_0 + x_j (orthographic, back faces culled). Singular vertices: black dots (the antipode in (a) is behind).
Self-checks: the curvature of xi is 2 pi k exactly; psi is parallel for xi on every edge; the field indices equal k;
max |xi| < pi; the energies xi^T star1^{-1} xi are 43.081 (a) and 47.169 (b).
Interactive version: (b) on the whole sphere.

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-4-1-icosphere-design.py
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
V, F = dm.icosphere(3)
D0, s1 = dd.d0(F, len(V)), dd.hodge1(V, F)
rho, delta = ch45.transport_angles(V, F), dm.angle_defect(V, F)
j90 = int(np.argmin(np.abs(V @ V[0] - np.cos(np.radians(90)))))   # as in Exercise 45.4.6
assert abs(V[j90] @ V[0]) < 0.05 and int(np.argmin(V @ V[0])) == 3
fields = []
for sing in ((0, 3), (0, j90)):
    k = np.zeros(len(V), int)
    k[list(sing)] = 1
    xi = ch45.trivial_connection(D0, s1, delta, rho, k)
    assert np.allclose(delta - D0.T @ xi, 2 * np.pi * k)
    psi = ch45.parallel_field(F, rho, xi)
    assert np.abs(ch45.wrap(ch45.connection_from_field(F, rho, psi) - xi)).max() < 1e-9
    ind, _ = ch45.field_indices(D0, delta, rho, psi, F)
    assert np.array_equal(ind, k) and np.abs(xi).max() < np.pi
    energy = xi @ (xi / s1)
    fields.append((sing, psi, energy, np.abs(xi).max()))
    print(f"singularities {sing}: energy {energy:.3f}, max |xi| = {np.abs(xi).max():.3f}")
assert abs(fields[0][2] - 43.081) < 5e-4 and abs(fields[1][2] - 47.169) < 5e-4

view = ch45.unit(V[0] + V[j90])
elev = np.degrees(np.arcsin(view[2]))
azim = np.degrees(np.arctan2(view[1], view[0]))
cam = dm.Ortho(elev=elev, azim=azim)
Nf = dm.face_normals(V, F)
cen = V[F].mean(1)
vis = Nf @ cam.view > 0
order = np.argsort(cam.depth(cen))
order = order[vis[order]]
light = np.clip(0.62 + 0.38 * (Nf @ ch45.unit(cam.view + 0.6 * cam.up - 0.3 * cam.right)), 0.4, 1.0)
fc = np.tile(np.array([0.9, 0.9, 0.9, 1.0]), (len(F), 1))
fc[:, :3] *= light[:, None]
sel = np.where(vis & (np.arange(len(F)) % 4 == 3) & (Nf @ cam.view > 0.15))[0]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3))
for ax, (sing, psi, energy, mx), title in zip(axs, fields, (r"(a) $k = +1$ at two antipodal vertices",
                                                             r"(b) $k = +1$ at two vertices $90^\circ$ apart")):
    ax.add_collection(PolyCollection(cam(V[F[order]]), facecolors=fc[order], edgecolors="#B4B4B4", linewidths=0.15))
    X = ch45.face_vectors(V, F, psi)
    for f in sel:
        ax.annotate("", xy=cam(cen[f] + 0.06 * X[f]), xytext=cam(cen[f] - 0.06 * X[f]),
                    arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=1.0, mutation_scale=7), zorder=4)
    for i in sing:
        if V[i] @ cam.view > 0:
            ax.plot(*cam(V[i]), "o", color=col["main"], ms=6, zorder=5)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.05)
dgfig.save(fig, __file__)

sing, psi, energy, mx = fields[1]
X = ch45.face_vectors(V, F, psi)
traces = [dm.plotly_mesh(V, F, color="#E2E2E2", name="icosphere(3)"), dm.plotly_edges(V, F, color="#B4B4B4", width=0.5)]
traces += dm.plotly_arrows(cen * 1.005, 0.06 * X, color=col["tangent"], width=3, cone=0.025, name="field")
import plotly.graph_objects as go  # noqa: E402
traces.append(go.Scatter3d(x=V[list(sing), 0] * 1.01, y=V[list(sing), 1] * 1.01, z=V[list(sing), 2] * 1.01,
                           mode="markers", marker=dict(size=6, color="#000000"), name="singularities"))
dm.plotly_save(traces, __file__, "icosphere(3): smoothest field with index +1 at two vertices 90 degrees apart",
               "fig-45-4-1")
