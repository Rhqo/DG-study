"""Figure 38.6.3: the same edge lengths, two shapes (ddg38.dimpled_icosahedron).

(a) The regular icosahedron inscribed in the unit sphere (dgmesh.icosahedron). (b) The same simplicial surface with
vertex x_0 reflected through the plane of its link (the five neighbors lie in a plane at height 1/sqrt 5 along x_0).
All 30 edge lengths (1.0515) and therefore all diagonal Hodge stars and the cotan Laplacian are the same; the
enclosed volume drops from 2.5362 to 1.8352 and the five faces at x_0 turn their normals by 74.75 degrees. The star of
x_0 is shaded blue, its link is drawn in orange. (a) is convex, (b) is not: Theorem 38.6.14 (Cauchy, Alexandrov) needs
both shapes to be convex. Also writes an interactive version.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-6-3-dimpled-icosahedron.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402

C = dgfig.COLORS
V, W, F = ddg38.dimpled_icosahedron()
e1, e2 = ddg38.edge_lengths(V, F), ddg38.edge_lengths(W, F)
assert len(e1) == 30 and max(abs(e1[k] - e2[k]) for k in e1) < 1e-14
assert np.abs(dm.cotan_laplacian(V, F) - dm.cotan_laplacian(W, F)).max() < 1e-12
assert abs(dm.signed_volume(V, F) - 2.5362) < 1e-4 and abs(dm.signed_volume(W, F) - 1.8352) < 1e-4
turn = ddg38.angle_deg(dm.face_normals(V, F), dm.face_normals(W, F))
star = [f for f in range(len(F)) if 0 in F[f]]
assert np.allclose(turn[star], 74.7547, atol=1e-4) and np.allclose(np.delete(turn, star), 0, atol=1e-9)
nb = sorted({int(j) for f in F[star] for j in f if j != 0})
n0 = V[0] / np.linalg.norm(V[0])
assert np.allclose(V[nb] @ n0, 1 / np.sqrt(5))
# the link in cyclic order
link = [nb[0]]
while len(link) < 5:
    link.append([j for j in nb if j not in link and any(set(f) >= {0, link[-1], j} for f in F.tolist())][0])

fig = plt.figure(figsize=(7.0, 3.5))
for k, (X, title) in enumerate(((V, "(a)"), (W, "(b)"))):
    ax = fig.add_axes([0.5 * k, 0.0, 0.5, 0.95], projection="3d")
    ax.view_init(elev=14, azim=40)
    ax.set_axis_off()
    col = dm.shade(X, F, dm.hex_to_rgb(C["surface"]))
    fc = [C["region"] if f in star else col[f] for f in range(len(F))]
    ax.add_collection3d(Poly3DCollection(X[F], facecolors=fc, edgecolors="#555555", linewidths=0.7, alpha=0.55))
    ax.plot(*X[link + link[:1]].T, color=C["accent"], lw=2.0, zorder=8)
    dgfig.point3d(ax, X[0], r"$\mathbf{x}_0$", size=14, label_offset=(0.05, 0.05, 0.1))
    dgfig.equal_aspect(ax, V, zoom=1.35)
    fig.text(0.02 + 0.5 * k, 0.93, title, fontsize=12)
dgfig.save(fig, __file__)

tr = []
for X, color, name, off in ((V, "#D0D0D0", "icosahedron", np.array([-1.3, 0, 0])),
                            (W, "#56B4E9", "dimpled icosahedron", np.array([1.3, 0, 0]))):
    tr += [dm.plotly_mesh(X + off, F, color=color, opacity=0.6, name=name), dm.plotly_edges(X + off, F)]
dm.plotly_save(tr, __file__, "Same 30 edge lengths: the icosahedron (gray) and the dimpled icosahedron (blue)",
               "fig-38-6-3")
