"""Figure 33.6.2: dual complexes of the octahedron and the icosahedron (Section 33.6).

(a) the octahedron of lib/ch33.py (gray) and its dual complex (orange): dual vertices drawn at the face centroids,
    dual edges joining the centroids of the two triangles at each edge. The dual is combinatorially a cube.
(b) the icosahedron dgmesh.icosphere(0) and its dual, combinatorially a dodecahedron.
Dual vertex positions are a drawing choice (combinatorial dual; geometry in Section 36.5). Orthographic view;
edges of the back half are dashed. An interactive version shows both pairs.
Self-check: n_k* = n_{2-k}; every dual face is a polygon with deg(i) sides (4 for the octahedron, 5 for the
icosahedron); the centroids of the octahedron are the 8 points (+-1, +-1, +-1)/3 and the dual edges have equal
length; every dual vertex is on 3 dual edges.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-6-2-dual-polyhedra.py``
"""

import itertools
import pathlib
import sys

import dgdec as dd
import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
ORANGE = C["accent"]


def dual_data(V, F):
    he = dd.HalfedgeMesh(F)
    E, _ = dd.edge_list(F)
    ef = dm.edge_faces(F)
    dE = np.array([ef[tuple(e)] for e in E.tolist()])
    dF = [he.vertex_faces(i) for i in range(len(V))]
    P = V[F].mean(1)
    return E, dE, dF, P


pairs = [("(a) octahedron and its dual", ch33.octahedron(), 4),
         ("(b) icosahedron and its dual", dm.icosphere(0), 5)]
data = []
for title, (V, F), sides in pairs:
    E, dE, dF, P = dual_data(V, F)
    assert (len(P), len(dE), len(dF)) == (len(F), len(E), len(V))
    assert all(len(p) == sides for p in dF)
    assert (np.bincount(dE.ravel(), minlength=len(F)) == 3).all()
    data.append((V, F, E, dE, P))
Vo, Fo, Eo, dEo, Po = data[0]
assert sorted(map(tuple, np.round(3 * Po).astype(int).tolist())) == sorted(itertools.product((-1, 1), repeat=3))
L = np.linalg.norm(Po[dEo[:, 0]] - Po[dEo[:, 1]], axis=1)
assert np.allclose(L, L[0])

fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.4))
for ax, (title, _, _), (V, F, E, dE, P) in zip(axs, pairs, data):
    cam = dm.Ortho(elev=18, azim=-58)
    nf = dm.face_normals(V, F)
    front = nf @ cam.view > 0
    ef = dm.edge_faces(F)
    Q, Pq = cam(V), cam(P)
    for (a, b) in E.tolist():
        vis = any(front[f] for f in ef[(a, b)])
        ax.plot(*Q[[a, b]].T, color="#444444" if vis else "#AAAAAA", lw=1.1 if vis else 0.7,
                ls="-" if vis else (0, (3, 2.5)), zorder=2 if vis else 1)
    for r, (f, g) in enumerate(dE.tolist()):
        a, b = E[r]
        vis = (V[a] @ cam.view > 0) or (V[b] @ cam.view > 0)       # the dual faces a*, b* at the edge
        ax.plot(*Pq[[f, g]].T, color=ORANGE, lw=2.0 if vis else 1.0, ls="-" if vis else (0, (3, 2.5)),
                zorder=4 if vis else 3, alpha=1.0 if vis else 0.8)
    vis_p = (P @ cam.view) > 0
    ax.plot(*Pq[vis_p].T, "o", color=ORANGE, ms=4.5, mec="#8a5a00", mew=0.6, zorder=5)
    ax.set_title(title, fontsize=11.5)
    dgfig.schematic_axes(ax, xlim=(Q[:, 0].min() - 0.1, Q[:, 0].max() + 0.1),
                         ylim=(Q[:, 1].min() - 0.1, Q[:, 1].max() + 0.1))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.08)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go  # noqa: E402

tr = []
for (title, _, _), (V, F, E, dE, P), off in zip(pairs, data, (np.array([-1.4, 0, 0]), np.array([1.4, 0, 0]))):
    tr.append(dm.plotly_mesh(V + off, F, color="#D8D8D8", opacity=0.35, name=title))
    tr.append(dm.plotly_edges(V + off, F, color="#555555", width=2))
    xs, ys, zs = [], [], []
    for f, g in dE.tolist():
        for p in (P[f] + off, P[g] + off):
            xs.append(p[0]); ys.append(p[1]); zs.append(p[2])
        xs.append(None); ys.append(None); zs.append(None)
    tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=ORANGE, width=6), name="dual edges",
                           hoverinfo="name"))
    tr.append(go.Scatter3d(x=P[:, 0] + off[0], y=P[:, 1] + off[1], z=P[:, 2] + off[2], mode="markers",
                           marker=dict(color=ORANGE, size=4), name="dual vertices", hoverinfo="name"))
dm.plotly_save(tr, __file__, "Primal (gray) and dual (orange): octahedron - cube, icosahedron - dodecahedron",
               "fig-33-6-2-interactive")
