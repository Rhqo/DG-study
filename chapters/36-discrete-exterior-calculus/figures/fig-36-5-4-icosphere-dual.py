"""Figure 36.5.4: the circumcentric dual cells of icosphere(1) and the Hodge star star0 (Section 36.5).

Mesh dgmesh.icosphere(1) (42 vertices on the unit sphere, 80 triangles, all acute). Each dual cell i* is made of
the right triangles (x_i, m_ij, c_f) and (x_i, c_f, m_ik), one pair in each face f = (i, j, k) at i (m = edge
midpoint, c_f = circumcenter of f, which lies in the plane of f). The cells are colored by their area star0_i.
(a) view from the side (elev 20, azim -60), back faces culled, primal edges thin gray, dual edges orange.
(b) star0_i for the 12 vertices of valence 5 (pentagons) and the 30 of valence 6 (hexagons): circumcentric
    (0.2560, 0.2865; colors as in (a)) and barycentric (one third of the adjacent triangle areas: 0.2322, 0.2960).
Self-check: every piece is a right triangle with legs |e|/2 and (|e|/2) cot(theta) > 0 (Lemma 36.5.9); the cell
areas equal dgdec.dual_areas(..., "circumcentric") and Corollary 36.5.11; the 42 areas sum to the total area of the
80 triangles (11.6659, less than 4 pi = 12.5664); the interactive version
fig-36-5-4-icosphere-dual-interactive.html is written with plotly.

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/figures/fig-36-5-4-icosphere-dual.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

V, F = dm.icosphere(1)
ang = dm.corner_angles(V, F)
assert ang.max() < np.pi / 2
C = dd.circumcenters(V, F)
pieces, owner, piece_face = [], [], []
for f, tri in enumerate(F):
    for c in range(3):
        i, j, k = tri[c], tri[(c + 1) % 3], tri[(c + 2) % 3]
        mij, mik = 0.5 * (V[i] + V[j]), 0.5 * (V[i] + V[k])
        for m in (mij, mik):
            assert abs((C[f] - m) @ (V[i] - m)) < 1e-12           # right angle at the midpoint
        pieces += [np.array([V[i], mij, C[f]]), np.array([V[i], C[f], mik])]
        owner += [i, i]
        piece_face += [f, f]
pieces = np.array(pieces)
owner = np.array(owner)
area_piece = 0.5 * np.linalg.norm(np.cross(pieces[:, 1] - pieces[:, 0], pieces[:, 2] - pieces[:, 0]), axis=1)
star0 = np.bincount(owner, area_piece, minlength=len(V))
assert np.allclose(star0, dd.dual_areas(V, F, "circumcentric"))
total = dm.face_areas(V, F).sum()
assert abs(star0.sum() - total) < 1e-12 and abs(total - 11.6659) < 1e-4
deg = np.bincount(dm.edges(F).ravel(), minlength=len(V))
assert sorted(set(deg.tolist())) == [5, 6] and (deg == 5).sum() == 12

col = dgfig.COLORS
fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1.15, 1.0]})
cam = dm.Ortho(elev=20, azim=-60)
normals = dm.face_normals(V, F)
vis_face = normals @ cam.view > 0
cmap = plt.get_cmap("YlOrBr")
lo, hi = star0.min(), star0.max()
depth = cam.depth(pieces.mean(1))
order = [p for p in np.argsort(depth) if vis_face[piece_face[p]]]
pc = PolyCollection(cam(pieces[order]), facecolors=cmap(0.15 + 0.7 * (star0[owner[order]] - lo) / (hi - lo)),
                    edgecolors="none", zorder=1)
axa.add_collection(pc)
for (a, b), fs in dm.edge_faces(F).items():
    if any(vis_face[f] for f in fs):
        axa.plot(*cam(V[[a, b]]).T, color="#8C8C8C", lw=0.5, zorder=2)
        m = 0.5 * (V[a] + V[b])
        for f in fs:
            if vis_face[f]:
                axa.plot(*cam(np.array([m, C[f]])).T, color=col["accent"], lw=1.0, zorder=3)
axa.set_aspect("equal")
axa.autoscale_view()
axa.set_axis_off()
axa.set_title(r"(a) dual cells $i^\star$, colored by $\star_0$", fontsize=11)
bary = dd.dual_areas(V, F, "barycentric")
vals = [(star0[deg == 5][0], star0[deg == 6][0]), (bary[deg == 5][0], bary[deg == 6][0])]
assert np.ptp(star0[deg == 5]) < 1e-12 and np.ptp(star0[deg == 6]) < 1e-12 and np.ptp(bary[deg == 5]) < 1e-12
xs = np.array([0.0, 1.0])
axb.bar(xs - 0.18, vals[0], width=0.34, color=[cmap(0.15), cmap(0.85)], edgecolor="#555555", lw=0.6,
        label="circumcentric")
axb.bar(xs + 0.18, vals[1], width=0.34, color="#D9D9D9", edgecolor="#555555", lw=0.6, hatch="//",
        label="barycentric")
for x, (a_, b_) in zip(xs, zip(*vals)):
    axb.text(x - 0.18, a_ + 0.003, f"{a_:.4f}", ha="center", fontsize=8.5)
    axb.text(x + 0.18, b_ + 0.003, f"{b_:.4f}", ha="center", fontsize=8.5)
axb.set_xticks(xs, ["valence 5\n(12 pentagons)", "valence 6\n(30 hexagons)"])
axb.set_ylim(0.2, 0.31)
axb.set_ylabel(r"$\star_0$")
axb.legend(fontsize=9, frameon=False, loc="upper left")
axb.set_title(r"(b) two choices of $\star_0$", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.88, bottom=0.18, wspace=0.3)
dgfig.save(fig, __file__)

import plotly.graph_objects as go  # noqa: E402

P = pieces.reshape(-1, 3)
I = np.arange(0, len(P), 3)
mesh = go.Mesh3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], i=I, j=I + 1, k=I + 2, intensity=star0[owner],
                 intensitymode="cell", colorscale="YlOrBr", cmin=lo - 0.15 * (hi - lo), cmax=hi + 0.15 * (hi - lo),
                 flatshading=True, showscale=True, colorbar=dict(title="star0"), name="dual cells",
                 hovertemplate="dual cell area %{intensity:.4f}<extra></extra>")
dual_lines_x, dual_lines_y, dual_lines_z = [], [], []
for (a, b), fs in dm.edge_faces(F).items():
    m = 0.5 * (V[a] + V[b])
    for f in fs:
        for q in (m, C[f], None):
            dual_lines_x.append(None if q is None else q[0])
            dual_lines_y.append(None if q is None else q[1])
            dual_lines_z.append(None if q is None else q[2])
duals = go.Scatter3d(x=dual_lines_x, y=dual_lines_y, z=dual_lines_z, mode="lines",
                     line=dict(color=col["accent"], width=3), hoverinfo="skip", showlegend=False)
dm.plotly_save([mesh, dm.plotly_edges(V, F, color="#777777", width=1.5), duals], __file__,
               "icosphere(1): circumcentric dual cells colored by star0 (dual cell area)", "fig-36-5-4")
