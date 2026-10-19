"""Figure 39.5.3: rebuilding a mesh from edge lengths and bending angles (Theorem 39.5.8, Implementation 39.5).

icosphere(1) (42 vertices, 80 faces). Edge lengths and bending angles are read off the mesh and the mesh is rebuilt
by gluing triangles in depth-first order from face 0 (ddg39.reconstruct, same algorithm as the page).
(a) the first 24 faces placed, shaded from dark (early) to light (late), over the outline of the full result.
(b) the same data with one bending angle on a tree edge (the edge used to place face number 60) increased by 0.3:
the faces placed after that step rotate as a block (light blue); edges whose rebuilt length differs from the given length by more
than 1e-6 are drawn thick orange. Also writes an interactive version of (b).

Run: PYTHONPATH=tools python3 chapters/39-surface-curvature/figures/fig-39-5-3-reconstruction.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg39  # noqa: E402

C = dgfig.COLORS
V, F = dm.icosphere(1)
ell, theta = ddg39.lengths_angles(V, F)
Y, seq, tree = ddg39.reconstruct(F, ell, theta)
assert ddg39.rigid_error(V, Y) < 1e-12 and len(seq) == len(F) and len(tree) == len(F) - 1
# align the rebuilt mesh with the original for drawing (proper rigid motion, Kabsch)
cx, cy = Y.mean(0), V.mean(0)
U_, S_, Vt_ = np.linalg.svd((Y - cx).T @ (V - cy))
Rm = Vt_.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt_.T @ U_.T))]) @ U_.T
assert np.linalg.det(Rm) > 0
Ya = (Y - cx) @ Rm.T + cy
assert np.abs(Ya - V).max() < 1e-12

# perturb the tree edge that placed face number 40
NPLACE = 60
F_list = F.tolist()
f_new = seq[NPLACE]
cands = [e for e in tree if set(e) <= set(F_list[f_new])]
# the tree edge shared by f_new and an earlier face
placed_before = set(seq[:NPLACE])
edge_star = None
for e in cands:
    others = [g for g, f in enumerate(F_list) if g != f_new and set(e) <= set(f)]
    if others and others[0] in placed_before:
        edge_star = e
assert edge_star is not None
theta2 = dict(theta)
theta2[edge_star] += 0.3
Y2, seq2, _ = ddg39.reconstruct(F, ell, theta2)
# align using face 0 only (it is placed first and is unaffected)
def frame(P, f):
    e = P[f[1]] - P[f[0]]
    e /= np.linalg.norm(e)
    n = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
    n /= np.linalg.norm(n)
    return np.column_stack([e, np.cross(n, e), n])
Fr = frame(V, F[0]) @ frame(Y2, F[0]).T
Y2a = (Y2 - Y2[F[0][0]]) @ Fr.T + V[F[0][0]]
bad = [(i, j) for (i, j), l in ell.items() if abs(np.linalg.norm(Y2[j] - Y2[i]) - l) > 1e-6]
assert 0 < len(bad) < len(ell) and edge_star not in bad          # the changed edge itself keeps its length
moved = np.linalg.norm(Y2a - V, axis=1) > 1e-6
print(f"perturbed edge {edge_star}, wrong edges {len(bad)}, moved vertices {moved.sum()}")

fig = plt.figure(figsize=(7.2, 3.6))
ax = dgfig.axes3d(fig, 121, elev=20, azim=-50)
ax.add_collection3d(Poly3DCollection(V[F], facecolors=(0.85, 0.85, 0.85, 0.08), edgecolors="#BBBBBB", linewidths=0.3))
cm = plt.get_cmap("Blues")
NSHOW = 24
for r_, g in enumerate(seq[:NSHOW]):
    col = cm(0.95 - 0.6 * r_ / NSHOW)
    ax.add_collection3d(Poly3DCollection([Ya[F[g]]], facecolors=[col], edgecolors="#333333", linewidths=0.6,
                                         alpha=0.95))
ax.text(*(Ya[F[seq[0]]].mean(0) * 1.15), r"$1$", fontsize=11)
dgfig.equal_aspect(ax, V, zoom=1.3)
ax.set_title("(a)", loc="left", fontsize=12, y=0.96)

ax = dgfig.axes3d(fig, 122, elev=20, azim=-50)
col = dm.shade(Y2a, F, dm.hex_to_rgb(C["surface"]))
col[seq2[NPLACE:], :3] = np.array(dm.hex_to_rgb(C["region"]))[None] * 0.9
ax.add_collection3d(Poly3DCollection(Y2a[F], facecolors=col, edgecolors="#555555", linewidths=0.3, alpha=0.75))
segs = [[Y2a[i], Y2a[j]] for i, j in bad]
ax.add_collection3d(Line3DCollection(segs, colors=C["accent"], linewidths=2.2, zorder=10))
dgfig.equal_aspect(ax, np.vstack([V, Y2a]), zoom=1.3)
ax.set_title("(b)", loc="left", fontsize=12, y=0.96)
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95, wspace=0.0)
dgfig.save(fig, __file__)

tr = [dm.plotly_mesh(Y2a, F, color="#D0D0D0", opacity=0.8, name="rebuilt mesh"), dm.plotly_edges(Y2a, F)]
import plotly.graph_objects as go  # noqa: E402
xs, ys, zs = [], [], []
for i, j in bad:
    for p in (Y2a[i], Y2a[j]):
        xs.append(p[0]), ys.append(p[1]), zs.append(p[2])
    xs.append(None), ys.append(None), zs.append(None)
tr.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=C["accent"], width=7), name="wrong lengths",
                       hoverinfo="skip"))
dm.plotly_save(tr, __file__, "icosphere(1) rebuilt with one bending angle changed by 0.3: orange edges have wrong length",
               "fig-39-5-3")
