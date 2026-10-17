"""Figure 33.4.2: the boundary of a simplicial surface is a union of loops (Section 33.4).

(a) the untwisted band lib/ch33.py cylinder_band(24) (center radius 1, half-width 0.35): two boundary loops of 24
    edges each; (b) the Moebius band dgmesh.mobius_mesh(24) (same size): one boundary loop of 48 edges.
Faces and boundary edges are drawn together with an orthographic camera in back-to-front order.
Self-check: the boundary edges (edges in exactly one triangle) form 2 loops / 1 loop; every boundary vertex lies on
exactly two boundary edges; chi = 0 for both.
An interactive version shows both bands.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-4-2-boundary-loops.py``
"""

import pathlib
import sys

import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
LOOP_COLORS = [C["accent"], C["tangent"]]
n = 24


def loops(F):
    """Boundary loops as lists of vertices (Proposition 33.4.8)."""
    be = dm.boundary_edges(F)
    adj = {}
    for a, b in be:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    assert all(len(v) == 2 for v in adj.values())         # two boundary edges at every boundary vertex
    out, seen = [], set()
    for s in sorted(adj):
        if s in seen:
            continue
        cyc, prev, cur = [s], None, s
        while True:
            seen.add(cur)
            nxt = [w for w in adj[cur] if w != prev][0] if prev is not None else adj[cur][0]
            if nxt == s:
                break
            cyc.append(nxt)
            prev, cur = cur, nxt
        out.append(cyc)
    return out


bands = [("(a) band: 2 boundary loops", ch33.cylinder_band(n)), ("(b) Möbius band: 1 boundary loop", dm.mobius_mesh(n))]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.2))
data = []
for ax, (title, (V, F)) in zip(axs, bands):
    L = loops(F)
    assert dm.euler_characteristic(F) == 0 and dm.is_manifold(F)
    assert [len(c) for c in L] == ([n, n] if "band:" in title and "Möbius" not in title else [2 * n])
    data.append((V, F, L))
    cam = dm.Ortho(elev=32, azim=-70)
    items = []
    base = dm.hex_to_rgb(C["surface"])
    cols = dm.shade(V, F, base, light=(0.3, -0.6, 0.8), ambient=0.5)
    for f, col in zip(F, cols):
        items.append((cam.depth(V[f].mean(0)), "face", f, col))
    ef = dm.edge_faces(F)
    for k, cyc in enumerate(L):
        for a, b in zip(cyc, cyc[1:] + cyc[:1]):
            (f0,) = ef[(min(a, b), max(a, b))]               # a boundary edge lies in exactly one face
            items.append((cam.depth(V[F[f0]].mean(0)) + 1e-6, "edge", (a, b), LOOP_COLORS[k]))
    items.sort(key=lambda it: it[0])
    for z, (d, kind, obj, col) in enumerate(items):
        if kind == "face":
            ax.add_patch(Polygon(cam(V[obj]), closed=True, facecolor=col, edgecolor="#8c8c8c", lw=0.3, zorder=z))
        else:
            ax.plot(*cam(V[list(obj)]).T, color=col, lw=2.6, zorder=z, solid_capstyle="round")
    P = cam(V)
    ax.set_title(title, fontsize=11.5)
    dgfig.schematic_axes(ax, xlim=(P[:, 0].min() - 0.1, P[:, 0].max() + 0.1),
                         ylim=(P[:, 1].min() - 0.1, P[:, 1].max() + 0.1))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go  # noqa: E402

tr = []
for (title, _), (V, F, L), off in zip(bands, data, (np.array([-1.5, 0, 0]), np.array([1.5, 0, 0]))):
    W = V + off
    tr.append(dm.plotly_mesh(W, F, color="#D0D0D0", name=title))
    tr.append(dm.plotly_edges(W, F, color="#888888", width=1))
    for k, cyc in enumerate(L):
        idx = cyc + cyc[:1]
        tr.append(go.Scatter3d(x=W[idx, 0], y=W[idx, 1], z=W[idx, 2], mode="lines",
                               line=dict(color=LOOP_COLORS[k], width=8), name=f"boundary loop {k + 1}",
                               hoverinfo="name"))
dm.plotly_save(tr, __file__, "Boundary loops: band (two loops) and Möbius band (one loop)", "fig-33-4-2-interactive")
