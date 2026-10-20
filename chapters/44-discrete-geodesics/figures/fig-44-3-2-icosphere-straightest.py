"""Figure 44.3.2: discrete straightest geodesics on icospheres approach great circles (Section 44.3, Example 44.3.11).

Start: the barycentre p of face 5 of dgmesh.icosphere(level), direction d = unit(N_5 x (0.3, -0.2, 1)) (tangent to the
face). Traced with ch44.trace_straightest for length 2 pi. n = unit(p x d) is the normal of the plane through 0, p,
p + d, i.e. of the great circle with the same initial data (projected).
(a) icosphere(2) seen from the direction cos(60 deg) n + sin(60 deg) p/|p| (front faces only, painter's algorithm):
    the traced path (orange, no vertex hit) and the great circle of the unit sphere in the plane n^perp (gray
    dashed; the mesh lies inside the unit sphere, so the circle is drawn on top).
(b) the deviation |<x(s), n>| from the plane against arc length s for level 2, 3, 4.
Interactive version: icosphere(2) with the path and the great circle.
Self-checks: the path stays on the mesh (all points in a face plane), its length is 2 pi, max deviations are
0.0862, 0.0338, 0.0052 (decreasing).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-3-2-icosphere-straightest.py
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
runs = {}
for level in (2, 3, 4):
    V, F = dm.icosphere(level)
    p = V[F[5]].mean(0)
    d = ch44.unit(np.cross(dm.face_normals(V, F)[5], [0.3, -0.2, 1.0]))
    res = ch44.trace_straightest(V, F, 5, p, d, 2 * np.pi)
    P = res["points"]
    assert abs(ch44.polyline_length(P) - 2 * np.pi) < 1e-9
    assert not any(e[0] == "vertex" for e in res["events"])
    n = ch44.unit(np.cross(p, d))
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    runs[level] = (V, F, P, n, s, np.abs(P @ n))
    print(f"level {level}: max deviation {np.abs(P @ n).max():.4f}")
assert np.allclose([runs[k][5].max() for k in (2, 3, 4)], [0.0862, 0.0338, 0.0052], atol=5e-5)

V, F, P, n, s, dev = runs[2]
# camera: look along a direction close to n so the great circle appears as an ellipse
e1 = ch44.unit(np.cross(n, [0, 0, 1.0]))
e2 = np.cross(n, e1)
view = ch44.unit(np.cos(np.radians(60)) * n + np.sin(np.radians(60)) * ch44.unit(P[0]))
right = ch44.unit(np.cross([0, 0, 1.0], view)) if abs(view[2]) < 0.99 else e1
up = np.cross(view, right)
proj = lambda X: np.stack([X @ right, X @ up], -1)  # noqa: E731
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
Nf = dm.face_normals(V, F)
vis = Nf @ view > 0
order = np.argsort((V[F].mean(1) @ view))
order = order[vis[order]]
shade = dm.shade(V, F, dm.hex_to_rgb(col["surface"]))
axa.add_collection(PolyCollection(proj(V[F[order]]), facecolors=shade[order], edgecolors="#888888", linewidths=0.3))
tt = np.linspace(0, 2 * np.pi, 400)
G = (np.outer(np.cos(tt), e1) + np.outer(np.sin(tt), e2))
for X, kw in ((P, dict(color=col["accent"], lw=2.2)), (G, dict(color="#333333", lw=0.9, ls="--"))):
    front = X @ view > 0
    Q = proj(X)
    axa.plot(np.where(front, Q[:, 0], np.nan), np.where(front, Q[:, 1], np.nan), **kw)
assert P[0] @ view > 0
axa.plot(*proj(P[0]), "o", color=col["main"], ms=4)
axa.text(*(proj(P[0]) + [0.05, 0.05]), r"$p$", fontsize=12)
axa.set_aspect("equal")
axa.set_axis_off()
axa.set_title("(a) icosphere(2), front half", fontsize=11)
for level, c in ((2, col["normal"]), (3, col["tangent"]), (4, col["third"])):
    axb.plot(runs[level][4], runs[level][5], color=c, lw=1.4, label=f"icosphere({level})")
axb.set_xlabel(r"arc length $s$")
axb.set_ylabel(r"$|\langle x(s), n\rangle|$")
axb.set_xlim(0, 2 * np.pi)
axb.set_xticks([0, np.pi, 2 * np.pi])
axb.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
axb.legend(frameon=False, loc="upper left")
axb.spines[["top", "right"]].set_visible(False)
axb.set_title("(b) distance from the great-circle plane", fontsize=11)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.16, wspace=0.25)
dgfig.save(fig, __file__)

import plotly.graph_objects as go  # noqa: E402
traces = [dm.plotly_mesh(V, F, color="#D0D0D0", name="icosphere(2)"), dm.plotly_edges(V, F, width=1.0),
          go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=col["accent"], width=6),
                       name="straightest geodesic"),
          go.Scatter3d(x=G[:, 0], y=G[:, 1], z=G[:, 2], mode="lines", line=dict(color=col["aux"], width=3, dash="dash"),
                       name="great circle")]
dm.plotly_save(traces, __file__, "icosphere(2): discrete straightest geodesic (orange) and great circle (dashed)",
               "fig-44-3-2")
