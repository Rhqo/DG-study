"""Figure 44.4.2: the discrete exponential map on icosphere(2) (Section 44.4, Example 44.4.11).

p = barycentre of face 0 of dgmesh.icosphere(2). 48 straightest geodesics from p (ch44.trace_straightest) in the
directions R(2 pi k/48) e (e = unit(N_0 x (1, 0.2, 0)), rotation about N_0), length 2.8. The images of the circles
|v| = 0.7, 1.4, 2.1, 2.8 (orange polylines through the ray end points) are compared with the geodesic circles of the
unit sphere about p/|p| with the same radii (black dashed). View from cos(1) p/|p| + sin(1) e2 (front half only;
e2 is a unit tangent vector at p/|p|).
Interactive version: the same rays and circles on icosphere(2).
Self-checks: every ray has the requested length; the angle between the discrete and the smooth image of the ray end
points at |v| = 1.4 is at most 0.0794 over the 48 directions.

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/figures/fig-44-4-3-discrete-exp-icosphere.py
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
V, F = dm.icosphere(2)
T = ch44.MeshTopo(V, F)
p = V[F[0]].mean(0)
N0 = dm.face_normals(V, F)[0]
e = ch44.unit(np.cross(N0, [1.0, 0.2, 0.0]))
radii = [0.7, 1.4, 2.1, 2.8]
rays, ends = [], {R_: [] for R_ in radii}
ph = ch44.unit(p)
worst = 0.0
for k in range(48):
    a = 2 * np.pi * k / 48
    d = np.cos(a) * e + np.sin(a) * np.cross(N0, e)
    res = ch44.trace_straightest(V, F, 0, p, d, radii[-1], topo=T)
    P = res["points"]
    assert abs(ch44.polyline_length(P) - radii[-1]) < 1e-9
    rays.append(P)
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    vh = ch44.unit(d - (d @ ph) * ph)
    for R_ in radii:
        j = min(np.searchsorted(s, R_) - 1, len(P) - 2)
        q = P[j] + (R_ - s[j]) * ch44.unit(P[j + 1] - P[j])
        ends[R_].append(q)
        if R_ == 1.4:
            worst = max(worst, np.arccos(np.clip(ch44.unit(q) @ ch44.sphere_exp(ph, R_ * vh), -1, 1)))
assert abs(worst - 0.0794) < 5e-4
print(f"max angle between discrete and smooth exp at |v| = 1.4: {worst:.4f}")

e1 = ch44.unit(np.cross(ph, [0, 0, 1.0]))
e2 = np.cross(ph, e1)
view = ch44.unit(np.cos(1.0) * ph + np.sin(1.0) * e2)
right = ch44.unit(np.cross([0, 0, 1.0], view))
up = np.cross(view, right)
proj = lambda X: np.stack([np.asarray(X) @ right, np.asarray(X) @ up], -1)  # noqa: E731
fig, ax = plt.subplots(figsize=(4.2, 4.2))
Nf = dm.face_normals(V, F)
vis = Nf @ view > 0
order = np.argsort(V[F].mean(1) @ view)
order = order[vis[order]]
shade = dm.shade(V, F, dm.hex_to_rgb(col["surface"]))
ax.add_collection(PolyCollection(proj(V[F[order]]), facecolors=shade[order], edgecolors="#999999", linewidths=0.3))


def plot_front(X, **kw):
    X = np.asarray(X)
    front = X @ view > 0
    Q = proj(X)
    ax.plot(np.where(front, Q[:, 0], np.nan), np.where(front, Q[:, 1], np.nan), **kw)


for P in rays:
    plot_front(P, color=col["tangent"], lw=0.5)
tt = np.linspace(0, 2 * np.pi, 300)
for R_ in radii:
    E = np.array(ends[R_] + ends[R_][:1])
    plot_front(E, color=col["accent"], lw=1.8)
    G = np.cos(R_) * ph + np.sin(R_) * (np.outer(np.cos(tt), e1) + np.outer(np.sin(tt), e2))
    plot_front(G, color=col["main"], lw=0.8, ls="--")
ax.plot(*proj(p), "o", color=col["main"], ms=4)
ax.text(*(proj(p) + [0.03, 0.04]), r"$p$", fontsize=12)
ax.set_aspect("equal")
ax.set_xlim(-1.05, 1.05)
ax.set_ylim(-1.05, 1.05)
ax.set_axis_off()
dgfig.save(fig, __file__)

import plotly.graph_objects as go  # noqa: E402
traces = [dm.plotly_mesh(V, F, color="#D0D0D0", name="icosphere(2)"), dm.plotly_edges(V, F, width=1.0)]
for P in rays:
    traces.append(go.Scatter3d(x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="lines", line=dict(color=col["tangent"], width=2),
                               hoverinfo="skip", showlegend=False))
for R_ in radii:
    E = np.array(ends[R_] + ends[R_][:1])
    traces.append(go.Scatter3d(x=E[:, 0], y=E[:, 1], z=E[:, 2], mode="lines", line=dict(color=col["accent"], width=5),
                               name=f"|v| = {R_}"))
dm.plotly_save(traces, __file__, "icosphere(2): discrete exponential map from p (blue rays, orange images of circles)",
               "fig-44-4-2")
