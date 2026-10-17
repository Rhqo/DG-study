"""Figure 34.1.3: the components of a 2-vector in R^3 are the signed areas of its shadows on the three
coordinate planes (Section 34.1, Example 34.1.10).

u = (1, 2, 3), v = (-1, 0, 2) (Example 1.6.18). The parallelogram with edges u, v (blue fill) is projected
orthogonally onto the coordinate planes; the shadows are drawn on parallel planes behind the parallelogram
(floor z = -1, wall y = 3, wall x = -2; moving a shadow to a parallel plane does not change it).
Components: u ^ v = 2 e1^e2 + 5 e1^e3 + 4 e2^e3, i.e. the 2x2 minors of the 3x2 matrix [u v].
Self-checks: each shadow's signed area (in the plane oriented by (e_i, e_j), i < j) equals the minor, and
2^2 + 5^2 + 4^2 = 45 = |u x v|^2. An interactive plotly version is written next to the SVG.
"""
from pathlib import Path

import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
u = np.array([1.0, 2.0, 3.0])
v = np.array([-1.0, 0.0, 2.0])
P = np.array([np.zeros(3), u, u + v, v])          # parallelogram corners in order
U = np.column_stack([u, v])
minors = {(i, j): float(np.linalg.det(U[[i, j], :])) for (i, j) in [(0, 1), (0, 2), (1, 2)]}
assert np.allclose([minors[(0, 1)], minors[(0, 2)], minors[(1, 2)]], [2, 5, 4])
assert abs(sum(m * m for m in minors.values()) - np.dot(np.cross(u, v), np.cross(u, v))) < 1e-12

WALL = {(0, 1): (2, -1.0), (0, 2): (1, 3.0), (1, 2): (0, -2.0)}   # plane (i,j) -> (fixed coordinate, value)
shadows = {}
for (i, j), (k, val) in WALL.items():
    S = P.copy()
    S[:, k] = val
    # signed area of the shadow in the (e_i, e_j) coordinates (shoelace formula)
    a = 0.5 * sum(S[m, i] * S[(m + 1) % 4, j] - S[(m + 1) % 4, i] * S[m, j] for m in range(4))
    assert abs(a - minors[(i, j)]) < 1e-12, (i, j, a)
    shadows[(i, j)] = S

fig = plt.figure(figsize=(5.2, 4.6))
ax = dgfig.axes3d(fig, elev=24, azim=-48)

# faint walls
walls = {
    (0, 1): [[-2, -0.6, -1], [1.6, -0.6, -1], [1.6, 3, -1], [-2, 3, -1]],
    (0, 2): [[-2, 3, -1], [1.6, 3, -1], [1.6, 3, 5.6], [-2, 3, 5.6]],
    (1, 2): [[-2, -0.6, -1], [-2, 3, -1], [-2, 3, 5.6], [-2, -0.6, 5.6]],
}
for wpoly in walls.values():
    ax.add_collection3d(Poly3DCollection([wpoly], facecolor="#F2F2F2", alpha=0.5, edgecolor="#BBBBBB",
                                         linewidth=0.5, zorder=0))

names = {(0, 1): r"$e_1\wedge e_2$", (0, 2): r"$e_1\wedge e_3$", (1, 2): r"$e_2\wedge e_3$"}
lab_pos = {(0, 1): (0.9, -0.5, -1.05), (0, 2): (0.6, 3.0, 5.0), (1, 2): (-2.0, -0.5, 4.2)}
for key, S in shadows.items():
    ax.add_collection3d(Poly3DCollection([S], facecolor=C["accent"], alpha=0.35, edgecolor=C["accent"],
                                         linewidth=1.0, zorder=1))
    ax.text(*lab_pos[key], names[key] + rf"$:\ {minors[key]:+.0f}$", fontsize=11, ha="left", zorder=20)
    for m in range(4):
        seg = np.array([P[m], S[m]])
        ax.plot(*seg.T, color=C["aux"], lw=0.5, ls=(0, (2, 2)), alpha=0.7, zorder=2)

ax.add_collection3d(Poly3DCollection([P], facecolor=C["region"], alpha=0.55, edgecolor=C["tangent"],
                                     linewidth=0.9, zorder=5))
dgfig.arrow3d(ax, P[0], u, role="tangent", label=r"$u$", label_offset=(0.25, 0.0, 0.0), zorder=12)
dgfig.arrow3d(ax, P[0], v, role="third", label=r"$v$", label_offset=(-0.15, 0.0, 0.25), zorder=12)
ax.text(-0.2, 0.9, 2.6, r"$u\wedge v$", fontsize=12, zorder=20)

# coordinate axes along the walls
for d, lab in zip(np.eye(3), ("x", "y", "z")):
    base = np.array([-2.0, -0.6, -1.0])
    ax.plot(*np.array([base, base + 1.0 * d]).T, color="#555555", lw=0.8)
    ax.text(*(base + 1.2 * d), rf"${lab}$", fontsize=11, color="#555555", ha="center")

dgfig.equal_aspect(ax, np.array([[-2, -0.6, -1], [1.6, 3, 5.6]]), zoom=0.98)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
dgfig.save(fig, __file__)

# interactive version (plotly) -------------------------------------------------------------------------
import plotly.graph_objects as go  # noqa: E402


def quad(S, color, opacity, name):
    return go.Mesh3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], i=[0, 0], j=[1, 2], k=[2, 3], color=color,
                     opacity=opacity, name=name, hoverinfo="name", showscale=False)


traces = [quad(P, C["region"], 0.6, "u ^ v")]
for key, S in shadows.items():
    nm = {(0, 1): "e1^e2 shadow: +2", (0, 2): "e1^e3 shadow: +5", (1, 2): "e2^e3 shadow: +4"}[key]
    traces.append(quad(S, C["accent"], 0.45, nm))
    loop = np.vstack([S, S[:1]])
    traces.append(go.Scatter3d(x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], mode="lines",
                               line=dict(color=C["accent"], width=3), hoverinfo="skip", showlegend=False))
for vec, col, nm in ((u, C["tangent"], "u = (1, 2, 3)"), (v, C["third"], "v = (-1, 0, 2)")):
    traces.append(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines+text",
                               line=dict(color=col, width=6), text=["", nm], hoverinfo="text",
                               showlegend=False))
pfig = go.Figure(traces)
# initial camera = the static view (elev 24, azim -48). plotly's default eye (1.25, 1.25, 1.25) is almost
# perpendicular to the normal u x v = (4, -5, 2), so the parallelogram would be seen edge-on (reviewer fix).
el, az = np.radians(24), np.radians(-48)
eye = 2.1 * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
assert np.dot(eye / np.linalg.norm(eye), np.cross(u, v) / np.linalg.norm(np.cross(u, v))) > 0.95
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z",
                              camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
                   margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                   title=dict(text="u ^ v = 2 e1^e2 + 5 e1^e3 + 4 e2^e3: signed areas of the three shadows",
                              x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-34-1-3-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
