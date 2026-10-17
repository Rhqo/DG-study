"""Figure 34.4.1: a 2-form measures the signed area of the shadow of a parallelogram on a plane
(Section 34.4, Proposition 34.4.1).

u = (1.1, 0, 0.25), v = (0.2, 0.9, 0.35). The covectors alpha, beta have orthonormal sharps
a = alpha^sharp = (cos t, 0, sin t), b = beta^sharp = (0, 1, 0) with t = 40 degrees; they span the plane P,
oriented by (a, b). The shadow pi(u ^ v) (orthogonal projection onto P) is drawn on the parallel copy P - 1.25 n
(n = a x b; shifted by 1.25 n), which does not change its shape. Its signed area equals
(alpha ^ beta)(u, v) = alpha(u) beta(v) - alpha(v) beta(u) = 0.903.
Self-checks: a, b orthonormal; the projected corners lie in the plane; the shoelace area in (a, b) coordinates equals
the 2-form value; |value| <= |u ^ v| (projection shrinks area). An interactive plotly version is written too.
"""
from pathlib import Path

import numpy as np

import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
u = np.array([1.1, 0.0, 0.25])
v = np.array([0.2, 0.9, 0.35])
t = np.deg2rad(40.0)
a = np.array([np.cos(t), 0.0, np.sin(t)])
b = np.array([0.0, 1.0, 0.0])
n = np.cross(a, b)
assert abs(a @ a - 1) < 1e-12 and abs(b @ b - 1) < 1e-12 and abs(a @ b) < 1e-12

value = (a @ u) * (b @ v) - (a @ v) * (b @ u)
P = np.array([np.zeros(3), u, u + v, v])
coords = np.stack([P @ a, P @ b], 1)              # (a, b)-coordinates of the projected corners
shoelace = 0.5 * sum(coords[m, 0] * coords[(m + 1) % 4, 1] - coords[(m + 1) % 4, 0] * coords[m, 1] for m in range(4))
assert abs(shoelace - value) < 1e-12
assert abs(value) <= np.linalg.norm(np.cross(u, v)) + 1e-12
off = -1.25 * n
S = coords[:, :1] * a + coords[:, 1:] * b + off  # shadow on the shifted plane
assert np.allclose((S - off) @ n, 0)
print(f"(alpha^beta)(u, v) = {value:.4f}, |u^v| = {np.linalg.norm(np.cross(u, v)):.4f}")

fig = plt.figure(figsize=(5.4, 4.4))
ax = dgfig.axes3d(fig, elev=28, azim=-125)

# the plane P (shifted copy), as a square patch
sq = 1.45
Pl = np.array([off - 0.25 * sq * a - 0.25 * sq * b, off + sq * a - 0.25 * sq * b, off + sq * a + sq * b,
               off - 0.25 * sq * a + sq * b])
ax.add_collection3d(Poly3DCollection([Pl], facecolor=C["region"], alpha=0.15, edgecolor=C["tangent"], linewidth=0.6,
                                     zorder=0))
ax.add_collection3d(Poly3DCollection([S], facecolor=C["accent"], alpha=0.45, edgecolor=C["accent"], linewidth=1.0,
                                     zorder=1))
for m in range(4):
    ax.plot(*np.array([P[m], S[m]]).T, color=C["aux"], lw=0.6, ls=(0, (2, 2)), zorder=2)
ax.add_collection3d(Poly3DCollection([P], facecolor=C["region"], alpha=0.5, edgecolor=C["tangent"], linewidth=0.9,
                                     zorder=4))
dgfig.arrow3d(ax, np.zeros(3), u, role="tangent", label=r"$u$", label_offset=(0.1, 0.0, 0.0), zorder=10)
dgfig.arrow3d(ax, np.zeros(3), v, role="third", label=r"$v$", label_offset=(0.0, 0.0, 0.1), zorder=10)
dgfig.arrow3d(ax, off, a, role="covector", label=r"$\alpha^\sharp$", label_offset=(0.12, 0.0, 0.0), zorder=9)
dgfig.arrow3d(ax, off, b, role="covector", label=r"$\beta^\sharp$", label_offset=(0.0, 0.1, 0.05), zorder=9)
ax.text(*(0.5 * (u + v) + np.array([0.05, -0.05, 0.12])), r"$u\wedge v$", fontsize=12, zorder=20)
ax.text(*(S[2] + np.array([0.15, 0.0, -0.15])), r"$(\alpha\wedge\beta)(u,v) = %.3f$" % value, fontsize=11,
        color="#9A6200", zorder=20)
ax.text(*(Pl[1] + np.array([0.0, 0.0, -0.1])), r"$P$", fontsize=12, color=C["tangent"])

dgfig.equal_aspect(ax, np.vstack([P, S, Pl]), zoom=1.0)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
dgfig.save(fig, __file__)

# interactive version (plotly) -------------------------------------------------------------------------
import plotly.graph_objects as go  # noqa: E402


def quad(Q, color, opacity, name):
    return go.Mesh3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], i=[0, 0], j=[1, 2], k=[2, 3], color=color, opacity=opacity,
                     name=name, hoverinfo="name")


def seg(p, q, color, text, width=6):
    return go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines+text", text=["", text],
                        line=dict(color=color, width=width), hoverinfo="text", showlegend=False)


traces = [quad(Pl, C["region"], 0.18, "plane P"), quad(S, C["accent"], 0.5, f"shadow: signed area {value:.3f}"),
          quad(P, C["region"], 0.6, "u ^ v"), seg(np.zeros(3), u, C["tangent"], "u"), seg(np.zeros(3), v, C["third"], "v"),
          seg(off, off + a, C["covector"], "alpha#"), seg(off, off + b, C["covector"], "beta#")]
for m in range(4):
    traces.append(go.Scatter3d(x=[P[m, 0], S[m, 0]], y=[P[m, 1], S[m, 1]], z=[P[m, 2], S[m, 2]], mode="lines",
                               line=dict(color=C["aux"], width=2, dash="dash"), hoverinfo="skip", showlegend=False))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                   margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                   title=dict(text=f"(alpha ^ beta)(u, v) = {value:.3f}: signed area of the shadow on P", x=0.5,
                              font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-34-4-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
