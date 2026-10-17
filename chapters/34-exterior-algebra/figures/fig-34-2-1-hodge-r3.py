"""Figure 34.2.1: the Hodge star in R^3 as an orthogonal complement (Section 34.2, Definition 34.2.5,
Proposition 34.2.8).

(a) xi = u ^ v with u = (1, 0.15, 0.25), v = (0.3, 0.9, 0.35): the 1-vector star(xi) = u x v is normal to the
    parallelogram, its length equals the area |u ^ v| and (u, v, star xi) is positively oriented.
(b) the 1-vector w = u x v and its Hodge star, drawn as a square in the plane w-perp with side sqrt|w|
    (area |w|), oriented so that (w, p, q) is positively oriented (counterclockwise seen from the tip of w).
    Since star star = 1 in R^3, star w = u ^ v: the dashed outline is the parallelogram of (a), the same 2-vector.
Self-checks: |u x v| = |u ^ v|, det(u, v, u x v) > 0, p, q orthonormal-scaled with p ^ q = u ^ v
(components compared), det(w, p, q) > 0. An interactive plotly version is written next to the SVG.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
import ch34  # noqa: E402

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

C = dgfig.COLORS
u = np.array([1.0, 0.15, 0.25])
v = np.array([0.3, 0.9, 0.35])
n = np.cross(u, v)
area = np.linalg.norm(n)
assert abs(area ** 2 - ch34.inner_t(ch34.vec_tensor(u, v), ch34.vec_tensor(u, v))) < 1e-12
assert np.linalg.det(np.column_stack([u, v, n])) > 0
assert np.allclose(ch34.hodge_t(ch34.vec_tensor(u, v), 3), n)     # star(u ^ v) = u x v

# square p ^ q in the plane perpendicular to w = n, area |w|, with (w, p, q) positive
w = n
s = np.sqrt(np.linalg.norm(w))
p = u - np.dot(u, w) / np.dot(w, w) * w
p = s * p / np.linalg.norm(p)
q = np.cross(w, p)
q = s * q / np.linalg.norm(q)
assert np.linalg.det(np.column_stack([w, p, q])) > 0
assert np.allclose(ch34.vec_tensor(p, q), ch34.vec_tensor(u, v))  # star w = u ^ v (star star = 1)
print(f"area |u^v| = {area:.4f}, u x v = {np.round(n, 4)}")


def arc(center, a, b, r, n_pts=40):
    e1 = a / np.linalg.norm(a)
    bp = b - np.dot(b, e1) * e1
    e2 = bp / np.linalg.norm(bp)
    ts = np.linspace(0.3, 1.25 * np.pi + 0.3, n_pts)
    return center + r * (np.cos(ts)[:, None] * e1 + np.sin(ts)[:, None] * e2)


def draw_arc(ax, pts):
    ax.plot(*pts.T, color=C["main"], lw=1.1, zorder=9)
    dgfig.arrow3d(ax, pts[-3], pts[-1] - pts[-3], role="main", lw=1.1, head=9, zorder=9)


fig = plt.figure(figsize=(6.4, 3.2))
axA = fig.add_subplot(121, projection="3d", computed_zorder=False)
axB = fig.add_subplot(122, projection="3d", computed_zorder=False)
for ax in (axA, axB):
    ax.view_init(elev=20, azim=-58)
    ax.set_axis_off()
o = np.zeros(3)
P = np.array([o, u, u + v, v])

# (a)
axA.add_collection3d(Poly3DCollection([P], facecolor=C["region"], alpha=0.4, edgecolor=C["tangent"],
                                      linewidth=0.8, zorder=2))
dgfig.arrow3d(axA, o, u, role="tangent", label=r"$u$", label_offset=(0.0, -0.15, -0.05), fontsize=13)
dgfig.arrow3d(axA, o, v, role="third", label=r"$v$", label_offset=(-0.2, -0.05, -0.08), fontsize=13)
draw_arc(axA, arc(0.5 * (u + v), u, v, 0.18))
c = 0.5 * (u + v)
dgfig.arrow3d(axA, c, n, role="normal", label=r"$\star(u\wedge v) = u\times v$", label_offset=(0.0, 0.0, 0.12),
              fontsize=12, zorder=11)
axA.set_title("(a) star of a 2-vector", fontsize=12, y=0.95)

# (b)
Q = np.array([o - 0.5 * (p + q), o + 0.5 * (p - q), o + 0.5 * (p + q), o - 0.5 * (p - q)])
axB.add_collection3d(Poly3DCollection([Q], facecolor=C["accent"], alpha=0.35, edgecolor=C["accent"],
                                      linewidth=0.9, zorder=2))
Pc = P - 0.5 * (u + v)
axB.plot(*np.vstack([Pc, Pc[:1]]).T, color=C["aux"], lw=0.8, ls=(0, (3, 2)), zorder=3)
draw_arc(axB, arc(o, p, q, 0.22))
dgfig.arrow3d(axB, o, w, role="normal", label=r"$w$", label_offset=(0.06, 0.0, 0.08), fontsize=13, zorder=11)
axB.text(*(0.5 * (p + q) + np.array([0.15, 0.0, -0.1])), r"$\star w$", fontsize=13, color="#9A6200")
axB.set_title("(b) star of a 1-vector", fontsize=12, y=0.95)

dgfig.equal_aspect(axA, np.vstack([P, c + n, [[0, 0, -0.1]]]), zoom=0.95)
dgfig.equal_aspect(axB, np.vstack([Q, Pc, w, [[0, 0, -0.5]]]), zoom=0.95)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
dgfig.save(fig, __file__)

# interactive version (plotly) -------------------------------------------------------------------------
import plotly.graph_objects as go  # noqa: E402


def quad(S, color, opacity, name):
    return go.Mesh3d(x=S[:, 0], y=S[:, 1], z=S[:, 2], i=[0, 0], j=[1, 2], k=[2, 3], color=color,
                     opacity=opacity, name=name, hoverinfo="name")


def seg(a, b, color, text):
    return go.Scatter3d(x=[a[0], b[0]], y=[a[1], b[1]], z=[a[2], b[2]], mode="lines+text", text=["", text],
                        line=dict(color=color, width=6), hoverinfo="text", showlegend=False)


traces = [quad(P, C["region"], 0.5, "u ^ v"), seg(o, u, C["tangent"], "u"), seg(o, v, C["third"], "v"),
          seg(c, c + n, C["normal"], "star(u ^ v) = u x v"),
          quad(Q + 2.2 * np.array([1.0, 0.0, 0.0]), C["accent"], 0.45, "star w (square, area |w|)"),
          seg(2.2 * np.array([1.0, 0, 0]), 2.2 * np.array([1.0, 0, 0]) + w, C["normal"], "w")]
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                   margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                   title=dict(text="Hodge star in R^3: star(u ^ v) = u x v (left), star w = square in w-perp (right)",
                              x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-34-2-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
