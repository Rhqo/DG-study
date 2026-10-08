"""Figure 27.4.1: an oriented line is a point of TS^2: its direction d and its closest point q (Section 27.4).

Line with unit direction d = (1, 0.3, 0)/|.| whose closest point to the origin is q (the component of (-0.25, 0.8, 1.5)
orthogonal to d); p = q + 1.3 d.
(a) 3D: the origin O, the line (black), d (blue arrow), the closest point q = p - <p, d> d (orange; orange segment O q),
    the point p, and the Plucker moment m = p x d (vermillion arrow from O; it is normal to the plane through O and the
    line, and |m| = |q|).
(b) The same line as a point of TS^2: the unit sphere, the point d on it, the tangent plane T_d S^2 (light blue) and
    q drawn as a tangent vector at d (orange).
Also writes an interactive plotly version of (a).
Self-checks: <q, d> = 0, <m, d> = 0, <m, q> = 0, |m| = |q|, q = d x m.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-4-1-line-coordinates.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
d = np.array([1.0, 0.3, 0.0])
d = d / np.linalg.norm(d)
q0 = np.array([-0.25, 0.8, 1.5])
q = q0 - (q0 @ d) * d
p = q + 1.3 * d
m = np.cross(p, d)
assert abs(q @ d) < 1e-12 and abs(m @ d) < 1e-12 and abs(m @ q) < 1e-12
assert abs(np.linalg.norm(m) - np.linalg.norm(q)) < 1e-12 and np.allclose(np.cross(d, m), q)

fig = plt.figure(figsize=(9.6, 4.8))
ax = dgfig.axes3d(fig, pos=121, elev=30, azim=-30)
s = np.linspace(-2.2, 2.6, 2)
L = q[None, :] + s[:, None] * d[None, :]
ax.plot(*L.T, color="k", lw=1.8)
ax.scatter(0, 0, 0, color="k", s=18)
ax.text(-0.15, -0.1, -0.35, r"$O$", fontsize=12)
ax.plot(*np.array([np.zeros(3), q]).T, color=C["accent"], lw=1.6, ls="--")
ax.scatter(*q, color=C["accent"], s=30, zorder=9)
ax.text(*(q + np.array([0.08, 0.05, -0.32])), r"$q$", fontsize=12, color=C["accent"])
ax.scatter(*p, color="k", s=14, zorder=9)
ax.text(*(p + np.array([0.08, 0.0, 0.1])), r"$p$", fontsize=12)
dgfig.arrow3d(ax, q, d, role="tangent")
ax.text(*(q + d + np.array([0.02, 0.0, 0.1])), r"$d$", fontsize=12, color=C["tangent"])
dgfig.arrow3d(ax, np.zeros(3), m, role="normal")
ax.text(*(m * 1.08 + np.array([0.05, 0, 0.05])), r"$m=p\times d$", fontsize=12, color=C["normal"])
for v, lab in ((np.array([1.5, 0, 0]), "x"), (np.array([0, 1.5, 0]), "y"), (np.array([0, 0, 1.5]), "z")):
    ax.plot(*np.array([np.zeros(3), v]).T, color=C["aux"], lw=0.7)
    ax.text(*(v * 1.05), rf"${lab}$", fontsize=11, color=C["aux"])
dgfig.equal_aspect(ax, np.vstack([L, np.zeros((1, 3)), m[None, :], q[None, :], np.eye(3) * 1.5]))
ax.set_title("(a) a line: direction $d$, closest point $q$", fontsize=12)

bx = dgfig.axes3d(fig, pos=122, elev=24, azim=72)
uu, vv = np.meshgrid(np.linspace(0, np.pi, 40), np.linspace(0, 2 * np.pi, 60), indexing="ij")
X, Y, Z = np.sin(uu) * np.cos(vv), np.sin(uu) * np.sin(vv), np.cos(uu)
dgfig.surface(bx, X, Y, Z, alpha=0.18)
bx.plot(*np.array([np.zeros(3), d]).T, color=C["tangent"], lw=1.0, ls="--", zorder=8)
bx.scatter(0, 0, 0, color="k", s=10, zorder=8)
bx.text(-0.05, 0, -0.2, r"$0$", fontsize=11)
e1 = np.cross(d, [0, 0, 1.0])
e1 /= np.linalg.norm(e1)
e2 = np.cross(d, e1)
dgfig.tangent_plane(bx, d, e1, e2, size=0.75)
bx.scatter(*d, color=C["tangent"], s=30, zorder=9)
bx.text(*(d * 1.12 + np.array([0, 0.1, -0.25])), r"$d\in S^2$", fontsize=12, color=C["tangent"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
qs = q / np.linalg.norm(q) * 0.65
dgfig.arrow3d(bx, d, qs, role="accent")
bx.text(*(d + qs + np.array([0.05, 0, 0.05])), r"$q\in T_dS^2$", fontsize=12, color=C["accent"],
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
dgfig.equal_aspect(bx, np.stack([X, Y, Z], -1).reshape(-1, 3), np.vstack([d + qs, d + 0.75 * e1, d - 0.75 * e1, d + 0.75 * e2]))
bx.set_title("(b) the same line as a point of $TS^2$", fontsize=12)
dgfig.save(fig, __file__)

import plotly.graph_objects as go

tr = [go.Scatter3d(x=L[:, 0], y=L[:, 1], z=L[:, 2], mode="lines", line=dict(color="black", width=6), name="line"),
      go.Scatter3d(x=[0, q[0]], y=[0, q[1]], z=[0, q[2]], mode="lines+markers", line=dict(color=C["accent"], width=5, dash="dash"),
                   marker=dict(size=[3, 6], color=C["accent"]), name="closest point q"),
      go.Scatter3d(x=[q[0], q[0] + d[0]], y=[q[1], q[1] + d[1]], z=[q[2], q[2] + d[2]], mode="lines",
                   line=dict(color=C["tangent"], width=8), name="direction d"),
      go.Scatter3d(x=[0, m[0]], y=[0, m[1]], z=[0, m[2]], mode="lines", line=dict(color=C["normal"], width=8),
                   name="moment m = p x d"),
      go.Scatter3d(x=[0, p[0]], y=[0, p[1]], z=[0, p[2]], mode="markers+text", text=["O", "p"],
                   marker=dict(size=4, color="black"), textposition="top center", showlegend=False)]
pf = go.Figure(tr)
pf.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=40, b=0),
                 title=dict(text="A line as (d, q) with q = closest point; Plucker moment m = p x d", x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pf.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-27-4-1-interactive", config={"displaylogo": False})
print(f"saved {out}", "q", q, "m", m, "|q|", np.linalg.norm(q))
