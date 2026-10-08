"""Figure 27.1.1: the fibers of the pinhole projection are rays, and ker J is the ray direction (Section 27.1).

(a) 3D: camera center O, the image plane t_z = 1 (normalized camera, f = 1, c = 0), the plane t_z = 3.5 (dashed), and
    the rays (fibers) through a 3 x 3 grid of image points and the four corners. One ray (orange) passes through ubar_0 = (0.5, 0.25); on it the point t_0 = 3 (0.5, 0.25, 1).
(b) The cross-section t_y = 0: rays through u_x = -0.6, ..., 0.9 (gray, every ray is a level set of u_x), the image
    line t_z = 1, the point t_0 = (1.2, 2.4) (u_x = 0.5). Blue arrow: k = 0.5 t_0/|t_0| (ker J). Green arrow: v, the
    same length, perpendicular to the ray. Dashed lines from O through t_0 + k and t_0 + v show where the moved points
    land on the image line: t_0 + k lands exactly on u_x = 0.5, t_0 + v lands at u_x = 0.5 + 0.257 (exact) and the
    linear prediction J v = 0.233.
Also writes an interactive plotly version with the same rays and arrows.
Self-checks: J t_0 = 0, phi(t_0 + k) = phi(t_0), the exact and linear displacements agree to first order.

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/figures/fig-27-1-1-fibers-kernel.py``
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


def phi(t):
    return np.array([t[0] / t[2], t[1] / t[2]])


def Jmat(t):
    return np.array([[1 / t[2], 0, -t[0] / t[2] ** 2], [0, 1 / t[2], -t[1] / t[2] ** 2]])


# ---------------------------------------------------------------- (a) 3D
ub0 = np.array([0.5, 0.25])
t0 = 3.0 * np.array([ub0[0], ub0[1], 1.0])
assert np.allclose(Jmat(t0) @ t0, 0)
assert np.allclose(phi(2.0 * t0), phi(t0))
k3 = 0.9 * t0 / np.linalg.norm(t0)
v3 = np.cross(t0, [0, 1.0, 0])
v3 = 0.9 * v3 / np.linalg.norm(v3)

fig = plt.figure(figsize=(9.6, 4.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.05)
ax = dgfig.axes3d(fig, pos=gs[0], elev=18, azim=-40)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# plot coordinates (t_x, t_z, -t_y): depth goes into the picture, t_y (image "down") points down
P = lambda q: np.array([q[0], q[2], -q[1]])
W2, H2 = 0.8, 0.6
rect = np.array([[-W2, -H2, 1], [W2, -H2, 1], [W2, H2, 1], [-W2, H2, 1]])
ax.add_collection3d(Poly3DCollection([np.array([P(q) for q in rect])], facecolor=C["region"], alpha=0.3,
                                     edgecolor=C["tangent"], lw=0.9))
DFAR = 3.5
far = DFAR * rect
farc = np.array([P(q) for q in np.vstack([far, far[:1]])])
ax.plot(farc[:, 0], farc[:, 1], farc[:, 2], color=C["aux"], lw=0.7, ls="--", zorder=1)
gx, gy = np.linspace(-0.6, 0.6, 3), np.linspace(-0.45, 0.45, 3)
ray_dirs = [np.array([a_, b_, 1.0]) for a_ in gx for b_ in gy] + [q for q in rect]
for dvec in ray_dirs:
    pts = np.array([P([0, 0, 0]), P(DFAR * dvec)])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=C["aux"], lw=0.6, zorder=2)
    ax.scatter(*P(dvec), color=C["tangent"], s=6, zorder=4)
end0 = 3.8 * np.array([ub0[0], ub0[1], 1.0])
pts = np.array([P([0, 0, 0]), P(end0)])
ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=C["accent"], lw=2.2, zorder=5)
ax.scatter(*P([0, 0, 0]), color="k", s=18, zorder=8)
ax.text(*(P([0, 0, 0]) + np.array([-0.45, 0.0, -0.1])), r"$O$", fontsize=12)
ax.scatter(*P(t0), color="k", s=18, zorder=8)
ax.text(*(P(t0) + np.array([-0.1, 0.0, -0.55])), r"$t_0$", fontsize=12)
dgfig.arrow3d(ax, P(t0), P(k3), role="tangent")
ax.text(*(P(t0 + k3) + np.array([0.1, 0.0, -0.2])), r"$\ker J$", fontsize=12, color=C["tangent"])
dgfig.arrow3d(ax, P(t0), P(v3), role="third")
ax.text(*(P(t0 + v3) + np.array([-0.35, 0.0, 0.1])), r"$v$", fontsize=12, color=C["third"])
ax.scatter(*P([ub0[0], ub0[1], 1.0]), color=C["accent"], s=22, zorder=8)
ax.text(*P([-1.0, -0.75, 1.0]), r"$t_z=1$", fontsize=11, color=C["tangent"])
ax.text(*P([2.9, 2.2, DFAR]), r"$t_z=3.5$", fontsize=11, color=C["aux"])
ax.set_xlim(-2.8, 2.8)
ax.set_ylim(0, 3.8)
ax.set_zlim(-2.1, 2.1)
ax.set_box_aspect((5.6, 3.8, 4.2), zoom=1.12)
ax.set_title(r"(a) fibers $\varphi^{-1}(u)$ are rays from $O$", fontsize=12)

# ---------------------------------------------------------------- (b) cross-section t_y = 0
ax2 = fig.add_subplot(gs[1])
t0b = np.array([1.2, 0.0, 2.4])
L = 0.5
kb = L * t0b / np.linalg.norm(t0b)
vb = L * np.array([t0b[2], 0, -t0b[0]]) / np.linalg.norm(t0b)        # perpendicular, pointing to larger u_x
assert abs(kb @ vb) < 1e-12
assert np.isclose(phi(t0b + kb)[0], phi(t0b)[0])
du_exact = phi(t0b + vb)[0] - phi(t0b)[0]
du_lin = (Jmat(t0b) @ vb)[0]
assert abs(du_exact - 0.2569) < 1e-3 and abs(du_lin - 0.2329) < 1e-3, (du_exact, du_lin)
for a in np.arange(-0.6, 0.91, 0.15):
    ax2.plot([0, 3.3 * a], [0, 3.3], color=C["aux"], lw=0.6, zorder=1)
ax2.plot([0, 3.3 * 0.5], [0, 3.3], color=C["accent"], lw=2.0, zorder=2)
ax2.plot([-1.3, 2.4], [1, 1], color="k", lw=1.2, zorder=2)
ax2.text(-1.25, 0.82, r"image line $t_z=1$", fontsize=11)
ax2.annotate("", xy=(t0b + kb)[[0, 2]], xytext=t0b[[0, 2]],
             arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=1.8, mutation_scale=14), zorder=5)
ax2.annotate("", xy=(t0b + vb)[[0, 2]], xytext=t0b[[0, 2]],
             arrowprops=dict(arrowstyle="-|>", color=C["third"], lw=1.8, mutation_scale=14), zorder=5)
ax2.plot(*t0b[[0, 2]], "ko", ms=4, zorder=6)
ax2.text(t0b[0] - 0.33, t0b[2] + 0.02, r"$t_0$", fontsize=12)
ax2.text((t0b + kb)[0] - 0.33, (t0b + kb)[2] + 0.02, r"$k\in\ker J$", fontsize=12, color=C["tangent"])
ax2.text((t0b + vb)[0] + 0.05, (t0b + vb)[2] - 0.12, r"$v$", fontsize=12, color=C["third"])
# where t0 + v lands
tv = t0b + vb
ax2.plot([0, tv[0] / tv[2] * 3.0], [0, 3.0], color=C["third"], lw=0.9, ls="--", zorder=3)
ax2.plot([0.5], [1], "o", color=C["accent"], ms=5, zorder=6)
ax2.plot([0.5 + du_exact], [1], "o", color=C["third"], ms=5, zorder=6)
ax2.annotate("", xy=(0.5 + du_exact, 0.9), xytext=(0.5, 0.9),
             arrowprops=dict(arrowstyle="-|>", color=C["third"], lw=1.4, mutation_scale=11))
ax2.text(0.82, 0.42, r"$Jv=0.233$" + "\n" + r"(exact $0.257$)", fontsize=11, color=C["third"])
ax2.text(0.02, 1.12, r"$Jk=0$", fontsize=12, color=C["tangent"])
ax2.plot(0, 0, "ko", ms=4)
ax2.text(-0.12, -0.2, r"$O$", fontsize=12)
ax2.text(1.75, 3.2, r"$u_x=0.5$", fontsize=11, color=C["accent"])
ax2.text(-1.25, 3.05, r"rays $=$" + "\n" + r"level sets of $u_x$", fontsize=11, color=C["aux"])
ax2.set_xlim(-1.3, 2.4)
ax2.set_ylim(-0.3, 3.4)
ax2.set_aspect("equal")
ax2.set_xlabel(r"$t_x$")
ax2.set_ylabel(r"$t_z$ (depth)")
ax2.set_title(r"(b) cross-section $t_y=0$", fontsize=12)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go

traces = []
for dvec in ray_dirs:
    end = DFAR * dvec
    traces.append(go.Scatter3d(x=[0, end[0]], y=[0, end[1]], z=[0, end[2]], mode="lines",
                               line=dict(color="#999999", width=2), hoverinfo="skip", showlegend=False))
traces.append(go.Scatter3d(x=[0, end0[0]], y=[0, end0[1]], z=[0, end0[2]], mode="lines",
                           line=dict(color=C["accent"], width=6), name="ray through u0"))
traces.append(go.Mesh3d(x=rect[:4, 0], y=rect[:4, 1], z=rect[:4, 2], i=[0, 0], j=[1, 2], k=[2, 3],
                        color=C["region"], opacity=0.3, hoverinfo="skip"))
for vec, col, nm in [(k3, C["tangent"], "ker J (along the ray)"), (v3, C["third"], "v (across the ray)")]:
    e = t0 + vec
    traces.append(go.Scatter3d(x=[t0[0], e[0]], y=[t0[1], e[1]], z=[t0[2], e[2]], mode="lines",
                               line=dict(color=col, width=7), name=nm))
    traces.append(go.Cone(x=[e[0]], y=[e[1]], z=[e[2]], u=[vec[0]], v=[vec[1]], w=[vec[2]], sizemode="absolute",
                          sizeref=0.15, anchor="tip", colorscale=[[0, col], [1, col]], showscale=False, hoverinfo="skip"))
traces.append(go.Scatter3d(x=[0, t0[0]], y=[0, t0[1]], z=[0, t0[2]], mode="markers+text", text=["O", "t0"],
                           marker=dict(size=4, color="black"), textposition="top center", hoverinfo="text"))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="t_x", yaxis_title="t_y", zaxis_title="t_z (depth)",
                              camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=1.4, y=-0.8, z=-1.2))),
                   margin=dict(l=0, r=0, t=40, b=0), showlegend=True,
                   title=dict(text="Fibers of the pinhole projection are rays; ker J points along the ray", x=0.5,
                              font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-27-1-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
