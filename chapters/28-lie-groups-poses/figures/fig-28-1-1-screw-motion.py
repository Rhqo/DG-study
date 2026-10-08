"""Figure 28.1.1: the twist xi = (v, w) generates a screw motion; v is the initial velocity, not the translation (Section 28.1).

xi = (v, w) with w = e_3 (unit angular velocity about z) and v = (0, a, b), a = 1, b = 0.3. The curve
T(tau) = Exp(tau xi), 0 <= tau <= pi, is applied to a body frame that starts at the world frame.
  * The origin of the body frame moves along c(tau) = (a cos tau - a, a sin tau, b tau): the running-example helix
    gamma(tau) = (a cos tau, a sin tau, b tau) (GUIDELINES section 7, Example 5.4.5) shifted by (-a, 0, 0). The screw axis is the
    vertical line x = -a, y = 0, the pitch is b.
  * c'(0) = v (blue arrow): v is the velocity of the origin at s = 0.
  * The translation of Exp(pi xi) is t = V(pi w) (pi v) = (-2a, 0, pi b) (orange chord), not pi v = (0, pi a, pi b)
    (gray dotted arrow).
(a) 3D view with body-frame triads at tau = 0, pi/4, ..., pi (x: blue, y: green, z: vermilion; length 0.45), v and t.
    The triad at tau = 0 is the world frame (labels x, y, z). Height cue: the shadow of c(tau) on the plane z = 0
    (light gray dotted) and the vertical drop at tau = pi of length b pi = 0.94 (the rise along the screw axis).
(b) top view (projection to the xy-plane) of the same objects, together with pi v.
Also writes an interactive plotly version (fig-28-1-1-screw-motion-interactive.html).
Self-check: Exp(s xi) from the closed form equals the matrix exponential; c(s) = gamma(s) - (a, 0, 0);
the screw axis point q = w x v / |w|^2 = (-a, 0, 0) is fixed up to the shift along the axis.

Run from the project root:
``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/figures/fig-28-1-1-screw-motion.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dglie as L  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
A, B = 1.0, 0.3
w = np.array([0.0, 0.0, 1.0])
v = np.array([0.0, A, B])
xi = np.concatenate([v, w])
S_END = np.pi
ss = np.linspace(0, S_END, 301)
Ts = [L.expSE3(s * xi) for s in ss]
path = np.array([T[:3, 3] for T in Ts])

# self-checks
for s, T in zip(ss[::30], Ts[::30]):
    assert np.allclose(T, L.expm(L.hat6(s * xi)), atol=1e-12)
helix = np.stack([A * np.cos(ss), A * np.sin(ss), B * ss], 1)
assert np.allclose(path, helix - np.array([A, 0, 0]), atol=1e-12)
q_axis = np.cross(w, v) / (w @ w)
assert np.allclose(q_axis, [-A, 0, 0])
t_end = Ts[-1][:3, 3]
assert np.allclose(t_end, [-2 * A, 0, np.pi * B], atol=1e-12)
assert np.allclose((path[1] - path[0]) / (ss[1] - ss[0]), v, atol=2e-2)

marks = np.linspace(0, S_END, 5)
TRI = 0.45
axcol = [C["tangent"], C["third"], C["normal"]]

fig = plt.figure(figsize=(8.4, 4.4))
ax = dgfig.axes3d(fig, pos=121, elev=24, azim=-62)
ax.set_position([0.0, 0.0, 0.55, 0.95])
ax.plot(*path.T, color=C["main"], lw=1.8, zorder=5)
# height cue: shadow of the path on z = 0 and the drop line at tau = pi (rise b pi along the screw axis)
ax.plot(path[:, 0], path[:, 1], 0 * path[:, 2], color="#AAAAAA", lw=1.0, ls=(0, (1.5, 2)), zorder=1)
ax.plot([path[-1, 0]] * 2, [path[-1, 1]] * 2, [0, path[-1, 2]], color="#AAAAAA", lw=1.0, ls=(0, (1.5, 2)), zorder=1)
ax.text(path[-1, 0] + 0.1, path[-1, 1] - 0.05, 0.12, r"rise $b\pi \approx 0.94$", fontsize=10,
        color=C["aux"], ha="left", zorder=13)
ax.plot([-A, -A], [0, 0], [-0.25, np.pi * B + 0.35], color=C["aux"], lw=1.0, ls=(0, (4, 3)), zorder=2)
ax.text(-A - 0.1, 0, np.pi * B + 0.5, "screw axis", fontsize=10.5, color=C["aux"], ha="right")
for s in marks:
    T = L.expSE3(s * xi)
    for k in range(3):
        dgfig.arrow3d(ax, T[:3, 3], T[:3, k], role=axcol[k], scale=TRI, lw=1.3, head=8, zorder=8)
dgfig.arrow3d(ax, [0, 0, 0], v, role="tangent", scale=1.0, lw=2.2, head=12, zorder=12)
ax.text(0.15, 1.05, 0.35, r"$v = c'(0)$", color=C["tangent"], fontsize=12, zorder=13)
dgfig.arrow3d(ax, [0, 0, 0], t_end, role="accent", lw=2.0, head=12, zorder=11)
ax.text(*(t_end * 0.5 + np.array([0.1, -0.2, -0.45])), r"$t = V(\pi\omega)\,\pi v$", color="#B07400", fontsize=12,
        zorder=13)
for k, lab in enumerate(["x", "y", "z"]):          # the triad at tau = 0 is the world frame
    tip = 1.12 * TRI * np.eye(3)[k] + np.array([0.0, 0.0, 0.04 if k < 2 else 0.0])
    ax.text(*tip, f"${lab}$", color=axcol[k], fontsize=10, ha="center", va="center", zorder=13)
pv = np.pi * v
ax.text(*(path[-1] + np.array([-0.15, 0.0, 0.25])), r"$\tau = \pi$", fontsize=11, zorder=13)
ax.text(0.1, -0.3, -0.12, r"$\tau = 0$", fontsize=11, zorder=13)
dgfig.equal_aspect(ax, path, np.array([[-2.3, -0.5, -0.3], [0.5, 1.4, 1.3]]), zoom=1.2)
ax.set_title(r"(a) $\mathrm{Exp}(\tau\,\xi)$, $0 \leq \tau \leq \pi$ (3D)", fontsize=12, y=0.98)

bx = fig.add_axes([0.62, 0.13, 0.36, 0.76])
bx.plot(path[:, 0], path[:, 1], color=C["main"], lw=1.8)
bx.plot(-A, 0, "+", color=C["aux"], ms=12, mew=1.4)
bx.text(-A - 0.2, -0.5, "screw axis", fontsize=10, color=C["aux"])
for s in marks:
    T = L.expSE3(s * xi)
    p = T[:3, 3]
    for k in range(2):
        d = T[:2, k] * TRI
        bx.annotate("", xy=p[:2] + d, xytext=p[:2], arrowprops=dict(arrowstyle="-|>", color=axcol[k], lw=1.2,
                                                                   mutation_scale=9, shrinkA=0, shrinkB=0))
akw = dict(arrowstyle="-|>", lw=2.0, mutation_scale=13, shrinkA=0, shrinkB=0)
bx.annotate("", xy=v[:2], xytext=(0, 0), arrowprops=dict(color=C["tangent"], **akw))
bx.text(0.08, 0.55, r"$v$", color=C["tangent"], fontsize=12)
bx.annotate("", xy=t_end[:2], xytext=(0, 0), arrowprops=dict(color=C["accent"], **akw))
bx.text(-1.35, -0.25, r"$t$", color="#B07400", fontsize=12)
bx.annotate("", xy=pv[:2], xytext=(0, 0), arrowprops=dict(color=C["aux"], ls=(0, (2, 2)), **akw))
bx.text(-2.6, 2.95, r"$\pi v$ (not the translation)", color=C["aux"], fontsize=11)
bx.text(-2.45, 0.12, r"$\tau = \pi$", fontsize=11)
bx.text(0.08, -0.3, r"$\tau = 0$", fontsize=11)
bx.set_aspect("equal")
bx.set_xlim(-2.7, 1.0)
bx.set_ylim(-0.7, 3.4)
bx.set_xlabel(r"$x$")
bx.set_ylabel(r"$y$")
bx.set_title(r"(b) top view ($xy$-plane)", fontsize=12)

dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go  # noqa: E402

tr = [go.Scatter3d(x=path[:, 0], y=path[:, 1], z=path[:, 2], mode="lines", line=dict(color="black", width=5),
                   name="origin of the body frame c(tau)")]
tr.append(go.Scatter3d(x=[-A, -A], y=[0, 0], z=[-0.25, np.pi * B + 0.35], mode="lines",
                       line=dict(color="#777777", width=3, dash="dash"), name="screw axis"))
hexc = ["#0072B2", "#009E73", "#D55E00"]
for s in np.linspace(0, S_END, 9):
    T = L.expSE3(s * xi)
    for k in range(3):
        q = T[:3, 3] + TRI * T[:3, k]
        tr.append(go.Scatter3d(x=[T[0, 3], q[0]], y=[T[1, 3], q[1]], z=[T[2, 3], q[2]], mode="lines",
                               line=dict(color=hexc[k], width=5), showlegend=False, hoverinfo="skip"))
for vec, col, name in [(v, "#0072B2", "v = c'(0)"), (t_end, "#E69F00", "t = translation of Exp(pi xi)"),
                       (pv, "#777777", "pi v")]:
    tr.append(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines+markers",
                           line=dict(color=col, width=7), marker=dict(size=[2, 5], color=col), name=name))
pfig = go.Figure(tr)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                   margin=dict(l=0, r=0, t=40, b=0),
                   title=dict(text="Screw motion Exp(tau xi), xi = (v, w), v = (0, 1, 0.3), w = (0, 0, 1), 0 <= tau <= pi",
                              x=0.5, font=dict(size=13)))
out = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-28-1-1-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
