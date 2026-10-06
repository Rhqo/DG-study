"""Figure 0.6.3: the rotation vector field X(x, y, z) = (−y, x, 0) on the unit sphere S² and its flow (Section 0.6).

Blue arrows: X at colatitudes 30°, 60°, 90°, 120°, 150° and longitudes every 30° (only the visible ones are drawn),
drawn at 0.3 times their true length. Grey circles of latitude are integral curves. Orange: the integral curve
through p (colatitude 60°, longitude −90°) for 0 ≤ t ≤ π/2, ending at θ_{π/2}(p). X vanishes at the poles.
The flow is θ_t = rotation about the z-axis by the angle t; this is checked against a numerical RK4 solution.

Also writes fig-0-6-3-flow-interactive.html (plotly, same data).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-6-3-flow.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

import dgnum

C = dgfig.COLORS
LW = dgfig.LW
SCALE = 0.3


def X(x):
    return np.array([-x[1], x[0], 0.0])


def flow(t, x):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ x


def sph(theta, lon):
    return np.array([np.sin(theta) * np.cos(lon), np.sin(theta) * np.sin(lon), np.cos(theta)])


p = sph(np.deg2rad(60), np.deg2rad(-90))
ts = np.linspace(0, np.pi / 2, 201)

# self-checks (§12.1): X is tangent to S², the flow is the solution of x' = X(x), and it stays on S²
rng = np.random.default_rng(0)
for _ in range(20):
    a = rng.normal(size=3)
    a /= np.linalg.norm(a)
    assert abs(np.dot(X(a), a)) < 1e-12
num = dgnum.rk4(lambda t, y: X(y), p, ts)
exact = np.array([flow(t, p) for t in ts])
assert np.max(np.abs(num - exact)) < 1e-8
assert np.allclose(np.linalg.norm(exact, axis=1), 1)
assert np.allclose(flow(np.pi / 2, p), sph(np.deg2rad(60), 0.0))

fig = plt.figure(figsize=(5.4, 4.8))
ax = dgfig.axes3d(fig, elev=18, azim=-90)
th, ph = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81), indexing="ij")
Xs, Ys, Zs = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(ax, Xs, Ys, Zs, alpha=0.3, grid=False, zorder=1)

view = dgfig.view_vector(ax)
t = np.linspace(0, 2 * np.pi, 361)
for thd in (30, 60, 90, 120, 150):
    tr = np.deg2rad(thd)
    circ = np.stack([np.sin(tr) * np.cos(t), np.sin(tr) * np.sin(t), np.cos(tr) + 0 * t], 1)
    dgfig.curve3d(ax, circ, role="aux", lw=0.8, visible=circ @ view > 0, zorder=3)
    for lon in np.deg2rad(np.arange(0, 360, 30)):
        a = sph(tr, lon)
        if a @ view > 0.15:
            dgfig.arrow3d(ax, a, X(a), role="tangent", scale=SCALE, head=8, lw=1.1, zorder=8)

seg = exact
dgfig.curve3d(ax, seg, role="accent", lw=2.2, zorder=9)
dgfig.point3d(ax, p, None, size=18, zorder=12)
dgfig.point3d(ax, flow(np.pi / 2, p), None, size=18, zorder=12)
dgfig.point3d(ax, np.array([0, 0, 1.0]), None, size=12, zorder=12)
dgfig.point3d(ax, np.array([0, 0, -1.0]), None, size=12, zorder=12)
dgfig.equal_aspect(ax, Xs, Ys, Zs, pad=0.06, zoom=1.0)

from mpl_toolkits.mplot3d import proj3d  # noqa: E402

box = dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85)


def crop_to(fig, ax, pts, pad=0.05):
    """The 3D box is taller than the drawing. Shrink the figure height to the projected extent of ``pts``
    (plus room ``pad`` for labels) without changing the projection scale, so the SVG has no empty band."""
    import matplotlib

    fig.canvas.draw()
    xy = np.array([ax.transData.transform(proj3d.proj_transform(*q, ax.get_proj())[:2]) for q in pts])
    y = fig.transFigure.inverted().transform(xy)[:, 1]
    y0, y1 = y.min() - pad, y.max() + pad
    W, H = fig.get_size_inches()
    H2 = (y1 - y0) * H
    pos = ax.get_position()
    ax.set_position([pos.x0, (pos.y0 - y0) * H / H2, pos.width, pos.height * H / H2])
    fig.set_size_inches(W, H2)
    matplotlib.rcParams["savefig.bbox"] = None


def label(pt, text, dx, dy, color="black"):
    x2, y2, _ = proj3d.proj_transform(*pt, ax.get_proj())
    ax.annotate(text, xy=(x2, y2), xytext=(dx, dy), textcoords="offset points", fontsize=12, color=color,
                ha="center", va="center", zorder=20, bbox=box)


label(p, r"$p$", -6, -12)
label(flow(np.pi / 2, p), r"$\theta_{\pi/2}(p)$", 26, -12, C["accent"])
label(sph(np.deg2rad(90), np.deg2rad(-60)) + SCALE * X(sph(np.deg2rad(90), np.deg2rad(-60))), r"$X$", 10, 8,
      C["tangent"])
label(np.array([0, 0, 1.0]), r"$N$", 0, 11)
label(np.array([0, 0, -1.0]), r"$S$", 0, -11)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
crop_to(fig, ax, np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], 1))
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive (plotly, same data)
import plotly.graph_objects as go  # noqa: E402

base, vecs = [], []
for thd in (30, 60, 90, 120, 150):
    for lon in np.deg2rad(np.arange(0, 360, 30)):
        a = sph(np.deg2rad(thd), lon)
        base.append(a)
        vecs.append(SCALE * X(a))
base, vecs = np.array(base), np.array(vecs)
traces = [go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.35, showscale=False, colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]],
                     hoverinfo="skip")]
for thd in (30, 60, 90, 120, 150):
    tr = np.deg2rad(thd)
    traces.append(go.Scatter3d(x=np.sin(tr) * np.cos(t), y=np.sin(tr) * np.sin(t), z=np.cos(tr) + 0 * t,
                               mode="lines", line=dict(color="#777777", width=2), hoverinfo="skip"))
for a, v in zip(base, vecs):
    traces.append(go.Scatter3d(x=[a[0], a[0] + v[0]], y=[a[1], a[1] + v[1]], z=[a[2], a[2] + v[2]], mode="lines",
                               line=dict(color=C["tangent"], width=4), hoverinfo="skip"))
traces.append(go.Cone(x=base[:, 0] + vecs[:, 0], y=base[:, 1] + vecs[:, 1], z=base[:, 2] + vecs[:, 2],
                      u=vecs[:, 0], v=vecs[:, 1], w=vecs[:, 2], sizemode="absolute", sizeref=0.08, anchor="tip",
                      showscale=False, colorscale=[[0, C["tangent"]], [1, C["tangent"]]], hoverinfo="skip"))
traces.append(go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", line=dict(color=C["accent"], width=7),
                           name="integral curve"))
q = flow(np.pi / 2, p)
traces.append(go.Scatter3d(x=[p[0], q[0], 0, 0], y=[p[1], q[1], 0, 0], z=[p[2], q[2], 1, -1], mode="markers+text",
                           marker=dict(size=4, color="black"), text=["p", "theta_{pi/2}(p)", "N", "S"],
                           textposition="top center"))
figp = go.Figure(traces)
figp.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                              zaxis=dict(visible=False)),
                   margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                   title="Figure 0.6.3: X = (-y, x, 0) on S^2")
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
figp.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-0-6-3-interactive")
print(f"saved {out}")
