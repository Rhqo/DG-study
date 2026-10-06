"""Figure 0.6.1: stereographic projection σ from the north pole N of the unit sphere S² (Section 0.6).

σ(x, y, z) = (x, y)/(1 − z) (GUIDELINES §7, dgsym.stereographic(2)); the image plane is z = 0.
p at colatitude 60°, longitude 30° (upper hemisphere, σ(p) outside the unit circle) and
q at colatitude 120°, longitude −130° (lower hemisphere, σ(q) inside). Their circles of latitude go to circles of
radius cot(θ/2): √3 for p, 1/√3 for q. N, p, σ(p) lie on one line (and N, q, σ(q)).

Also writes the interactive version fig-0-6-1-stereographic-interactive.html (plotly, same data).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-6-1-stereographic.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

import dgsym

C = dgfig.COLORS
LW = dgfig.LW

# σ from the canonical definition (§3.6: do not retype the formula)
st = dgsym.stereographic(2)
X = st["coords"]
sigma = sp.lambdify(X, list(st["sigma"]), "numpy")


def sph(theta, lon):
    """point of the unit sphere with colatitude theta and longitude lon"""
    return np.array([np.sin(theta) * np.cos(lon), np.sin(theta) * np.sin(lon), np.cos(theta)])


N = np.array([0.0, 0.0, 1.0])
p = sph(np.deg2rad(60), np.deg2rad(30))
q = sph(np.deg2rad(120), np.deg2rad(-130))
sp_ = np.array(list(sigma(*p)) + [0.0])
sq_ = np.array(list(sigma(*q)) + [0.0])

# self-checks (§12.1)
for a, s, th in ((p, sp_, 60), (q, sq_, 120)):
    assert abs(np.linalg.norm(a) - 1) < 1e-12
    # N, a, σ(a) are collinear
    assert np.linalg.norm(np.cross(a - N, s - N)) < 1e-12
    # |σ(a)| = cot(θ/2)
    assert abs(np.linalg.norm(s) - 1 / np.tan(np.deg2rad(th) / 2)) < 1e-12
assert abs(np.linalg.norm(sp_) - np.sqrt(3)) < 1e-12 and abs(np.linalg.norm(sq_) - 1 / np.sqrt(3)) < 1e-12

# ---------------------------------------------------------------- static SVG
fig = plt.figure(figsize=(6.0, 4.6))
ax = dgfig.axes3d(fig, elev=24, azim=-62)

L = 2.0
P = np.array([[-L, -L, 0], [L, -L, 0], [L, L, 0], [-L, L, 0]])
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ax.add_collection3d(Poly3DCollection([P], facecolor=C["region"], alpha=0.18, edgecolor=C["aux"],
                                     linewidth=LW["aux"], zorder=0))

th, ph = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81), indexing="ij")
Xs, Ys, Zs = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(ax, Xs, Ys, Zs, alpha=0.32, grid_every=8, zorder=1)

t = np.linspace(0, 2 * np.pi, 361)
# equator (fixed by σ), circles of latitude of p and q, and their images
eq = np.stack([np.cos(t), np.sin(t), 0 * t], 1)
ax.plot(*eq.T, color=C["main"], lw=0.9, zorder=3)
for a, th_deg, col in ((p, 60, C["tangent"]), (q, 120, C["accent"])):
    thr = np.deg2rad(th_deg)
    circ = np.stack([np.sin(thr) * np.cos(t), np.sin(thr) * np.sin(t), np.cos(thr) + 0 * t], 1)
    rad = 1 / np.tan(thr / 2)
    img = np.stack([rad * np.cos(t), rad * np.sin(t), 0 * t], 1)
    ax.plot(*circ.T, color=col, lw=1.2, zorder=4)
    ax.plot(*img.T, color=col, lw=1.2, ls=(0, (4, 2)), zorder=2)

# projection lines N → p → σ(p), N → q → σ(q)
for a, s, col in ((p, sp_, C["tangent"]), (q, sq_, C["accent"])):
    far = s if np.linalg.norm(s - N) > np.linalg.norm(a - N) else a
    ax.plot(*np.stack([N, far]).T, color=col, lw=LW["vector"], zorder=6)
    dgfig.point3d(ax, a, None, size=16, zorder=8)
    dgfig.point3d(ax, s, None, size=16, zorder=8)

dgfig.point3d(ax, N, None, size=22, zorder=9)
dgfig.equal_aspect(ax, P, np.array([[0, 0, -1.0], [0, 0, 1.15]]), zoom=1.18)

from mpl_toolkits.mplot3d import proj3d  # noqa: E402


def label(pt, text, dx, dy, color="black", box=False):
    """label a 3D point with an offset in screen points (robust placement)"""
    x2, y2, _ = proj3d.proj_transform(*pt, ax.get_proj())
    bbox = dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85) if box else None
    ax.annotate(text, xy=(x2, y2), xytext=(dx, dy), textcoords="offset points", fontsize=12, color=color,
                ha="center", va="center", zorder=20, bbox=bbox)


def crop_to(fig, ax, pts, pad=0.06):
    """The 3D box is taller than the drawing. Shrink the figure height to the projected extent of ``pts``
    (plus labels' room ``pad``) without changing the projection scale, so the SVG has no empty band."""
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


label(N, r"$N$", 8, 9)
label(p, r"$p$", 1, 14)
label(sp_, r"$\sigma(p)$", 18, -11, C["tangent"])
label(q, r"$q$", -9, -9)
label(sq_, r"$\sigma(q)$", 22, -13, C["accent"], box=True)
label(np.array([L * 0.92, -L * 0.95, 0]), r"$\mathbb{R}^2$", 0, 0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
crop_to(fig, ax, np.concatenate([P, np.array([[0, 0, 1.0], [0, 0, -1.0]]), sp_[None], sq_[None]]))
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive (plotly, same data)
import plotly.graph_objects as go  # noqa: E402

blue, orange = C["tangent"], C["accent"]
traces = [
    go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.35, showscale=False, colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]],
               hoverinfo="skip", name="S^2"),
    go.Surface(x=[[-L, L], [-L, L]], y=[[-L, -L], [L, L]], z=[[0, 0], [0, 0]], opacity=0.18, showscale=False,
               colorscale=[[0, "#56B4E9"], [1, "#56B4E9"]], hoverinfo="skip", name="R^2"),
    go.Scatter3d(x=eq[:, 0], y=eq[:, 1], z=eq[:, 2], mode="lines", line=dict(color="black", width=3),
                 name="equator"),
]
for a, s, th_deg, col, nm in ((p, sp_, 60, blue, "p"), (q, sq_, 120, orange, "q")):
    thr = np.deg2rad(th_deg)
    rad = 1 / np.tan(thr / 2)
    traces += [
        go.Scatter3d(x=np.sin(thr) * np.cos(t), y=np.sin(thr) * np.sin(t), z=np.cos(thr) + 0 * t, mode="lines",
                     line=dict(color=col, width=4), name=f"circle of latitude through {nm}"),
        go.Scatter3d(x=rad * np.cos(t), y=rad * np.sin(t), z=0 * t, mode="lines",
                     line=dict(color=col, width=4, dash="dash"), name=f"its image under sigma"),
        go.Scatter3d(x=[N[0], s[0]], y=[N[1], s[1]], z=[N[2], s[2]], mode="lines+markers+text",
                     line=dict(color=col, width=5), marker=dict(size=3, color="black"),
                     text=["", f"sigma({nm})"], textposition="top center", name=f"line N-{nm}"),
        go.Scatter3d(x=[a[0]], y=[a[1]], z=[a[2]], mode="markers+text", marker=dict(size=4, color="black"),
                     text=[nm], textposition="top center", name=nm),
    ]
traces.append(go.Scatter3d(x=[0], y=[0], z=[1], mode="markers+text", marker=dict(size=5, color="black"),
                           text=["N"], textposition="top center", name="N"))
figp = go.Figure(traces)
figp.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                              zaxis=dict(visible=False)),
                   margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                   title="Figure 0.6.1: stereographic projection")
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
figp.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-0-6-1-interactive")
print(f"saved {out}")
