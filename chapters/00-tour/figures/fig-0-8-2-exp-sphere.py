"""Figure 0.8.2: the exponential map exp_N : T_N S² → S² of the unit sphere at the north pole N (Section 0.8).

exp_p(v) = cos|v| p + sin|v| v/|v| (unit sphere). Left: the tangent plane T_N S² (identified with the xy-plane) with the
circles |v| = π/4, π/2, 3π/4 (coloured), the circle |v| = π (black) and eight rays (grey). Right: their images on S²:
the circles go to the circles of latitude with colatitude π/4, π/2 (equator), 3π/4, the rays go to meridians, and the
whole circle |v| = π goes to the south pole S (cut point). Blue: v = 2(cos(−30°), sin(−30°)) and the geodesic
t ↦ exp_N(tv), 0 ≤ t ≤ 1 (orange) ending at exp_N(v); its length is |v| = 2.

Also writes fig-0-8-2-exp-sphere-interactive.html (plotly, same data).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/figures/fig-0-8-2-exp-sphere.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
LW = dgfig.LW

N = np.array([0.0, 0.0, 1.0])


def exp_N(v2):
    """v2 = (a, b) ∈ T_N S² ≅ xy-plane"""
    v = np.array([v2[0], v2[1], 0.0])
    r = np.linalg.norm(v)
    if r < 1e-14:
        return N.copy()
    return np.cos(r) * N + np.sin(r) * v / r


radii = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
cols = [C["accent"], C["main"], C["third"]]
ang = np.deg2rad(-30)
v = 2.0 * np.array([np.cos(ang), np.sin(ang)])
geo = np.array([exp_N(t * v) for t in np.linspace(0, 1, 200)])

# self-checks (§12.1)
for r in radii + [np.pi]:
    for a in np.linspace(0, 2 * np.pi, 13):
        q = exp_N(r * np.array([np.cos(a), np.sin(a)]))
        assert abs(np.linalg.norm(q) - 1) < 1e-12
        assert abs(np.arccos(np.clip(q @ N, -1, 1)) - r) < 1e-9          # distance from N is |v|
assert np.allclose(exp_N(np.pi * np.array([0.6, 0.8])), -N)              # |v| = π goes to S
seg = np.linalg.norm(np.diff(geo, axis=0), axis=1).sum()
assert abs(seg - 2.0) < 1e-3                                             # length of the geodesic = |v|
assert np.allclose(geo @ np.array([-np.sin(ang), np.cos(ang), 0]), 0)               # a great circle through N

fig = plt.figure(figsize=(6.6, 3.3))
ax0 = fig.add_subplot(121)
dgfig.schematic_axes(ax0, (-3.5, 3.5), (-3.5, 3.5))
t = np.linspace(0, 2 * np.pi, 300)
ax0.fill(np.pi * np.cos(t), np.pi * np.sin(t), color=C["region"], alpha=0.2, lw=0)
for a in np.deg2rad(np.arange(0, 360, 45)):
    ax0.plot([0, np.pi * np.cos(a)], [0, np.pi * np.sin(a)], color=C["aux"], lw=0.7)
for r, c in zip(radii, cols):
    ax0.plot(r * np.cos(t), r * np.sin(t), color=c, lw=1.4, ls=(0, (4, 2)))
ax0.plot(np.pi * np.cos(t), np.pi * np.sin(t), color=C["main"], lw=1.8)
ax0.annotate("", xy=tuple(v), xytext=(0, 0),
             arrowprops=dict(arrowstyle="-|>", color=C["tangent"], lw=2.0, mutation_scale=13))
ax0.plot(0, 0, "o", color=C["main"], ms=4)
box = dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.85)
ax0.text(v[0] + 0.1, v[1] - 0.45, r"$v$", fontsize=13, color=C["tangent"], bbox=box)
ax0.text(0.1, -0.45, r"$0$", fontsize=11, bbox=box)
ax0.text(-3.4, 3.2, r"$T_N S^2$", fontsize=12)
ax0.text(2.35, -2.75, r"$|v| = \pi$", fontsize=11)

ax1 = dgfig.axes3d(fig, pos=122, elev=18, azim=-60)
th, ph = np.meshgrid(np.linspace(0, np.pi, 41), np.linspace(0, 2 * np.pi, 81), indexing="ij")
Xs, Ys, Zs = np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)
dgfig.surface(ax1, Xs, Ys, Zs, alpha=0.3, grid=False, zorder=1)
view = dgfig.view_vector(ax1)
for a in np.deg2rad(np.arange(0, 360, 45)):
    mer = np.array([exp_N(s * np.array([np.cos(a), np.sin(a)])) for s in np.linspace(0, np.pi, 120)])
    dgfig.curve3d(ax1, mer, role="aux", lw=0.7, visible=mer @ view > 0, zorder=3)
for r, c in zip(radii, cols):
    circ = np.array([exp_N(r * np.array([np.cos(a), np.sin(a)])) for a in t])
    dgfig.curve3d(ax1, circ, role=c, lw=1.5, visible=circ @ view > 0, zorder=4)
dgfig.curve3d(ax1, geo, role="accent", lw=2.4, zorder=8)
dgfig.arrow3d(ax1, N, 0.55 * np.array([v[0], v[1], 0]) / 2, role="tangent", head=10, zorder=9)
dgfig.point3d(ax1, N, None, size=16, zorder=10)
dgfig.point3d(ax1, geo[-1], None, size=16, zorder=10)
dgfig.point3d(ax1, -N, None, size=16, zorder=10)
dgfig.equal_aspect(ax1, Xs, Ys, Zs, zoom=1.15)

from mpl_toolkits.mplot3d import proj3d  # noqa: E402


def label(pt, text, dx, dy, color="black"):
    x2, y2, _ = proj3d.proj_transform(*pt, ax1.get_proj())
    ax1.annotate(text, xy=(x2, y2), xytext=(dx, dy), textcoords="offset points", fontsize=12, color=color,
                 ha="center", va="center", zorder=20, bbox=box)


label(N, r"$N$", -8, 10)
label(geo[-1], r"$\exp_N(v)$", 30, -4, C["accent"])
label(-N, r"$S$", 0, -11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.12)
dgfig.map_arrow(fig, ax0, ax1, r"$\exp_N$", xy_from=(0.98, 0.75), xy_to=(0.08, 0.75), rad=-0.35, fontsize=12)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive (plotly, same data)
import plotly.graph_objects as go  # noqa: E402

traces = [go.Surface(x=Xs, y=Ys, z=Zs, opacity=0.35, showscale=False, colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]],
                     hoverinfo="skip")]
# the tangent plane z = 1 with the same circles and rays, and their images
for r, c in zip(radii + [np.pi], cols + [C["main"]]):
    traces.append(go.Scatter3d(x=r * np.cos(t), y=r * np.sin(t), z=1 + 0 * t, mode="lines",
                               line=dict(color=c, width=3, dash="dash"), hoverinfo="skip"))
    circ = np.array([exp_N(r * np.array([np.cos(a), np.sin(a)])) for a in t])
    traces.append(go.Scatter3d(x=circ[:, 0], y=circ[:, 1], z=circ[:, 2], mode="lines", line=dict(color=c, width=5),
                               hoverinfo="skip"))
for a in np.deg2rad(np.arange(0, 360, 45)):
    traces.append(go.Scatter3d(x=[0, np.pi * np.cos(a)], y=[0, np.pi * np.sin(a)], z=[1, 1], mode="lines",
                               line=dict(color="#777777", width=2), hoverinfo="skip"))
    mer = np.array([exp_N(s * np.array([np.cos(a), np.sin(a)])) for s in np.linspace(0, np.pi, 120)])
    traces.append(go.Scatter3d(x=mer[:, 0], y=mer[:, 1], z=mer[:, 2], mode="lines",
                               line=dict(color="#777777", width=2), hoverinfo="skip"))
traces.append(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[1, 1], mode="lines+text", text=["", "v"],
                           line=dict(color=C["tangent"], width=7)))
traces.append(go.Scatter3d(x=geo[:, 0], y=geo[:, 1], z=geo[:, 2], mode="lines", line=dict(color=C["accent"], width=8)))
traces.append(go.Scatter3d(x=[0, geo[-1, 0], 0], y=[0, geo[-1, 1], 0], z=[1, geo[-1, 2], -1], mode="markers+text",
                           marker=dict(size=4, color="black"), text=["N", "exp_N(v)", "S"], textposition="top center"))
figp = go.Figure(traces)
figp.update_layout(scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False),
                              zaxis=dict(visible=False)),
                   margin=dict(l=0, r=0, t=30, b=0), showlegend=False,
                   title="Figure 0.8.2: exp_N on S^2 (T_N S^2 drawn at z = 1)")
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
figp.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-0-8-2-interactive")
print(f"saved {out}")
