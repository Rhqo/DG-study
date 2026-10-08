"""Figure 30.3.1: a 2D Gaussian (2DGS primitive) is a local parametrization of a plane piece (Section 30.3).

P(u, v) = p_k + s_u t_u u + s_v t_v v with s_u = 0.6, s_v = 0.3, p_k = (0, 0, 0), and R = [t_u, t_v, t_w] with
t_w = (0.05, -0.05, 1)/norm (a nearly horizontal plane, seen obliquely from above so that the normal visibly leaves
the plane) and t_u at 40 deg in that plane. G(u, v) = exp(-(u^2 + v^2)/2).
(a) the (u, v)-plane: gray grid (spacing 0.5), circles u^2 + v^2 = 1 (black) and 4 (gray), shading = G; arrows e_u (blue),
    e_v (green).
(b) the image P(U) in space: the same grid and circles mapped by P (ellipses), shading = G, arrows s_u t_u (blue),
    s_v t_v (green), unit normal t_w = t_u x t_v (vermillion, drawn with length 0.9); the curved arrow between the
    panels is P. View: elevation 32 deg, azimuth -60 deg.
Self-checks: P_u x P_v = s_u s_v t_w; E = s_u^2, F = 0, G = s_v^2; the drawn normal makes > 50 deg with the viewing
direction (so it does not look like an in-plane vector).
Also writes an interactive plotly version (same splat).

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-3-1-splat-parametrization.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
SU, SV = 0.6, 0.3


def rot_z(a):
    a = np.radians(a)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def rot_x(a):
    a = np.radians(a)
    return np.array([[1.0, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


tw = np.array([0.05, -0.05, 1.0])
tw /= np.linalg.norm(tw)                            # nearly horizontal plane, seen obliquely from above
t0 = np.cross(np.array([0.0, 0.0, 1.0]), tw)
t0 /= np.linalg.norm(t0)
t1 = np.cross(tw, t0)
ang = np.radians(40.0)
tu = np.cos(ang) * t0 + np.sin(ang) * t1
tv = np.cross(tw, tu)
R = np.column_stack([tu, tv, tw])
assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
pk = np.zeros(3)


def P(u, v):
    return pk + SU * np.asarray(u)[..., None] * tu + SV * np.asarray(v)[..., None] * tv


Pu, Pv = SU * tu, SV * tv
assert np.allclose(np.cross(Pu, Pv), SU * SV * tw)
assert np.allclose([Pu @ Pu, Pu @ Pv, Pv @ Pv], [SU ** 2, 0, SV ** 2])

fig = plt.figure(figsize=(5.8, 8.4))
axa = fig.add_axes([0.16, 0.565, 0.66, 0.385])
axb = fig.add_axes([-0.08, -0.07, 1.16, 0.58], projection="3d", computed_zorder=False)
axb.patch.set_visible(False)                        # do not hide the u-label of panel (a)

# (a) parameter plane
L = 2.6
g = np.linspace(-L, L, 300)
U, V = np.meshgrid(g, g)
axa.imshow(np.exp(-(U ** 2 + V ** 2) / 2), extent=(-L, L, -L, L), origin="lower", cmap="Blues", vmin=0, vmax=1.3,
           alpha=0.9)
for k in np.arange(-2.5, 2.51, 0.5):
    axa.plot([k, k], [-L, L], color="#999999", lw=0.5)
    axa.plot([-L, L], [k, k], color="#999999", lw=0.5)
tt = np.linspace(0, 2 * np.pi, 300)
axa.plot(np.cos(tt), np.sin(tt), color=C["main"], lw=1.8)
axa.plot(2 * np.cos(tt), 2 * np.sin(tt), color=C["aux"], lw=1.2)
kw = dict(arrowstyle="-|>", lw=1.8, mutation_scale=14, shrinkA=0, shrinkB=0)
axa.annotate("", xy=(1, 0), xytext=(0, 0), arrowprops=dict(color=C["tangent"], **kw))
axa.annotate("", xy=(0, 1), xytext=(0, 0), arrowprops=dict(color=C["third"], **kw))
axa.text(1.05, -0.3, r"$e_u$", color=C["tangent"], fontsize=13)
axa.text(-0.45, 0.55, r"$e_v$", color=C["third"], fontsize=13)
axa.text(1.5, 1.95, r"$u^2 + v^2 = 4$", color=C["aux"], fontsize=11.5, ha="center",
         bbox=dict(fc="white", ec="none", pad=0.5))
axa.text(0.75, -1.25, r"$u^2 + v^2 = 1$", color=C["main"], fontsize=11.5, bbox=dict(fc="white", ec="none", pad=0.5))
axa.set_aspect("equal")
axa.set_xlim(-L, L)
axa.set_ylim(-L, L)
axa.set_xlabel(r"$u$", labelpad=1)
axa.set_ylabel(r"$v$")
axa.set_title(r"(a) parameter plane, shading $G(u,v) = e^{-(u^2+v^2)/2}$", fontsize=13)

# (b) the splat in space
axb.view_init(elev=32, azim=-60)
axb.set_axis_off()
NLEN = 0.9                                          # drawn length of the unit normal t_w
assert np.degrees(np.arccos(abs(tw @ dgfig.view_vector(axb)))) > 50
gu = np.linspace(-L, L, 120)
UU, VV = np.meshgrid(gu, gu)
XYZ = P(UU, VV)
Gv = np.exp(-(UU ** 2 + VV ** 2) / 2)
axb.plot_surface(XYZ[..., 0], XYZ[..., 1], XYZ[..., 2], facecolors=cm.Blues(Gv / 1.3), alpha=0.85, linewidth=0,
                 shade=False, rstride=2, cstride=2, rasterized=True, zorder=1)
for k in np.arange(-2.5, 2.51, 0.5):
    li = P(np.full(50, k), np.linspace(-L, L, 50))
    axb.plot(*li.T, color="#999999", lw=0.5, zorder=2)
    li = P(np.linspace(-L, L, 50), np.full(50, k))
    axb.plot(*li.T, color="#999999", lw=0.5, zorder=2)
axb.plot(*P(np.cos(tt), np.sin(tt)).T, color=C["main"], lw=1.8, zorder=3)
axb.plot(*P(2 * np.cos(tt), 2 * np.sin(tt)).T, color=C["aux"], lw=1.2, zorder=3)
dgfig.arrow3d(axb, pk, Pu, role="tangent", zorder=10, head=13, lw=1.8)
dgfig.arrow3d(axb, pk, Pv, role="third", zorder=10, head=13, lw=1.8)
dgfig.arrow3d(axb, pk, NLEN * tw, role="normal", zorder=11, head=13, lw=1.8)
axb.text(*(1.25 * Pu + 0.05 * tw), r"$s_u t_u$", color=C["tangent"], fontsize=13, zorder=12)
axb.text(*(1.5 * Pv - 0.15 * tu + 0.05 * tw), r"$s_v t_v$", color=C["third"], fontsize=13, zorder=12, ha="right")
axb.text(*(NLEN * tw + 0.08 * tu), r"$t_w = t_u \times t_v$", color=C["normal"], fontsize=13, zorder=12)
axb.plot(*pk[:, None], "o", color=C["main"], ms=4, zorder=12)
axb.text(*(pk - 0.36 * tu - 0.02 * tv), r"$p_k$", fontsize=13, zorder=12, ha="center", va="top")
pts = np.concatenate([XYZ.reshape(-1, 3), (1.05 * NLEN * tw)[None]], 0)
dgfig.equal_aspect(axb, pts, zoom=1.62)
fig.text(0.5, 0.475, r"(b) the splat $P(u, v) = p_k + s_u t_u u + s_v t_v v$", ha="center", fontsize=13)
dgfig.map_arrow(fig, axa, axb, r"$P$", xy_from=(1.02, 0.15), xy_to=(0.86, 0.70), rad=-0.35, fontsize=15,
                label_offset=(-0.06, 0.0))
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go

traces = [go.Surface(x=XYZ[..., 0], y=XYZ[..., 1], z=XYZ[..., 2], surfacecolor=Gv, colorscale="Blues", cmin=0, cmax=1.3,
                     showscale=False, opacity=0.9, customdata=np.stack([UU, VV, Gv], -1),
                     hovertemplate="u = %{customdata[0]:.2f}, v = %{customdata[1]:.2f}<br>G = %{customdata[2]:.3f}"
                                   "<extra></extra>")]
for rad, col in [(1, "black"), (2, "gray")]:
    ring = P(rad * np.cos(tt), rad * np.sin(tt))
    traces.append(go.Scatter3d(x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines", line=dict(color=col, width=4),
                               hoverinfo="skip", showlegend=False))
for vec, col, name in [(Pu, "#0072B2", "s_u t_u"), (Pv, "#009E73", "s_v t_v"), (NLEN * tw, "#D55E00", "t_w")]:
    traces.append(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines+text", text=["", name],
                               line=dict(color=col, width=7), textfont=dict(color=col, size=14), showlegend=False,
                               hoverinfo="skip"))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=40, b=0),
                   title=dict(text="2D Gaussian splat (s_u = 0.6, s_v = 0.3): local parametrization and normal t_w",
                              x=0.5, font=dict(size=14)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-30-3-1-interactive", config={"displaylogo": False})
print(f"saved {out}")
