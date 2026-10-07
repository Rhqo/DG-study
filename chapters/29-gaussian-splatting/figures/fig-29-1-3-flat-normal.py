"""Figure 29.1.3: a flat Gaussian behaves like a surface patch; its shortest axis is the normal (Section 29.1).

(a) 3D: a spherical cap of radius 2 (gray) and seven flat Gaussians (1-sigma ellipsoids, light blue) placed on it with
    scales s = (0.32, 0.20, 0.04). Each Gaussian has r_1, r_2 tangent to the cap (random angle in the tangent plane,
    seed 3) and r_3 = the unit normal of the sphere. Vermillion arrows: s-independent length 0.45 along +r_3.
(b) The cross-section y = 0 drawn in 2D: the circle of radius 2 and three flat 2D Gaussians (s = (0.32, 0.04))
    centered on it; blue = long axis s_1 r_1, vermillion = shortest axis r_3 (drawn with length 0.45).
Also writes an interactive plotly version (fig-29-1-3-flat-normal-interactive.html) with the same Gaussians.
Self-check: for every Gaussian the eigenvector of the smallest eigenvalue of Sigma is parallel to the sphere normal.

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/figures/fig-29-1-3-flat-normal.py``
"""

from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib.pyplot as plt
import numpy as np

C = dgfig.COLORS
rng = np.random.default_rng(3)
RHO = 2.0
S3 = np.array([0.32, 0.20, 0.04])
NLEN = 0.45


def frame_at(p, ang):
    n = p / np.linalg.norm(p)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = a - (a @ n) * n
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    r1 = np.cos(ang) * t1 + np.sin(ang) * t2
    r2 = np.cross(n, r1)
    return np.stack([r1, r2, n], 1)


# centers on the cap: colatitude, longitude (degrees)
spots = [(0, 0), (22, 10), (22, 130), (22, 250), (40, 60), (40, 190), (40, 310)]
gauss = []
for th, ph in spots:
    th, ph = np.radians(th), np.radians(ph)
    p = RHO * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    R = frame_at(p, rng.uniform(0, np.pi))
    assert np.isclose(np.linalg.det(R), 1.0)
    Sigma = R @ np.diag(S3 ** 2) @ R.T
    lam, V = np.linalg.eigh(Sigma)
    assert abs(abs(V[:, 0] @ (p / RHO)) - 1.0) < 1e-12      # shortest axis = sphere normal
    gauss.append((p, R))

# ---------------------------------------------------------------- static figure
fig = plt.figure(figsize=(7.6, 3.3))
ax = dgfig.axes3d(fig, pos=121, elev=38, azim=-58)
uu, vv = np.meshgrid(np.linspace(0, np.radians(52), 30), np.linspace(0, 2 * np.pi, 61), indexing="ij")
X = RHO * np.sin(uu) * np.cos(vv)
Y = RHO * np.sin(uu) * np.sin(vv)
Z = RHO * np.cos(uu)
dgfig.surface(ax, X, Y, Z, alpha=0.35, grid_every=5, zorder=1)
eu, ev = np.meshgrid(np.linspace(0, np.pi, 13), np.linspace(0, 2 * np.pi, 25), indexing="ij")
unit = np.stack([np.sin(eu) * np.cos(ev), np.sin(eu) * np.sin(ev), np.cos(eu)], -1)
for i, (p, R) in enumerate(gauss):
    E = p + (unit * S3) @ R.T
    ax.plot_surface(E[..., 0], E[..., 1], E[..., 2], color=C["region"], alpha=0.9, linewidth=0, shade=False,
                    zorder=3 + i, rasterized=True)
    rim = p + np.stack([S3[0] * np.cos(ev[0]), S3[1] * np.sin(ev[0])], 1) @ R[:, :2].T
    ax.plot(*rim.T, color=C["tangent"], lw=0.9, zorder=3 + i)
    dgfig.arrow3d(ax, p, R[:, 2], role="normal", scale=NLEN, zorder=20, head=9, lw=1.4)
p0, R0 = gauss[0]
ax.text(*(p0 + 0.62 * R0[:, 2]), r"$r_3$", color=C["normal"], fontsize=12, zorder=30)
dgfig.equal_aspect(ax, np.stack([X, Y, Z], -1).reshape(-1, 3), np.array([[0, 0, RHO + 0.55]]), zoom=1.4)
ax.set_title("(a) flat Gaussians on a sphere", fontsize=11, y=0.92)

# (b) cross-section y = 0
axb = fig.add_subplot(122)
tt = np.linspace(np.radians(-42), np.radians(42), 200)
axb.plot(RHO * np.sin(tt), RHO * np.cos(tt), color=C["main"], lw=1.6, zorder=1)
s2 = np.array([0.32, 0.04])
circ = np.linspace(0, 2 * np.pi, 120)
arrow_kw = dict(arrowstyle="-|>", lw=1.4, mutation_scale=10, shrinkA=0, shrinkB=0)
for k, a in enumerate(np.radians([-25, 0, 25])):
    c = RHO * np.array([np.sin(a), np.cos(a)])
    n = c / RHO
    tvec = np.array([n[1], -n[0]])
    R2 = np.stack([tvec, n], 1)
    E = c + (np.stack([np.cos(circ), np.sin(circ)], 1) * s2) @ R2.T
    axb.fill(*E.T, color=C["region"], alpha=0.6, lw=0.8, ec=C["tangent"], zorder=3)
    axb.annotate("", xy=c + s2[0] * tvec, xytext=c, arrowprops=dict(color=C["tangent"], **arrow_kw), zorder=4)
    axb.annotate("", xy=c + NLEN * n, xytext=c, arrowprops=dict(color=C["normal"], **arrow_kw), zorder=4)
    assert abs(R2[:, 1] @ n - 1) < 1e-12
cmid = RHO * np.array([np.sin(np.radians(25)), np.cos(np.radians(25))])
nmid = cmid / RHO
axb.text(*(cmid + (NLEN + 0.06) * nmid + np.array([0.02, 0.0])), r"$r_3$", color=C["normal"], fontsize=12)
tmid = np.array([nmid[1], -nmid[0]])
axb.text(*(cmid - 0.05 * tmid - 0.26 * nmid), r"$s_1 r_1$", color=C["tangent"], fontsize=11)
axb.text(-1.25, 1.36, "surface", color=C["main"], fontsize=10.5)
dgfig.schematic_axes(axb, (-1.45, 1.75), (1.25, 2.6))
axb.set_title(r"(b) cross-section $y = 0$", fontsize=11)
fig.subplots_adjust(wspace=0.0, left=0.0, right=1.0)

dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go

traces = [go.Surface(x=X, y=Y, z=Z, colorscale=[[0, "#D0D0D0"], [1, "#D0D0D0"]], showscale=False, opacity=0.45,
                     hoverinfo="skip", name="surface")]
eu2, ev2 = np.meshgrid(np.linspace(0, np.pi, 21), np.linspace(0, 2 * np.pi, 41), indexing="ij")
unit2 = np.stack([np.sin(eu2) * np.cos(ev2), np.sin(eu2) * np.sin(ev2), np.cos(eu2)], -1)
for i, (p, R) in enumerate(gauss):
    E = p + (unit2 * S3) @ R.T
    traces.append(go.Surface(x=E[..., 0], y=E[..., 1], z=E[..., 2], colorscale=[[0, "#56B4E9"], [1, "#56B4E9"]],
                             showscale=False, opacity=0.9, name=f"Gaussian {i + 1}",
                             hovertemplate=f"Gaussian {i + 1}<br>s = (0.32, 0.20, 0.04)<extra></extra>"))
    q = p + NLEN * R[:, 2]
    traces.append(go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines",
                               line=dict(color="#D55E00", width=6), showlegend=False, hoverinfo="skip"))
    traces.append(go.Cone(x=[q[0]], y=[q[1]], z=[q[2]], u=[R[0, 2]], v=[R[1, 2]], w=[R[2, 2]], sizemode="absolute",
                          sizeref=0.08, anchor="tip", colorscale=[[0, "#D55E00"], [1, "#D55E00"]], showscale=False,
                          hoverinfo="skip"))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=40, b=0), showlegend=False,
                   title=dict(text="Flat Gaussians (s = 0.32, 0.20, 0.04) on a sphere of radius 2: "
                                   "shortest axis r3 (vermillion) = normal", x=0.5, font=dict(size=13)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-29-1-3-interactive",
                config={"displaylogo": False})
print(f"saved {out}")
