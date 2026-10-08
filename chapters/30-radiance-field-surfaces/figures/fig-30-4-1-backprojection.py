"""Figure 30.4.1: a depth map is a parametrized surface; it breaks at depth discontinuities (Section 30.4).

Synthetic scene seen by a pinhole camera at the origin looking along +z: image 96 x 72 px, f_x = f_y = 100 px,
(c_x, c_y) = (47.5, 35.5). Background: a tilted plane whose inverse depth is 0.25 + 0.002 (v - 35.5) (closer at the
bottom). Foreground: a sphere of radius 0.6 centered at (0.3, -0.1, 2.5). d(u, v) = z-depth of the first hit.
(a) the depth map d(u, v) as an image (sequential colormap), orange: pixels next to a depth jump > 0.3; white dashed:
    the pixel row v = 32 used in (b).
(b) top view of that row: the back-projected points X(u, 32) = d (xb, yb, 1) drawn in the (x, z) plane (every pixel;
    black = sphere, gray = background), the rays through every 6th pixel (thin gray), the full sphere cross-section and
    the background line behind the sphere (dotted: not seen by the camera), and the two orange segments a naive mesh
    adds between the neighbouring pixels across each depth jump. The camera is below the panel (rays converge to it).
(c) the same in 3D: the back-projected points drawn as the images of every 3rd pixel row and column (black on the
    sphere, gray on the plane), in plot axes (x, z, -y); orange: all quads a naive mesh would create across the
    silhouette. They form part of the cone of camera rays through the occluding contour.
Self-checks: depth positive; every orange quad spans a depth jump > 0.3; on smooth parts the back-projection is an
immersion (<X_u x X_v, r> = d^2/(f_x f_y) > 0); in (b) the orange segments are nearly parallel to the rays (< 1.2 deg).
Also writes an interactive plotly version (same data).

Run from the project root: ``PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/figures/fig-30-4-1-backprojection.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
from pathlib import Path

import dgfig

dgfig.setup()

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

matplotlib.rcParams.update({"font.size": 13, "axes.titlesize": 13.5, "axes.labelsize": 13,
                            "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 11.5})
C = dgfig.COLORS
Wd, Ht, F = 96, 72, 100.0
CX, CY = (Wd - 1) / 2, (Ht - 1) / 2
uu, vv = np.meshgrid(np.arange(Wd, dtype=float), np.arange(Ht, dtype=float), indexing="xy")
R = np.stack([(uu - CX) / F, (vv - CY) / F, np.ones_like(uu)], -1)          # K^{-1}(u, v, 1)

rho_bg = 0.25 + 0.002 * (vv - CY)
d_bg = 1 / rho_bg
Cs, Rs = np.array([0.3, -0.1, 2.5]), 0.6
a = (R * R).sum(-1)
b = -2 * R @ Cs
c = Cs @ Cs - Rs ** 2
disc = b * b - 4 * a * c
t_s = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
D = np.minimum(t_s, d_bg)                                                    # z-depth (r has z = 1)
on_sphere = t_s < d_bg
assert np.all(D > 0)
X = D[..., None] * R

jump_h = np.abs(np.diff(D, axis=1)) > 0.3
jump_v = np.abs(np.diff(D, axis=0)) > 0.3
edge = np.zeros_like(D, bool)
edge[:, :-1] |= jump_h
edge[:, 1:] |= jump_h
edge[:-1, :] |= jump_v
edge[1:, :] |= jump_v
assert np.all(np.diff(on_sphere.astype(int), axis=1)[jump_h] != 0)

# immersion check on a smooth sphere pixel and a plane pixel (central differences)
for (i, j) in [(36, 52), (5, 10)]:
    xu = (X[i, j + 1] - X[i, j - 1]) / 2
    xv = (X[i + 1, j] - X[i - 1, j]) / 2
    assert abs(np.cross(xu, xv) @ R[i, j] - D[i, j] ** 2 / F ** 2) < 2e-3 * D[i, j] ** 2 / F ** 2

V0 = 32                                                                      # the row shown in (b)
fig = plt.figure(figsize=(5.8, 11.8))
axa = fig.add_axes([0.11, 0.758, 0.70, 0.222])
im = axa.imshow(D, cmap="viridis", origin="upper", extent=(-0.5, Wd - 0.5, Ht - 0.5, -0.5), vmin=2.0, vmax=5.8)
ey, ex = np.nonzero(edge)
axa.plot(ex, ey, "s", color=C["accent"], ms=2.2, mew=0)
axa.plot([-0.5, Wd - 0.5], [V0, V0], color="white", lw=1.3, ls=(0, (4, 2)))
axa.text(2, V0 - 1.5, "row for (b)", color="white", fontsize=11, va="bottom")
axa.set_xlabel(r"pixel $u$")
axa.set_ylabel(r"pixel $v$")
axa.set_title(r"(a) depth map $d(u, v)$, orange: depth jumps", fontsize=13)
cb = fig.colorbar(im, ax=axa, fraction=0.04, pad=0.03)
cb.set_label(r"$d$")

# (b) top view of one row
axr = fig.add_axes([0.11, 0.418, 0.84, 0.255])
row = X[V0]                                                                  # (Wd, 3)
sph = on_sphere[V0]
jumps = np.nonzero(np.abs(np.diff(D[V0])) > 0.3)[0]
assert len(jumps) == 2
for j in range(0, Wd, 6):
    axr.plot([0, row[j, 0]], [0, row[j, 2]], color=C["aux"], lw=0.5, alpha=0.6, zorder=1)
yb0 = (V0 - CY) / F
tt = np.linspace(0, 2 * np.pi, 400)
# the sphere's cross-section by the plane of this row's rays (y = yb0 z), seen from above (x, z)
nrm_pl = np.array([0.0, 1.0, -yb0]) / np.hypot(1.0, yb0)
dist_c = Cs @ nrm_pl
cen = Cs - dist_c * nrm_pl
rad = np.sqrt(Rs ** 2 - dist_c ** 2)
e1 = np.array([1.0, 0.0, 0.0])
e2 = np.cross(nrm_pl, e1)
circ = cen[None] + rad * (np.cos(tt)[:, None] * e1[None] + np.sin(tt)[:, None] * e2[None])
axr.plot(circ[:, 0], circ[:, 2], color=C["main"], lw=0.9, ls=":", zorder=2)
d_bg_row = d_bg[V0, 0]
axr.plot([-0.475 * d_bg_row, 0.475 * d_bg_row], [d_bg_row, d_bg_row], color=C["aux"], lw=0.9, ls=":", zorder=2)
axr.plot(row[~sph, 0], row[~sph, 2], "o", color=C["aux"], ms=2.4, zorder=3)
axr.plot(row[sph, 0], row[sph, 2], "o", color=C["main"], ms=2.4, zorder=4)
for j in jumps:
    a_, b_ = row[j], row[j + 1]
    axr.plot([a_[0], b_[0]], [a_[2], b_[2]], color=C["accent"], lw=2.6, zorder=5)
    seg = (b_ - a_)[[0, 2]]
    ray = ((a_ + b_) / 2)[[0, 2]]
    ang = np.degrees(np.arccos(abs(seg @ ray) / np.linalg.norm(seg) / np.linalg.norm(ray)))
    assert ang < 1.2, ang
axr.annotate("depth jump: a naive mesh\nadds this edge", xy=row[jumps[1]][[0, 2]] * 0.5 + row[jumps[1] + 1][[0, 2]] * 0.5,
             xytext=(0.98, 1.98), fontsize=11, color=C["accent"], arrowprops=dict(arrowstyle="->", lw=0.9, color=C["accent"]),
             bbox=dict(fc="white", ec="none", pad=0.5))
axr.annotate("smooth part:\na regular curve", xy=(row[sph][len(row[sph]) // 2, 0], row[sph][len(row[sph]) // 2, 2]),
             xytext=(-2.08, 2.5), fontsize=11, arrowprops=dict(arrowstyle="->", lw=0.9),
             bbox=dict(fc="white", ec="none", pad=0.5))
axr.text(0.15, 4.2, "occluded (not in the depth map)", fontsize=10.5, color=C["aux"], ha="center")
axr.text(-2.08, 1.62, "gray rays: to the camera\nat z = 0 (below the panel)", fontsize=10.5, color=C["aux"], va="bottom",
         bbox=dict(fc="white", ec="none", pad=0.5))
axr.set_aspect("equal")
axr.set_xlim(-2.15, 2.15)
axr.set_ylim(1.55, 4.45)
axr.set_xlabel(r"$x$")
axr.set_ylabel(r"depth $z$")
axr.set_title(rf"(b) row $v = {V0}$ seen from above: back-projected points", fontsize=13)

# (c) the same in 3D
axb = fig.add_axes([-0.02, -0.055, 1.04, 0.385], projection="3d", computed_zorder=False)
axb.patch.set_visible(False)
axb.view_init(elev=25, azim=-110)
axb.set_axis_off()
P = np.stack([X[..., 0], X[..., 2], -X[..., 1]], -1)                         # plot axes (x, z, -y)
step = 3
for i in range(0, Ht, step):
    rowp = P[i]
    brk = np.abs(np.diff(D[i])) > 0.3
    segs = np.split(np.arange(Wd), np.nonzero(brk)[0] + 1)
    for sgm in segs:
        if len(sgm) > 1:
            axb.plot(*rowp[sgm].T, color=C["main"] if on_sphere[i, sgm[0]] else C["aux"],
                     lw=0.8 if on_sphere[i, sgm[0]] else 0.6, zorder=4 if on_sphere[i, sgm[0]] else 2)
for j in range(0, Wd, step):
    col = P[:, j]
    brk = np.abs(np.diff(D[:, j])) > 0.3
    segs = np.split(np.arange(Ht), np.nonzero(brk)[0] + 1)
    for sgm in segs:
        if len(sgm) > 1:
            axb.plot(*col[sgm].T, color=C["main"] if on_sphere[sgm[0], j] else C["aux"],
                     lw=0.8 if on_sphere[sgm[0], j] else 0.6, zorder=4 if on_sphere[sgm[0], j] else 2)
quads = []
for i in range(Ht - 1):
    for j in range(Wd - 1):
        ds = D[i:i + 2, j:j + 2]
        if ds.max() - ds.min() > 0.3:
            quads.append([P[i, j], P[i, j + 1], P[i + 1, j + 1], P[i + 1, j]])
pc = Poly3DCollection(quads, facecolor=C["accent"], edgecolor="none", alpha=0.4, zorder=3)
axb.add_collection3d(pc)
axb.plot([0], [0], [0], "o", color=C["main"], ms=6, zorder=5)
axb.text(0.15, 0.0, -0.3, "camera", fontsize=11.5, zorder=6)
dgfig.equal_aspect(axb, P.reshape(-1, 3), np.zeros((1, 3)), zoom=1.3)
fig.text(0.5, 0.338, r"(c) the same in 3D: $X(u, v) = d\,\mathsf{K}^{-1}(u, v, 1)^{\mathsf{T}}$", ha="center",
         fontsize=13)
fig.text(0.04, 0.285, "orange: faces of a naive mesh\nacross the silhouette", color=C["accent"], fontsize=11.5, ha="left")
dgfig.save(fig, __file__)
print(f"{len(quads)} jump quads, {edge.sum()} edge pixels")

# ---------------------------------------------------------------- interactive version
import plotly.graph_objects as go

idx = np.arange(Ht * Wd).reshape(Ht, Wd)
I, J, K = [], [], []
Ij, Jj, Kj = [], [], []
for i in range(Ht - 1):
    for j in range(Wd - 1):
        q = [idx[i, j], idx[i, j + 1], idx[i + 1, j + 1], idx[i + 1, j]]
        ds = D[i:i + 2, j:j + 2]
        tgt = (Ij, Jj, Kj) if ds.max() - ds.min() > 0.3 else (I, J, K)
        tgt[0].extend([q[0], q[0]])
        tgt[1].extend([q[1], q[2]])
        tgt[2].extend([q[2], q[3]])
Pf = P.reshape(-1, 3)
uv = np.stack([uu.ravel(), vv.ravel(), D.ravel()], -1)
traces = [go.Mesh3d(x=Pf[:, 0], y=Pf[:, 1], z=Pf[:, 2], i=I, j=J, k=K, intensity=D.ravel(), colorscale="Viridis",
                    cmin=2.0, cmax=5.8, colorbar=dict(title="d"), customdata=uv, name="depth surface",
                    hovertemplate="u = %{customdata[0]:.0f}, v = %{customdata[1]:.0f}<br>d = %{customdata[2]:.3f}<extra></extra>"),
          go.Mesh3d(x=Pf[:, 0], y=Pf[:, 1], z=Pf[:, 2], i=Ij, j=Jj, k=Kj, color="#E69F00", opacity=0.5,
                    name="faces across depth jumps", hoverinfo="skip"),
          go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["camera"], marker=dict(size=4, color="black"),
                       hoverinfo="skip", showlegend=False)]
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="z (depth)", zaxis_title="-y"),
                   margin=dict(l=0, r=0, t=40, b=0),
                   title=dict(text="Back-projected depth map; orange = faces a naive mesh puts across the silhouette",
                              x=0.5, font=dict(size=14)))
out = Path(__file__).with_name(Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-30-4-1-interactive", config={"displaylogo": False})
print(f"saved {out}")
