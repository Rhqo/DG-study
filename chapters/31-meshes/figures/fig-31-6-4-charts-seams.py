"""Figure 31.6.4: cutting a closed surface into charts trades seam length for distortion (Section 31.6).

The unit sphere as the level-4 icosphere (2562 vertices, 5120 faces). Four ways to cut it into disk-like charts
(faces assigned by their centroid c):
  1 chart:   the sphere minus a small cap around the south pole (faces with c_z < -cos 10 deg removed),
  2 charts:  the two hemispheres (sign of c_z),
  6 charts:  the faces of a cube (largest |coordinate| of c and its sign),
  20 charts: the faces of an icosahedron (nearest face centre of the icosahedron).
Each chart is flattened by LSCM [Levy02] with its farthest pair of boundary vertices pinned. For each face, sigma_1 sigma_2
is the surface area per unit UV area (sigma_i: singular values of the Jacobian UV -> surface).
(a) The 6-chart cut, coloured by chart, chart boundaries (seams) in black.
(b) For each cut: the total seam length (chart boundaries; each seam counted once; for 1 chart the rim of the hole)
    against the worst chart's ratio max sigma_1 sigma_2 / min sigma_1 sigma_2 (log scale).
Self-check: every chart is a disk (chi = 1) and no face is flipped; the ratios are about 14470, 4.8, 1.47, 1.10.
Also writes an interactive version (menu over the four cuts).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-6-4-charts-seams.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

C = dgfig.COLORS
V, F = dm.icosphere(4)
CEN = V[F].mean(1)
CEN /= np.linalg.norm(CEN, axis=1, keepdims=True)
Vi, Fi = dm.icosahedron()
Ci = Vi[Fi].mean(1)
Ci /= np.linalg.norm(Ci, axis=1, keepdims=True)
ax_ = np.argmax(np.abs(CEN), 1)
CUTS = {
    1: np.where(CEN[:, 2] < -np.cos(np.radians(10)), -1, 0),
    2: (CEN[:, 2] < 0).astype(int),
    6: 2 * ax_ + (CEN[np.arange(len(CEN)), ax_] < 0),
    20: np.argmax(CEN @ Ci.T, 1),
}


def farthest_pair(Vc, loop):
    P = Vc[loop]
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    a, b = np.unravel_index(np.argmax(D), D.shape)
    return loop[a], loop[b]


def seam_edges(lab):
    """Edges whose two faces lie in different charts (or next to a removed face)."""
    out = []
    for (i, j), fs in dm.edge_faces(F).items():
        if len(fs) == 2 and lab[fs[0]] != lab[fs[1]]:
            out.append((i, j))
    return np.array(out)


results = {}
for k, lab in CUTS.items():
    ratios = []
    for c in np.unique(lab):
        if c < 0:
            continue
        Vc, Fc, _ = dm.submesh(V, F, lab == c)
        assert dm.euler_characteristic(Fc) == 1 and dm.is_manifold(Fc)
        loop = dm.boundary_loop(Fc)
        UV = dm.lscm(Vc, Fc, farthest_pair(Vc, loop), [(0.0, 0.0), (1.0, 0.0)])
        s, det = dm.uv_distortion(Vc, Fc, UV)
        assert (det > 0).all()
        a = s[:, 0] * s[:, 1]
        ratios.append(a.max() / a.min())
    E = seam_edges(lab)
    length = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).sum()
    results[k] = (max(ratios), length)
    print(f"{k:2d} chart(s): worst max/min of s1 s2 = {max(ratios):.2f}, seam length = {length:.2f}")
expect = {1: 14470.06, 2: 4.84, 6: 1.47, 20: 1.10}
for k, r in expect.items():
    assert abs(results[k][0] / r - 1) < 0.01

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.8), gridspec_kw=dict(width_ratios=[0.85, 1.15], wspace=0.25))
cam = dm.Ortho(elev=25, azim=-55)
palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
lab = CUTS[6]
base = np.array([dm.hex_to_rgb(palette[c]) for c in lab])
light = np.array([0.4, -0.6, 0.7])
light /= np.linalg.norm(light)
lam = 0.55 + 0.45 * np.clip(dm.face_normals(V, F) @ light, 0, 1)
cols = np.clip(base * lam[:, None] + 0.15 * (1 - base), 0, 1)
dm.draw_mesh_2d(axa, V, F, cam, cols, edgecolor="none", lw=0, cull=True, rasterized=True)
E = seam_edges(lab)
P2 = cam(V)
for (i, j), ok in zip(E, cam.depth(V[E].mean(1)) > 0):   # seams on the visible hemisphere
    if ok:
        axa.plot(*P2[[i, j]].T, color="black", lw=1.2, zorder=5)
axa.set_xlim(-1.05, 1.05)
axa.set_ylim(-1.05, 1.05)
axa.set_aspect("equal")
axa.set_axis_off()
axa.set_title("(a) 6 charts (cube), seams in black", fontsize=12)

ks = [1, 2, 6, 20]
xs = [results[k][1] for k in ks]
ys = [results[k][0] for k in ks]
axb.semilogy(xs, ys, "-", color=C["aux"], lw=1.0)
axb.semilogy(xs, ys, "o", color=C["normal"], ms=7)
for k, x, y in zip(ks, xs, ys):
    txt = "1 chart\n(hole of 10°)" if k == 1 else f"{k} charts"
    off = (-30, 10) if k == 20 else (8, 4)
    axb.annotate(txt, (x, y), xytext=off, textcoords="offset points", fontsize=10.5)
axb.set_xlabel("total seam length")
axb.set_ylabel(r"worst $\max \sigma_1\sigma_2 / \min \sigma_1\sigma_2$")
axb.set_xlim(-2, 42)
axb.set_ylim(0.8, 1e5)
axb.grid(True, which="major", lw=0.3, alpha=0.5)
axb.set_title("(b) more seams, less area distortion", fontsize=12)
fig.subplots_adjust(left=0.01, right=0.97, top=0.9, bottom=0.14)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive: menu over the cuts
import plotly.graph_objects as go  # noqa: E402

pal = palette + ["#999999", "#444444"] * 10
traces, tags = [], []
for k, lab in CUTS.items():
    colors = [pal[c % len(pal)] if c >= 0 else "#FFFFFF" for c in lab]
    keepf = lab >= 0
    tr = go.Mesh3d(x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[keepf, 0], j=F[keepf, 1], k=F[keepf, 2],
                   facecolor=np.array(colors)[keepf], flatshading=True, hoverinfo="skip",
                   lighting=dict(ambient=0.6, diffuse=0.5, specular=0.05, facenormalsepsilon=0, vertexnormalsepsilon=0))
    E = seam_edges(lab)
    seg = np.full((len(E), 3, 3), np.nan)
    seg[:, 0], seg[:, 1] = 1.002 * V[E[:, 0]], 1.002 * V[E[:, 1]]
    seg = seg.reshape(-1, 3)
    ln = go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", line=dict(color="black", width=3),
                      hoverinfo="skip", showlegend=False)
    traces += [tr, ln]
    tags += [k, k]
for t_, k in zip(traces, tags):
    t_.visible = (k == 6)


def title_of(k):
    r, length = results[k]
    return f"{k} chart{'s' if k > 1 else ''}: seam length {length:.1f}, worst area ratio {r:.3g}"


buttons = [dict(label=f"{k} chart{'s' if k > 1 else ''}", method="update",
                args=[{"visible": [kk == k for kk in tags]}, {"title.text": title_of(k)}]) for k in CUTS]
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=60, b=0), showlegend=False,
                   updatemenus=[dict(buttons=buttons, active=2, direction="down", x=0.02, y=1.0, xanchor="left",
                                    yanchor="top")],
                   title=dict(text=title_of(6), x=0.5, font=dict(size=13)))
out = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-31-6-4-interactive", config={"displaylogo": False})
print(f"saved {out}")
