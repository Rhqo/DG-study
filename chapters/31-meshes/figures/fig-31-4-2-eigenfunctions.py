"""Figure 31.4.2: eigenfunctions of the cotan Laplacian are a Fourier basis on the surface (Section 31.4).

Top row: unit icosphere level 3 (642 vertices), eigenfunctions phi_1, phi_4, phi_9 (one from each of l = 1, 2, 3).
Bottom row: the ellipsoid x^2/1.6^2 + y^2/1.0^2 + z^2/0.7^2 = 1 (the same icosphere scaled), phi_1, phi_2, phi_3.
Mixed Voronoi mass; -L phi = lambda M phi; colour = phi_k / max|phi_k| (red positive, blue negative, white 0).
The axes x (to the lower right), y (to the right, away), z (up) follow the camera of the panels.
Also writes an interactive version with a menu over phi_1..phi_15 for both shapes.
Self-check: M-orthonormality and the eigen-equation residual; on the ellipsoid phi_1, phi_2, phi_3 change sign
along x, y, z respectively (their correlation with that coordinate is above 0.95 in absolute value).

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-4-2-eigenfunctions.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import cm  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

C = dgfig.COLORS
V0, F = dm.icosphere(3)
shapes = {"sphere": V0, "ellipsoid": V0 * np.array([1.6, 1.0, 0.7])}
eig = {}
for name, V in shapes.items():
    L = dm.cotan_laplacian(V, F)
    m = dm.mass_voronoi(V, F)
    lam, Phi = dm.generalized_eigs(L, m, 16)
    assert np.allclose(Phi.T @ (m[:, None] * Phi), np.eye(16), atol=1e-9)
    assert np.abs(-L @ Phi - (m[:, None] * Phi) * lam).max() < 1e-9
    eig[name] = (lam, Phi)
lamE, PhiE = eig["ellipsoid"]
VE = shapes["ellipsoid"]
for k, axis in ((1, 0), (2, 1), (3, 2)):
    assert abs(np.corrcoef(PhiE[:, k], VE[:, axis])[0, 1]) > 0.95

cam = dm.Ortho(elev=22, azim=-55)
picks = [("sphere", [1, 4, 9]), ("ellipsoid", [1, 2, 3])]
fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.0))
norm = Normalize(-1, 1)
for row, (name, ks) in enumerate(picks):
    V = shapes[name]
    lam, Phi = eig[name]
    for col, k in enumerate(ks):
        ax = axs[row, col]
        f = Phi[:, k] / np.abs(Phi[:, k]).max()
        rgba = cm.RdBu_r(norm(f[F].mean(1)))
        n = dm.face_normals(V, F)
        light = np.array([0.3, -0.5, 0.9]) / np.linalg.norm([0.3, -0.5, 0.9])
        rgba[:, :3] *= (0.75 + 0.25 * np.abs(n @ light))[:, None]
        P = dm.draw_mesh_2d(ax, V, F, cam, rgba, edgecolor="none", lw=0, cull=True, rasterized=True)
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.15, 1.15)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(rf"$\varphi_{{{k}}}$,  $\lambda_{{{k}}} = {lam[k]:.2f}$", fontsize=11.5)
    axs[row, 0].text(-0.12, 0.5, name, transform=axs[row, 0].transAxes, rotation=90, va="center", ha="center",
                     fontsize=12)
# small axis gizmo
ax = axs[1, 2]
o = np.array([1.2, -0.75])
for vec, lab in ((np.array([1, 0, 0]), r"$x$"), (np.array([0, 1, 0]), r"$y$"), (np.array([0, 0, 1]), r"$z$")):
    q = o + 0.5 * cam(vec)
    ax.annotate("", xy=q, xytext=o, arrowprops=dict(arrowstyle="-|>", color=C["aux"], lw=0.9))
    ax.text(*(o + 0.66 * cam(vec)), lab, fontsize=11, color=C["aux"], ha="center", va="center")
sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.04, pad=0.03, shrink=0.5)
cb.set_label(r"$\varphi_k / \max|\varphi_k|$", fontsize=10.5)
fig.subplots_adjust(left=0.04, right=0.99, top=0.94, bottom=0.14, wspace=0.02, hspace=0.12)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive: menu over k
import plotly.graph_objects as go  # noqa: E402

traces, labels = [], []
for si, (name, V) in enumerate(shapes.items()):
    lam, Phi = eig[name]
    off = np.array([0.0 if si == 0 else 3.6, 0, 0])
    for k in range(1, 16):
        f = Phi[:, k] / np.abs(Phi[:, k]).max()
        traces.append(dm.plotly_mesh(V + off, F, intensity=f, colorscale="RdBu_r", cmin=-1, cmax=1,
                                     name=f"{name} phi_{k}", hovertemplate=f"{name}, phi_{k}: %{{intensity:.3f}}<extra></extra>"))
        labels.append((si, k))
for t_, (si, k) in zip(traces, labels):
    t_.visible = (k == 1)
buttons = []
for k in range(1, 16):
    vis = [kk == k for (_, kk) in labels]
    buttons.append(dict(label=f"k = {k}", method="update",
                        args=[{"visible": vis},
                              {"title.text": f"Eigenfunction phi_{k}: sphere lambda = {eig['sphere'][0][k]:.3f}, "
                                             f"ellipsoid lambda = {eig['ellipsoid'][0][k]:.3f}"}]))
pfig = go.Figure(traces)
pfig.update_layout(scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=60, b=0), showlegend=False,
                   updatemenus=[dict(buttons=buttons, direction="down", x=0.02, y=1.0, xanchor="left", yanchor="top")],
                   title=dict(text=f"Eigenfunction phi_1: sphere lambda = {eig['sphere'][0][1]:.3f}, ellipsoid lambda = "
                                   f"{eig['ellipsoid'][0][1]:.3f}", x=0.5, font=dict(size=13)))
out = pathlib.Path(__file__).with_name(pathlib.Path(__file__).stem + "-interactive.html")
pfig.write_html(out, include_plotlyjs="cdn", full_html=True, div_id="fig-31-4-2-interactive", config={"displaylogo": False})
print(f"saved {out}")
