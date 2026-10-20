"""Figure 45.1.2: the harmonic part of du on the torus of revolution (Section 45.1, Example 45.1.17).

Mesh: dgmesh.torus_mesh(2, 0.8, 24, 48), i.e. T_{R,r} with (R, r) = (2, 0.8) sampled at u = 2 pi i/24 (around the
tube) and v = 2 pi j/48 (around the z-axis). omega = R_1(du) = wrap(d0 u) (exact on every edge, closed, not exact).
gamma = omega - d0 alpha with d0^T star1 d0 alpha = d0^T star1 omega (Theorem 45.1.16(a)); the smooth harmonic form
in the class of du is gamma_s = (sqrt(R^2 - r^2) / (R + r cos u)) du (Example 45.1.17).
(a) the mesh (view elev 30, azim -60, painter's algorithm, faces facing away culled) shaded by |gamma^sharp| at the face
    barycenters (Whitney interpolant; light = small, dark = large), with blue arrows (0.3 times the true length) on
    the faces 2(i nv + j) with i, j even that face the viewer.
(b) the coefficient gamma_e / Delta u on the 24 x 48 meridian edges (Delta v = 0) against u at the edge midpoint
    (black dots, all 48 longitudes on top of each other), the exact coefficient sqrt(R^2 - r^2)/(R + r cos u)
    (orange) and the coefficient 1 of du itself (gray dashed).
Self-checks: max |gamma - R_1(gamma_s)| < 1e-3; d1 gamma = 0 and d0^T star1 gamma = 0 to 1e-12; the period of gamma
around a meridian loop is 2 pi.
Interactive version: the same mesh with the arrows on every face.

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-1-2-torus-harmonic.py
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
R, r, nu, nv = 2.0, 0.8, 24, 48
V, F = dm.torus_mesh(R, r, nu, nv)
E, _ = dd.edge_list(F)
D0, D1, s1 = dd.d0(F, len(V)), dd.d1(F), dd.hodge1(V, F)
u = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]) - R)
v = np.arctan2(V[:, 1], V[:, 0])
du, dv = ch45.wrap(D0 @ u), ch45.wrap(D0 @ v)
gamma = ch45.harmonic_basis(D0, s1, du[None])[0]
c = np.sqrt(R * R - r * r)
ex = np.zeros(len(E))
for t, w in zip(ch45.GL_T, ch45.GL_W):
    ex += w * c / (R + r * np.cos(u[E[:, 0]] + t * du)) * du
err = np.abs(gamma - ex).max()
assert err < 1e-3, err
assert np.abs(D1 @ gamma).max() < 1e-12 and np.abs(D0.T @ (s1 * gamma)).max() < 1e-12
loop = np.zeros(len(E))                              # meridian loop j = 0: vertices i*nv, i = 0..nu-1
_, idx = dd.edge_list(F)
for i in range(nu):
    a, b = i * nv, ((i + 1) % nu) * nv
    loop[idx[(min(a, b), max(a, b))]] += 1.0 if a < b else -1.0
assert abs(abs(loop @ gamma) - 2 * np.pi) < 1e-10
mer = (dv == 0) & (du != 0)
print(f"max |gamma - R_1(gamma_s)| = {err:.2e}; meridian coefficient {np.min(gamma[mer] / du[mer]):.4f} .. "
      f"{np.max(gamma[mer] / du[mer]):.4f}")

X = dd.whitney_at_barycenters(V, F, gamma)
mag = np.linalg.norm(X, axis=1)
cam = dm.Ortho(elev=30, azim=-60)
Nf = dm.face_normals(V, F)
cen = V[F].mean(1)
vis = Nf @ cam.view > 0
order = np.argsort(cam.depth(cen))
order = order[vis[order]]
light = np.clip(0.6 + 0.4 * (Nf @ ch45.unit(np.array([0.3, -0.4, 0.9]))), 0.35, 1.0)
cm = plt.get_cmap("Greys")
fc = cm(0.08 + 0.62 * mag / mag.max())
fc[:, :3] *= light[:, None]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.15, 1.0]})
axa.add_collection(PolyCollection(cam(V[F[order]]), facecolors=fc[order], edgecolors="#9A9A9A", linewidths=0.15))
fi, fj = (np.arange(len(F)) // 2) // nv, (np.arange(len(F)) // 2) % nv   # quad (i, j) of each face
front = vis & (Nf @ cam.view > 0.25) & (np.arange(len(F)) % 2 == 0) & (fi % 2 == 0) & (fj % 2 == 0)
for f in np.where(front)[0]:
    a0, a1 = cam(cen[f]), cam(cen[f] + 0.3 * X[f])
    axa.annotate("", xy=a1, xytext=a0, arrowprops=dict(arrowstyle="-|>", color=col["tangent"], lw=0.8,
                                                       mutation_scale=7), zorder=4)
axa.set_aspect("equal")
axa.autoscale_view()
axa.set_axis_off()
axa.set_title(r"(a) harmonic part $\gamma$ of $du$", fontsize=11)
mer = (dv == 0) & (du != 0)
um = ch45.wrap(u[E[mer, 0]] + du[mer] / 2)
tt = np.linspace(-np.pi, np.pi, 400)
axb.axhline(1.0, color=col["aux"], lw=1.0, ls="--", zorder=1)
axb.plot(tt, c / (R + r * np.cos(tt)), color=col["accent"], lw=2.6, alpha=0.8, zorder=2)
axb.plot(um, gamma[mer] / du[mer], ".", color=col["main"], ms=2.5, zorder=3)
axb.text(-2.95, 1.05, r"$du$", fontsize=10, color=col["aux"])
axb.text(-1.25, 1.36, r"$\sqrt{R^2 - r^2}\,/\,(R + r\cos u)$", fontsize=10, color=col["accent"])
axb.set_xlabel(r"$u$ (angle around the tube)")
axb.set_ylabel(r"$\gamma_e\,/\,\Delta u$")
axb.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
axb.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
axb.set_xlim(-np.pi, np.pi)
axb.set_title(r"(b) coefficient on meridian edges", fontsize=11)
axb.spines[["top", "right"]].set_visible(False)
fig.subplots_adjust(left=0.01, right=0.98, top=0.9, bottom=0.17, wspace=0.18)
dgfig.save(fig, __file__)

mesh = dm.plotly_mesh(V, F, intensity=mag, colorscale="Greys", cmin=0.0, cmax=float(mag.max()) * 1.6,
                      name="|gamma|", hovertemplate="|gamma| = %{intensity:.3f}<extra></extra>")
arrows = dm.plotly_arrows(cen, 0.12 * X, color=col["tangent"], width=3, cone=0.05, name="gamma")
dm.plotly_save([mesh, dm.plotly_edges(V, F, color="#9A9A9A", width=0.6)] + arrows, __file__,
               "torus_mesh(2, 0.8, 24, 48): harmonic part of du (arrows 0.12 x true length)", "fig-45-1-2")
