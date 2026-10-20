"""Figure 40.4.2: the mollified regular tetrahedron P + eps B and its Steiner polynomial.

P = ddg40.tetrahedron() (vertices (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1), edge 2 sqrt 2): V = 8/3, A = 8 sqrt 3,
every bending angle theta = pi - arccos(1/3) = 1.9106, every angle defect pi.
(a) the boundary of P + eps B for eps = 0.35, cut into its pieces (ddg40.mollified_pieces): translated faces (gray),
cylinder pieces along the six edges (sky blue), sphere pieces at the four vertices (orange).
(b) vol(P + eps B) for 0 <= eps <= 0.6 as the stack of the four terms of Theorem 40.4.10: V (gray), eps A (light gray),
(eps^2 / 2) sum l theta (sky blue), (eps^3 / 3) sum delta (orange); black dots: the volume counted on a grid of spacing
h = 0.02 (ddg40.mollified_volume_grid). Also writes an interactive version of (a).

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-4-2-mollified-tetrahedron.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg40  # noqa: E402

C = dgfig.COLORS
V, F = ddg40.tetrahedron()
vol, area, M, G = ddg40.steiner_coefficients(V, F)
assert abs(vol - 8 / 3) < 1e-12 and abs(area - 8 * np.sqrt(3)) < 1e-12 and abs(G - 4 * np.pi) < 1e-12
assert abs(M - 0.5 * 6 * 2 * np.sqrt(2) * (np.pi - np.arccos(1 / 3))) < 1e-12
EPS = 0.35
pieces = ddg40.mollified_pieces(V, F, EPS, n_arc=16, n_sph=10)
tot = {"face": 0.0, "edge": 0.0, "vertex": 0.0}
for P, Fp, k in pieces:
    tot[k] += dm.face_areas(P, Fp).sum()
assert abs(tot["face"] - area) < 1e-9 and abs(tot["edge"] - 2 * EPS * M) < 0.01 and abs(tot["vertex"] - EPS ** 2 * G) < 0.01
es = np.array([0.15, 0.3, 0.45, 0.6])
grid = np.array([ddg40.mollified_volume_grid(V, F, e, 0.02) for e in es])
poly = vol + es * area + es ** 2 * M + es ** 3 / 3 * G
assert np.all(np.abs(grid / poly - 1) < 0.015)                  # grid counting error (h = 0.02)

fig = plt.figure(figsize=(7.0, 3.4))
ax = fig.add_axes([0.0, 0.03, 0.44, 0.9])
cam = dm.Ortho(elev=28, azim=-75)
cols = {"face": C["surface"], "edge": C["region"], "vertex": C["accent"]}
allV, allF, allC = [], [], []
off = 0
for P, Fp, k in pieces:
    allV.append(P)
    allF.append(Fp + off)
    allC += [k] * len(Fp)
    off += len(P)
Va, Fa = np.vstack(allV), np.vstack(allF)
nrm = dm.face_cross(Va, Fa)
nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
cent = Va[Fa].mean(1)
flip = np.sum(nrm * cent, 1) < 0                                   # make every piece face outward for culling and shading
Fa[flip] = Fa[flip][:, ::-1]
rgba = []
light = np.array([0.4, -0.5, 0.8]) / np.linalg.norm([0.4, -0.5, 0.8])
nrm = dm.face_normals(Va, Fa)
for f, k in enumerate(allC):
    base = np.array(dm.hex_to_rgb(cols[k]))
    s = 0.55 + 0.45 * max(0.0, nrm[f] @ light)
    rgba.append(list(np.clip(base * s, 0, 1)) + [1.0])
dm.draw_mesh_2d(ax, Va, Fa, cam, np.array(rgba), edgecolor="none", lw=0.0, cull=True)
ax.set_aspect("equal")
lim = 1.9
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_axis_off()
ax.set_title("(a)", loc="left", fontsize=12)

ax2 = fig.add_axes([0.56, 0.16, 0.42, 0.74])
e = np.linspace(0, 0.6, 200)
terms = [vol + 0 * e, e * area, e ** 2 * M, e ** 3 / 3 * G]
labels = [r"$V$", r"$\varepsilon A$", r"$\frac{\varepsilon^2}{2}\sum\ell\theta$", r"$\frac{\varepsilon^3}{3}\sum\delta$"]
colors = ["#9a9a9a", C["surface"], C["region"], C["accent"]]
ax2.stackplot(e, *terms, labels=labels, colors=colors)
ax2.plot(es, grid, "o", color=C["main"], ms=4, label="grid count")
ax2.set_xlabel(r"$\varepsilon$")
ax2.set_ylabel(r"$\mathrm{vol}(P + \varepsilon B)$")
ax2.set_xlim(0, 0.6)
ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
ax2.set_title("(b)", loc="left", fontsize=12)
dgfig.save(fig, __file__)

tr = [dm.plotly_mesh(P, Fp, color=cols[k], name=k + " pieces", opacity=1.0) for P, Fp, k in pieces]
dm.plotly_save(tr, __file__, "The mollified tetrahedron P + 0.35 B: faces (gray), edge cylinders (blue), vertex spheres (orange)",
               "fig-40-4-2")
print(f"V {vol:.4f}, A {area:.4f}, (1/2) sum l theta {M:.4f}, sum delta {G:.4f}")
print("eps", es, "grid", np.round(grid, 4), "polynomial", np.round(poly, 4))
print("areas of the pieces at eps = 0.35:", {k: round(v, 4) for k, v in tot.items()})
