"""Figure 31.2.3: angle defects on the torus of revolution and the discrete Gauss-Bonnet theorem (Section 31.2).

(a) T_{R,r} with R = 2, r = 0.8 sampled on the 24 x 48 (u, v) grid; faces coloured by the mean angle defect of
    their three vertices (diverging colormap, white = 0). Sum of all 1152 defects: about 4e-13.
(b) K_i = delta_i / A_i (A_i = barycentric area) against u for the 12 x 24 grid (open circles) and the 24 x 48 grid
    (dots), and the exact K(u) = cos u / (r (R + r cos u)) (black curve, GUIDELINES 7).
Also writes an interactive plotly version coloured by delta_i.
Self-check: sum of delta is 0 to 1e-11; the maximal error of K_i is below 0.03 and 0.008 on the two grids.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-2-3-torus-defect.py``
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
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

C = dgfig.COLORS
R, r = 2.0, 0.8


def torus_data(nu, nv):
    V, F = dm.torus_mesh(R, r, nu, nv)
    u = np.repeat(2 * np.pi * np.arange(nu) / nu, nv)
    d = dm.angle_defect(V, F)
    A = dm.mass_barycentric(V, F)
    return V, F, u, d, A


fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.9), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.28))

# ---------------------------------------------------------------- (a)
V, F, u, d, A = torus_data(24, 48)
assert abs(d.sum()) < 1e-11
fval = d[F].mean(1)
norm = TwoSlopeNorm(vcenter=0.0, vmin=-np.abs(d).max(), vmax=np.abs(d).max())
rgba = cm.RdBu_r(norm(fval))
cam = dm.Ortho(elev=40, azim=-60)
nrm = dm.face_normals(V, F)
light = np.array([0.3, -0.5, 0.9]) / np.linalg.norm([0.3, -0.5, 0.9])
k = 0.7 + 0.3 * np.abs(nrm @ light)
rgba[:, :3] *= k[:, None]
P = dm.draw_mesh_2d(ax, V, F, cam, rgba, edgecolor="#777777", lw=0.15, cull=True)
ax.set_xlim(P[:, 0].min() - 0.1, P[:, 0].max() + 0.1)
ax.set_ylim(P[:, 1].min() - 0.1, P[:, 1].max() + 0.1)
ax.set_aspect("equal")
ax.set_axis_off()
sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.7)
cb.ax.tick_params(labelsize=8.5)
ax.set_title(r"(a) $\delta_i$ on a $24 \times 48$ torus mesh", fontsize=11.5)
mant, expo = ("%.0e" % d.sum()).split("e")
cb.set_label(r"angle defect $\delta_i$;  $\sum_i \delta_i = %s\times 10^{%d}$" % (mant, int(expo)), fontsize=10.5)

# ---------------------------------------------------------------- (b)
uu = np.linspace(0, 2 * np.pi, 400)
axb.plot(uu, np.cos(uu) / (r * (R + r * np.cos(uu))), color=C["main"], lw=1.6, label=r"exact $K(u)$", zorder=1)
for (nu, nv), sty in (((12, 24), dict(mfc="none", mec=C["tangent"], ms=7, marker="o", ls="none")),
                      ((24, 48), dict(color=C["accent"], ms=4, marker="o", ls="none"))):
    V2, F2, u2, d2, A2 = torus_data(nu, nv)
    Ki = d2 / A2
    Kex = np.cos(u2) / (r * (R + r * np.cos(u2)))
    err = np.abs(Ki - Kex).max()
    assert err < (0.03 if nu == 12 else 0.008)
    sel = np.arange(0, len(u2), nv)  # one vertex per ring of constant u (all vertices on a ring agree)
    axb.plot(u2[sel], Ki[sel], label=rf"$\delta_i/A_i$, ${nu}\times{nv}$", zorder=2, **sty)
axb.axhline(0, color=C["aux"], lw=0.8)
axb.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
axb.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
axb.set_xlabel(r"$u$ (angle around the tube; $u = 0$ outside)")
axb.set_ylabel("Gaussian curvature")
axb.legend(fontsize=10.5, frameon=False, loc="upper center")
axb.set_title(r"(b) $\delta_i / A_i$ against $u$", fontsize=11.5)
axb.set_ylim(-1.2, 0.85)

fig.subplots_adjust(left=0.01, right=0.98, top=0.86, bottom=0.15)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
lim = float(np.abs(d).max())
tr = [dm.plotly_mesh(V, F, intensity=d, colorscale="RdBu_r", cmin=-lim, cmax=lim, showscale=True,
                     name="angle defect", colorbar_title="delta_i",
                     hovertemplate="angle defect %{intensity:.4f}<extra></extra>"),
      dm.plotly_edges(V, F, color="#888888", width=1)]
dm.plotly_save(tr, __file__, "Angle defects on the torus T_{2,0.8} (24 x 48 grid): positive outside, negative inside, "
               "sum = 0", "fig-31-2-3-interactive")
