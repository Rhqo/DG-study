"""Figure 41.5.2: mean curvature flow moves the edges of a cube, Gauss curvature flow only its corners.

cube_mesh(8, center 0, side 2) (each face an 8 x 8 grid of squares, two triangles each; faces outward). Explicit flows
x_i' = (L x)_i / A_i and x_i' = -(K N)_i / A_i with h = 0.2 x (2 / lambda_max) up to t = 0.1. The meshes at t = 0.1,
colored by the displacement |x_i(t) - x_i(0)| (same color scale, 0 to 0.7), seen obliquely (elevation 22, azimuth 30 degrees).
Also writes an interactive version with both meshes side by side.

Run: PYTHONPATH=tools python3 chapters/41-discrete-curvature-variational/figures/fig-41-5-2-cube-flows.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm, colors  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg41  # noqa: E402

V0, F = ddg41.cube_mesh(8, center=(0, 0, 0), side=2.0)
on_face = np.isclose(np.abs(V0), 1).sum(1)
center = int(np.argmin(np.linalg.norm(V0 - np.array([0, 0, 1.0]), axis=1)))
KN0 = ddg41.gauss_curvature_normal(V0, F)
assert np.abs(KN0[on_face < 3]).max() < 1e-14 and np.allclose(np.abs(KN0[on_face == 3]), np.pi / 4)
Lx0 = dm.cotan_apply(V0, F, V0)
assert np.abs(Lx0[on_face == 1]).max() < 1e-12 and np.abs(Lx0[on_face == 2]).max() > 0.2
hc = 2 / ddg41.largest_eigenvalue(V0, F)
h = 0.2 * hc
steps = int(round(0.1 / h))
h = 0.1 / steps
out = {}
for name, vel in (("mean", ddg41.mcf_velocity), ("gauss", ddg41.gcf_velocity)):
    V = ddg41.explicit_flow(V0, F, vel, h, steps)
    out[name] = (V, np.linalg.norm(V - V0, axis=1))
dM, dG = out["mean"][1], out["gauss"][1]
assert dG[center] < 1e-6 < dM[center] and dM[on_face == 3].max() > dG[on_face == 3].max()

norm = colors.Normalize(0, 0.7)
fig = plt.figure(figsize=(7.2, 3.9))
for k, (name, title) in enumerate((("mean", "(a) mean curvature flow"), ("gauss", "(b) Gauss curvature flow"))):
    V, d = out[name]
    ax = dgfig.axes3d(fig, 121 + k, elev=22, azim=30)
    fc = cm.viridis(norm(d[F].mean(1)))
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolor="#444444", lw=0.15))
    dgfig.equal_aspect(ax, V0, zoom=1.25)
    ax.set_title(title, loc="left", fontsize=11)
cax = fig.add_axes([0.36, 0.13, 0.28, 0.03])
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), cax=cax, orientation="horizontal")
cb.set_label(r"$|\mathbf{x}_i(t)-\mathbf{x}_i(0)|$", fontsize=10)
cb.set_ticks([0, 0.35, 0.7])
fig.subplots_adjust(left=0.0, right=1.0, bottom=0.22, top=0.93, wspace=0.0)
dgfig.save(fig, __file__)

shift = np.array([3.0, 0, 0])
tr = [dm.plotly_mesh(out["mean"][0] - shift, F, intensity=dM, colorscale="Viridis", cmin=0, cmax=0.7, showscale=True,
                     name="mean curvature flow", colorbar_title="displacement"),
      dm.plotly_mesh(out["gauss"][0] + shift, F, intensity=dG, colorscale="Viridis", cmin=0, cmax=0.7, name="Gauss curvature flow")]
dm.plotly_save(tr, __file__, "Cube at t = 0.1: mean curvature flow (left) and Gauss curvature flow (right), colored by displacement",
               "fig-41-5-2")
for name in ("mean", "gauss"):
    V, d = out[name]
    print(name, "steps", steps, "h", h, "volume", dm.signed_volume(V, F), "area", dm.face_areas(V, F).sum(), "corner", d[on_face == 3].max(),
          "edge max", d[on_face == 2].max(), "face center", d[center], "face max", d[on_face == 1].max())
