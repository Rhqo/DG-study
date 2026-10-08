"""Figure 31.5.2: what a level-set mesh gets right and wrong (Section 31.5).

The unit sphere as the zero set of its exact SDF |x| - 1, sampled on a grid of spacing h (offset 0.0137 so that no
sample is exactly on the sphere) and extracted by marching tetrahedra (linear interpolation on grid edges).
(a) A patch of the h = 0.1 mesh seen from outside (orthographic, looking at the point (1, 1, 1)/sqrt 3). Faces with a
    corner angle below 5 degrees are vermillion and circled.
(b) Errors against h (log-log), h = 0.2, 0.14, 0.1, 0.07, 0.05:
    vertex position max | |x_i| - 1 |,  face normal error (mean angle to the exact normal, radians),
    pointwise mean curvature: median of |<(M^-1 L x)_i, N_i> - 2H| with 2H = -2 (mixed Voronoi M),
    integrated mean curvature: |sum_i A_i <(M^-1 L x)_i, N_i> / sum_i A_i - 2H|.
Self-check: position error ~ h^2, normal error ~ h, the pointwise curvature error stays near 1 (50% of |2H|),
the integrated one ~ h^2; about 5.7% of the faces have a corner angle below 5 degrees at every h.
Also writes an interactive version: the whole h = 0.1 sphere, faces with an angle below 5 degrees in vermillion.

Run from the project root: ``PYTHONPATH=tools python3 chapters/31-meshes/figures/fig-31-5-2-level-set-accuracy.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import dgmesh as dm  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

C = dgfig.COLORS


def sphere_mesh(h):
    xs = np.arange(-1.3, 1.3 + 1e-9, h) + 0.0137
    G = np.stack(np.meshgrid(xs, xs, xs, indexing="ij"), -1)
    return dm.marching_tetrahedra(np.linalg.norm(G, axis=-1) - 1, xs, xs, xs)


hs = [0.2, 0.14, 0.1, 0.07, 0.05]
rows = []
for h in hs:
    V, F = sphere_mesh(h)
    assert dm.euler_characteristic(F) == 2 and dm.is_manifold(F)
    r = np.linalg.norm(V, axis=1)
    fn = dm.face_normals(V, F)
    cen = V[F].mean(1)
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    nerr = np.arccos(np.clip((fn * cen).sum(1), -1, 1)).mean()
    m = dm.mass_voronoi(V, F)
    nc = ((dm.cotan_apply(V, F, V) / m[:, None]) * (V / r[:, None])).sum(1)
    sliver = (np.degrees(dm.corner_angles(V, F)).min(1) < 5).mean()
    rows.append((h, np.abs(r - 1).max(), nerr, np.median(np.abs(nc + 2)), abs((nc * m).sum() / m.sum() + 2), sliver))
    if h == 0.1:
        V10, F10 = V, F
R = np.array(rows)
rate = lambda col: np.polyfit(np.log(R[:, 0]), np.log(R[:, col]), 1)[0]  # noqa: E731
assert 1.8 < rate(1) < 2.2 and 0.85 < rate(2) < 1.15 and 1.8 < rate(4) < 2.2
assert (R[:, 3] > 0.9).all() and (R[:, 3] < 1.2).all()
assert (np.abs(R[:, 5] - 0.057) < 0.006).all()

fig, (ax, axb) = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw=dict(width_ratios=[0.9, 1.1], wspace=0.3))
d = np.ones(3) / np.sqrt(3)
cam = dm.Ortho(elev=np.degrees(np.arcsin(d[2])), azim=45)
near = (V10[F10].mean(1) @ d) > 0.93
Fp = F10[near]
order = np.argsort(cam.depth(V10[Fp].mean(1)))
Fp = Fp[order]
sl = np.degrees(dm.corner_angles(V10, Fp)).min(1) < 5
cols = np.where(sl[:, None], dm.hex_to_rgb(C["normal"])[None, :], dm.hex_to_rgb(C["surface"])[None, :])
pc = PolyCollection(cam(V10[Fp]), facecolors=cols, edgecolors="#333333", linewidths=0.6)
ax.add_collection(pc)
cs = cam(V10[Fp[sl]].mean(1))
ax.plot(cs[:, 0], cs[:, 1], "o", ms=7, mfc="none", mec=C["normal"], mew=1.6, zorder=5)
P = cam(V10[np.unique(Fp)])
ax.set_xlim(P[:, 0].min() - 0.01, P[:, 0].max() + 0.01)
ax.set_ylim(P[:, 1].min() - 0.01, P[:, 1].max() + 0.01)
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title(f"(a) a patch, h = 0.1: {sl.sum()} of {len(sl)} faces\nhave an angle < 5° (circled)", fontsize=11)

styles = [(1, "vertex position", C["main"], "o-"), (2, "face normal (rad)", C["tangent"], "s-"),
          (3, r"pointwise $2H$ (median)", C["normal"], "^--"), (4, r"integrated $2H$", C["accent"], "D-")]
for col, lab, color, mk in styles:
    axb.loglog(R[:, 0], R[:, col], mk, color=color, ms=5, lw=1.5, label=lab)
axb.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2]))
axb.xaxis.set_major_formatter(ScalarFormatter())
axb.xaxis.set_minor_formatter(NullFormatter())
axb.set_xlabel(r"grid spacing $h$")
axb.set_ylabel("error")
axb.set_ylim(1e-6, 3)
axb.legend(fontsize=10, frameon=False, loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.6)
axb.grid(True, which="major", lw=0.3, alpha=0.5)
axb.set_title("(b) errors against grid spacing", fontsize=11.5)

fig.subplots_adjust(left=0.01, right=0.98, top=0.86, bottom=0.14)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive: the whole h = 0.1 sphere
slv = np.degrees(dm.corner_angles(V10, F10)).min(1) < 5
tr = dm.plotly_mesh(V10.astype(np.float32), F10.astype(np.int32), intensity=slv.astype(float),
                    colorscale=[[0, "#D0D0D0"], [1, C["normal"]]], cmin=0, cmax=1, name="faces",
                    hovertemplate="min angle < 5 deg: %{intensity}<extra></extra>")
tr.lighting.facenormalsepsilon = 0
tr.lighting.vertexnormalsepsilon = 0
tr.lightposition = dict(x=1e5, y=-1e5, z=1e5)
edges_ = np.array(dm.edges(F10))
seg = np.full((len(edges_), 3, 3), np.nan, dtype=np.float32)
seg[:, 0], seg[:, 1] = V10[edges_[:, 0]], V10[edges_[:, 1]]
seg = seg.reshape(-1, 3)
import plotly.graph_objects as go  # noqa: E402
et = go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines", line=dict(color="#444444", width=1.0),
                  hoverinfo="skip", showlegend=False)
dm.plotly_save([tr, et], __file__, f"Marching tetrahedra of the unit sphere SDF, h = 0.1: {slv.sum()} of {len(slv)} faces "
               f"have an angle < 5 deg (vermillion)", "fig-31-5-2-interactive")
