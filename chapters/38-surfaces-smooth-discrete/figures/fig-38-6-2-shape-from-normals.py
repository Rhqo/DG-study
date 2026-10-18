"""Figure 38.6.2: what the discrete Gauss map determines.

(a) Two faces f = (0, 1, 2), g = (1, 0, 3) of the hinge (ddg38.hinge, psi = 100 degrees) with unit normals N_f, N_g
(vermilion, at the barycenters) and the direction N_f x N_g (blue, at the edge midpoint): it is parallel to the shared
edge x_1 - x_0, because the edge is perpendicular to both normals (Lemma 38.6.7).
(b) Two triangulated boxes (ddg38.cube({}) scaled by (1, 1, 1) and by (1.5, 0.7, 0.45)): the same simplicial surface,
the same face normals +-e_k, but not similar. The diagonals of the squares are flat edges (N_f = N_g).
(c) torus_mesh(2, 0.8, 12, 8) and torus_mesh(1.2, 0.6, 12, 8): every quad of the grid is a planar trapezoid and its
normal does not depend on (R, r), so the two meshes have the same face normals (to 1e-15) but are not similar. One
quad (i, j) = (1, 7) and its normal are marked in both.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-6-2-shape-from-normals.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import dgfig  # noqa: E402
dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import dgmesh as dm  # noqa: E402
import ddg38  # noqa: E402

C = dgfig.COLORS

# ---------------------------------------------------------------- data and self-checks
Vh, Fh = ddg38.hinge(np.deg2rad(100))
Nh = dm.face_normals(Vh, Fh)
edge_dir = np.cross(Nh[0], Nh[1])
assert np.linalg.norm(np.cross(edge_dir, Vh[1] - Vh[0])) < 1e-12           # N_f x N_g is parallel to the edge
assert abs(np.linalg.norm(edge_dir) - np.sin(np.deg2rad(100))) < 1e-12      # |N_f x N_g| = sin(bending angle)

Vb, Fb = ddg38.cube({})
SCALES = [np.array([1.0, 1.0, 1.0]), np.array([1.5, 0.7, 0.45])]
boxes = [Vb * s for s in SCALES]
assert np.allclose(dm.face_normals(boxes[0], Fb), dm.face_normals(boxes[1], Fb))
nb = dm.face_normals(boxes[0], Fb)
flat = sum(1 for (i, j) in ddg38.edge_lengths(boxes[0], Fb)
           if np.allclose(*nb[[f for f in range(len(Fb)) if i in Fb[f] and j in Fb[f]]]))
assert flat == 6                                                            # the six square diagonals

TORI = [(2.0, 0.8), (1.2, 0.6)]
tori = [dm.torus_mesh(R, r, 12, 8) for R, r in TORI]
assert np.abs(dm.face_normals(*tori[0]) - dm.face_normals(*tori[1])).max() < 1e-14
Nt = dm.face_normals(*tori[0])
assert np.abs(Nt[0::2] - Nt[1::2]).max() < 1e-14                           # each quad is planar (flat diagonal)
q = 1 * 8 + 7                                                               # quad (i, j) = (1, 7): faces 2q, 2q + 1

fig = plt.figure(figsize=(7.2, 5.4))


def panel(rect, elev, azim):
    ax = fig.add_axes(rect, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.computed_zorder = False
    return ax


# ---------------------------------------------------------------- (a)
ax = panel([0.0, 0.5, 0.34, 0.47], 20, -62)
ax.add_collection3d(Poly3DCollection(Vh[Fh], facecolors=[C["surface"], C["region"]], edgecolors="#555555",
                                     linewidths=0.7, alpha=0.45))
ax.plot(*Vh[[0, 1]].T, color=C["main"], lw=2.2, zorder=6)
for f, lab, off in ((0, r"$\mathbf{N}_f$", (0.04, 0.0, 0.04)), (1, r"$\mathbf{N}_g$", (0.04, 0.0, 0.04))):
    dgfig.arrow3d(ax, Vh[Fh[f]].mean(0), 0.6 * Nh[f], role="normal", label=lab, zorder=10, head=9, label_offset=off)
m = (Vh[0] + Vh[1]) / 2
dgfig.arrow3d(ax, m, 0.5 * edge_dir / np.linalg.norm(edge_dir), role="tangent",
              label=r"$\mathbf{N}_f\times\mathbf{N}_g$", zorder=11, head=9, label_offset=(0.22, 0.0, -0.2))
dgfig.equal_aspect(ax, np.vstack([Vh, Vh[Fh[0]].mean(0) + 0.6 * Nh[0], Vh[Fh[1]].mean(0) + 0.6 * Nh[1]]), zoom=1.15)
fig.text(0.02, 0.95, "(a)", fontsize=12)

# ---------------------------------------------------------------- (b)
for k, V in enumerate(boxes):
    ax = panel([0.34 + 0.33 * k, 0.5, 0.33, 0.47], 22, -58)
    col = dm.shade(V, Fb, dm.hex_to_rgb(C["surface"]))
    ax.add_collection3d(Poly3DCollection(V[Fb], facecolors=col, edgecolors="#555555", linewidths=0.6, alpha=0.5))
    seen = []
    for f in range(len(Fb)):
        n = nb[f]
        if n @ np.array([0.5, -0.8, 0.4]) > 0 and not any(np.allclose(n, s) for s in seen):   # visible squares
            seen.append(n)
            sq = [g for g in range(len(Fb)) if np.allclose(nb[g], n)]
            dgfig.arrow3d(ax, V[np.unique(Fb[sq])].mean(0), 0.6 * n, role="normal", zorder=10, head=8)
    dgfig.equal_aspect(ax, np.vstack([Vb * 1.5, Vb * 0.6 * 1.5]), zoom=1.2)
fig.text(0.36, 0.95, "(b)", fontsize=12)

# ---------------------------------------------------------------- (c)
for k, (V, F) in enumerate(tori):
    ax = panel([0.5 * k, 0.0, 0.5, 0.44], 40, -90)
    col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
    fc = [C["region"] if f in (2 * q, 2 * q + 1) else col[f] for f in range(len(F))]
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="#666666", linewidths=0.4, alpha=0.6))
    c = V[np.unique(F[[2 * q, 2 * q + 1]])].mean(0)
    dgfig.arrow3d(ax, c, 0.9 * Nt[2 * q], role="normal", zorder=12, head=8)
    dgfig.equal_aspect(ax, tori[0][0], zoom=1.25)
fig.text(0.02, 0.40, "(c)", fontsize=12)
dgfig.save(fig, __file__)
