"""Figure 38.4.2: the hinge, two triangles sharing the edge x_0 x_1 (ddg38.hinge).

Triangle (0, 1, 2) is fixed in the plane z = 0 with x_2 = (0.5, 0.8, 0). Triangle (1, 0, 3) has
x_3(psi) = (0.3, -0.9 cos psi, 0.9 sin psi); the angle between the two face normals is psi. Drawn for
psi = 0, pi/2, 3 pi/4 (blue) and psi = pi (orange, folded onto the side of the fixed triangle). The edge lengths do not
depend on psi. For psi < pi the two triangles meet only along the common edge (a discrete immersion); at psi = pi they
overlap in a region of area 36/101 = 0.356.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-4-2-hinge.py
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
PSIS = [0.0, np.pi / 2, 3 * np.pi / 4, np.pi]

# self-checks: edge lengths constant, normal angle = psi, overlap only at psi = pi
ell0 = ddg38.edge_lengths(*ddg38.hinge(0.0))
for psi in np.linspace(0, np.pi, 9):
    V, F = ddg38.hinge(psi)
    ell = ddg38.edge_lengths(V, F)
    assert all(abs(ell[k] - ell0[k]) < 1e-12 for k in ell0)
    n = dm.face_normals(V, F)
    assert abs(np.arccos(np.clip(n[0] @ n[1], -1, 1)) - psi) < 1e-9
V, F = ddg38.hinge(np.pi)
ov = ddg38.planar_overlap(V[F[0]], V[F[1]])
assert abs(ov - 36 / 101) < 1e-12
V0, F0 = ddg38.hinge(0.0)
assert ddg38.planar_overlap(V0[F0[0]], V0[F0[1]]) < 1e-12

fig = plt.figure(figsize=(6.0, 4.0))
ax = dgfig.axes3d(fig, 111, elev=20, azim=-35)
V, F = ddg38.hinge(0.0)
ax.add_collection3d(Poly3DCollection([V[F[0]]], facecolors=C["surface"], edgecolors=C["main"], linewidths=1.0,
                                     alpha=0.7))
for psi in PSIS:
    V, F = ddg38.hinge(psi)
    col = C["accent"] if psi == np.pi else C["region"]
    ax.add_collection3d(Poly3DCollection([V[F[1]]], facecolors=col, edgecolors=C["tangent"] if psi < np.pi else
                                         C["accent"], linewidths=0.9, alpha=0.35))
    lab = {0.0: r"$\psi = 0$", np.pi / 2: r"$\psi = \pi/2$", 3 * np.pi / 4: r"$\psi = 3\pi/4$", np.pi: r"$\psi = \pi$"}[psi]
    off = {0.0: (0.0, -0.18, -0.08), np.pi / 2: (0.05, 0.0, 0.1), 3 * np.pi / 4: (-0.05, 0.08, 0.1),
           np.pi: (0.1, 0.12, 0.02)}[psi]
    ax.text(*(V[3] + np.array(off)), lab, fontsize=11, color=C["accent"] if psi == np.pi else C["tangent"])
ax.plot([0, 1], [0, 0], [0, 0], color=C["main"], lw=2.2)
V, F = ddg38.hinge(0.0)
for i, o in ((0, (-0.12, -0.05, -0.05)), (1, (0.05, -0.05, -0.05)), (2, (0.05, 0.05, 0.0))):
    ax.text(*(V[i] + np.array(o)), rf"$\mathbf{{x}}_{i}$", fontsize=12)
pts = np.vstack([ddg38.hinge(p)[0] for p in PSIS])
dgfig.equal_aspect(ax, pts, zoom=1.0)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
dgfig.save(fig, __file__)
