"""Figure 38.5.1: the discrete Gauss map near the vertex x_4 = e_3 of the octahedron (ddg38.octahedron).

(a) the octahedron with the unit normals N_f of the four faces at x_4, drawn from the face barycenters.
(b) the unit sphere with the four normals (+-1, +-1, 1)/sqrt 3 and the great arcs joining the normals of faces that share
an edge at x_4. The arcs bound a spherical quadrilateral; by Girard's formula its area is 2 pi / 3, which equals the
angle defect 2 pi - 4 (pi / 3) of the vertex (Section 40.1).

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-5-1-discrete-gauss-map.py
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
V, F = ddg38.octahedron()
i = 4
star = ddg38.star_faces(F, i)
N = dm.face_normals(V, F)
Ns = N[[np.where((F == f).all(1))[0][0] for f in star]]

# order the faces around x_4: consecutive faces share an edge (i, j)
order = [0]
while len(order) < len(star):
    last = star[order[-1]]
    k = list(last).index(i)
    nxt = last[(k + 2) % 3]                         # face (i, a, b) is followed by the face containing edge (i, b) as (i, b, c)
    for t, f in enumerate(star):
        kk = list(f).index(i)
        if f[(kk + 1) % 3] == nxt:
            order.append(t)
            break
Ns = Ns[order]
# self-checks: the normals, consecutive normals share a perpendicular edge, Girard area = angle defect = 2 pi / 3
assert np.allclose(np.sort(np.round(Ns * np.sqrt(3), 12), axis=0), np.sort(np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1],
                                                                                     [1, -1, 1]], float), axis=0))
angles = []
m = len(Ns)
for k in range(m):
    a, b, c = Ns[k - 1], Ns[k], Ns[(k + 1) % m]
    ta = a - (a @ b) * b
    tc = c - (c @ b) * b
    angles.append(np.arccos(np.clip(ta @ tc / (np.linalg.norm(ta) * np.linalg.norm(tc)), -1, 1)))
area = sum(angles) - (m - 2) * np.pi
defect = dm.angle_defect(V, F)[i]
assert np.isclose(area, 2 * np.pi / 3) and np.isclose(defect, 2 * np.pi / 3)


def arc(a, b, n=60):
    t = np.linspace(0, 1, n)
    P = (1 - t)[:, None] * a + t[:, None] * b
    return P / np.linalg.norm(P, axis=1, keepdims=True)


fig = plt.figure(figsize=(7.0, 3.6))
ax = dgfig.axes3d(fig, 121, elev=24, azim=-55)
ax.set_position([0.0, 0.0, 0.5, 1.0])
col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
ax.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="#555555", linewidths=0.6, alpha=0.45))
for f in star:
    c = V[f].mean(0)
    n = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
    dgfig.arrow3d(ax, c, 0.6 * n / np.linalg.norm(n), role="normal", zorder=9, head=8)
dgfig.point3d(ax, V[i], None, size=12)
ax.text(*(V[i] + np.array([0.05, 0.05, 0.12])), r"$\mathbf{x}_4$", fontsize=12)
dgfig.equal_aspect(ax, np.vstack([V, V * 1.25]), zoom=1.55)
ax.set_title("(a)", loc="left", fontsize=12, y=0.93)

ax2 = dgfig.axes3d(fig, 122, elev=35, azim=-55)
TH, PH = np.meshgrid(np.linspace(0, np.pi, 37), np.linspace(0, 2 * np.pi, 73), indexing="ij")
dgfig.surface(ax2, *ddg38.sphere(TH, PH), alpha=0.18, grid_every=6)
P = np.vstack([arc(Ns[k], Ns[(k + 1) % m]) for k in range(m)])
ax2.add_collection3d(Poly3DCollection([P], facecolors=C["accent"], alpha=0.35, edgecolors="none"))
ax2.plot(*P.T, color=C["accent"], lw=1.8, zorder=8)
for n in Ns:
    dgfig.point3d(ax2, n, None, role="normal", size=22)
ax2.text(*(Ns[0] * 1.12), r"$\mathbf{N}_f$", color=C["normal"], fontsize=11)
dgfig.point3d(ax2, [0, 0, 1.0], None, size=12)
dgfig.equal_aspect(ax2, np.array([[-1, -1, -1], [1, 1, 1.0]]), zoom=1.25)
ax2.set_title("(b)", loc="left", fontsize=12, y=0.93)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
dgfig.save(fig, __file__)
