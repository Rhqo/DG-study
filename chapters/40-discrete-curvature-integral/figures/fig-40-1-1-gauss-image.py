"""Figure 40.1.1: the angle defect of a convex vertex is the area of its Gauss image.

A valence-6 star (ddg40.STAR_LINK below): vertex x_i = (0, 0, 0.55), link points at the angles 10, 65, 130, 185, 250,
305 degrees, radii 1.0, 0.85, 1.05, 0.9, 1.1, 0.8, heights 0.1, -0.12, 0.05, 0, -0.06, 0.08. Faces (i, j_k, j_{k+1}).
(a) the star with the six face normals N_k drawn at the face centroids (length 0.45).
(b) the unit sphere with the six normals and the great arcs between consecutive ones (Proposition 38.5.2): a convex
spherical hexagon whose interior angles are pi - theta_k (theta_k the corner angle of face k at x_i) and whose area
equals the angle defect delta_i = 2 pi - sum theta_k = 0.836 (Proposition 40.1.7).

Run: PYTHONPATH=tools python3 chapters/40-discrete-curvature-integral/figures/fig-40-1-1-gauss-image.py
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
import ddg40  # noqa: E402

C = dgfig.COLORS
V, F = ddg40.example_star()
i = 0
fs = ddg40.faces_around(F, i)
N = dm.face_normals(V, F)[fs]
theta = dm.corner_angles(V, F)[fs, 0]
delta = 2 * np.pi - theta.sum()
inner = ddg40.spherical_interior_angles(N)
area = ddg40.spherical_polygon_area(N)
# self-checks (Proposition 40.1.7): convex vertex, interior angles pi - theta, area = Girard = angle defect
assert ddg40.is_convex_vertex(V, F, i)
assert np.allclose(inner, np.pi - theta, atol=1e-12)
assert abs(area - delta) < 1e-12 and abs(ddg40.girard_area(inner) - delta) < 1e-12
assert abs(delta - 0.8362637747907984) < 1e-12


def arc(a, b, n=40):
    t = np.linspace(0, 1, n)
    P = (1 - t)[:, None] * a + t[:, None] * b
    return P / np.linalg.norm(P, axis=1, keepdims=True)


fig = plt.figure(figsize=(7.0, 3.5))
ax = dgfig.axes3d(fig, 121, elev=24, azim=-62)
ax.set_position([0.0, 0.0, 0.5, 1.0])
col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
ax.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="#555555", linewidths=0.7, alpha=0.75))
for k, f in enumerate(fs):
    c = V[F[f]].mean(0)
    dgfig.arrow3d(ax, c, 0.45 * N[k], role="normal", zorder=9, head=8)
k0 = 0
ax.text(*(V[F[fs[k0]]].mean(0) + 0.5 * N[k0] + np.array([0.04, -0.02, 0.04])), r"$\mathbf{N}_1$", color=C["normal"],
        fontsize=11)
dgfig.point3d(ax, V[i], None, size=14)
ax.text(*(V[i] + np.array([-0.05, 0.08, 0.1])), r"$\mathbf{x}_i$", fontsize=12)
dgfig.equal_aspect(ax, np.vstack([V, V[F].mean(1) + 0.5 * dm.face_normals(V, F)]), zoom=1.15)
ax.set_title("(a)", loc="left", fontsize=12, y=0.93)

ax2 = dgfig.axes3d(fig, 122, elev=58, azim=-62)
TH, PH = np.meshgrid(np.linspace(0, np.pi, 37), np.linspace(0, 2 * np.pi, 73), indexing="ij")
dgfig.surface(ax2, np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH), alpha=0.16, grid_every=6)
m = len(N)
P = np.vstack([arc(N[k], N[(k + 1) % m]) for k in range(m)])
ax2.add_collection3d(Poly3DCollection([P * 1.002], facecolors=C["accent"], alpha=0.45, edgecolors="none", zorder=6))
ax2.plot(*(P * 1.003).T, color=C["accent"], lw=1.6, zorder=8)
for n in N:
    dgfig.point3d(ax2, n * 1.004, None, role="normal", size=20)
ax2.text(*(N[k0] * 1.08 + np.array([0.06, -0.1, 0.0])), r"$\mathbf{N}_1$", color=C["normal"], fontsize=11)
# interior angle mark at N_1
a, b = N[k0], N[(k0 + 1) % m]
c = N[k0 - 1]
ta = (b - (b @ a) * a)
tc = (c - (c @ a) * a)
ta, tc = ta / np.linalg.norm(ta), tc / np.linalg.norm(tc)
tt = np.linspace(0, 1, 20)
mark = np.array([a + 0.12 * ((1 - s) * ta + s * tc) / np.linalg.norm((1 - s) * ta + s * tc) for s in tt])
mark /= np.linalg.norm(mark, axis=1, keepdims=True)
ax2.plot(*(mark * 1.005).T, color=C["main"], lw=1.0, zorder=9)
ax2.text(*(a * 1.0 + 0.3 * (ta + tc) / np.linalg.norm(ta + tc) + np.array([-0.1, 0.0, 0.05])), r"$\pi - \theta_1$",
         fontsize=10.5, zorder=10)
ax2.text2D(0.08, 0.06, r"area $= \delta_i = %.3f$" % delta, transform=ax2.transAxes, fontsize=11)
dgfig.equal_aspect(ax2, np.array([[-1, -1, -1], [1, 1, 1.0]]), zoom=1.25)
ax2.set_title("(b)", loc="left", fontsize=12, y=0.93)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
dgfig.save(fig, __file__)
print(f"theta = {np.round(theta, 4)}, sum {theta.sum():.4f}, delta {delta:.4f}, interior angles {np.round(inner, 4)}")
