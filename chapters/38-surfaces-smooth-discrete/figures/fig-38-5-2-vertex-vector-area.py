"""Figure 38.5.2: the barycentric discrete vector area of a vertex does not depend on the position of the vertex.

An irregular valence-5 star: link vertices at angles 0, 70, 150, 200, 290 degrees, radii 1.0, 0.8, 1.1, 0.9, 1.0 and
heights 0.1, 0.3, -0.1, 0.2, -0.2. The center sits at x_i = (0.05, 0, 0.5) in (a) and at (-0.25, 0.2, 0) in (b).
Shaded: the barycentric dual cell C_i (in each face, the quadrilateral x_i, edge midpoint, barycenter, edge midpoint).
Arrow: the discrete vector area A_i = (1/3) sum_f A_f N_f = (1/6) sum_k x_{j_k} x x_{j_{k+1}}, drawn at twice its
length. Both stars pass the projection test of Corollary 38.4.12 with n = e_3. Also writes an interactive version.

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-5-2-vertex-vector-area.py
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
phi = np.deg2rad([0, 70, 150, 200, 290])
rad = np.array([1.0, 0.8, 1.1, 0.9, 1.0])
hgt = np.array([0.1, 0.3, -0.1, 0.2, -0.2])
link = np.c_[rad * np.cos(phi), rad * np.sin(phi), hgt]
CENTERS = [np.array([0.05, 0.0, 0.5]), np.array([-0.25, 0.2, 0.0])]
F = np.array([[0, 1 + k, 1 + (k + 1) % 5] for k in range(5)])


def star(c):
    return np.vstack([c, link])


A_link = sum(np.cross(link[k], link[(k + 1) % 5]) for k in range(5)) / 6
results = []
for c in CENTERS:
    V = star(c)
    A = ddg38.vector_area(V, F, "barycentric")[0]
    Ai = dm.mass_barycentric(V, F)[0]
    assert np.allclose(A, A_link, atol=1e-14)                       # Proposition 38.5.5(b)
    assert np.allclose(A, dm.face_cross(V, F).sum(0) / 6)           # (1/3) sum A_f N_f
    assert ddg38.star_passes(V, F, 0, np.array([0, 0, 1.0]))
    results.append((A, Ai))
assert abs(results[0][1] - results[1][1]) > 0.05                   # the dual-cell areas differ


def dual_cell_pieces(V):
    out = []
    for f in F:
        a, b, c = V[f]
        g = (a + b + c) / 3
        out.append([a, (a + b) / 2, g, (a + c) / 2])
    return out


fig = plt.figure(figsize=(7.0, 3.5))
for k, (c, title) in enumerate(zip(CENTERS, ("(a)", "(b)"))):
    ax = dgfig.axes3d(fig, 121 + k, elev=28, azim=-60)
    V = star(c)
    col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=col, edgecolors="#555555", linewidths=0.7, alpha=0.5))
    ax.add_collection3d(Poly3DCollection(dual_cell_pieces(V), facecolors=C["region"], edgecolors=C["tangent"],
                                         linewidths=0.6, alpha=0.55))
    dgfig.arrow3d(ax, c, 2 * results[k][0], role="normal", zorder=12, head=10, lw=1.8)
    dgfig.point3d(ax, c, None, size=12)
    ax.text(*(c + np.array([0.08, -0.05, -0.12])), r"$\mathbf{x}_i$", fontsize=12)
    ax.text(*(c + 2 * results[k][0] + np.array([0.05, 0.0, 0.05])), r"$2\mathbf{A}_i$", color=C["normal"], fontsize=12)
    dgfig.equal_aspect(ax, np.vstack([link, np.array([[0, 0, -0.3], [0, 0, 1.6]])]), zoom=1.3)
    ax.set_title(title, loc="left", fontsize=12, y=0.95)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.0)
dgfig.save(fig, __file__)

tr = []
for c, color, name in ((CENTERS[0], "#D0D0D0", "x_i at (0.05, 0, 0.5)"), (CENTERS[1], "#56B4E9", "x_i at (-0.25, 0.2, 0)")):
    V = star(c)
    tr += [dm.plotly_mesh(V, F, color=color, opacity=0.55, name=name), dm.plotly_edges(V, F)]
    tr += dm.plotly_arrows(c[None], 2 * A_link[None], color=C["normal"], cone=0.1, name="2 A_i")
dm.plotly_save(tr, __file__, "Two positions of x_i with the same link: the vector area A_i (vermilion, x2) is the same",
               "fig-38-5-2")
