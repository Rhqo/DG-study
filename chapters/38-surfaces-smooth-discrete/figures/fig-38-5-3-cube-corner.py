"""Figure 38.5.3: four vertex normals at the corner (1, 1, 1) of the triangulated cube [-1, 1]^3 (ddg38.cube).

(a) pattern A: the diagonal of the square x = 1 passes through the corner (two triangles there), the squares y = 1 and
z = 1 use the other diagonal (one triangle each). Uniform and area-weighted normals are (2, 1, 1)/sqrt 6, 19.47 degrees
from (1, 1, 1)/sqrt 3; the angle-weighted and the vector-area normal with mixed Voronoi cells are (1, 1, 1)/sqrt 3.
(b) pattern B: no diagonal passes through the corner; all four normals are (1, 1, 1)/sqrt 3.
Arrows of length 0.9 from the corner: uniform (blue), area-weighted (green, dashed), angle-weighted (orange),
vector-area with mixed cells (purple, dashed).

Run: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/figures/fig-38-5-3-cube-corner.py
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
KINDS = [("uniform", C["tangent"], "-"), ("area", C["third"], "--"), ("angle", C["accent"], "-"),
         ("vector-mixed", C["covector"], "--")]
corner = 7
diag = np.ones(3) / np.sqrt(3)
PATTERNS = [{"x+": True}, {}]

normals = []
for pat in PATTERNS:
    V, F = ddg38.cube(pat)
    assert np.allclose(V[corner], [1, 1, 1]) and dm.euler_characteristic(F) == 2 and dm.oriented_consistently(F)
    normals.append({k: ddg38.vertex_normal(V, F, k)[corner] for k, _, _ in KINDS})
nA, nB = normals
w = np.array([2.0, 1.0, 1.0]) / np.sqrt(6)
assert np.allclose(nA["uniform"], w) and np.allclose(nA["area"], w)
assert np.allclose(nA["angle"], diag) and np.allclose(nA["vector-mixed"], diag)
assert abs(np.degrees(np.arccos(w @ diag)) - 19.4712) < 1e-4
assert all(np.allclose(v, diag) for v in nB.values())

fig = plt.figure(figsize=(7.0, 3.6))
for k, (pat, title) in enumerate(zip(PATTERNS, ("(a)", "(b)"))):
    ax = dgfig.axes3d(fig, 121 + k, elev=20, azim=-60)
    V, F = ddg38.cube(pat)
    col = dm.shade(V, F, dm.hex_to_rgb(C["surface"]))
    near = [f for f in range(len(F)) if corner in F[f]]
    fc = [C["region"] if f in near else col[f] for f in range(len(F))]
    ax.add_collection3d(Poly3DCollection(V[F], facecolors=fc, edgecolors="#555555", linewidths=0.7, alpha=0.45))
    for name, color, ls in KINDS:
        n = normals[k][name]
        tip = V[corner] + 0.9 * n
        ax.plot(*np.c_[V[corner], tip], color=color, lw=1.8, ls=ls, zorder=10)
        ax.scatter(*tip, color=color, s=14, zorder=11, depthshade=False)
    dgfig.point3d(ax, V[corner], None, size=12)
    dgfig.equal_aspect(ax, np.vstack([V, V[corner] + np.array([1.0, 0.5, 0.5])]), zoom=1.15)
    ax.set_title(title, loc="left", fontsize=12, y=0.95)
handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.8) for _, c, ls in KINDS]
fig.legend(handles, ["uniform", "area-weighted", "angle-weighted", "vector area (mixed)"], loc="lower center", ncol=4,
           frameon=False, fontsize=10)
fig.subplots_adjust(left=0, right=1, bottom=0.08, top=1, wspace=0.0)
dgfig.save(fig, __file__)
