"""Figure 33.3.1: oriented simplices and the orientation they induce on their faces (Section 33.3).

(a) the oriented edge [i, j] (induced: -i, +j) and the oriented triangle [i, j, k] (induced: [j,k], [k,i], [i,j]),
    drawn in the plane with i = (0, 0), j = (1, 0), k = (0.45, 0.8) (counterclockwise, det > 0).
(b) the oriented tetrahedron [0, 1, 2, 3] with x_0 = 0, x_1 = e_1, x_2 = e_2, x_3 = e_3 (det = 1 > 0) and the unit
    normals (x_b - x_a) x (x_c - x_a) / |.| of the four induced faces [1,2,3], [0,3,2], [0,1,3], [0,2,1],
    drawn at the face centroids with length 0.35.
Self-check: dgdec.induced_boundary gives these faces with signs (+, -, +, -); every induced normal points away
from the centroid of the tetrahedron.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-3-1-induced-orientation.py``
"""

import dgdec as dd
import dgmesh as dm

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Polygon  # noqa: E402

C = dgfig.COLORS
BLUE, RED = C["tangent"], C["normal"]


def arrow2(ax, p, q, color=BLUE, lw=1.6, ms=13, shrink=0.0, z=5):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    p2, q2 = p + shrink * d, q - shrink * d
    ax.add_patch(FancyArrowPatch(p2, q2, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=z))


fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 3.4), gridspec_kw=dict(width_ratios=[1.25, 1.0]))

# ---------------------------------------------------------------- (a) edge and triangle
ei, ej = np.array([0.0, 0.25]), np.array([1.0, 0.25])
axa.plot(*np.c_[ei, ej], color="black", lw=1.2, zorder=3)
arrow2(axa, ei, ej, shrink=0.18)
for p, lab, sgn, col in ((ei, "i", "-", RED), (ej, "j", "+", C["third"])):
    axa.plot(*p, "o", color="black", ms=5, zorder=6)
    axa.text(p[0], p[1] - 0.17, rf"${lab}$", ha="center", fontsize=13)
    axa.text(p[0], p[1] + 0.15, rf"$\mathbf{{{sgn}}}$", ha="center", fontsize=18, color=col)
axa.text(0.5, -0.25, r"$[i, j]$", ha="center", fontsize=12.5)

T = np.array([[1.45, 0.0], [2.45, 0.0], [1.9, 0.8]])
assert np.linalg.det(np.c_[T[1] - T[0], T[2] - T[0]]) > 0          # counterclockwise
axa.add_patch(Polygon(T, closed=True, facecolor=C["region"], alpha=0.3, edgecolor="none", zorder=1))
for a, b in ((0, 1), (1, 2), (2, 0)):
    axa.plot(*T[[a, b]].T, color="black", lw=1.2, zorder=3)
    arrow2(axa, T[a], T[b], shrink=0.22)
cen = T.mean(0)
th = np.linspace(np.deg2rad(-60), np.deg2rad(230), 60)
arc = cen + 0.14 * np.c_[np.cos(th), np.sin(th)]
axa.plot(*arc.T, color=BLUE, lw=1.2, zorder=4)
arrow2(axa, arc[-2], arc[-1], lw=1.2, ms=10)
for p, lab, off in zip(T, ("i", "j", "k"), ((-0.08, -0.1), (0.08, -0.1), (0.0, 0.12))):
    axa.plot(*p, "o", color="black", ms=5, zorder=6)
    axa.text(p[0] + off[0], p[1] + off[1], rf"${lab}$", ha="center", va="center", fontsize=13)
axa.text(1.95, -0.25, r"$[i, j, k]$", ha="center", fontsize=12.5)
axa.set_title("(a) induced orientations", fontsize=12)
dgfig.schematic_axes(axa, xlim=(-0.2, 2.65), ylim=(-0.45, 1.05))

# ---------------------------------------------------------------- (b) tetrahedron [0,1,2,3] and its induced faces
X = np.vstack([np.zeros(3), np.eye(3)])
assert np.linalg.det(X[1:] - X[0]) > 0
bd = dd.induced_boundary((0, 1, 2, 3))
assert [s for s, _ in bd] == [1, -1, 1, -1]
faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]       # the induced orientations, written positively
for (s, f), g in zip(bd, faces):
    assert dd.induced_orientation((0, 1, 2, 3), g) == 1 and set(f) == set(g)
cen3 = X.mean(0)
cam = dm.Ortho(elev=12, azim=120)
P = cam(X)
N = []
for f in faces:
    a, b, c = X[list(f)]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n)
    assert n @ ((a + b + c) / 3 - cen3) > 0                  # outward
    N.append(n)
front = [n @ cam.view > 0 for n in N]
for f, fr in zip(faces, front):
    if fr:
        axb.add_patch(Polygon(P[list(f)], closed=True, facecolor=C["region"], alpha=0.25, edgecolor="none", zorder=1))
for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
    vis = any(fr for f, fr in zip(faces, front) if a in f and b in f)
    axb.plot(*P[[a, b]].T, color="black" if vis else C["aux"], lw=1.3 if vis else 0.9,
             ls="-" if vis else (0, (3, 2.5)), zorder=3)
for f, n, fr in zip(faces, N, front):
    c0 = X[list(f)].mean(0)
    p0, p1 = cam(c0), cam(c0 + 0.35 * n)
    axb.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, color=RED, lw=1.5 if fr else 1.0,
                                  ls="-" if fr else (0, (2.5, 2)), alpha=1.0 if fr else 0.7, zorder=6))
offs = [(-0.1, -0.06), (0.03, -0.12), (0.1, -0.04), (0.0, 0.1)]
for v in range(4):
    axb.plot(*P[v], "o", color="black", ms=5, zorder=7)
    axb.text(P[v, 0] + offs[v][0], P[v, 1] + offs[v][1], rf"${v}$", fontsize=13, ha="center", va="center")
axb.set_title(r"(b) $[0, 1, 2, 3]$: induced faces", fontsize=12)
allp = np.vstack([P] + [cam(X[list(f)].mean(0) + 0.35 * n)[None] for f, n in zip(faces, N)])
dgfig.schematic_axes(axb, xlim=(allp[:, 0].min() - 0.15, allp[:, 0].max() + 0.15),
                     ylim=(allp[:, 1].min() - 0.15, allp[:, 1].max() + 0.12))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.1)
dgfig.save(fig, __file__)
