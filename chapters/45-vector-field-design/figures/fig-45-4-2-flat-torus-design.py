"""Figure 45.4.2: designed direction fields on the flat torus (Section 45.4, Example 45.4.15).

Mesh: ch45.FlatTorus(24, 24) (side 1/24; 576 vertices, 1152 faces, star1 = 1/sqrt 3, delta = 0, transport rho from the
plane). Generators from ch45.generators (BFS from vertex 0, face 0); for this mesh c*_1 crosses the horizontal vertex
row c_h once and c*_2 crosses the vertical zigzag c_v once, and the horizontal face strip h* is -c*_2 in homology.
(a) k = 0 everywhere, n = (0, -1): holonomy 0 around c*_1 and -2 pi around c*_2, i.e. +2 pi along every horizontal row of
    faces: the field turns once counterclockwise as one goes once around in x and does not turn in y.
(b) k = +1 at vertex p = 6 * 24 + 6 and k = -1 at vertex q = 17 * 24 + 17, n = (0, 0).
Arrows (blue, length 0.035, centered) at the centroids of the first face of the pairs (r, c) with r, c even; black dot: k = +1,
black square: k = -1; dashed: the fundamental rectangle. psi = 0 in face 0.
Self-checks: the fields are parallel for xi on every edge; field indices = k; winding of the global angle along row 3 of
faces is +1 in (a) and 0 in (b); max |xi| < pi.

Run from the project root: PYTHONPATH=tools python3 chapters/45-vector-field-design/figures/fig-45-4-2-flat-torus-design.py
"""

import pathlib
import sys

import dgdec as dd
import dgfig

dgfig.setup()
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch45  # noqa: E402

col = dgfig.COLORS
ft = ch45.FlatTorus(24, 24)
F, n = ft.F, ft.n
D0 = dd.d0(F)
s0, s1, s2 = ch45.stars_from_corners(F, ft.C)
rho = ch45.transport_angles_flat(F, ft.C)
delta = np.zeros(D0.shape[1])
tc, C, Cs = ch45.generators(F)
G = ch45.harmonic_basis(D0, s1, Cs)
E, left, right = ch45.left_right_faces(F)
_, idx = dd.edge_list(F)


def vloop(vs):
    c = np.zeros(len(E))
    for a, b in zip(vs, vs[1:] + vs[:1]):
        c[idx[(min(a, b), max(a, b))]] += 1.0 if a < b else -1.0
    return c


ch, cv = vloop(list(range(n))), vloop([r * n for r in range(ft.rows)])
assert np.allclose(Cs @ ch, [1, 0]) and np.allclose(Cs @ cv, [0, 1])
d = ft.C[:, 1] - ft.C[:, 0]
ref = np.arctan2(d[:, 1], d[:, 0])
p, q = 6 * n + 6, 17 * n + 17
cases = [(np.zeros(len(ft.P), int), (0, -1), r"(a) $k = 0$, one turn around $x$"),
         (None, (0, 0), r"(b) $k = +1$ at $p$, $k = -1$ at $q$")]
k2 = np.zeros(len(ft.P), int)
k2[p], k2[q] = 1, -1
cases[1] = (k2, (0, 0), cases[1][2])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.3))
segs = [[ft.C[f, a], ft.C[f, (a + 1) % 3]] for f in range(len(F)) for a in range(3)]
sel = np.array([2 * (r * n + c) for r in range(0, ft.rows, 2) for c in range(0, n, 2)])
cen = ft.centroids()
for ax, (k, nn, title) in zip(axs, cases):
    xi = ch45.trivial_connection(D0, s1, delta, rho, k, Cs, G, nn)
    psi = ch45.parallel_field(F, rho, xi)
    assert np.abs(ch45.wrap(ch45.connection_from_field(F, rho, psi) - xi)).max() < 1e-9 and np.abs(xi).max() < np.pi
    ind, _ = ch45.field_indices(D0, delta, rho, psi, F)
    assert np.array_equal(ind, k)
    glob = ref + psi
    row = [2 * (3 * n + c) + t for c in range(n) for t in (0, 1)]
    wind = sum(ch45.wrap(glob[b] - glob[a]) for a, b in zip(row, row[1:] + row[:1])) / (2 * np.pi)
    assert abs(wind - round(wind)) < 1e-9 and (nn != (0, -1) or round(wind) == 1)
    print(f"{title}: winding along row 3 = {wind:+.3f}, max |xi| = {np.abs(xi).max():.3f}")
    X = ch45.face_vectors_flat(ft.C, psi)
    ax.add_collection(LineCollection(segs, colors="#E0E0E0", linewidths=0.3, zorder=0, rasterized=True))
    ax.quiver(cen[sel, 0] - 0.0175 * X[sel, 0], cen[sel, 1] - 0.0175 * X[sel, 1], 0.035 * X[sel, 0], 0.035 * X[sel, 1],
              angles="xy", scale_units="xy", scale=1, color=col["tangent"], width=0.005, headwidth=3.6,
              headlength=4.0, headaxislength=3.6, zorder=3)
    ax.plot([0, ft.W, ft.W, 0, 0], [0, 0, ft.H, ft.H, 0], color=col["aux"], lw=0.8, ls="--", zorder=1)
    if k[p]:
        ax.plot(*ft.P[p], "o", color=col["main"], ms=6, zorder=5)
        ax.plot(*ft.P[q], "s", color=col["main"], ms=6, zorder=5)
        ax.text(*(ft.P[p] + [0.025, 0.02]), r"$p$", fontsize=11, zorder=6)
        ax.text(*(ft.P[q] + [0.025, 0.02]), r"$q$", fontsize=11, zorder=6)
    ax.set_xlim(-0.04, ft.W + 0.04)
    ax.set_ylim(-0.04, ft.H + 0.04)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06)
dgfig.save(fig, __file__)
