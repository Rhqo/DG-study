"""Figure 33.2.2: boundary and interior of a pure subcomplex (Section 33.2).

(a) dgmesh.grid_mesh on the 5 x 5 vertices {0,...,4}^2 (pattern "same"); K' = the closure of the 24 triangles in
    the L-shaped union of squares [0,4]x[0,2] and [0,2]x[2,4]. bd(K') in orange, int(K') in blue.
(b) the octahedron of lib/ch33.py seen from above; the closed star Cl(St(4)) of the top vertex 4 = +e3 (four
    triangles) in blue and its boundary in orange, which equals Lk(4) (Proposition 33.2.17).
Self-check: bd(K') is one closed loop of 16 edges (dgdec.boundary_simplices); int(K') is not a subcomplex;
bd(Cl(St(4))) = Lk(4) is the 4-cycle 0-2-1-3.
An interactive version shows the octahedron and the icosahedron (dgmesh.icosphere(0)) with the closed star of one
vertex and its link.

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/figures/fig-33-2-2-boundary-interior.py``
"""

import pathlib
import sys

import dgdec as dd
import dgmesh as dm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402

import dgfig  # noqa: E402

dgfig.setup()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

C = dgfig.COLORS
BLUE, ORANGE, GRAY = C["tangent"], C["accent"], "#BBBBBB"

# ---------------------------------------------------------------- (a) grid and an L-shaped pure subcomplex
xs = np.arange(5.0)
V, F = dm.grid_mesh(xs, xs, pattern="same")
X = V[:, :2]
sq = lambda f: (int(X[list(f)][:, 0].min()), int(X[list(f)][:, 1].min()))  # noqa: E731  (square of a triangle)
inL = [f for f in F if (sq(f)[1] < 2) or (sq(f)[0] < 2)]
Kp = dd.generate_complex([tuple(int(v) for v in f) for f in inL])
assert len(inL) == 24 and dd.is_pure(Kp, 2)
bd = dd.boundary_simplices(Kp)
it = dd.interior(Kp)
assert [len(dd.k_simplices(bd, k)) for k in range(2)] == [16, 16]
assert dm.link_type([s for s in bd if len(s) == 2]) == "cycle"        # one closed loop
assert not dd.is_complex(it)                                            # int is not a subcomplex
assert [len(dd.k_simplices(Kp, k)) for k in range(3)] == [21, 44, 24]
assert [len(dd.k_simplices(it, k)) for k in range(3)] == [5, 28, 24]

fig = plt.figure(figsize=(6.6, 3.5))
axa = fig.add_subplot(1, 2, 1)
axb = fig.add_subplot(1, 2, 2)
for f in F:
    axa.add_patch(Polygon(X[f], closed=True, facecolor="#F4F4F4", edgecolor=GRAY, lw=0.6, zorder=0))
for f in inL:
    axa.add_patch(Polygon(X[f], closed=True, facecolor=BLUE, alpha=0.28, edgecolor="none", zorder=1))
for s in it:
    if len(s) == 2:
        axa.plot(*X[list(s)].T, color=BLUE, lw=1.6, zorder=2)
for s in bd:
    if len(s) == 2:
        axa.plot(*X[list(s)].T, color=ORANGE, lw=3.0, zorder=3, solid_capstyle="round")
for s in bd:
    if len(s) == 1:
        axa.plot(*X[s[0]], "o", color=ORANGE, ms=6, zorder=4)
for s in it:
    if len(s) == 1:
        axa.plot(*X[s[0]], "o", color=BLUE, ms=6, zorder=4)
axa.plot(*X.T, "o", color=GRAY, ms=2.5, zorder=0.5)
axa.set_title(r"(a) $\mathrm{bd}(\mathcal{K}')$ and $\mathrm{int}(\mathcal{K}')$", fontsize=12)
dgfig.schematic_axes(axa, xlim=(-0.3, 4.3), ylim=(-0.3, 4.3))

# ---------------------------------------------------------------- (b) octahedron: boundary of a closed star = link
Vo, Fo = ch33.octahedron()
Ko = dd.complex_from_faces(Fo)
top = 4
Cst = dd.closure(Ko, dd.star(Ko, top))
Lk = dd.link(Ko, top)
assert dd.boundary_simplices(Cst) == Lk and dm.link_type([s for s in Lk if len(s) == 2]) == "cycle"
assert sorted(s for s in Lk if len(s) == 1) == [(0,), (1,), (2,), (3,)]
cam = dm.Ortho(elev=28, azim=-62)
P = cam(Vo)
front = (dm.face_cross(Vo, Fo) @ cam.view) > 0
starF = np.array([top in f for f in Fo])
order = np.argsort(cam.depth(Vo[Fo].mean(1)))
for fi in order:
    if not front[fi]:
        continue
    col = BLUE if starF[fi] else "#DDDDDD"
    axb.add_patch(Polygon(P[Fo[fi]], closed=True, facecolor=col, alpha=0.32 if starF[fi] else 0.55,
                          edgecolor="none", zorder=1))
for a_, b_ in dm.edges(Fo):
    vis = any(front[fi] for fi in range(len(Fo)) if a_ in Fo[fi] and b_ in Fo[fi])
    inLk = (min(a_, b_), max(a_, b_)) in Lk
    col = ORANGE if inLk else (BLUE if top in (a_, b_) else "#888888")
    lw = 3.0 if inLk else (1.6 if top in (a_, b_) else 1.0)
    axb.plot(*P[[a_, b_]].T, color=col, lw=lw if vis else lw * 0.7, ls="-" if vis else (0, (3, 2.2)), zorder=2,
             alpha=1.0 if vis else 0.8)
for v in range(6):
    col = ORANGE if (v,) in Lk else (BLUE if v == top else "#888888")
    axb.plot(*P[v], "o", color=col, ms=6, zorder=4)
axb.text(*(P[top] + np.array([0.08, 0.06])), r"$4$", fontsize=12, color=BLUE)
for v, off in zip(range(4), [(0.07, -0.12), (-0.12, 0.05), (0.1, 0.03), (-0.12, -0.1)]):
    axb.text(*(P[v] + np.array(off)), rf"${v}$", fontsize=12, color="#8a5a00")
axb.set_title(r"(b) $\mathrm{bd}(\mathrm{Cl}(\mathrm{St}(4))) = \mathrm{Lk}(4)$", fontsize=12)
dgfig.schematic_axes(axb, xlim=(P[:, 0].min() - 0.25, P[:, 0].max() + 0.25),
                     ylim=(P[:, 1].min() - 0.2, P[:, 1].max() + 0.2))
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.1)
dgfig.save(fig, __file__)

# ---------------------------------------------------------------- interactive version
tr = []
for (Vm, Fm, v0, off, name) in [(Vo, Fo, 4, np.array([-1.4, 0, 0]), "octahedron"),
                               (*dm.icosphere(0), 0, np.array([1.4, 0, 0]), "icosahedron")]:
    Km = dd.complex_from_faces(Fm)
    Lm = dd.link(Km, v0)
    sf = np.array([v0 in f for f in Fm])
    tr.append(dm.plotly_mesh(Vm + off, Fm[sf], color="#9CC3E0", name=f"{name}: St({v0})"))
    tr.append(dm.plotly_mesh(Vm + off, Fm[~sf], color="#E6E6E6", name=f"{name}: other triangles", opacity=0.6))
    tr.append(dm.plotly_edges(Vm + off, Fm, color="#666666", width=1.5))
    Le = np.array([s for s in Lm if len(s) == 2])
    xs_, ys_, zs_ = [], [], []
    for a_, b_ in Le:
        for c_ in (a_, b_):
            xs_.append(Vm[c_, 0] + off[0]); ys_.append(Vm[c_, 1] + off[1]); zs_.append(Vm[c_, 2] + off[2])
        xs_.append(None); ys_.append(None); zs_.append(None)
    import plotly.graph_objects as go  # noqa: E402
    tr.append(go.Scatter3d(x=xs_, y=ys_, z=zs_, mode="lines", line=dict(color=ORANGE, width=9),
                           name=f"{name}: Lk({v0})", hoverinfo="name"))
    assert len(Le) == (4 if name == "octahedron" else 5)
dm.plotly_save(tr, __file__, "Closed star (blue) and link (orange): octahedron (4-cycle), icosahedron (5-cycle)",
               "fig-33-2-2-interactive")
