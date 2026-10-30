"""Animation 33.6.1: building the dual of the octahedron one dimension at a time, and seeing that it is a cube
(Section 33.6, Definition of the dual complex, Proposition n_k* = n_{2-k}, Figure 33.6.2(a)).

The regular octahedron of lib/ch33.py (vertices +-e_1, +-e_2, +-e_3, faces outward), orthographic view, the camera
turning slowly about the z-axis (azimuth -68 -> -28 degrees, elevation 22). Dual positions are a drawing choice
(the dual of Section 33.6 is combinatorial): a dual vertex at each face centroid, a dual edge drawn on the surface
from centroid to the midpoint of the primal edge to the other centroid (so it visibly crosses its primal edge), a dual
face is the cell of the surface around a primal vertex bounded by its dual edges.
Steps: 8 faces -> 8 dual vertices; 12 edges -> 12 dual edges; 6 vertices -> 6 dual faces (squares: deg = 4);
then the dual is pushed out radially, centroids (+-1, +-1, +-1)/3 -> (+-1, +-1, +-1), the primal fades and the dual is
the cube. Back elements are drawn faint (opacity from the direction of the element toward the viewer).
Self-check: n_k* = n_{2-k}; every dual face has deg(i) = 4 sides and every dual vertex is on 3 dual edges; the
centroids are (+-1, +-1, +-1)/3; at the end the dual vertices are the cube corners, every dual face is a planar
square of side 2 (the bend points are the cube edge midpoints); chi = 2 for both.

Render: ``bash tools/render_anims.sh chapters/33-combinatorial-surfaces/anim/anim-33-6-1-octahedron-dual.py``
"""

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ch33  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, FadeOut, Polygon, ValueTracker, VGroup, always_redraw, linear, smooth)  # noqa: E402

V, F = ch33.octahedron()
HE = dd.HalfedgeMesh(F)
E, _ = dd.edge_list(F)
EF = dm.edge_faces(F)
DE = np.array([EF[tuple(e)] for e in E.tolist()])               # dual edge k joins the two faces of edge k
DF = [HE.vertex_faces(i) for i in range(len(V))]                 # dual face i: faces around vertex i (ccw)
CEN0 = V[F].mean(1)
MID = V[E].mean(1)
n0, n1, n2 = len(V), len(E), len(F)
assert (n0, n1, n2) == (6, 12, 8) and (len(CEN0), len(DE), len(DF)) == (n2, n1, n0)          # n_k* = n_{2-k}
assert all(len(c) == 4 for c in DF) and (np.bincount(DE.ravel(), minlength=n2) == 3).all()
assert sorted(map(tuple, np.round(3 * CEN0).astype(int).tolist())) == sorted(itertools.product((-1, 1), repeat=3))
assert n0 - n1 + n2 == 2 == n2 - n1 + n0
# the dual edge around primal vertex i: consecutive faces of DF[i] share an edge containing i
DFE = []
for i, cyc in enumerate(DF):
    es = []
    for a, b in zip(cyc, cyc[1:] + cyc[:1]):
        k = [k for k in range(n1) if set(DE[k]) == {a, b}]
        assert len(k) == 1 and i in E[k[0]]
        es.append(k[0])
    DFE.append(es)


def geometry(s):
    """dual vertices, bend points of the dual edges, for the push parameter s in [0, 1] (0: on the octahedron)."""
    c = CEN0 * (1 + 2 * s)                     # (+-1, +-1, +-1)/3 -> (+-1, +-1, +-1)
    b = MID * (1 + s)                          # primal edge midpoints -> cube edge midpoints
    return c, b


_c, _b = geometry(1.0)
assert np.allclose(np.abs(_c), 1)
for k, (a, b) in enumerate(DE):                # the bend point is the midpoint of the straight cube edge
    assert np.allclose(_b[k], (_c[a] + _c[b]) / 2) and abs(np.linalg.norm(_c[a] - _c[b]) - 2) < 1e-12
for i in range(n0):                            # dual faces are planar squares at the end
    P = np.array([_c[f] for f in DF[i]])
    nrm = np.cross(P[1] - P[0], P[2] - P[0])
    assert np.allclose((P - P[0]) @ nrm, 0) and np.allclose(np.linalg.norm(np.roll(P, -1, 0) - P, axis=1), 2)

ORIGIN2 = (-2.75, 0.0)
SC0, SC1 = 2.25, 1.38                          # zoom out while the dual is pushed out to the cube
AZ0, AZ1, ELEV = -68.0, -28.0, 22.0
ORANGE = hp.C["accent"]
for _s, _sc in ((0.0, SC0), (1.0, SC1)):         # layout: the polyhedra stay inside the left part of the frame
    for _az in (AZ0, AZ1):
        _cam = hp.Cam(ELEV, _az, _sc, ORIGIN2)
        _P = _cam(np.vstack([V, geometry(_s)[0]]) if _s == 0 else geometry(_s)[0])
        assert _P[:, 1].min() > -2.75 and _P[:, 1].max() < 3.0 and _P[:, 0].max() < 0.6 and _P[:, 0].min() > -6.8


def fade(direction, view):
    """opacity of an element on the convex surface from its outward direction (smooth front/back cue)."""
    d = np.asarray(direction, float)
    t = np.clip((d @ view / np.linalg.norm(d) + 0.12) / 0.24, 0, 1)
    return 0.22 + 0.78 * t * t * (3 - 2 * t)


class Anim(hp.Scene2D):
    def construct(self):
        az = ValueTracker(AZ0)
        push = ValueTracker(0.0)
        prim = ValueTracker(1.0)                  # primal visibility
        show = {"dv": ValueTracker(0.0), "de": ValueTracker(0.0), "df": ValueTracker(0.0)}

        def scene():
            cam = hp.Cam(ELEV, az.get_value(), SC0 + (SC1 - SC0) * push.get_value(), ORIGIN2)
            view = cam.view
            g = VGroup()
            po = prim.get_value()
            c, b = geometry(push.get_value())
            # primal faces (front ones, light fill) and edges
            if po > 0.01:
                for f in F:
                    nrm = np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]])
                    if nrm @ view > 0:
                        g.add(Polygon(*cam(V[f]), fill_color=hp.C["surface"], fill_opacity=0.45 * po, stroke_width=0))
                for k, (i, j) in enumerate(E):          # seen if one of its two faces is
                    op = max(fade(CEN0[f], view) for f in DE[k]) * po
                    g.add(hp.seg(cam(V[i]), cam(V[j]), color="#555555", width=2.5).set_stroke(opacity=op))
                for i in range(n0):
                    op = max(fade(CEN0[f], view) for f in DF[i]) * po
                    g.add(hp.dot(cam(V[i]), color=hp.BLACK, r=0.06).set_opacity(op))
            # dual faces
            o = show["df"].get_value()
            if o > 0.01:
                for i in range(n0):
                    if V[i] @ view > 0:
                        ring = []
                        for f, k in zip(DF[i], DFE[i]):
                            ring += [c[f], b[k]]
                        g.add(Polygon(*cam(np.array(ring)), fill_color=ORANGE, fill_opacity=0.28 * o, stroke_width=0))
            # visibility: on the octahedron a point is seen if its primal face faces the viewer; on the final cube an
            # element is seen if one of its cube faces (= dual faces, one per primal vertex) does. Blend with s.
            sp = push.get_value()
            o = show["de"].get_value()
            if o > 0.01:
                for k, (a, a2) in enumerate(DE):
                    cube = max(fade(V[E[k][0]], view), fade(V[E[k][1]], view))
                    for f, (p0, p1) in ((a, (c[a], b[k])), (a2, (b[k], c[a2]))):     # half on face a, half on a2
                        op = ((1 - sp) * fade(CEN0[f], view) + sp * cube) * o
                        g.add(hp.polyline(cam(np.array([p0, p1])), color=ORANGE, width=4.5).set_stroke(opacity=op))
            o = show["dv"].get_value()
            if o > 0.01:
                for f in range(n2):
                    cube = max(fade(V[i], view) for i in F[f])
                    op = ((1 - sp) * fade(CEN0[f], view) + sp * cube) * o
                    g.add(hp.dot(cam(c[f]), color=ORANGE, r=0.085).set_opacity(op))
            return g
        sc = always_redraw(scene)
        self.add(sc)
        title = hp.text("octahedron and its dual", size=32).move_to([ORIGIN2[0], 3.45, 0])
        self.add(title)
        XL = 0.95
        rows = [("8 faces", "8 dual vertices"), ("12 edges", "12 dual edges"), ("6 vertices", "6 dual faces")]
        row_m = []
        for k, (a, b2) in enumerate(rows):
            y = 2.4 - 0.75 * k
            row_m.append(VGroup(hp.text(a, size=30).move_to([XL, y, 0], aligned_edge=np.array([-1, 0, 0])),
                                hp.text("→", size=30).move_to([XL + 2.35, y, 0]),
                                hp.text(b2, size=30, color=ORANGE).move_to([XL + 2.75, y, 0],
                                                                           aligned_edge=np.array([-1, 0, 0]))))
        T_TOTAL = 21.7
        rate = (AZ1 - AZ0) / T_TOTAL
        az.add_updater(lambda m, dt: m.increment_value(rate * dt))     # the camera turns the whole time
        self.add(az)
        self.wait(1.5)
        self.play(show["dv"].animate.set_value(1.0), FadeIn(row_m[0]), run_time=1.2)
        self.wait(2.0)
        self.play(show["de"].animate.set_value(1.0), FadeIn(row_m[1]), run_time=1.5)
        self.wait(2.0)
        self.play(show["df"].animate.set_value(1.0), FadeIn(row_m[2]), run_time=1.5)
        self.wait(2.0)
        title2 = hp.text("the dual is a cube", size=32).move_to(title)
        self.play(FadeOut(title), run_time=0.5)
        self.play(push.animate.set_value(1.0), prim.animate.set_value(0.0), run_time=4.0, rate_func=smooth)
        self.play(FadeIn(title2), run_time=0.6)
        self.wait(0.6)
        chi = VGroup(hp.math(r"$\chi = 6 - 12 + 8 = 2$", size=32),
                     hp.math(r"$\chi^\star = 8 - 12 + 6 = 2$", size=32, color=ORANGE)) \
            .arrange(np.array([0, -1.0, 0]), buff=0.35, aligned_edge=np.array([-1, 0, 0])) \
            .move_to([XL, -0.75, 0], aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(chi), run_time=1.0)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=20.0)
