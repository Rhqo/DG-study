"""Animation 33.5.1: faces, edges and vertices of a halfedge mesh are orbits -- of next, of twin, of next o twin
(Section 33.5, Definition 33.5.10, Proposition 33.5.13, the boundary subsection, Figure 33.5.2).

The closed fan of Figure 33.5.1: vertex 0 at the origin, vertices 1-5 on the unit circle at 90 + 72 (k - 1) degrees,
faces (0, t+1, t+2) (counterclockwise), stored as dgdec.HalfedgeMesh: 15 interior halfedges (blue, drawn inside
their triangle) and 5 boundary halfedges (grey dashed, outside the rim, face "none").
Part 1: from h = 0 -> 1, next (rho) runs 0->1, 1->2, 2->0 and returns after 3 steps: the face (0, 1, 2).
Part 2: twin (eta) runs 0->1, 1->0 and returns after 2 steps: the edge {0, 1}.
Part 3: next o twin (rho o eta): each step first jumps to the twin (vermillion), then to the next halfedge of the twin's
face (orange). The orange halfedges 0->1, 0->5, 0->4, 0->3, 0->2 all leave vertex 0, the faces passed are shaded, and
after 5 = deg 0 steps the walk is back at h: the vertex 0 (clockwise, Conventions box of Section 33.5).
Part 4: at the boundary vertex 1 the same walk uses a boundary halfedge: 1->2, (2->1) 1->5, (5->1) 1->0, back after
3 = deg 1 steps.
Self-check: orbits computed with dgdec.HalfedgeMesh agree with the lists shown; next^3 = id on interior halfedges,
twin is a fixed-point-free involution, the rho o eta orbits are the vertices with lengths = valences, chi = 1.

Render: ``bash tools/render_anims.sh chapters/33-combinatorial-surfaces/anim/anim-33-5-1-halfedge-walk.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402
import dgdec as dd  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, FadeOut, Polygon, VGroup)  # noqa: E402

ANG = np.deg2rad(90 + 72 * np.arange(5))
X = np.vstack([[0, 0], np.c_[np.cos(ANG), np.sin(ANG)]])
F = np.array([(0, 1 + t, 1 + (t + 1) % 5) for t in range(5)])
HE = dd.HalfedgeMesh(F)
NI = 3 * len(F)
assert np.all(HE.next[HE.next[HE.next[:NI]]] == np.arange(NI))
assert np.all(HE.twin[HE.twin] == np.arange(HE.nh)) and np.all(HE.twin != np.arange(HE.nh))
assert HE.nh == 20 and HE.euler_characteristic() == 1 == dm.euler_characteristic(F)


def find(a, b):
    return [h for h in range(HE.nh) if HE.vertex[h] == a and HE.head[h] == b][0]


def orbit(perm, h0):
    out, h = [h0], int(perm[h0])
    while h != h0:
        out.append(h)
        h = int(perm[h])
    return out


ROT = HE.next[HE.twin]                                   # rho o eta (Crane's vertex orbit)
H0, H1 = find(0, 1), find(1, 2)
FACE = orbit(HE.next, H0)
EDGE = orbit(HE.twin, H0)
V0 = orbit(ROT, H0)
V1 = orbit(ROT, H1)
assert [(HE.vertex[h], HE.head[h]) for h in FACE] == [(0, 1), (1, 2), (2, 0)]
assert [(HE.vertex[h], HE.head[h]) for h in EDGE] == [(0, 1), (1, 0)]
assert [int(HE.head[h]) for h in V0] == [1, 5, 4, 3, 2] and all(HE.vertex[h] == 0 for h in V0)
assert [int(HE.head[h]) for h in V1] == [2, 5, 0] and HE.face[V1[1]] < 0           # passes a boundary halfedge
nb = dm.neighbors(F)
assert len(V0) == len(nb[0]) == 5 and len(V1) == len(nb[1]) == 3
for v in range(6):                                                                   # every vertex is one orbit
    assert sorted(orbit(ROT, int(HE.vertex_halfedge[v]))) == sorted(np.nonzero(HE.vertex == v)[0].tolist())

CEN = np.array([-3.35, -0.1, 0.0])
SC = 2.45


def P(x):
    return CEN + SC * hp.to3(np.asarray(x, float))


def endpoints(h):
    a, b = X[HE.vertex[h]], X[HE.head[h]]
    if HE.face[h] >= 0:
        c = X[F[HE.face[h]]].mean(0)
        p, q = a + 0.2 * (c - a), b + 0.2 * (c - b)
    else:
        nrm = (a + b) / 2 / np.linalg.norm((a + b) / 2)
        p, q = a + 0.12 * nrm, b + 0.12 * nrm
    return P(p + 0.12 * (q - p)), P(q - 0.12 * (q - p))


def he_arrow(h, color=None, width=3.0):
    p, q = endpoints(h)
    if color is None:
        if HE.face[h] < 0:
            return hp.seg(p, q, color=hp.C["aux"], width=2.5, dashed=True)
        color = hp.C["tangent"]
    a = hp.arrow(p, q, color=color, width=width, tip=0.2 if width < 5 else 0.27)
    return a.set_z_index(10) if width >= 5 else a            # the highlighted halfedge stays on top of the shading


def name(h):
    s = f"{HE.vertex[h]}→{HE.head[h]}"
    return s + "  (boundary)" if HE.face[h] < 0 else s


class Anim(hp.Scene2D):
    def construct(self):
        tris = VGroup(*[Polygon(*P(X[f]), fill_color=hp.C["surface"], fill_opacity=0.45, stroke_color="#9A9A9A",
                                stroke_width=1.5) for f in F])
        labs = VGroup()
        for v in range(6):
            q = X[v] * 1.17 if v else np.array([-0.11, -0.12])
            labs.add(hp.text(str(v), size=30).move_to(P(q)))
        base = VGroup(*[he_arrow(h) for h in range(HE.nh)])
        self.add(tris, base, labs, *[hp.dot(P(x), r=0.06) for x in X])
        XL = 0.9                                                   # left edge of the text column

        def heading(expr):
            return hp.math(expr, size=34).move_to([XL, 3.2, 0], aligned_edge=np.array([-1, 0, 0]))

        def item(k, s, color=hp.C["accent"]):
            return hp.text(s, size=30, color=color).move_to([XL + 0.3, 2.35 - 0.56 * k, 0],
                                                             aligned_edge=np.array([-1, 0, 0]))
        self.wait(1.0)

        # ---- part 1: a face is an orbit of next
        head = heading(r"face = orbit of next $\rho$")
        cur = he_arrow(H0, hp.C["accent"], 7)
        shade = Polygon(*P(X[F[0]]), fill_color=hp.C["region"], fill_opacity=0.35, stroke_width=0)
        items = VGroup(item(0, "h = " + name(H0)))
        self.play(FadeIn(head), FadeIn(cur), FadeIn(items[0]), FadeIn(shade), run_time=0.8)
        self.wait(0.7)
        for h in FACE[1:] + [H0]:
            new = he_arrow(h, hp.C["accent"], 7)
            it = item(len(items), name(h) if h != H0 else name(h) + "  = h, back", hp.C["accent"])
            self.play(FadeOut(cur), FadeIn(new), FadeIn(it), run_time=0.8)
            cur = new
            items.add(it)
        res = item(len(items) + 0.4, "3 steps: one face", hp.C["third"])
        self.play(FadeIn(res), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(head), FadeOut(items), FadeOut(res), FadeOut(shade), run_time=0.6)

        # ---- part 2: an edge is an orbit of twin
        head = heading(r"edge = orbit of twin $\eta$")
        items = VGroup(item(0, "h = " + name(H0)))
        self.play(FadeIn(head), FadeIn(items[0]), run_time=0.6)
        for h in EDGE[1:] + [H0]:
            new = he_arrow(h, hp.C["normal"] if h != H0 else hp.C["accent"], 7)
            it = item(len(items), name(h) if h != H0 else name(h) + "  = h, back",
                      hp.C["normal"] if h != H0 else hp.C["accent"])
            self.play(FadeOut(cur), FadeIn(new), FadeIn(it), run_time=0.8)
            cur = new
            items.add(it)
        res = item(len(items) + 0.4, "2 steps: one edge", hp.C["third"])
        self.play(FadeIn(res), run_time=0.6)
        self.wait(1.0)
        self.play(FadeOut(head), FadeOut(items), FadeOut(res), run_time=0.6)

        # ---- part 3: a vertex is an orbit of next o twin
        head = heading(r"vertex = orbit of $\rho\circ\eta$")
        items = VGroup(item(0, "h = " + name(H0)))
        self.play(FadeIn(head), FadeIn(items[0]), run_time=0.6)
        visited = VGroup()
        self.add(visited)
        for h in V0[1:] + [H0]:
            tw = he_arrow(int(HE.twin[ROT_INV[h]]), hp.C["normal"], 7)       # eta of the previous halfedge
            self.play(FadeOut(cur), FadeIn(tw), run_time=0.7)
            cur = tw
            f = int(HE.face[HE.twin[ROT_INV[h]]])
            visited.add(Polygon(*P(X[F[f]]), fill_color=hp.C["accent"], fill_opacity=0.18, stroke_width=0))
            new = he_arrow(h, hp.C["accent"], 7)
            it = item(len(items), name(h) if h != H0 else name(h) + "  = h, back")
            self.play(FadeOut(cur), FadeIn(new), FadeIn(visited[-1]), FadeIn(it), run_time=0.8)
            cur = new
            items.add(it)
        res = item(len(items) + 0.4, "5 steps = deg 0: vertex 0", hp.C["third"])
        self.play(FadeIn(res), run_time=0.6)
        self.wait(1.5)
        self.play(FadeOut(head), FadeOut(items), FadeOut(res), FadeOut(visited), run_time=0.6)

        # ---- part 4: the boundary vertex 1
        head = heading(r"vertex 1: orbit of $\rho\circ\eta$")
        new = he_arrow(H1, hp.C["accent"], 7)
        items = VGroup(item(0, "h = " + name(H1)))
        self.play(FadeIn(head), FadeOut(cur), FadeIn(new), FadeIn(items[0]), run_time=0.8)
        cur = new
        for h in V1[1:] + [H1]:
            tw_h = int(HE.twin[ROT_INV[h]])
            tw = he_arrow(tw_h, hp.C["normal"], 7)
            self.play(FadeOut(cur), FadeIn(tw), run_time=0.7)
            cur = tw
            new = he_arrow(h, hp.C["accent"], 7)
            it = item(len(items), name(h) if h != H1 else name(h) + "  = h, back")
            self.play(FadeOut(cur), FadeIn(new), FadeIn(it), run_time=0.8)
            cur = new
            items.add(it)
        res = item(len(items) + 0.4, "3 steps = deg 1: vertex 1", hp.C["third"])
        self.play(FadeIn(res), run_time=0.6)
        self.wait(3.0)


ROT_INV = np.empty(HE.nh, int)
ROT_INV[ROT] = np.arange(HE.nh)                     # h = rot(ROT_INV[h]) = next(twin(ROT_INV[h]))

if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.5)
