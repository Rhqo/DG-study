"""Animation 45.2.1: the two breadth-first searches of the tree-cotree decomposition, the two edges they leave over,
and the two loops those edges close (Section 45.2, Definition 45.2.9, Theorem 45.2.11, Figure 45.2.1).

Mesh and drawing as Figure 45.2.1: dgmesh.torus_mesh(2, 0.8, 8, 16) (128 vertices, 384 edges, 256 faces) in the
parameter rectangle, vertex (i, j) (index 16 i + j) at (j, i); opposite sides of the dashed rectangle are glued, so
edges on them are drawn twice. ch45.tree_cotree(F) with root vertex 0 and root face 0.
1. The primal BFS: the tree edges appear in the order in which BFS reaches their child vertex (blue, 127 edges).
2. The dual BFS over the dual edges of the edges not in T: cotree dual edges in BFS order (orange, 255, drawn from a
   face centroid through the edge midpoint to the next centroid).
3. The leftover edges (vermilion): 384 - 127 - 255 = 2 = 2g.
4. Each leftover edge e_a closes a primal loop c_a with T (solid) and a dual loop c*_a with T* (dashed); blue a = 1,
   green a = 2; c*_a meets c_b only for a = b, at the leftover edge (black ring).
Self-check (as Figure 45.2.1): |T| = n0 - 1, |T*| = n2 - 1, two leftover edges, d0^T c_a = 0, d1 c*_a = 0,
c*_a . c_b = delta_ab, and the periods of R_1(du), R_1(dv) on c_1, c_2 form an invertible matrix.

Render: bash tools/render_anims.sh chapters/45-vector-field-design/anim/anim-45-2-1-tree-cotree.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _helpers import *  # noqa: E402,F401,F403
import dgdec as dd  # noqa: E402
import ch45  # noqa: E402

# ---------------------------------------------------------------- data (Figure 45.2.1)
nu, nv = 8, 16
V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
tc, Cp, Cs = ch45.generators(F)
E, LEFTF, RIGHTF = tc["E"], tc["left"], tc["right"]
n0, n1, n2 = dm.counts(F)
D0, D1 = dd.d0(F, n0), dd.d1(F)
assert len(tc["T"]) == n0 - 1 == 127 and len(tc["Ts"]) == n2 - 1 == 255 and len(tc["rest"]) == 2 and n1 == 384
assert np.abs(D0.T @ Cp.T).max() == 0 and np.abs(D1 @ Cs.T).max() == 0 and np.array_equal(Cs @ Cp.T, np.eye(2))
uu = 2 * np.pi * (np.arange(n0) // nv) / nu
vv = 2 * np.pi * (np.arange(n0) % nv) / nv
per = np.array([[c @ ch45.wrap(D0 @ uu), c @ ch45.wrap(D0 @ vv)] for c in Cp])
assert abs(np.linalg.det(per)) > 1.0
TREE_ORDER = [e for v, (p, e) in tc["vpar"].items() if p >= 0]           # BFS discovery order of the child vertex
COTREE_ORDER = [e for f, (g, e) in tc["fpar"].items() if g >= 0]
assert set(TREE_ORDER) == tc["T"] and set(COTREE_ORDER) == tc["Ts"]
DEPTH_V = {}
for v, (p, e) in tc["vpar"].items():
    DEPTH_V[v] = 0 if p < 0 else DEPTH_V[p] + 1
DEPTH_F = {}
for f, (g, e) in tc["fpar"].items():
    DEPTH_F[f] = 0 if g < 0 else DEPTH_F[g] + 1

# unwrapped face corners (as the figure): face 2 (i nv + j) + t of quad (i, j)
Cf = np.zeros((n2, 3, 2))
for f in range(n2):
    q, t = divmod(f, 2)
    i, j = divmod(q, nv)
    pos = {(i % nu) * nv + (j % nv): (j, i), (i % nu) * nv + (j + 1) % nv: (j + 1, i),
           ((i + 1) % nu) * nv + j % nv: (j, i + 1), ((i + 1) % nu) * nv + (j + 1) % nv: (j + 1, i + 1)}
    Cf[f] = [pos[int(a)] for a in F[f]]
CEN = Cf.mean(1)


def edge_segments(e):
    out = []
    for f in (LEFTF[e], RIGHTF[e]):
        k = [list(F[f]).index(E[e, 0]), list(F[f]).index(E[e, 1])]
        seg = (tuple(Cf[f, k[0]]), tuple(Cf[f, k[1]]))
        if seg not in out and seg[::-1] not in out:
            out.append(seg)
    return out


def dual_segments(e):
    out = []
    for f in (RIGHTF[e], LEFTF[e]):
        k = [list(F[f]).index(E[e, 0]), list(F[f]).index(E[e, 1])]
        out.append([CEN[f], Cf[f, k].mean(0)])
    return out


# ---------------------------------------------------------------- layout
S = 0.74
O = np.array([-nv * S / 2, -3.35])


def sc(xy):
    xy = np.asarray(xy, float)
    return np.concatenate([O + S * xy, np.zeros(xy.shape[:-1] + (1,))], -1)


def edges_mob(es, color, width, dual=False, dashed=False):
    segs = []
    for e in es:
        segs += dual_segments(e) if dual else edge_segments(e)
    if not segs:
        return VMobject()
    S2 = sc(np.array(segs, float))
    if not dashed:
        return segments(S2, color=color, width=width)
    return VGroup(*[DashedVMobject(Line(a, b, color=color, stroke_width=width), num_dashes=3) for a, b in S2])


class Anim(Scene2D):
    def construct(self):
        base = edges_mob(range(n1), "#D2D2D2", 1.6)
        rect = DashedVMobject(polyline(sc(np.array([[0, 0], [nv, 0], [nv, nu], [0, nu], [0, 0]])), color=HEX["aux"],
                                       width=2.5), num_dashes=80)
        ttl = lab("tree-cotree on a torus mesh (opposite sides glued)", 32).move_to([-6.85, 3.5, 0],
                                                                                  aligned_edge=LEFT)
        root = Dot(sc([0, 0]), radius=0.1, color=C["tangent"]).set_z_index(5)
        self.add(base, rect, ttl, root)
        self.wait(1.2)
        # 1. primal BFS
        s1 = lab("1. BFS on vertices: spanning tree T", 30, color=C["tangent"]).move_to([-6.85, 2.95, 0],
                                                                                      aligned_edge=LEFT)
        kv = ValueTracker(0.0)
        tmob = always_redraw(lambda: edges_mob(TREE_ORDER[:int(kv.get_value())], C["tangent"], 4.0).set_z_index(2))
        cnt1 = readout(lambda: f"{int(kv.get_value())} edges", [3.4, 2.95, 0], size=30, color=C["tangent"])
        self.play(FadeIn(s1), run_time=0.4)
        self.add(tmob, cnt1)
        self.play(kv.animate.set_value(len(TREE_ORDER)), run_time=5.0, rate_func=linear)
        self.wait(1.0)
        # 2. dual BFS
        s2 = lab("2. BFS on faces across the other edges: T*", 30, color=C["accent"]).move_to([-6.85, 2.95, 0],
                                                                                            aligned_edge=LEFT)
        kd = ValueTracker(0.0)
        dmob = always_redraw(lambda: edges_mob(COTREE_ORDER[:int(kd.get_value())], C["accent"], 3.0,
                                               dual=True).set_z_index(3))
        cnt2 = readout(lambda: f"{int(kd.get_value())} dual edges", [3.4, 2.95, 0], size=30, color=C["accent"])
        froot = Dot(sc(CEN[0]), radius=0.09, color=C["accent"]).set_z_index(5)
        self.play(FadeOut(s1), FadeOut(cnt1), FadeIn(s2), FadeIn(froot), run_time=0.5)
        self.add(dmob, cnt2)
        self.play(kd.animate.set_value(len(COTREE_ORDER)), run_time=5.0, rate_func=linear)
        self.wait(0.8)
        # 3. leftover edges
        rest = edges_mob(tc["rest"], C["normal"], 9.0).set_z_index(6)
        s3 = mt(r"3. left over: $384 - 127 - 255 = 2 = 2g$ edges", 30, color=C["normal"])
        s3.move_to([-6.85, 2.95, 0], aligned_edge=LEFT)
        self.play(FadeOut(s2), FadeOut(cnt2), FadeIn(s3), FadeIn(rest), run_time=0.6)
        self.play(rest.animate.set_stroke(width=6.0), run_time=0.6)
        self.wait(1.5)
        # 4. the loops
        tmob.clear_updaters()
        dmob.clear_updaters()
        self.play(tmob.animate.set_stroke(opacity=0.25), dmob.animate.set_stroke(opacity=0.25), run_time=0.6)
        s4 = mt(r"4. each closes a loop $c_a$ (solid) and a dual loop $c^\star_a$ (dashed)", 30)
        s4.move_to([-6.85, 2.95, 0], aligned_edge=LEFT)
        self.play(FadeOut(s3), FadeIn(s4), run_time=0.5)
        for a, col in enumerate((C["tangent"], C["third"])):
            prim = edges_mob(np.where(Cp[a] != 0)[0], col, 6.0).set_z_index(7)
            dual = edges_mob(np.where(Cs[a] != 0)[0], col, 4.0, dual=True, dashed=True).set_z_index(7)
            rings = VGroup(*[Circle(radius=0.22, color=BLACK, stroke_width=3).move_to(sc(np.mean(sgm, 0)))
                             for sgm in edge_segments(tc["rest"][a])]).set_z_index(8)
            self.play(Create(prim), Create(dual), FadeIn(rings), run_time=1.6)
            self.wait(0.6)
        self.wait(2.4)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=22.0)
