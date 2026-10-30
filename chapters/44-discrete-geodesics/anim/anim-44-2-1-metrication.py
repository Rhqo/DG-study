"""Animation 44.2.1: refining the grid does not remove the metrication error of edge-graph shortest paths
(Section 44.2, Proposition 44.2.1, (44.2.1), Implementation box).

Mesh: the unit square, dgmesh.grid_mesh(xs, xs, pattern="same") with n = 4, 8, 16, 32, 64 squares per side (every
square cut along the (1, 1) diagonal, h = 1/n), source = the centre vertex (1/2, 1/2) (orange), edge-graph distances
d_G by ch44.dijkstra (the Implementation box's algorithm).
Left: the mesh (light gray). Blue: an edge path of length d_G to the vertex (1, 3/4) (direction (2, 1)); green: an
edge path of length d_G to the vertex (1, 0) (direction (1, -1)). Among the many edge paths of that length the drawn
one is the staircase that stays closest to the segment (interleaved steps). Orange: the straight segments (the true
distance d), the circle of radius 0.35 about the source, and, thin blue, the level set d_G = 0.35 (marching
triangles on the piecewise linear interpolant of d_G).
Right: d_G / d against n (log axis) for the four directions of the Implementation box: (1, 0) and (1, 1) (black, 1),
(2, 1) (blue, (sqrt 2 + 1)/sqrt 5 = 1.0797) and (1, -1) (green, sqrt 2 = 1.4142).
Self-check: Dijkstra's d_G equals (44.2.1) at the four targets for every n; the drawn staircases use only mesh edges
and have length d_G; the ratios are independent of n; the level set is not close to
the circle: for every n it lies on the hexagon {graph norm = 0.35} of (44.2.1) to 1e-12 (d_G is linear on every
face that does not straddle a sector boundary), and its largest distance from the circle is 0.08 to 0.10.

Render: bash tools/render_anims.sh chapters/44-discrete-geodesics/anim/anim-44-2-1-metrication.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _helpers import *  # noqa: E402,F401,F403
import ch44  # noqa: E402

# ---------------------------------------------------------------- data
NS = (4, 8, 16, 32, 64)
RHO = 0.35
SQ2 = np.sqrt(2)


def dG_formula(a, b, h):
    """(44.2.1): graph distance from 0 to (a h, b h)."""
    if a * b >= 0:
        m, M = min(abs(a), abs(b)), max(abs(a), abs(b))
        return h * (SQ2 * m + (M - m))
    return h * (abs(a) + abs(b))


def staircase(start, steps):
    """vertex positions of an edge path: steps is a list of unit grid steps (dx, dy) in {(1,0), (1,1), (0,-1)}."""
    P = [np.array(start, float)]
    for s in steps:
        P.append(P[-1] + np.array(s, float))
    return np.array(P)


def interleave(n_a, step_a, n_b, step_b):
    """n_a steps step_a and n_b steps step_b, interleaved so that the path stays closest to the straight segment."""
    out, ka, kb = [], 0, 0
    total = np.array(step_a, float) * n_a + np.array(step_b, float) * n_b
    for _ in range(n_a + n_b):
        cand = []
        for s, ok in ((step_a, ka < n_a), (step_b, kb < n_b)):
            if ok:
                q = np.sum(out, axis=0) + np.array(s, float) if out else np.array(s, float)
                cand.append((abs(q[0] * total[1] - q[1] * total[0]), s))
        s = min(cand, key=lambda c: c[0])[1]
        out.append(np.array(s, float))
        if s == step_a:
            ka += 1
        else:
            kb += 1
    return [tuple(s) for s in out]


def level_segments(V, F, d, c):
    segs = []
    for tri in F:
        vals = d[tri]
        pts = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (vals[a] - c) * (vals[b] - c) < 0:
                w = (c - vals[a]) / (vals[b] - vals[a])
                pts.append(V[tri[a], :2] + w * (V[tri[b], :2] - V[tri[a], :2]))
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)


def graph_norm(Q):
    """the norm of (44.2.1) at the offsets Q from the source (divided by h it is d_G at grid points)."""
    a, b = Q[:, 0], Q[:, 1]
    m, M = np.minimum(abs(a), abs(b)), np.maximum(abs(a), abs(b))
    return np.where(a * b >= 0, SQ2 * m + (M - m), abs(a) + abs(b))


DATA = {}
for n in NS:
    xs = np.linspace(0, 1, n + 1)
    V, F = dm.grid_mesh(xs, xs, pattern="same")
    h = 1.0 / n
    c = (n // 2) * (n + 1) + n // 2
    d, _ = ch44.dijkstra(V, F, c)
    idx = lambda i, j: i * (n + 1) + j  # noqa: E731
    ratios = {}
    for name, (i, j) in {"(1,0)": (n, n // 2), "(1,1)": (n, n), "(2,1)": (n, 3 * n // 4), "(1,-1)": (n, 0)}.items():
        a, b = i - n // 2, j - n // 2
        assert abs(d[idx(i, j)] - dG_formula(a, b, h)) < 1e-12
        ratios[name] = d[idx(i, j)] / np.linalg.norm(V[idx(i, j)] - V[c])
    st21 = staircase((n // 2, n // 2), interleave(n // 4, (1, 1), n // 4, (1, 0)))
    st1m = staircase((n // 2, n // 2), interleave(n // 2, (1, 0), n // 2, (0, -1)))
    E = {tuple(sorted(e)) for e in dm.edges(F).tolist()}
    for st, tgt in ((st21, (n, 3 * n // 4)), (st1m, (n, 0))):
        assert tuple(st[-1].astype(int)) == tgt
        for p, q in zip(st[:-1].astype(int), st[1:].astype(int)):
            assert tuple(sorted((idx(*p), idx(*q)))) in E
        L = np.sum(np.linalg.norm(np.diff(st, axis=0), axis=1)) * h
        assert abs(L - d[idx(*tgt)]) < 1e-12
    segs = level_segments(V, F, d, RHO)
    Q = segs.reshape(-1, 2) - 0.5
    assert np.abs(graph_norm(Q) - RHO).max() < 1e-12              # the level set IS the hexagon, for every n
    assert np.abs(np.linalg.norm(Q, axis=1) - RHO).max() > 0.08    # and stays away from the circle
    DATA[n] = dict(V=V, F=F, h=h, ratios=ratios, st21=st21 * h, st1m=st1m * h, segs=segs, E=dm.edges(F))
for key, val in (("(1,0)", 1.0), ("(1,1)", 1.0), ("(2,1)", (SQ2 + 1) / np.sqrt(5)), ("(1,-1)", SQ2)):
    assert all(abs(DATA[n]["ratios"][key] - val) < 1e-12 for n in NS)
assert round((SQ2 + 1) / np.sqrt(5), 4) == 1.0797

# ---------------------------------------------------------------- layout
SQ0, SQS = np.array([-6.6, -2.2]), 5.0                  # lower-left corner and side of the unit square


def sc(xy):
    xy = np.asarray(xy, float)
    return np.concatenate([SQ0 + SQS * xy, np.zeros(xy.shape[:-1] + (1,))], -1)


PL = Plot(1.15, -1.3, 5.0, 3.6, (3.4, 76), (0.9, 1.5), xlog=True)


def mesh_mob(n):
    D = DATA[n]
    V, E = D["V"], D["E"]
    w = {4: 2.2, 8: 1.8, 16: 1.4, 32: 1.0, 64: 0.7}[n]
    g = VGroup(segments(sc(V[E][:, :, :2]), color="#C9C9C9", width=w))
    g.add(segments(sc(D["segs"]), color=C["tangent"], width=2.5))
    g.add(polyline(sc(D["st21"]), color=C["tangent"], width=6))
    g.add(polyline(sc(D["st1m"]), color=C["third"], width=6))
    return g


class Anim(Scene2D):
    def construct(self):
        ttl = lab("edge-graph shortest paths, refined", 34).move_to([-6.85, 3.5, 0], aligned_edge=LEFT)
        frame = polyline(sc(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])), color=BLACK, width=2.5)
        tt = np.linspace(0, 2 * np.pi, 200)
        circ = polyline(sc(0.5 + RHO * np.stack([np.cos(tt), np.sin(tt)], 1)), color=C["accent"], width=3.5)
        segs = VGroup(Line(sc([0.5, 0.5]), sc([1, 0.75]), color=C["accent"], stroke_width=4),
                      Line(sc([0.5, 0.5]), sc([1, 0.0]), color=C["accent"], stroke_width=4))
        src = Dot(sc([0.5, 0.5]), radius=0.09, color=C["accent"]).set_z_index(5)
        over = VGroup(circ, segs, src).set_z_index(3)
        axes = PL.axes(xticks=[(n, str(n)) for n in NS], yticks=[(1.0, "1.0"), (1.2, "1.2"), (1.4, "1.4")],
                       xlabel="n (squares per side)")
        ylab = mt(r"$d_G\,/\,d$", 32).next_to(PL.p(3.4, 1.5), UP, buff=0.15).shift(RIGHT * 0.2)
        ex = DashedVMobject(Line(PL.p(3.4, 1.0), PL.p(76, 1.0), stroke_width=1.5, color=HEX["aux"]), num_dashes=40)
        nl = [lab(f"n = {n}, h = 1/{n}", 32).move_to([6.85, 3.5, 0], aligned_edge=RIGHT) for n in NS]
        kl = VGroup(lab("(1, −1): 1.4142", 30, color=C["third"]).move_to(PL.p(4.6, 1.468), aligned_edge=LEFT),
                    lab("(2, 1): 1.0797", 30, color=C["tangent"]).move_to(PL.p(4.6, 1.135), aligned_edge=LEFT),
                    lab("(1, 0), (1, 1): 1", 30).move_to(PL.p(4.6, 0.95), aligned_edge=LEFT))
        cur = mesh_mob(NS[0])
        self.add(cur, frame, over, ttl, axes, ylab, ex, nl[0])
        pts = VGroup()
        for k, n in enumerate(NS):
            new = VGroup(*[Dot(PL.p(n, DATA[n]["ratios"][key]), radius=0.08, color=col).set_z_index(4)
                           for key, col in (("(1,0)", BLACK), ("(1,1)", BLACK), ("(2,1)", C["tangent"]),
                                            ("(1,-1)", C["third"]))])
            if k == 0:
                self.play(FadeIn(new), run_time=0.5)
                self.wait(2.0)
            else:
                nxt = mesh_mob(n)
                self.play(FadeOut(cur), FadeIn(nxt), FadeOut(nl[k - 1]), FadeIn(nl[k]), FadeIn(new), run_time=0.7)
                cur = nxt
                self.wait(1.6)
            pts.add(new)
            if k == 1:
                self.play(FadeIn(kl), run_time=0.5)
        fin = lab("the staircase approaches the segment, its length does not", 30)
        fin.move_to([-6.85, -2.6, 0], aligned_edge=LEFT)
        fin2 = mt(r"thin blue: $d_G = 0.35$ is a hexagon, not the circle", 30, color=C["tangent"])
        fin2.move_to([-6.85, -2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.5)
        self.wait(2.0)
        self.play(FadeOut(fin), FadeIn(fin2), run_time=0.5)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=13.0)
