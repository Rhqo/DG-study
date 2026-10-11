"""Animation 31.5.1: marching triangles, the 2D version of marching tetrahedra (Proposition 31.5.1).

Field: f(p) = |p| - r(theta(p)) with r(theta) = 1 + 0.15 cos 3 theta + 0.1 sin 2 theta (the closed curve of Figure
31.5.3; f < 0 inside). Samples on the grid {-1.5 + 0.3 k}^2 (11 x 11 points, 10 x 10 cells). Every square cell is
split by the same diagonal (from its lower-left to its upper-right corner, the 2D Kuhn split), f is interpolated
linearly on each triangle, so the zero set inside a triangle is empty or one segment whose ends lie on the two edges
with a sign change, at p = p_a + f_a / (f_a - f_b) (p_b - p_a) ((30.2.9)).
Left: the grid, samples (filled = inside, hollow = outside), the cell being processed (orange box) and the extracted
segments (orange). Right: a zoom of one cell with the corner values, the crossing points and the two segments, which
meet on the shared diagonal.
Self-check: no sample is 0; each triangle gives 0 or 1 segment; every crossing point (one per sign-changing edge) is
the end of exactly two segments, and the segments form one closed polygon; its vertices are within 0.031 of the true
curve; the zoomed cell has a sign change on its diagonal (two segments meeting there) and all its crossing
points at 0.2 < t < 0.8 along their edges.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-5-1-marching.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data
H = 0.3
XS = -1.5 + H * np.arange(11)
NX = len(XS)


def r_of(th):
    return 1 + 0.15 * np.cos(3 * th) + 0.1 * np.sin(2 * th)


def field(x, y):
    return np.hypot(x, y) - r_of(np.arctan2(y, x))


GX, GY = np.meshgrid(XS, XS, indexing="ij")
FV = field(GX, GY)
assert np.abs(FV).min() > 1e-3


def gid(i, j):
    return i * NX + j


PTS = np.stack([GX.ravel(), GY.ravel()], 1)
FF = FV.ravel()


def cell_tris(i, j):
    a, b, c, d = gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1)
    return [(a, b, c), (a, c, d)]           # diagonal a-c (lower-left to upper-right)


def crossing(a, b):
    t = FF[a] / (FF[a] - FF[b])
    return PTS[a] + t * (PTS[b] - PTS[a])


CELLS = [(i, j) for j in range(NX - 1) for i in range(NX - 1)]      # raster order: rows from the bottom
SEGS = {}                                     # cell -> list of (edge key 1, edge key 2)
for (i, j) in CELLS:
    out = []
    for tri in cell_tris(i, j):
        es = [(tri[k], tri[(k + 1) % 3]) for k in range(3)]
        ch = [tuple(sorted(e)) for e in es if (FF[e[0]] > 0) != (FF[e[1]] > 0)]
        assert len(ch) in (0, 2)
        if ch:
            out.append((ch[0], ch[1]))
    SEGS[(i, j)] = out
deg = {}
for segs in SEGS.values():
    for e1, e2 in segs:
        deg[e1] = deg.get(e1, 0) + 1
        deg[e2] = deg.get(e2, 0) + 1
assert all(v == 2 for v in deg.values())
# one closed polygon
adj = {}
for segs in SEGS.values():
    for e1, e2 in segs:
        adj.setdefault(e1, []).append(e2)
        adj.setdefault(e2, []).append(e1)
start = next(iter(adj))
loop, prev, cur = [start], None, start
while True:
    nxt = [e for e in adj[cur] if e != prev][0] if prev is not None else adj[cur][0]
    if nxt == start:
        break
    loop.append(nxt)
    prev, cur = cur, nxt
assert len(loop) == len(adj)
POLY = np.array([crossing(*e) for e in loop])
th_ = np.linspace(0, 2 * np.pi, 2000)
TRUE = np.stack([r_of(th_) * np.cos(th_), r_of(th_) * np.sin(th_)], 1)
dist = np.min(np.linalg.norm(POLY[:, None] - TRUE[None], axis=-1), axis=1)
assert round(dist.max(), 3) == 0.031          # crossing points vs the true curve (about h / 10, h = 0.3)
# the zoomed cell: its diagonal changes sign, so its two triangles each give a segment, meeting on the diagonal
ZOOM = None
for (i, j) in CELLS:
    a, c = gid(i, j), gid(i + 1, j + 1)
    ts = [FF[e[0]] / (FF[e[0]] - FF[e[1]]) for seg in SEGS[(i, j)] for e in seg]
    if len(SEGS[(i, j)]) == 2 and (FF[a] > 0) != (FF[c] > 0) and min(ts) > 0.2 and max(ts) < 0.8:
        ZOOM = (i, j)
        break
assert ZOOM is not None


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        S0, SC = np.array([-3.45, -0.5]), 2.0         # main grid: world -> scene

        def w2s(p):
            p = np.asarray(p, float)
            return np.array([S0[0] + SC * p[0], S0[1] + SC * p[1], 0.0])

        ttl = title("Marching triangles (2D marching tetrahedra)", 34)
        grid = VGroup()
        for x in XS:
            grid.add(Line(w2s((x, XS[0])), w2s((x, XS[-1])), stroke_width=1.2, color=GREY_B))
            grid.add(Line(w2s((XS[0], x)), w2s((XS[-1], x)), stroke_width=1.2, color=GREY_B))
        for (i, j) in CELLS:
            grid.add(Line(w2s(PTS[gid(i, j)]), w2s(PTS[gid(i + 1, j + 1)]), stroke_width=0.8, color=GREY_A))
        dots = VGroup()
        for k, p in enumerate(PTS):
            if FF[k] < 0:
                dots.add(Dot(w2s(p), radius=0.065, color=BLACK))
            else:
                dots.add(Circle(radius=0.06, stroke_width=1.6, color=BLACK, fill_color=WHITE, fill_opacity=1).move_to(w2s(p)))
        leg = VGroup(VGroup(Dot(radius=0.065, color=BLACK), mt(r"$f < 0$ (inside)", 30)).arrange(RIGHT, buff=0.15),
                     VGroup(Circle(radius=0.06, stroke_width=1.6, color=BLACK), mt(r"$f > 0$", 30)).arrange(RIGHT, buff=0.15))
        leg.arrange(RIGHT, buff=0.5).move_to([-3.45, 2.95, 0])
        self.add(ttl, grid, dots, leg)
        self.wait(1.6)

        # ---------------------------------------------------------- zoom on one cell
        i, j = ZOOM
        ZC, ZW = np.array([3.85, 0.3]), 2.8         # zoom centre and cell width (scene units)
        org = PTS[gid(i, j)]

        def z2s(p):
            q = (np.asarray(p, float) - org) / H - 0.5
            return np.array([ZC[0] + ZW * q[0], ZC[1] + ZW * q[1], 0.0])

        box = Square(SC * H, stroke_width=4, color=C["accent"]).move_to(w2s(org + H / 2)).set_z_index(4)
        corners = [gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1)]
        zcell = VGroup(Polygon(*[z2s(PTS[c]) for c in corners], stroke_width=2.5, color=BLACK),
                       Line(z2s(PTS[corners[0]]), z2s(PTS[corners[2]]), stroke_width=2, color=GREY_B))
        zlines = VGroup(DashedLine(w2s(org + (H, H)), z2s(PTS[corners[2]]) + np.array([-0.05, -0.05, 0]),
                                   stroke_width=1.5, color=C["aux"]),
                        DashedLine(w2s(org + (H, 0)), z2s(PTS[corners[1]]) + np.array([-0.05, 0.05, 0]),
                                   stroke_width=1.5, color=C["aux"]))
        zdots, zvals = VGroup(), VGroup()
        for c, off in zip(corners, (DL, DR, UR, UL)):
            if FF[c] < 0:
                zdots.add(Dot(z2s(PTS[c]), radius=0.11, color=BLACK).set_z_index(3))
            else:
                zdots.add(Circle(radius=0.1, stroke_width=2.5, color=BLACK, fill_color=WHITE, fill_opacity=1)
                          .move_to(z2s(PTS[c])).set_z_index(3))
            zvals.add(da.label(f"{FF[c]:+.2f}", 30).next_to(z2s(PTS[c]), off, buff=0.12))
        step = da.label("1. corner values and signs", 30).move_to([0.8, 2.75, 0], aligned_edge=LEFT)
        self.play(Create(box), FadeIn(zcell), Create(zlines), FadeIn(step), run_time=1.0)
        self.play(FadeIn(zdots), FadeIn(zvals), run_time=0.8)
        self.wait(1.5)
        # edges with a sign change -> crossing points
        ch_edges = sorted({e for seg in SEGS[ZOOM] for e in seg})
        hl = VGroup(*[Line(z2s(PTS[a]), z2s(PTS[b]), stroke_width=7, color=C["region"]).set_z_index(1)
                      for a, b in ch_edges])
        xpts = VGroup(*[Dot(z2s(crossing(a, b)), radius=0.1, color=C["accent"], stroke_color=BLACK, stroke_width=1.5)
                        .set_z_index(4) for a, b in ch_edges])
        step2 = da.label("2. a point per sign change", 30).move_to([0.8, 2.75, 0], aligned_edge=LEFT)
        formula = mt(r"$p = p_a + \frac{f_a}{f_a - f_b}\,(p_b - p_a)$", 34).move_to([3.85, -2.05, 0])
        self.play(FadeTransform(step, step2), Create(hl), run_time=0.8)
        self.play(FadeIn(xpts, scale=0.5), FadeIn(formula), run_time=0.9)
        self.wait(1.8)
        step3 = da.label("3. join them per triangle", 30).move_to([0.8, 2.75, 0], aligned_edge=LEFT)
        zsegs = VGroup(*[Line(z2s(crossing(*e1)), z2s(crossing(*e2)), stroke_width=7, color=C["accent"]).set_z_index(3)
                         for e1, e2 in SEGS[ZOOM]])
        diag_note = da.label("pieces meet on the diagonal", 30).move_to([3.85, -2.72, 0])
        self.play(FadeTransform(step2, step3), Create(zsegs), run_time=1.0)
        self.play(FadeIn(diag_note), run_time=0.5)
        self.wait(1.8)

        # ---------------------------------------------------------- march over all cells
        zoom_all = VGroup(zcell, zlines, zdots, zvals, hl, xpts, zsegs, diag_note, formula)
        step4 = da.label("4. march over all cells", 30).move_to([0.8, 2.75, 0], aligned_edge=LEFT)
        self.play(FadeOut(zoom_all), FadeTransform(step3, step4), run_time=0.7)
        cnt = {"n": 0}
        found = VGroup()
        self.add(found)
        for n_, (ci, cj) in enumerate(CELLS):
            segs = SEGS[(ci, cj)]
            for e1, e2 in segs:
                found.add(Line(w2s(crossing(*e1)), w2s(crossing(*e2)), stroke_width=6, color=C["accent"]).set_z_index(3))
            cnt["n"] += len(segs)
            if n_ % 2 == 1:                     # two cells per step (keeps the file under 4 MB)
                box.move_to(w2s(PTS[gid(ci, cj)] + H / 2))
                self.wait(2 / 15)
        self.play(FadeOut(box), run_time=0.4)
        true = DashedVMobject(polyline([w2s(p) for p in np.vstack([TRUE, TRUE[:1]])], color=GREY_D, width=2.5),
                              num_dashes=90).set_z_index(5)
        res = VGroup(da.label(f"{cnt['n']} segments, 1 closed curve", 30),
                     da.label("dashed: the true curve", 30),
                     da.label("each edge point uses only", 30),
                     da.label("that edge's two samples,", 30),
                     da.label("so neighbours agree", 30))
        res.arrange(DOWN, aligned_edge=LEFT, buff=0.16).move_to([0.8, 2.1, 0], aligned_edge=UL)
        res[2:].shift(DOWN * 0.25)
        self.play(Create(true), FadeIn(res[:2]), run_time=1.2)
        self.wait(1.0)
        self.play(FadeIn(res[2:]), run_time=0.6)
        tet = VGroup(da.label("3D: 6 tetrahedra per cube,", 30), da.label("pieces: triangles or quads", 30))
        tet.arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([0.8, -1.3, 0], aligned_edge=UL)
        self.play(FadeIn(tet), run_time=0.5)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.0)
