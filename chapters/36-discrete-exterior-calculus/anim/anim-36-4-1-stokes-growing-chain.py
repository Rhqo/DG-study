"""Animation 36.4.1: discrete Stokes's theorem, one face at a time (Section 36.4, Theorem 36.4.5, Corollary 36.4.6(b)).

Mesh: the unit square as lib/ch36.grid(3) (16 vertices, 33 edges, 18 counterclockwise triangles, alternating
diagonals; the planar grid 4 x 4 = (16, 33, 18) of the running examples). Discrete 1-form: integers alpha_e in
{-3, ..., 3} (seeded), a discrete form that is not given by a formula (Example 36.1.7(a)).
The 2-chain b grows from one face to all 18 faces in a fixed order; every new face shares an edge with b.
Shown: faces of b (light blue) with (d1 alpha)_f, the oriented boundary chain partial b (blue arrows, ccw) with the
value of alpha along each arrow, and the new face's boundary (orange). Edges shared by b and the new face are traversed
in opposite directions and cancel. Right: sum of d1 alpha over b and the circulation <alpha, partial b>.
Self-check: for every intermediate b, sum_f in b (d1 alpha)_f = <alpha, partial b> (= alpha . d1^T b) exactly
(integers), the drawn arrow values add up to the same number, and partial b is a cycle.

Render: bash tools/render_anims.sh chapters/36-discrete-exterior-calculus/anim/anim-36-4-1-stokes-growing-chain.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ch36  # noqa: E402
import dganim as da  # noqa: E402
import dgdec as dd  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, FadeOut, VGroup  # noqa: E402

# ---------------------------------------------------------------- data
V, F = ch36.grid(3)
E, EIDX = dd.edge_list(F)
D1 = dd.d1(F).astype(int)                       # (18, 33)
rng = np.random.default_rng(8)
ALPHA = rng.integers(1, 4, len(E)) * rng.choice([-1, 1], len(E))
DA = D1 @ ALPHA                                   # (d1 alpha)_f

# growth order: centre cell, then a plus shape, then the corners (each face shares an edge with b)
ORDER = [8, 9, 10, 3, 2, 14, 7, 6, 11, 15, 1, 0, 4, 5, 13, 12, 16, 17]
assert sorted(ORDER) == list(range(len(F)))


def face_edges(f):
    """[(edge index, sign along the face orientation, tail, head)] for the three edges of face f."""
    out = []
    t = F[f]
    for c in range(3):
        a, b = int(t[c]), int(t[(c + 1) % 3])
        e = EIDX[(min(a, b), max(a, b))]
        out.append((e, 1 if a < b else -1, a, b))
    return out


STEPS = []
chain = np.zeros(len(F), int)
for k, f in enumerate(ORDER):
    old_bd = D1.T @ chain
    chain[f] = 1
    bd = D1.T @ chain                              # boundary coefficients on edges
    assert set(np.unique(bd)) <= {-1, 0, 1}
    assert not (dd.d0(F).T @ bd).any()             # partial b is a cycle
    lhs = int(DA @ chain)
    rhs = int(ALPHA @ bd)
    assert lhs == rhs                              # Theorem 36.4.5 (exact, integers)
    shared = [e for e, s, a, b in face_edges(f) if old_bd[e] != 0]
    if k > 0:
        assert shared and all(old_bd[e] == -s for e, s, a, b in face_edges(f) if e in shared)
    STEPS.append(dict(f=f, bd=bd.copy(), lhs=lhs, rhs=rhs, shared=shared))
assert len(STEPS[-1]["shared"]) >= 2 and max(len(s["shared"]) for s in STEPS) == 2
assert abs(STEPS[-1]["bd"]).sum() == 12           # the square's boundary: 12 edges
FINAL = STEPS[-1]["lhs"]

# ---------------------------------------------------------------- layout
X0, Y0, S = -6.4, -2.25, 5.1                       # the unit square -> [X0, X0 + S] x [Y0, Y0 + S]
PX = -0.5                                         # left edge of the text column


def P(v):
    return np.array([X0 + S * v[0], Y0 + S * v[1], 0.0])


PV = np.array([P(v) for v in V])
CEN = np.array([PV[t].mean(0) for t in F])
J = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # +90 degrees


def edge_arrow(a, b, color, side=0.06):
    """Arrow along a -> b (scene points), shifted by `side` to its left; value label on its right."""
    pa, pb = PV[a], PV[b]
    d = (pb - pa) / np.linalg.norm(pb - pa)
    off = side * (J @ d)
    return hp.arrow(pa + 0.16 * (pb - pa) + off, pa + 0.84 * (pb - pa) + off, color=color, width=5, tip=0.2)


def edge_value(a, b, value, color):
    pa, pb = PV[a], PV[b]
    d = (pb - pa) / np.linalg.norm(pb - pa)
    m = 0.5 * (pa + pb) - 0.27 * (J @ d)
    return hp.text(hp.num(value, 0), 30, color=color).move_to(m)


def oriented(e, sign):
    i, j = E[e]
    return (i, j) if sign > 0 else (j, i)


def face_label_pos(f):
    """Centroid moved a quarter of the way toward the right-angle corner (keeps clear of the edge labels)."""
    t = PV[F[f]]
    for c in range(3):
        u, w = t[(c + 1) % 3] - t[c], t[(c + 2) % 3] - t[c]
        if abs(u @ w) < 1e-9:
            return CEN[f] + 0.25 * (t[c] - CEN[f]) / np.linalg.norm(t[c] - CEN[f])
    raise AssertionError("grid triangles are right triangles")


NOTES = {0: "one face: the sum around it = d\u2081\u03b1",
         1: "a shared edge: +a and \u2212a",
         2: "they cancel; \u2202b is what is left"}


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("discrete Stokes's theorem, one face at a time", 34).move_to([0, 3.52, 0])
        grid = VGroup(*[hp.seg(PV[i], PV[j], color=hp.HEX["aux"], width=2) for i, j in E],
                      *[hp.dot(p, color=hp.HEX["aux"], r=0.045) for p in PV])
        eq = hp.mt(r"$\langle d_1\alpha,\ b\rangle = \langle \alpha,\ \partial b\rangle$", 38).move_to([PX, 2.65, 0], aligned_edge=hp.LEFT)
        row_l = hp.mt(r"$\langle d_1\alpha,\ b\rangle\ =$", 36).move_to([PX, 1.2, 0], aligned_edge=hp.LEFT)
        sub_l = hp.text("sum of the numbers in the faces", 30, color=hp.C["aux"])
        sub_l.move_to([PX, 0.7, 0], aligned_edge=hp.LEFT)
        row_r = hp.mt(r"$\langle \alpha,\ \partial b\rangle\ =$", 36).move_to([PX, 0.05, 0], aligned_edge=hp.LEFT)
        sub_r = hp.text("sum of the blue arrow numbers", 30, color=hp.C["aux"])
        sub_r.move_to([PX, -0.45, 0], aligned_edge=hp.LEFT)
        key = VGroup(
            hp.text("light blue: faces of b (d\u2081\u03b1)", 30),
            hp.text("blue: \u2202b (\u03b1 along the arrow)", 30, color=hp.C["tangent"]),
            hp.text("orange: the new face's boundary", 30, color=hp.C["accent"]),
        ).arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.1).move_to([PX, -2.05, 0], aligned_edge=hp.LEFT)
        self.add(title, grid, eq, row_l, sub_l, row_r, sub_r, key)
        faces_g, bd_g, dyn_g, cnt_g, note_g = VGroup(), VGroup(), VGroup(), VGroup(), VGroup()
        world = VGroup(faces_g, bd_g, dyn_g, cnt_g, note_g)
        self.add(world)

        def counters(k, lhs, rhs):
            return [hp.text(hp.num(lhs, 0), 36, color=hp.C["tangent"]).next_to(row_l, hp.RIGHT, buff=0.25),
                    hp.text(hp.num(rhs, 0), 36, color=hp.C["tangent"]).next_to(row_r, hp.RIGHT, buff=0.25),
                    hp.text(f"faces in b: {k}", 30).move_to([PX, 1.85, 0], aligned_edge=hp.LEFT)]

        setter = hp.setter

        cnt_g.submobjects = counters(0, 0, 0)
        self.mark("intro")
        self.wait(2.0)

        bd = {}                  # edge -> (arrow, label) of the current boundary
        for k, st in enumerate(STEPS):
            f = st["f"]
            slow = k < 3
            # the new face's boundary, counterclockwise, with alpha along each arrow
            new = {}
            for e, s, a, b in face_edges(f):
                new[e] = (edge_arrow(a, b, hp.C["accent"]), edge_value(a, b, s * ALPHA[e], hp.C["accent"]))
            outline = hp.polyline(PV[list(F[f]) + [F[f][0]]], color=hp.C["accent"], width=3)
            orange = [outline] + [m for v in new.values() for m in v]
            tri = hp.poly_fill(PV[F[f]], hp.C["region"], opacity=0.35)
            fval = hp.text(hp.num(DA[f], 0), 30).move_to(face_label_pos(f))
            # the boundary after the step: shared edges cancel, the other edges of the face join
            bd_new = {e: v for e, v in bd.items() if e not in st["shared"]}
            for e in new:
                if e not in st["shared"]:
                    sg = st["bd"][e]
                    a, b = oriented(e, sg)
                    bd_new[e] = (edge_arrow(a, b, hp.C["tangent"]), edge_value(a, b, sg * ALPHA[e], hp.C["tangent"]))
            gone = [m for e in st["shared"] for m in (*bd[e], *new[e])]
            joined = [m for e in new if e not in st["shared"] for m in bd_new[e]]
            mid = counters(k + 1, st["lhs"], STEPS[k - 1]["rhs"] if k > 0 else 0)
            end = counters(k + 1, st["lhs"], st["rhs"])
            faces_after = faces_g.submobjects + [tri, fval]
            bd_mid = [m for v in bd.values() for m in v]
            bd_after = [m for v in bd_new.values() for m in v]
            if slow:
                note = hp.text(NOTES[k], 30, color=hp.C["normal"]).move_to([PX, -0.98, 0], aligned_edge=hp.LEFT)
                T = [0.0, 0.5, 1.1, 1.7, 2.6, 3.4]       # note, orange in, fill, hold, cancel, hold
                f_note, f_orange, f_face = hp.fade([note]), hp.fade(orange), hp.fade([tri, fval])
                f_join = hp.fade(joined)
                f_gone = hp.fade(gone + [outline] + [new[e][i] for e in new if e not in st["shared"] for i in (0, 1)],
                                 start_hidden=False)
                ev = [(T[0], T[0], setter(note_g, [note])), (T[0], T[1], f_note),
                      (T[1], T[1], setter(dyn_g, orange)), (T[1], T[2], f_orange),
                      (T[2], T[2], setter(faces_g, faces_after)), (T[2], T[2], setter(cnt_g, mid)),
                      (T[2], T[3], f_face),
                      (T[4], T[4], setter(bd_g, bd_mid + joined)),
                      (T[4], T[5], lambda u, g=f_gone: g(1 - u)), (T[4], T[5], f_join),
                      (T[5], T[5], setter(dyn_g, [])), (T[5], T[5], setter(bd_g, bd_after)),
                      (T[5], T[5], setter(cnt_g, end))]
                self.play(hp.timeline(world, ev, T[5] + 1.0))
            else:
                ev = [(0.0, 0.0, setter(dyn_g, orange)),
                      (0.3, 0.3, setter(faces_g, faces_after)), (0.3, 0.3, setter(cnt_g, mid)),
                      (0.6, 0.6, setter(dyn_g, [])), (0.6, 0.6, setter(bd_g, bd_after)),
                      (0.6, 0.6, setter(cnt_g, end))]
                self.play(hp.timeline(world, ev, 1.2))
            if k == 2:
                hp.hide(note_g.submobjects)
                note_g.submobjects = []
            bd = bd_new
            # the drawn arrow values add up to the circulation
            assert sum(int(lab.text.replace("\u2212", "-")) for arr, lab in bd.values()) == st["rhs"]
            self.mark(f"face {k + 1} done")
        end_txt = hp.text("inner edges cancel in pairs", 30, color=hp.C["normal"])
        end_txt.move_to([PX, -0.98, 0], aligned_edge=hp.LEFT)
        self.play(FadeIn(end_txt), run_time=0.6)
        self.mark("end text")
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=23.4)
