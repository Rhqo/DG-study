"""Animation 40.4.1: the signed offset of a nonconvex polygon winds twice near the concave vertex (Section 40.4,
Non-example 40.4.6, Remark 40.4.16, Figure 40.4.1(b)).

P = the L-shaped polygon (0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2) (counterclockwise, A = 3, L = 8). For eps from 0
to 0.7 the signed offset curve c_eps is built as in Remark 40.4.16: every edge translated by eps along its outward
normal, consecutive edges joined by the arc of radius eps about the vertex with the signed exterior angle tau_i
(+pi/2 at the five convex vertices, -pi/2 at the concave vertex (1, 1), where the arc turns clockwise).
Drawing: grey P; sky blue = points of winding number 1 outside P, i.e. P + eps B minus P; dark blue = winding number 2
(the square [1, 1 + eps]^2 minus the quarter disk about (1, 1)); black = c_eps, orange = its arcs. At the end a dot
runs once along c_eps.
Readouts: the signed area of c_eps (shoelace of the sampled curve) = A + L eps + pi eps^2, the true area of P + eps B =
A + L eps + (5 pi / 4 - 1) eps^2 (eps <= 1), and their difference (1 - pi/4) eps^2 = the dark region counted twice.
Self-check: A = 3, L = 8, sum tau = 2 pi; for eps = 0.2, 0.3, 0.7 the shoelace area of c_eps (fine sampling) equals
A + L eps + pi eps^2 to 1e-6; at eps = 0.3 the grid count of P + eps B (lib/ddg40.py) is 5.663 and the two values are
5.663 and 5.683 (Non-example 40.4.6); winding numbers 2, 1, 1, 1, 0 at test points in the dark region, the quarter
disk, P, the band and outside.

Render: bash tools/render_anims.sh chapters/40-discrete-curvature-integral/anim/anim-40-4-1-signed-offset.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg40  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, MoveAlongPath, TracedPath, ValueTracker, VGroup, always_redraw, linear, smooth  # noqa: E402

# ---------------------------------------------------------------- data
PL = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
M = len(PL)
EDGE = np.roll(PL, -1, 0) - PL
LEN = np.linalg.norm(EDGE, axis=1)
TANG = EDGE / LEN[:, None]
NRM = np.c_[TANG[:, 1], -TANG[:, 0]]                    # outward = right of a counterclockwise boundary
TAU = np.array([np.arctan2(TANG[k - 1, 0] * TANG[k, 1] - TANG[k - 1, 1] * TANG[k, 0], TANG[k - 1] @ TANG[k])
                for k in range(M)])                      # exterior angle at vertex k
A0 = 0.5 * np.sum(PL[:, 0] * np.roll(PL[:, 1], -1) - np.roll(PL[:, 0], -1) * PL[:, 1])
L0 = LEN.sum()
assert abs(A0 - 3) < 1e-12 and abs(L0 - 8) < 1e-12 and abs(TAU.sum() - 2 * np.pi) < 1e-12
assert np.allclose(TAU, [np.pi / 2] * 3 + [-np.pi / 2] + [np.pi / 2] * 2)
CONCAVE = 3


def offset_curve(eps, n_arc=40):
    """c_eps as a list of pieces: ("arc", points about vertex k) then ("edge", points of edge k) for k = 0..M-1."""
    pieces = []
    for k in range(M):
        phi0 = np.arctan2(NRM[k - 1, 1], NRM[k - 1, 0])
        s = np.linspace(0, TAU[k], n_arc + 1)
        pieces.append(("arc", k, PL[k] + eps * np.c_[np.cos(phi0 + s), np.sin(phi0 + s)]))
        pieces.append(("edge", k, np.array([PL[k] + eps * NRM[k], PL[(k + 1) % M] + eps * NRM[k]])))
    return pieces


def curve_points(eps, n_arc=40):
    pts = np.vstack([p for _, _, p in offset_curve(eps, n_arc)])
    return pts


def shoelace(P):
    return 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])


def winding(P, q):
    d = P - q
    ang = np.arctan2(d[:, 1], d[:, 0])
    return int(round(np.sum(np.angle(np.exp(1j * (np.roll(ang, -1) - ang)))) / (2 * np.pi)))


def signed_poly(eps):
    return A0 + L0 * eps + np.pi * eps ** 2


def true_area(eps):
    return A0 + L0 * eps + (5 * np.pi / 4 - 1) * eps ** 2


for eps in (0.2, 0.3, 0.7):
    assert abs(shoelace(curve_points(eps, 4000)) - signed_poly(eps)) < 1e-6
assert abs(ddg40.polygon_minkowski_area_grid(PL, 0.3, 0.004) - 5.663) < 2e-3
assert abs(true_area(0.3) - 5.663) < 5e-4 and abs(signed_poly(0.3) - 5.683) < 5e-4
_c = curve_points(0.3, 400)
assert [winding(_c, np.array(q)) for q in ((1.27, 1.27), (1.09, 1.09), (0.5, 0.5), (2.15, 0.5), (3.0, 3.0))] \
    == [2, 1, 1, 1, 0]
EPS_MAX = 0.7


def outer_boundary(eps, n_arc=40):
    """The boundary {d(., P) = eps} of P + eps B: c_eps without its small loop (cut at X = (1 + eps, 1 + eps))."""
    X = np.array([1 + eps, 1 + eps])
    out = []
    for kind, k, p in offset_curve(eps, n_arc):
        if kind == "arc" and k == CONCAVE:
            continue
        if kind == "edge" and k == CONCAVE - 1:          # (2, 1 + eps) -> (1, 1 + eps): stop at X
            p = np.array([p[0], X])
        if kind == "edge" and k == CONCAVE:              # (1 + eps, 1) -> (1 + eps, 2): start at X
            p = np.array([X, p[1]])
        out.append(p)
    return np.vstack(out)


def loop_region(eps, n_arc=40):
    """The winding-2 region: X -> (1, 1 + eps) -> clockwise arc about (1, 1) -> (1 + eps, 1) -> X."""
    X = np.array([1 + eps, 1 + eps])
    s = np.linspace(np.pi / 2, 0, n_arc + 1)
    return np.vstack([X, PL[CONCAVE] + eps * np.c_[np.cos(s), np.sin(s)]])


assert abs(shoelace(loop_region(0.7, 4000)) - (1 - np.pi / 4) * 0.49) < 1e-6
assert abs(shoelace(outer_boundary(0.3, 4000)) - true_area(0.3)) < 1e-6


class Anim(hp.Scene2D):
    def construct(self):
        e = ValueTracker(0.0)
        S, O = 1.9, np.array([-5.3, -2.3])                 # data (x, y) -> scene O + S (x, y)

        def sc(P):
            P = np.atleast_2d(P)
            return np.c_[O[0] + S * P[:, 0], O[1] + S * P[:, 1], np.zeros(len(P))]

        poly = hp.filled(sc(PL), hp.C["surface"], opacity=1.0, stroke_width=3, stroke_color=hp.C["main"])
        plab = hp.math(r"$P$", 40).move_to(sc([0.5, 0.75])[0])

        def drawing():
            eps = e.get_value()
            g = VGroup()
            if eps < 1e-3:
                return g
            g.add(hp.filled(sc(outer_boundary(eps)), hp.C["region"], opacity=0.35))
            g.add(hp.filled(sc(PL), hp.C["surface"], opacity=1.0))
            g.add(hp.filled(sc(loop_region(eps)), hp.C["tangent"], opacity=0.75))
            for kind, k, p in offset_curve(eps):
                if kind == "edge":
                    g.add(hp.polyline(sc(p), color=hp.C["main"], width=4))
                else:
                    g.add(hp.polyline(sc(p), color=hp.C["accent"], width=5))
            g.add(hp.polyline(np.vstack([sc(PL), sc(PL[:1])]), color=hp.C["main"], width=2.5))
            return g
        dr = always_redraw(drawing)
        celab = always_redraw(lambda: hp.math(r"$c_\varepsilon$", 36)
                              .move_to(sc([2 + e.get_value() + 0.28, 0.5])[0]).set_opacity(min(1, 8 * e.get_value())))
        # tau label inside the lower arm of P, clear of its edges
        taul = always_redraw(lambda: hp.math(r"$\tau = -\pi/2$", 30)
                             .move_to(sc([1.04, 0.95])[0], aligned_edge=np.array([-1.0, 1.0, 0]))
                             .set_opacity(min(1, 8 * e.get_value())))

        # ---- readouts (right)
        x0 = 0.75
        el = hp.math(r"$\varepsilon =$", 34).move_to([x0 + 0.45, 3.3, 0])
        en = always_redraw(lambda: hp.text(hp.num(e.get_value(), 2), 34).next_to(el, np.array([1.0, 0, 0]), buff=0.18))
        LEFT_ = np.array([-1.0, 0, 0])
        a1 = hp.math(r"signed area of $c_\varepsilon$ =", 32).move_to([x0, 2.45, 0], aligned_edge=LEFT_)
        n1 = always_redraw(lambda: hp.text(hp.num(signed_poly(e.get_value())), 32).next_to(a1, np.array([1.0, 0, 0]),
                                                                                             buff=0.18))
        f1 = hp.math(r"$= 3 + 8\varepsilon + \pi\varepsilon^2$", 32, color=hp.C["aux"]).move_to(
            [x0 + 0.5, 1.85, 0], aligned_edge=LEFT_)
        a2 = hp.math(r"area of $P + \varepsilon B$ =", 32).move_to([x0, 1.0, 0], aligned_edge=LEFT_)
        n2 = always_redraw(lambda: hp.text(hp.num(true_area(e.get_value())), 32).next_to(a2, np.array([1.0, 0, 0]),
                                                                                           buff=0.18))
        f2 = hp.math(r"$= 3 + 8\varepsilon + (5\pi/4 - 1)\,\varepsilon^2$", 32, color=hp.C["aux"]).move_to(
            [x0 + 0.5, 0.4, 0], aligned_edge=LEFT_)
        a3 = hp.text("difference =", 32, color=hp.C["tangent"]).move_to([x0, -0.45, 0], aligned_edge=LEFT_)
        n3 = always_redraw(lambda: hp.text(hp.num((1 - np.pi / 4) * e.get_value() ** 2), 32, color=hp.C["tangent"])
                           .next_to(a3, np.array([1.0, 0, 0]), buff=0.18))
        f3 = hp.math(r"$= (1 - \pi/4)\,\varepsilon^2$", 32, color=hp.C["tangent"]).move_to(
            [x0 + 0.5, -1.05, 0], aligned_edge=LEFT_)
        title = hp.text("L-shaped polygon: A = 3, L = 8", 32).move_to([-3.2, 3.3, 0])

        items = [poly, dr, plab, celab, taul, el, en, a1, f1, n1, a2, f2, n2, a3, n3, f3, title]
        self.add(*items)
        for m in (dr, celab, taul, en, n1, n2, n3):
            m.update()
        self.wait(2.0)
        self.play(e.animate.set_value(0.3), run_time=6.0, rate_func=smooth)
        self.wait(2.0)
        self.play(e.animate.set_value(EPS_MAX), run_time=4.0, rate_func=smooth)
        for m in (dr, celab, taul):
            m.clear_updaters()
        note = hp.text("dark blue: winding number 2", 30, color=hp.C["tangent"]).move_to(
            [x0, -1.9, 0], aligned_edge=np.array([-1.0, 0, 0]))
        self.play(FadeIn(note), run_time=0.5)
        path = hp.polyline(sc(curve_points(EPS_MAX, 60)), width=0)
        runner = hp.dot(path.get_start(), color=hp.C["normal"], r=0.1)
        trail = TracedPath(runner.get_center, stroke_color=hp.C["normal"], stroke_width=7, dissipating_time=1.2)
        self.add(trail, runner)
        self.play(MoveAlongPath(runner, path), run_time=5.0, rate_func=linear)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.5)
