"""Animation 34.1.1: a 2-vector forgets the shape of the parallelogram -- only det of the change of basis matters
(Section 34.1, Proposition 34.1.6(a), Proposition 34.1.8, Proposition 34.1.9, Example 34.1.7, Figure 34.1.2).

Chapter 1's basis b1 = (2, 1), b2 = (-1, 1) of R^2 (b1 ^ b2 = 3 e1 ^ e2). The pair (u, v) = (b1, b2) A(t) moves
continuously, (u, v) are the columns of [b1 b2] A(t):
  part 1  A = [[1, s], [0, 1]], s: 0 -> 1        shear: v slides along u to b1 + b2           (det A = 1)
  part 2  A = S(1 - r) R(r pi/2), r: 0 -> 1      to (b2, -b1) of Figure 34.1.2(a)              (det A = 1)
  part 3  A = [[0, -(1 - c)], [1, 2c]], c: 0 -> 1   u = b2 stays, v goes from -b1 to 2 b2    (det A = 1 - c)
In part 3 det A falls linearly to 0, the value with it, and the parallelogram collapses onto the line of u.
Left: the parallelogram (sky blue while the value is positive), u blue, v green, the curved arrow from u to v (the
orientation), the unit grid. Right, live: det A and the coefficient of u ^ v = det[u v] e1 ^ e2 = 3 det A e1 ^ e2.
Self-check: det[u v] = 3 det A at every sampled t; det A = 1 on parts 1 and 2 (so the value stays 3.000); the end pairs of
parts 1 and 2 are (b1, b1 + b2) and (b2, -b1) (Figure 34.1.2(a)); part 3 ends at the dependent pair (b2, 2 b2) with value 0
(Proposition 34.1.8).

Render: ``bash tools/render_anims.sh chapters/34-exterior-algebra/anim/anim-34-1-1-same-two-vector.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, FadeOut, Polygon, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402

B = np.array([[2.0, -1.0], [1.0, 1.0]])         # columns b1, b2
B1, B2 = B[:, 0], B[:, 1]
assert abs(np.linalg.det(B) - 3.0) < 1e-12


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def shear(s):
    return np.array([[1.0, s], [0.0, 1.0]])


def A_of(t):
    """t in [0, 3]: part 1 (0..1), part 2 (1..2), part 3 (2..3)."""
    if t <= 1:
        return shear(t)
    if t <= 2:
        r = t - 1
        return shear(1 - r) @ rot(r * np.pi / 2)
    c = t - 2
    # u stays b2 = B (0, 1)^T; v goes from -b1 = B (-1, 0)^T to 2 b2 = B (0, 2)^T
    return np.array([[0.0, -(1 - c)], [1.0, 2 * c]])


for t in np.linspace(0, 3, 301):
    A = A_of(t)
    U = B @ A
    assert abs(np.linalg.det(U) - 3 * np.linalg.det(A)) < 1e-12                   # Proposition 34.1.6(a)
    if t <= 2:
        assert abs(np.linalg.det(A) - 1) < 1e-12
assert np.allclose(B @ A_of(1), np.c_[B1, B1 + B2]) and np.allclose(B @ A_of(2), np.c_[B2, -B1])
assert np.allclose(B @ A_of(2.0 + 1e-12), np.c_[B2, -B1], atol=1e-9)              # continuous at t = 2
assert np.allclose(B @ A_of(3), np.c_[B2, 2 * B2]) and abs(np.linalg.det(A_of(3))) < 1e-12
assert abs(np.linalg.det(A_of(2.5)) - 0.5) < 1e-12                                 # det A = 1 - c

O = np.array([-3.35, -0.55, 0.0])
SC = 0.92


def P(x):
    return O + SC * hp.to3(np.asarray(x, float))


# layout: every corner of every parallelogram stays in the left part of the frame
_c = np.array([P(B @ A_of(t) @ w) for t in np.linspace(0, 3, 61) for w in ([1, 0], [0, 1], [1, 1])])
assert _c[:, 0].min() > -6.7 and _c[:, 0].max() < 0.3 and _c[:, 1].min() > -2.75 and _c[:, 1].max() < 2.9


def turn_arc(center, x, y, r=0.38):
    """curved arrow from the direction of x toward y (about 3/4 of a turn), as in Figure 34.1.2."""
    s = np.sign(x[0] * y[1] - x[1] * y[0]) or 1.0
    a0 = np.arctan2(x[1], x[0])
    ts = a0 + s * np.linspace(0.15, 1.35 * np.pi, 50)
    pts = center + r * np.stack([np.cos(ts), np.sin(ts), 0 * ts], 1)
    d = (pts[-1] - pts[-4]) / np.linalg.norm(pts[-1] - pts[-4])        # tangent at the end of the arc
    return VGroup(hp.polyline(pts[:-3], width=3), hp.arrow(pts[-1] - 0.3 * d, pts[-1], width=3, tip=0.16))


class Anim(hp.Scene2D):
    def construct(self):
        tt = ValueTracker(0.0)
        grid = VGroup()
        for k in range(-3, 4):
            grid.add(hp.seg(P([k, -2.3]), P([k, 3.3]), color="#E3E3E3", width=1.5))
        for k in range(-2, 4):
            grid.add(hp.seg(P([-3.4, k]), P([3.4, k]), color="#E3E3E3", width=1.5))
        self.add(grid, hp.dot(O, r=0.05))

        def figure():
            A = A_of(tt.get_value())
            U = B @ A
            u, v = U[:, 0], U[:, 1]
            d = np.linalg.det(U)
            g = VGroup()
            if abs(d) > 1e-3:
                col = hp.C["region"] if d > 0 else hp.C["accent"]
                g.add(Polygon(P([0, 0]), P(u), P(u + v), P(v), fill_color=col, fill_opacity=0.35, stroke_width=0))
                if abs(d) > 0.6:
                    g.add(turn_arc(P((u + v) / 2), u, v))
            g.add(hp.arrow(P([0, 0]), P(v), color=hp.C["third"], width=6, tip=0.25),
                  hp.arrow(P([0, 0]), P(u), color=hp.C["tangent"], width=6, tip=0.25))
            du, dv = u / np.linalg.norm(u), v / (np.linalg.norm(v) + 1e-9)
            Jm = np.array([[0.0, 1.0], [-1.0, 0.0]])            # rotation by -90 degrees
            g.add(hp.text("u", size=32, color=hp.C["tangent"]).move_to(P(u) + hp.to3(0.2 * du + 0.3 * Jm @ du)),
                  hp.text("v", size=32, color=hp.C["third"]).move_to(P(v) + hp.to3(0.2 * dv - 0.3 * Jm @ dv)))
            return g
        fig = always_redraw(figure)
        self.add(fig)

        XL = 1.0
        self.add(hp.math(r"$(u,\,v) = (b_1,\,b_2)\,A$", size=34).move_to([XL, 3.2, 0], aligned_edge=np.array([-1, 0, 0])))
        phase = [hp.text("shear: v → v + s u", size=30)]
        lab_det = hp.math(r"$\det A\ =$", size=34).move_to([XL, 1.6, 0], aligned_edge=np.array([-1, 0, 0]))
        lab_val = hp.math(r"$u\wedge v\ =$", size=34).move_to([XL, 0.75, 0], aligned_edge=np.array([-1, 0, 0]))
        e12 = hp.math(r"$e_1\wedge e_2$", size=34)

        def nums():
            A = A_of(tt.get_value())
            dA = np.linalg.det(A)
            val = np.linalg.det(B @ A)
            n1 = hp.text(f"{dA:.3f}".replace("-", "−"), size=32).next_to(lab_det, np.array([1.0, 0, 0]), buff=0.25)
            n2 = hp.text(f"{val:.3f}".replace("-", "−"), size=32).next_to(lab_val, np.array([1.0, 0, 0]), buff=0.25)
            return VGroup(n1, n2, e12.copy().next_to(n2, np.array([1.0, 0, 0]), buff=0.2))
        self.add(lab_det, lab_val, always_redraw(nums))
        ph = phase[0].move_to([XL, 2.4, 0], aligned_edge=np.array([-1, 0, 0]))
        self.add(ph)
        self.wait(2.0)
        self.play(tt.animate.set_value(1.0), run_time=4.0, rate_func=smooth)
        self.wait(1.5)
        ph2 = hp.text("turn the pair, det A = 1", size=30).move_to(ph, aligned_edge=np.array([-1, 0, 0]))
        self.play(*hp.swap(ph, ph2), run_time=0.6)
        self.play(tt.animate.set_value(2.0), run_time=4.5, rate_func=smooth)
        self.wait(1.5)
        same = hp.text("same 2-vector: 3 e₁∧e₂", size=30, color=hp.C["third"]).move_to([XL, -0.4, 0],
                                                                                    aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(same), run_time=0.7)
        self.wait(1.3)
        ph3 = hp.text("v → 2u: dependent", size=30).move_to(ph2, aligned_edge=np.array([-1, 0, 0]))
        self.play(*hp.swap(ph2, ph3), FadeOut(same), run_time=0.6)
        self.play(tt.animate.set_value(3.0), run_time=4.5, rate_func=smooth)
        zero = VGroup(hp.math(r"$u\wedge v = 0$", size=32),
                      hp.text("u, v linearly dependent", size=30)).arrange(np.array([0, -1.0, 0]), buff=0.25,
                                                                           aligned_edge=np.array([-1, 0, 0]))
        zero.move_to([XL, -0.75, 0], aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(zero), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=15.5)
