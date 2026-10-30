"""Animation 34.2.1: turning a parallelogram changes its three shadow areas but not the sum of their squares
(Section 34.2, Proposition 34.2.1(b), (c), Example 34.2.2; Section 34.1, Example 34.1.10, Figure 34.1.3).

u = (1, 2, 3), v = (-1, 0, 2) as in Figure 34.1.3: u ^ v = 2 e12 + 5 e13 + 4 e23 and |u ^ v|^2 = 45. The parallelogram is
turned rigidly once (2 pi) about the axis (1, 1, 1)/sqrt 3 through its centre (0, 1, 2.5). Its orthogonal shadows are
drawn on three planes parallel to the coordinate planes behind it (floor z = -0.5 for e12, wall y = 4 for e13, wall
x = -3 for e23; moving a shadow to a parallel plane does not change it). Shadow colours: e12 orange, e13 green,
e23 purple. Right: the three signed shadow areas (the components of u ^ v, (34.1.2)) and a stacked bar of their squares.
The components change sign (each one passes through 0), the bar always reaches the dashed line 45.
Self-check: the shadow polygons' signed areas (shoelace, orientation (e_i, e_j)) equal the 2 x 2 minors of [Ru Rv];
the sum of squares is 45 (1e-12) at every sampled angle and equals |Ru x Rv|^2; every component changes sign;
the parallelogram stays between the walls.

Render: ``bash tools/render_anims.sh chapters/34-exterior-algebra/anim/anim-34-2-1-shadow-pythagoras.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, Polygon, Rectangle, ValueTracker, VGroup, always_redraw, linear)  # noqa: E402

U0, V0 = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 2.0])
CEN = (U0 + V0) / 2                                         # centre of the parallelogram with corner at 0
AXIS = np.ones(3) / np.sqrt(3)
PAIRS = [(0, 1), (0, 2), (1, 2)]                            # e12, e13, e23
WALL = {(0, 1): (2, -0.5), (0, 2): (1, 4.0), (1, 2): (0, -3.0)}      # plane (i, j): fixed coordinate k = value
COL = {(0, 1): hp.C["accent"], (0, 2): hp.C["third"], (1, 2): hp.C["covector"]}
NAME = {(0, 1): r"$e_1\wedge e_2$", (0, 2): r"$e_1\wedge e_3$", (1, 2): r"$e_2\wedge e_3$"}


def rot(th):
    a = AXIS
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def corners(th):
    R = rot(th)
    s = (U0 + V0) / 2
    return np.array([CEN + R @ (w - s) for w in (np.zeros(3), U0, U0 + V0, V0)])


def minors(th):
    R = rot(th)
    M = np.c_[R @ U0, R @ V0]
    return {(i, j): M[i, 0] * M[j, 1] - M[j, 0] * M[i, 1] for i, j in PAIRS}


def shoelace(P2):
    x, y = P2[:, 0], P2[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


assert [round(minors(0)[p], 12) for p in PAIRS] == [2, 5, 4]                       # Example 34.1.10
sign_seen = {p: set() for p in PAIRS}
for th in np.linspace(0, 2 * np.pi, 721):
    m = minors(th)
    R = rot(th)
    assert abs(sum(x * x for x in m.values()) - 45) < 1e-12                          # Proposition 34.2.1(b)
    assert abs(np.sum(np.cross(R @ U0, R @ V0) ** 2) - 45) < 1e-12                   # = |u x v|^2 (Example 34.2.2)
    Q = corners(th)
    for (i, j) in PAIRS:
        # the parallelogram u ^ v has corners 0, u, u + v, v; its shadow on plane (i, j) has signed area = the minor
        assert abs(shoelace(Q[:, [i, j]]) - m[(i, j)]) < 1e-12
        sign_seen[(i, j)].add(np.sign(round(m[(i, j)], 9)))
    assert Q[:, 0].min() > -3 and Q[:, 1].max() < 4 and Q[:, 2].min() > -0.5             # in front of the walls
assert all({-1.0, 1.0} <= s for s in sign_seen.values())                            # every component changes sign
_mx = max(abs(v) for th in np.linspace(0, 2 * np.pi, 2001) for v in minors(th).values())
assert abs(_mx - 5.79) < 0.005                                                      # |component| <= 5.79 (caption)

CAM = dict(elev=24, azim=-48, scale=0.6, origin=(-2.45, -0.35))
XB = (-3.0, 3.0)
YB = (-2.0, 4.0)
ZB = (-0.5, 5.6)


def wall_quads():
    x0, x1 = XB
    y0, y1 = YB
    z0, z1 = ZB
    return [np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]),      # floor z = -0.5
            np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]),      # wall y = 4
            np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]])]      # wall x = -3


def shadow(Q, ij):
    k, val = WALL[ij]
    S = Q.copy()
    S[:, k] = val
    return S


BAR_X, BAR_Y0, BAR_H = 1.55, -2.35, 4.3          # stacked bar of the squares, total 45 -> height BAR_H


class Anim(hp.Scene2D):
    def construct(self):
        cam = hp.Cam(**CAM)
        th = ValueTracker(0.0)
        for q in wall_quads():
            self.add(Polygon(*cam(q), fill_color="#F2F2F2", fill_opacity=1.0, stroke_color="#BBBBBB", stroke_width=1.5))

        def scene():
            Q = corners(th.get_value())
            g = VGroup()
            for ij in PAIRS:
                g.add(Polygon(*cam(shadow(Q, ij)), fill_color=COL[ij], fill_opacity=0.55, stroke_color=COL[ij],
                              stroke_width=2))
            g.add(Polygon(*cam(Q), fill_color=hp.C["region"], fill_opacity=0.85, stroke_color=hp.C["tangent"],
                          stroke_width=2))
            g.add(hp.arrow(cam(Q[0]), cam(Q[3]), color=hp.C["third"], width=5, tip=0.2),
                  hp.arrow(cam(Q[0]), cam(Q[1]), color=hp.C["tangent"], width=5, tip=0.2))
            return g
        self.add(always_redraw(scene))

        # ---- right: components and the stacked bar of squares
        XL = 3.0
        heads = VGroup()
        for k, ij in enumerate(PAIRS):
            heads.add(hp.math(NAME[ij], size=32, color=COL[ij]).move_to([XL, 2.25 - 0.6 * k, 0],
                                                                     aligned_edge=np.array([-1, 0, 0])))
        self.add(heads)
        self.add(hp.text("turn u ∧ v about (1, 1, 1)", size=30).move_to([1.6, 3.45, 0],
                                                                       aligned_edge=np.array([-1, 0, 0])),
                 hp.text("shadow areas", size=30).move_to([XL, 2.85, 0], aligned_edge=np.array([-1, 0, 0])))

        def numbers():
            m = minors(th.get_value())
            g = VGroup()
            for k, ij in enumerate(PAIRS):
                g.add(hp.text(f"{m[ij]:+.2f}".replace("-", "−"), size=30, color=COL[ij])
                      .move_to([XL + 2.85, 2.25 - 0.6 * k, 0], aligned_edge=np.array([1, 0, 0])))
            tot = sum(x * x for x in m.values())
            g.add(hp.math(r"$\Sigma\,(\mathrm{area})^2 = %.2f$" % tot, size=30).move_to([XL, 0.3, 0],
                                                                                    aligned_edge=np.array([-1, 0, 0])))
            return g
        self.add(always_redraw(numbers))

        def bar():
            m = minors(th.get_value())
            g = VGroup()
            y = BAR_Y0
            for ij in PAIRS:
                h = max(1e-4, m[ij] ** 2 / 45 * BAR_H)
                g.add(Rectangle(width=0.7, height=h, fill_color=COL[ij], fill_opacity=0.9, stroke_color=hp.BLACK,
                                stroke_width=1).move_to([BAR_X, y + h / 2, 0]))
                y += h
            return g
        self.add(always_redraw(bar))
        top = BAR_Y0 + BAR_H
        self.add(hp.seg([BAR_X - 0.6, BAR_Y0, 0], [BAR_X + 0.6, BAR_Y0, 0], width=3),
                 hp.dashed([[BAR_X - 0.6, top], [BAR_X + 0.75, top]], color=hp.C["aux"], width=2.5, closed=False),
                 hp.math(r"$45 = |u\wedge v|^2$", size=30).move_to([BAR_X, top + 0.45, 0]))
        self.wait(2.0)
        self.play(th.animate.set_value(2 * np.pi), run_time=16.0, rate_func=linear)
        fin = hp.math(r"$|u\wedge v| = \sqrt{45}$", size=30).move_to([XL, -0.5, 0],
                                                                            aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(fin), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=12.0)
