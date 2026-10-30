"""Animation 34.3.1: one covector, a moving inner product -- the level lines stay, the sharp moves
(Section 34.3, (34.3.1), (34.3.2), Proposition 34.3.2, Example 34.3.3, Figure 34.3.1).

alpha = e^1 + e^2 on R^2, drawn by its level lines {alpha = c}, c = -3, ..., 3 (purple, c = 0 thicker); they never move.
The inner product g_s(x, y) = x^T G_s y with G_s = (1 - s) I + s G_1, G_1 = [[2, 1], [1, 1]] (Example 34.3.3), s: 0 -> 1;
det G_s = 1 + s - s^2 > 0. Then alpha^sharp = G_s^{-1} (1, 1) = (1 - s, 1) / det G_s goes from (1, 1) to (0, 1) (blue);
the grey arrow is the Euclidean (1, 1). The dashed ellipse is the g_s-circle of radius |alpha^sharp|_g through
alpha^sharp; it touches the level line {alpha = |alpha^sharp|_g^2} (highlighted) exactly at alpha^sharp
(Proposition 34.3.2(c)), so the touching level runs from 2 down to 1. The orange test vector u = (2.2, 0.4) of
Figure 34.3.1: alpha(u) = 2.6 for every s, and g_s(alpha^sharp, u) = 2.6 as well (the definition (34.3.1)).
Self-check: g_s(alpha^sharp, w) = alpha(w) for w = e_1, e_2, u; g_s(alpha^sharp, d) = 0 for the level direction
d = (1, -1); alpha(alpha^sharp) = |alpha^sharp|_g^2; the ellipse is tangent to the level line at alpha^sharp (the
g-gradient of x^T G x at alpha^sharp is parallel to (1, 1)); end values (1, 1), |.|^2 = 2 and (0, 1), |.|^2 = 1.

Render: ``bash tools/render_anims.sh chapters/34-exterior-algebra/anim/anim-34-3-1-sharp-metric.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402

A = np.array([1.0, 1.0])                      # alpha = e^1 + e^2
G1 = np.array([[2.0, 1.0], [1.0, 1.0]])
U = np.array([2.2, 0.4])
D = np.array([1.0, -1.0])


def G(s):
    return (1 - s) * np.eye(2) + s * G1


def sharp(s):
    return np.linalg.solve(G(s), A)


for s in np.linspace(0, 1, 101):
    g, a = G(s), sharp(s)
    assert np.all(np.linalg.eigvalsh(g) > 0) and abs(np.linalg.det(g) - (1 + s - s * s)) < 1e-12
    for w in (np.array([1.0, 0]), np.array([0, 1.0]), U):
        assert abs(a @ g @ w - A @ w) < 1e-12                                  # (34.3.1)
    assert abs(a @ g @ D) < 1e-12 and abs(A @ a - a @ g @ a) < 1e-12            # g-perpendicular; alpha(a) = |a|^2
    grad = 2 * g @ a                                                           # normal of the ellipse at a
    assert abs(grad[0] * A[1] - grad[1] * A[0]) < 1e-12                        # tangent to the level line
    assert np.allclose(a, np.array([1 - s, 1.0]) / (1 + s - s * s))
assert np.allclose(sharp(0), [1, 1]) and np.allclose(sharp(1), [0, 1]) and abs(A @ U - 2.6) < 1e-12
assert abs(A @ sharp(0) - 2) < 1e-12 and abs(A @ sharp(1) - 1) < 1e-12

O = np.array([-3.35, -0.25, 0.0])
SC = 1.05
BOX = 2.75                                    # the picture is clipped to [-BOX, BOX]^2 (math units)


def P(x):
    return O + SC * hp.to3(np.asarray(x, float))


def level_segment(c):
    """the part of {x + y = c} inside the box."""
    pts = []
    for x in (-BOX, BOX):
        y = c - x
        if -BOX <= y <= BOX:
            pts.append([x, y])
    for y in (-BOX, BOX):
        x = c - y
        if -BOX <= x <= BOX:
            pts.append([x, y])
    pts = np.unique(np.round(np.array(pts), 12), axis=0)
    return pts[[0, -1]]


def ellipse(g, r2, m=200):
    """{x : x^T g x = r2}."""
    w, Q = np.linalg.eigh(g)
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    return (Q @ (np.sqrt(r2 / w)[:, None] * np.stack([np.cos(t), np.sin(t)]))).T


_e = ellipse(G(0), 2.0)
assert np.abs(_e).max() < BOX and np.abs(ellipse(G(1), 1.0)).max() < BOX


def matrix_mob(g):
    """'G =' and the 2 x 2 matrix with drawn brackets (Text cannot stack rows)."""
    cells = VGroup(*[hp.text(f"{g[i, j]:.2f}", size=30) for i in range(2) for j in range(2)])
    for k, m in enumerate(cells):
        m.move_to([0.75 + 1.25 * (k % 2), 0.24 - 0.5 * (k // 2), 0])
    h0, h1 = -0.5, 0.48
    br = VGroup(hp.polyline([[0.25, h1, 0], [0.15, h1, 0], [0.15, h0, 0], [0.25, h0, 0]], width=2.5),
                hp.polyline([[2.5, h1, 0], [2.6, h1, 0], [2.6, h0, 0], [2.5, h0, 0]], width=2.5))
    lab = hp.text("G =", size=30).move_to([-0.45, -0.01, 0])
    return VGroup(lab, cells, br)


class Anim(hp.Scene2D):
    def construct(self):
        st = ValueTracker(0.0)
        frame = hp.polyline(P(np.array([[-BOX, -BOX], [BOX, -BOX], [BOX, BOX], [-BOX, BOX]])), color="#CCCCCC",
                            width=1.5, closed=True)
        lines = VGroup()
        for c in range(-3, 4):
            seg = level_segment(c)
            lines.add(hp.seg(P(seg[0]), P(seg[1]), color=hp.C["covector"], width=4 if c == 0 else 2.2))
        self.add(frame, lines, hp.dot(P([0, 0]), r=0.05))
        self.add(hp.math(r"level lines of $\alpha = e^1 + e^2$", size=30, color=hp.C["covector"])
                 .move_to([O[0], 3.45, 0]))
        uarr = hp.arrow(P([0, 0]), P(U), color=hp.C["accent"], width=5, tip=0.22)
        self.add(uarr, hp.text("u", size=30, color=hp.C["accent"]).move_to(P(U) + np.array([0.25, -0.2, 0])))

        def moving():
            s = st.get_value()
            g, a = G(s), sharp(s)
            r2 = float(A @ a)
            seg = level_segment(r2)
            return VGroup(hp.seg(P(seg[0]), P(seg[1]), color=hp.C["covector"], width=7),
                          hp.dashed(P(ellipse(g, r2))[:, :2], color=hp.C["tangent"], width=2.5),
                          hp.arrow(P([0, 0]), P([1, 1]), color=hp.C["aux"], width=3, tip=0.18),
                          hp.arrow(P([0, 0]), P(a), color=hp.C["tangent"], width=6, tip=0.24),
                          hp.dot(P(a), r=0.07))
        self.add(always_redraw(moving))

        XL = 1.25

        def panel():
            s = st.get_value()
            g, a = G(s), sharp(s)
            rows = [hp.text(f"s = {s:.2f}", size=30),
                    matrix_mob(g),
                    VGroup(hp.math(r"$\alpha^\sharp =$", size=32, color=hp.C["tangent"]),
                           hp.text(f"({a[0]:.2f}, {a[1]:.2f})", size=30, color=hp.C["tangent"]))
                    .arrange(np.array([1.0, 0, 0]), buff=0.2),
                    VGroup(hp.math(r"$|\alpha^\sharp|_g^2 = \alpha(\alpha^\sharp) =$", size=32),
                           hp.text(f"{A @ a:.3f}", size=30)).arrange(np.array([1.0, 0, 0]), buff=0.2),
                    VGroup(hp.math(r"$\alpha(u) =$", size=32, color=hp.C["accent"]),
                           hp.text(f"{A @ U:.3f}", size=30, color=hp.C["accent"])).arrange(np.array([1.0, 0, 0]), buff=0.2),
                    VGroup(hp.math(r"$g(\alpha^\sharp, u) =$", size=32),
                           hp.text(f"{a @ g @ U:.3f}", size=30)).arrange(np.array([1.0, 0, 0]), buff=0.2)]
            return VGroup(*[m.move_to([XL, y, 0], aligned_edge=np.array([-1, 0, 0]))
                            for m, y in zip(rows, (3.25, 2.35, 1.4, 0.65, -0.15, -0.85))])
        self.add(always_redraw(panel))
        self.wait(2.5)
        self.play(st.animate.set_value(1.0), run_time=8.0, rate_func=smooth)
        self.wait(1.0)
        fin = VGroup(hp.text("same level lines,", size=30),
                     hp.text("the sharp depends on g", size=30, color=hp.C["tangent"])) \
            .arrange(np.array([0, -1.0, 0]), buff=0.18, aligned_edge=np.array([-1, 0, 0])) \
            .move_to([XL, -1.9, 0], aligned_edge=np.array([-1, 0, 0]))
        self.play(FadeIn(fin), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=14.5)
