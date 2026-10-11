"""Animation 0.7.3: 작은 칸의 circulation을 더하면 안쪽 edge가 상쇄되고 boundary만 남는다 (0.7절, Theorem 0.7.3, Figure 0.7.3).

M = [0, 3] × [0, 2](Figure 0.7.3과 같다)를 단위 정사각형 6칸으로 나누고, 1-form ω = −x y² dx + x² y dy를 쓴다.
dω = (∂_x(x² y) − ∂_y(−x y²)) dx ∧ dy = 4xy dx ∧ dy ((0.7.5)).
칸 [i, i+1] × [j, j+1]의 boundary를 반시계 방향으로 돈 circulation은 ∬ 4xy = (2i + 1)(2j + 1)이다: 1, 3, 5 / 3, 9, 15.
안쪽 edge 7개마다 두 이웃 칸이 같은 edge를 반대 방향으로 지나므로 두 기여(+a, −a)가 지워진다.
  세로 edge x = k, y ∈ [j, j+1]: 위로 ∫ x² y dy = k²(2j + 1)/2,  가로 edge y = 1, x ∈ [i, i+1]: 오른쪽으로 ∫ −x dx = −(2i + 1)/2.
남는 것은 ∂M뿐이다: 아래 0, 오른쪽 18, 위 18, 왼쪽 0. 따라서 1 + 3 + 5 + 3 + 9 + 15 = 36 = ∮_∂M ω = ∬_M dω.
자기검사: 칸마다 boundary 선적분(수치)과 ∬ 4xy가 같음, 안쪽 edge 기여의 합 0, 네 변의 값, 전체 36.

실행: bash tools/render_anims.sh chapters/00-tour/anim/anim-0-7-3-stokes-cancel.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dganim as da  # noqa: E402
from manim import *  # noqa: E402,F403

from _helpers_b import arrow, polyline  # noqa: E402

C = da.C
NX, NY = 3, 2


def omega(x, y, dx, dy):
    return -x * y ** 2 * dx + x ** 2 * y * dy


def edge_int(p0, p1, n=2001):
    """선분 p0 → p1 위의 ∫ ω (사다리꼴)."""
    t = np.linspace(0, 1, n)
    x = p0[0] + (p1[0] - p0[0]) * t
    y = p0[1] + (p1[1] - p0[1]) * t
    val = omega(x, y, p1[0] - p0[0], p1[1] - p0[1])
    return float(np.sum((val[1:] + val[:-1]) / 2 * np.diff(t)))


def cell_edges(i, j):
    a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
    return [("b", a, b), ("r", b, c), ("t", c, d), ("l", d, a)]


# ---------------------------------------------------------------- 자기검사
circ = {}
for i in range(NX):
    for j in range(NY):
        s = sum(edge_int(p, q) for _, p, q in cell_edges(i, j))
        assert np.isclose(s, (2 * i + 1) * (2 * j + 1), atol=1e-6)            # = ∬_cell 4xy
        circ[(i, j)] = (2 * i + 1) * (2 * j + 1)
assert sum(circ.values()) == 36
INTERIOR = [((k, j), (k, j + 1)) for k in (1, 2) for j in (0, 1)] + [((i, 1), (i + 1, 1)) for i in range(3)]
for p, q in INTERIOR:
    assert np.isclose(edge_int(p, q) + edge_int(q, p), 0)
SIDES = {"bottom": ((0, 0), (3, 0)), "right": ((3, 0), (3, 2)), "top": ((3, 2), (0, 2)), "left": ((0, 2), (0, 0))}
SIDE_VAL = {k: edge_int(*v) for k, v in SIDES.items()}
assert np.allclose([SIDE_VAL[k] for k in ("bottom", "right", "top", "left")], [0, 18, 18, 0], atol=1e-6)

SC = 2.05
O = np.array([-6.4, -2.3, 0])             # 아래 90px(control bar)를 비운다(§12.8)
INSET = 0.12


def P(x, y):
    return O + SC * np.array([x, y, 0])


def fmt(v):
    v = 0.0 if abs(v) < 1e-9 else v
    s = f"{v:+.1f}".replace("-", "−")
    return s[:-2] if s.endswith(".0") else s


class Anim(da.Scene2D):
    def construct(self):
        M = Rectangle(width=NX * SC, height=NY * SC, stroke_width=0, fill_color=C["region"], fill_opacity=0.22)
        M.move_to(P(NX / 2, NY / 2))
        grid = VGroup(*[DashedLine(P(i, 0), P(i, NY), stroke_color=C["aux"], stroke_width=1.5, dash_length=0.08)
                        for i in range(NX + 1)],
                      *[DashedLine(P(0, j), P(NX, j), stroke_color=C["aux"], stroke_width=1.5, dash_length=0.08)
                        for j in range(NY + 1)])
        self.add(M, grid)
        title = da.label("Why Stokes's theorem holds: interior edges cancel", 34).to_edge(UP, buff=0.25)
        X0 = 0.8
        f1 = da.mtex(r"$\omega = -xy^2\,dx + x^2y\,dy$", 36).move_to([X0, 2.55, 0], aligned_edge=LEFT)
        f2 = da.mtex(r"$d\omega = 4xy\,dx\wedge dy$", 36).move_to([X0, 1.85, 0], aligned_edge=LEFT)
        self.add(title, f1, f2)
        self.wait(1.5)

        # 칸의 고리와 circulation
        segs, nums = {}, VGroup()
        for i in range(NX):
            for j in range(NY):
                x0, x1, y0, y1 = i + INSET, i + 1 - INSET, j + INSET, j + 1 - INSET
                corners = {"b": ((x0, y0), (x1, y0)), "r": ((x1, y0), (x1, y1)), "t": ((x1, y1), (x0, y1)),
                           "l": ((x0, y1), (x0, y0))}
                for side, (s, e) in corners.items():
                    m = (0.35 * s[0] + 0.65 * e[0], 0.35 * s[1] + 0.65 * e[1])
                    segs[(i, j, side)] = VGroup(Line(P(*s), P(*e), stroke_color=C["accent"], stroke_width=3.5),
                                                arrow(P(*s), P(*m), C["accent"], 3.5, 0.17))
                nums.add(da.label(str(circ[(i, j)]), 36).move_to(P(i + 0.5, j + 0.5)))
        loops = VGroup(*segs.values())
        cap = da.label("circulation of each cell", 30).move_to([X0, 1.15, 0], aligned_edge=LEFT)
        self.play(Create(loops), FadeIn(nums), FadeIn(cap), run_time=1.5)
        self.wait(1.5)

        # 안쪽 edge 상쇄
        cap2 = da.label("shared edges cancel: +a, −a", 30).move_to([X0, 0.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(cap2), run_time=0.4)
        for p, q in INTERIOR:
            if p[0] == q[0]:                       # 세로 edge x = k
                k, j = p[0], p[1]
                a_left = edge_int((k, j), (k, j + 1))          # 왼쪽 칸의 오른쪽 변: 위로
                pair = [(k - 1, j, "r", a_left, LEFT), (k, j, "l", -a_left, RIGHT)]
            else:                                  # 가로 edge y = 1
                i = p[0]
                a_low = edge_int((i + 1, 1), (i, 1))           # 아래 칸의 위 변: 왼쪽으로
                pair = [(i, 0, "t", a_low, DOWN), (i, 1, "b", -a_low, UP)]
            labs, hl = VGroup(), VGroup()
            for ci, cj, side, val, d in pair:
                seg = segs[(ci, cj, side)]
                hl.add(seg)
                L0 = seg[0].get_start() + 0.8 * (seg[0].get_end() - seg[0].get_start())
                if side in ("l", "b"):
                    L0 = seg[0].get_start() + 0.2 * (seg[0].get_end() - seg[0].get_start())
                lab = da.label(fmt(val), 30, color=C["normal"])
                lab.move_to(L0 + d * (0.12 + 0.5 * (lab.width if d[0] != 0 else lab.height)))
                labs.add(lab)
            self.play(hl.animate.set_color(C["normal"]).set_stroke(width=6), FadeIn(labs), run_time=0.4)
            self.wait(0.75)                                      # edge마다 1.5초(§12.8): 두 수를 읽을 시간
            self.play(FadeOut(hl), FadeOut(labs), run_time=0.35)
        self.wait(0.8)

        # 남는 boundary
        bnd = VGroup()
        Pts = [(0, 0), (3, 0), (3, 2), (0, 2), (0, 0)]
        for s, e in zip(Pts[:-1], Pts[1:]):
            m = (0.42 * s[0] + 0.58 * e[0], 0.42 * s[1] + 0.58 * e[1])
            bnd.add(Line(P(*s), P(*e), stroke_color=C["main"], stroke_width=6), arrow(P(*s), P(*m), C["main"], 6, 0.25))
        # 변의 값은 화살촉(변의 58% 지점)과 겹치지 않는 곳에 둔다
        sv = VGroup(da.label("0", 30).next_to(P(0.75, 0), DOWN, buff=0.12),
                    da.label("18", 30).next_to(P(3, 1.5), RIGHT, buff=0.12),     # 오른쪽 글줄의 수와 붙어 읽히지 않게
                    da.label("18", 30).next_to(P(2.25, 2), UP, buff=0.12),
                    da.label("0", 30).next_to(P(0, 1.5), LEFT, buff=0.12))
        cap3 = VGroup(da.mtex(r"only $\partial M$ is left:", 30),
                      da.mtex(r"$0 + 18 + 18 + 0 = 36$", 34)).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(
            [X0, -0.25, 0], aligned_edge=LEFT)
        self.play(FadeIn(bnd), FadeIn(sv), FadeIn(cap3), run_time=1.0)
        self.wait(1.5)
        fin = VGroup(da.mtex(r"$1 + 3 + 5 + 3 + 9 + 15 = 36$", 36),
                     da.mtex(r"$\iint_M d\omega = 36 = \oint_{\partial M}\omega$", 38)).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2).move_to([X0, -1.95, 0], aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.0)
