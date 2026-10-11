"""Animation 31.5.6: where silhouette gradients come from (Proposition 31.5.5, Figure 31.5.5, Exercise 31.5.5).

Part 1 (Figure 31.5.5): pixels with centres x = 0, 1, 2, 3 (pixel k covers [k - 0.5, k + 0.5]); a surface of colour 1
covers x < p, the background is 0. The edge position p sweeps from 0.05 to 2.95.
  point sampling: pixel k shows 1 iff its centre is covered (k < p);
  analytic antialiasing (the nvdiffrast rule [Laine20]): for horizontally adjacent pixels whose samples differ, the
    pixel containing the crossing receives the other's colour with a factor linear in the crossing position (0 at
    the midpoint, 1/2 at its centre);
  exact coverage: the covered fraction clamp(p - (k - 0.5), 0, 1).
  Right: the colour c_1 of pixel 1 against p, drawn as p moves.
Part 2: the unit pixel [0, 1]^2 and an edge {<x, n> = p} moving with unit normal speed, n = (1, 0) (blue) or
n = (1, 1)/sqrt 2 (orange). Covered area against p; at the moment each edge passes the pixel centre, the slope equals
the length of the edge inside the pixel: 1 and sqrt 2 ((31.5.5)).
Self-check: antialiased = exact for this vertical edge at every p (away from the measure-zero ties at pixel
centres); d c_1 / d p = 1 on (0.5, 1.5) and point sampling has derivative 0 except at p = 1; the areas are
min(max(p, 0), 1) and s^2/2 (s <= 1), 1 - (2 - s)^2/2 (1 <= s <= 2) with s = sqrt 2 p; their slopes at the centre
crossing are 1 and sqrt 2, equal to the edge lengths inside the pixel.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-5-6-silhouette.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

K = 4


def point_sampled(p):
    return np.array([1.0 if k < p else 0.0 for k in range(K)])


def antialiased(p):
    c = point_sampled(p)
    out = c.copy()
    for k in range(K - 1):
        if c[k] != c[k + 1] and k < p < k + 1:
            mid = k + 0.5
            if p > mid:
                w = p - mid
                out[k + 1] = (1 - w) * c[k + 1] + w * c[k]
            else:
                w = mid - p
                out[k] = (1 - w) * c[k] + w * c[k + 1]
    return out


def exact(p):
    return np.array([np.clip(p - (k - 0.5), 0, 1) for k in range(K)])


PS = np.linspace(0.05, 2.95, 1161) + 1e-4
for p in PS:
    if abs(p - round(p)) > 1e-9:
        assert np.allclose(antialiased(p), exact(p), atol=1e-12)
C1_AA = np.array([antialiased(p)[1] for p in PS])
C1_PT = np.array([point_sampled(p)[1] for p in PS])
C1_EX = np.array([exact(p)[1] for p in PS])
d_aa = np.gradient(C1_AA, PS)
inside = (PS > 0.52) & (PS < 1.48)
assert np.allclose(d_aa[inside], 1, atol=1e-6) and np.allclose(d_aa[(PS < 0.48) | (PS > 1.52)], 0, atol=1e-6)
d_pt = np.gradient(C1_PT, PS)
assert np.allclose(d_pt[np.abs(PS - 1) > 0.01], 0)


def area_v(p):
    return float(np.clip(p, 0, 1))


def area_d(p):
    s = np.sqrt(2) * p
    if s <= 0:
        return 0.0
    if s <= 1:
        return s * s / 2
    if s <= 2:
        return 1 - (2 - s) ** 2 / 2
    return 1.0


def poly_d(p):
    """The covered part {x + y < sqrt 2 p} of the unit pixel, as a polygon."""
    s = np.sqrt(2) * p
    if s <= 1:
        return [(0, 0), (s, 0), (0, s)]
    return [(0, 0), (1, 0), (1, s - 1), (s - 1, 1), (0, 1)]


def chord_d(p):
    s = np.sqrt(2) * p
    if s <= 1:
        return (s, 0), (0, s)
    return (1, s - 1), (s - 1, 1)


hh = 1e-6
for f, pc, slope in ((area_v, 0.5, 1.0), (area_d, np.sqrt(2) / 2, np.sqrt(2))):
    assert abs((f(pc + hh) - f(pc - hh)) / (2 * hh) - slope) < 1e-4
a, b = chord_d(np.sqrt(2) / 2)
assert np.isclose(np.hypot(a[0] - b[0], a[1] - b[1]), np.sqrt(2))
# numerical check of the area formula by sampling
g = (np.arange(400) + 0.5) / 400
GXp, GYp = np.meshgrid(g, g)
for p in (0.3, 0.7, 1.1):
    assert abs(((GXp + GYp) < np.sqrt(2) * p).mean() - area_d(p)) < 5e-3


class Anim(da.Scene2D):
    def construct(self):
        # ---------------------------------------------------------- part 1
        ttl = title("Silhouette gradients: point sampling vs. coverage", 34)
        X0, SQ = -6.0, 1.3

        def px(k, y):
            return np.array([X0 + (k + 0.5) * SQ, y, 0.0])

        rows = {"pt": 1.4, "aa": -0.6}
        pv = ValueTracker(0.05)

        def row_mobs():
            p = pv.get_value()
            g = VGroup()
            for key, fn in (("pt", point_sampled), ("aa", antialiased)):
                vals = fn(p)
                for k in range(K):
                    v = vals[k]
                    sq = Square(SQ, stroke_width=1.5, color=GREY_B, fill_color=rgb_hex([1 - v] * 3), fill_opacity=1)
                    sq.move_to(px(k, rows[key]))
                    g.add(sq)
                    g.add(da.label(f"{v:.1f}", 30, color=WHITE if v > 0.5 else BLACK).move_to(px(k, rows[key])))
            x = X0 + (p + 0.5) * SQ
            for y in rows.values():
                g.add(Line([x, y - SQ / 2 - 0.12, 0], [x, y + SQ / 2 + 0.12, 0], stroke_width=5, color=C["normal"]))
            return g

        rowm = always_redraw(row_mobs)
        rl = VGroup(da.label("point-sampled", 30).next_to(px(1.5, rows["pt"] + SQ / 2), UP, buff=0.18),
                    da.label("antialiased (nvdiffrast)", 30).next_to(px(1.5, rows["aa"] + SQ / 2), UP, buff=0.18))
        cen = VGroup(*[Dot(px(k, -1.8), radius=0.05, color=BLACK) for k in range(K)])
        cen.add(*[da.label(str(k), 30).next_to(px(k, -1.8), DOWN, buff=0.1) for k in range(K)])
        cen.add(da.label("pixel", 30).next_to(px(-0.1, -1.8), LEFT, buff=0.2).shift(DOWN * 0.2))
        elab = readout(lambda: f"edge at p = {pv.get_value():.2f}", [X0, -2.75, 0], size=30, color=C["normal"])

        P = Plot(0.75, -1.7, 4.7, 3.6, (0, 3), (-0.05, 1.1))
        axes = P.axes(xticks=[(0, "0"), (0.5, "0.5"), (1, "1"), (1.5, "1.5"), (3, "3")], yticks=[(0, "0"), (1, "1")],
                      xlabel=mt(r"edge position $p$", 30))
        ylab = mt(r"colour of pixel 1, $c_1$", 30).next_to(P.p(0, 1.1), UP, buff=0.12).shift(RIGHT * 1.2)
        band = Rectangle(width=P.X(1.5) - P.X(0.5), height=P.h, stroke_width=0, fill_color=C["region"],
                         fill_opacity=0.2).move_to([(P.X(0.5) + P.X(1.5)) / 2, P.y0 + P.h / 2, 0])
        slope = mt(r"$\partial c_1/\partial p = 1$", 30).move_to(P.p(1.55, 0.25), aligned_edge=LEFT)

        def traced():
            p = pv.get_value()
            n = max(2, int(np.searchsorted(PS, p)) + 1)
            g = VGroup(P.curve(PS[:n], C1_PT[:n], color=C["tangent"], width=5),
                       P.curve(PS[:n], C1_EX[:n], color=BLACK, width=2.5).set_z_index(2),
                       P.curve(PS[:n], C1_AA[:n], color=C["accent"], width=7))
            g.add(DashedLine(P.p(p, -0.05), P.p(p, 1.1), stroke_width=1.5, color=C["normal"]))
            return g

        tr = always_redraw(traced)
        leg = VGroup(da.label("point-sampled", 30, color=C["tangent"]),
                     da.label("antialiased", 30, color=C["accent"]),
                     da.label("exact coverage", 30))
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to(P.p(1.62, 0.86), aligned_edge=UL)
        surf = da.label("surface (1) | background (0)", 30).move_to([X0, 2.95, 0], aligned_edge=LEFT)
        self.add(ttl, rowm, rl, cen, elab, axes, ylab, band, tr, leg, surf)
        self.wait(1.6)
        self.play(pv.animate.set_value(0.5), run_time=1.6, rate_func=linear)
        self.play(pv.animate.set_value(1.5), run_time=4.0, rate_func=linear)
        self.play(FadeIn(slope), run_time=0.4)
        self.play(pv.animate.set_value(2.95), run_time=2.6, rate_func=linear)
        self.wait(1.6)

        # ---------------------------------------------------------- part 2
        part1 = VGroup(rowm, rl, cen, elab, axes, ylab, band, tr, leg, slope, surf)
        rowm.clear_updaters()
        tr.clear_updaters()
        elab.clear_updaters()
        ttl2 = mt(r"Coverage rate = edge length inside the pixel", 34).to_edge(UP, buff=0.3)
        self.play(FadeOut(part1), FadeTransform(ttl, ttl2), run_time=0.8)
        S2 = 2.3
        CV, CD = np.array([-5.6, -0.12]), np.array([-2.55, -0.12])

        def u2s(c, q):
            return np.array([c[0] + S2 * (q[0] - 0.5), c[1] + S2 * (q[1] - 0.5), 0.0])

        qv = ValueTracker(0.0)

        def pix():
            p = qv.get_value()
            g = VGroup()
            for c in (CV, CD):
                g.add(Square(S2, stroke_width=2, color=BLACK).move_to([c[0], c[1], 0]))
                g.add(Dot(u2s(c, (0.5, 0.5)), radius=0.05, color=BLACK))
            if p > 1e-3:
                pv_ = min(p, 1)
                g.add(Polygon(*[u2s(CV, q) for q in [(0, 0), (pv_, 0), (pv_, 1), (0, 1)]], stroke_width=0,
                              fill_color=GREY_D, fill_opacity=0.55))
                g.add(Polygon(*[u2s(CD, q) for q in poly_d(min(p, np.sqrt(2)))], stroke_width=0,
                              fill_color=GREY_D, fill_opacity=0.55))
            if p < 1:
                g.add(Line(u2s(CV, (p, -0.15)), u2s(CV, (p, 1.15)), stroke_width=2, color=C["tangent"]))
                g.add(Line(u2s(CV, (p, 0)), u2s(CV, (p, 1)), stroke_width=7, color=C["tangent"]))
            s = np.sqrt(2) * p
            if s < 2:
                a_, b_ = chord_d(p)
                x_lo, x_hi = max(-0.15, s - 1.15), min(1.15, s + 0.15)     # the line x + y = s inside [-0.15, 1.15]^2
                g.add(Line(u2s(CD, (x_lo, s - x_lo)), u2s(CD, (x_hi, s - x_hi)), stroke_width=2, color=C["accent"]))
                if s > 1e-3:
                    g.add(Line(u2s(CD, a_), u2s(CD, b_), stroke_width=7, color=C["accent"]))
            return g

        pm = always_redraw(pix)
        labs = VGroup(mt(r"$n = (1, 0)$", 30, color=C["tangent"]).next_to([CV[0], CV[1] + S2 / 2, 0], UP, buff=0.5),
                      mt(r"$n = (1, 1)/\sqrt{2}$", 30, color=C["accent"]).next_to([CD[0], CD[1] + S2 / 2, 0], UP, buff=0.5))
        cov = da.label("grey: covered part of the pixel", 30).move_to([-6.85, 2.7, 0], aligned_edge=LEFT)
        Q = Plot(1.2, -1.75, 5.3, 3.6, (0, 1.45), (0, 1.08))
        qax = Q.axes(xticks=[(0, "0"), (0.5, "0.5"), (1, "1"), (1.414, mt(r"$\sqrt{2}$", 30))],
                     yticks=[(0, "0"), (1, "1")], xlabel=mt(r"edge moved by $p$ along $n$", 30))
        qlab = da.label("covered area", 30).next_to(Q.p(0, 1.08), UP, buff=0.12).shift(RIGHT * 0.8)
        pp = np.linspace(0, 1.45, 600)
        AV = np.array([area_v(x) for x in pp])
        AD = np.array([area_d(x) for x in pp])

        def qtr():
            p = qv.get_value()
            n = max(2, int(np.searchsorted(pp, p)) + 1)
            g = VGroup(Q.curve(pp[:n], AV[:n], color=C["tangent"], width=5),
                       Q.curve(pp[:n], AD[:n], color=C["accent"], width=5))
            g.add(Dot(Q.p(min(p, 1.45), area_v(p)), radius=0.07, color=C["tangent"]),
                  Dot(Q.p(min(p, 1.45), area_d(p)), radius=0.07, color=C["accent"]))
            return g

        qt = always_redraw(qtr)
        self.add(pm, labs, cov, qax, qlab, qt)
        self.wait(1.2)
        self.play(qv.animate.set_value(0.5), run_time=2.0, rate_func=linear)
        tv = Line(Q.p(0.25, 0.25), Q.p(0.75, 0.75), stroke_width=3, color=C["tangent"])
        tvl = VGroup(mt(r"slope $1$", 30, color=C["tangent"]), mt(r"$=$ length $1$", 30, color=C["tangent"]))
        tvl.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([CV[0], CV[1] - S2 / 2 - 0.5, 0], aligned_edge=UP)
        self.play(Create(tv), FadeIn(tvl), run_time=0.6)
        self.wait(1.5)
        self.play(qv.animate.set_value(np.sqrt(2) / 2), run_time=1.0, rate_func=linear)
        c0 = area_d(np.sqrt(2) / 2)
        td = Line(Q.p(np.sqrt(2) / 2 - 0.18, c0 - 0.18 * np.sqrt(2)), Q.p(np.sqrt(2) / 2 + 0.18, c0 + 0.18 * np.sqrt(2)),
                  stroke_width=3, color=C["accent"])
        tdl = VGroup(mt(r"slope $\sqrt{2}$", 30, color=C["accent"]), mt(r"$=$ length $\sqrt{2}$", 30, color=C["accent"]))
        tdl.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([CD[0], CD[1] - S2 / 2 - 0.5, 0], aligned_edge=UP)
        self.play(Create(td), FadeIn(tdl), run_time=0.6)
        self.wait(1.8)
        self.play(qv.animate.set_value(1.45), run_time=2.4, rate_func=linear)
        self.wait(1.8)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=8.0)
