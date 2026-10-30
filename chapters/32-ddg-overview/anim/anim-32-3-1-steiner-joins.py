"""Animation 32.3.1: the discrete Steiner formula -- three ways to close an offset polygon give three exact linear
laws (Section 32.3, Proposition 32.3.7, (32.3.6), (32.3.7), Figure 32.3.2).

The counterclockwise convex pentagon of Figure 32.3.2, vertices (0, 0), (2, -0.1), (2.6, 1.1), (1.3, 2), (-0.3, 1.1),
length L = 7.9012. Every edge is translated by eps J t_i (eps < 0: outward). The translated edges (grey) are common to
the three closed curves; they differ only at the corners: (A) circular arc of radius |eps| (blue), (B) chord (green),
(C) the two offset edges extended until they meet at q_i (purple).
Part 1: eps = 0 -> -0.6 (stop at -0.35, the value of Figure 32.3.2). Right: the added length L_X(eps) - L against the
outward distance -eps; the three dots run on straight lines of slopes sum kappa^A = 2 pi = 6.283,
sum kappa^B = 5.868, sum kappa^C = 7.301 (32.3.4).
Part 2: only (C) makes sense inward: eps = -0.6 -> +0.4, the mitered polygon shrinks and its dot continues on the
same line below 0 (Proposition 32.3.7(b), valid while every new edge has positive length, here eps < 0.758).
Self-check: lengths of the drawn curves (the arc as a 40-point polyline) equal (32.3.6)/(32.3.7); at eps = -0.35 they are
10.1003, 9.9552, 10.4567 (Figure 32.3.2); the miter points q_i lie on both offset lines; inward edges stay positive.

Render: ``bash tools/render_anims.sh chapters/32-ddg-overview/anim/anim-32-3-1-steiner-joins.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import ddg32  # noqa: E402
import dgdec as dd  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, FadeOut, Polygon, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402

J = ddg32.J
X = np.array([[0.0, 0.0], [2.0, -0.1], [2.6, 1.1], [1.3, 2.0], [-0.3, 1.1]])
NV = len(X)
T = dd.curve_tangents(X, True)
NE = T @ J.T                                   # edge normals J t_i (inward for this counterclockwise polygon)
TH = dd.turning_angles(X, True)
NVX = dd.curve_vertex_normals(X, True)
L0 = dd.curve_length(X, True)
SUM = {"A": TH.sum(), "B": (2 * np.sin(TH / 2)).sum(), "C": (2 * np.tan(TH / 2)).sum()}
assert np.all(TH > 0) and abs(SUM["A"] - 2 * np.pi) < 1e-12 and abs(L0 - 7.9012) < 1e-4


def pieces(eps, m=40):
    """offset edges [(p_i^+, p_{i+1}^-)], corner arcs, chords, miter points q_i for an outward eps < 0."""
    edges, arcs, chords, q = [], [], [], []
    for i in range(NV):
        pm, pp = X[i] + eps * NE[i - 1], X[i] + eps * NE[i]          # end of offset edge i-1, start of offset edge i
        a0 = np.arctan2(*(pm - X[i])[::-1])
        s = a0 + np.linspace(0, TH[i], m)
        arcs.append(X[i] + abs(eps) * np.stack([np.cos(s), np.sin(s)], 1))
        chords.append(np.array([pm, pp]))
        q.append(X[i] + eps * NVX[i] / np.cos(TH[i] / 2))
        edges.append(np.array([pp, X[(i + 1) % NV] + eps * NE[i]]))
    return edges, arcs, chords, np.array(q)


def lengths(eps):
    """L_A, L_B, L_C measured on the drawn curves."""
    edges, arcs, chords, q = pieces(eps)
    le = sum(np.linalg.norm(e[1] - e[0]) for e in edges)
    LA = le + sum(np.linalg.norm(np.diff(a, axis=0), axis=1).sum() for a in arcs)
    LB = le + sum(np.linalg.norm(c[1] - c[0]) for c in chords)
    LC = np.linalg.norm(np.roll(q, -1, 0) - q, axis=1).sum()
    return LA, LB, LC


for eps in (-0.1, -0.35, -0.6):
    LA, LB, LC = lengths(eps)
    assert abs(LA - (L0 - eps * SUM["A"])) < 1e-3 and abs(LB - (L0 - eps * SUM["B"])) < 1e-12 \
        and abs(LC - (L0 - eps * SUM["C"])) < 1e-12                                       # (32.3.6), (32.3.7)
    _, _, _, q = pieces(eps)
    for i in range(NV):                       # q_i lies on both offset lines (32.3.8)
        for k in (i - 1, i):
            d = q[i] - (X[i] + eps * NE[k])
            assert abs(T[k, 0] * d[1] - T[k, 1] * d[0]) < 1e-12
assert [round(L0 + 0.35 * SUM[k], 4) for k in "ABC"] == [10.1003, 9.9552, 10.4567]          # Figure 32.3.2
EPS_IN = 0.4
tq = np.tan(TH / 2)
assert np.all(dd.curve_edge_lengths(X, True) - EPS_IN * (tq + np.roll(tq, -1)) > 0.2)      # Prop 32.3.7(b) condition
_, _, _, qin = pieces(EPS_IN)
assert abs(np.linalg.norm(np.roll(qin, -1, 0) - qin, axis=1).sum() - (L0 - EPS_IN * SUM["C"])) < 1e-12

# layout
CEN = np.array([-3.55, -0.05, 0.0])
SC = 1.42
MID = X.mean(0)


def P(x):
    x = np.asarray(x, float)
    y = (x - MID) * SC
    return CEN + (np.c_[y, np.zeros(len(y))] if y.ndim == 2 else np.r_[y, 0.0])


_ext = np.vstack([np.vstack(pieces(-0.6)[1]), pieces(-0.6)[3]])
_ps = P(_ext)
assert _ps[:, 0].min() > -6.9 and _ps[:, 0].max() < -0.3 and _ps[:, 1].min() > -2.75 and _ps[:, 1].max() < 2.9
PLOT = hp.Plot((-0.45, 0.66), (-3.2, 4.8), ((1.35, -2.2), (6.75, 2.05)))   # tick labels above the bottom 15%
YB = [PLOT(0, -3.2)[1], PLOT(0, 4.8)[1]]


class Anim(hp.Scene2D):
    def construct(self):
        eps = ValueTracker(0.0)
        show_ab = ValueTracker(1.0)
        fill = Polygon(*P(X), fill_color=hp.C["region"], fill_opacity=0.18, stroke_width=0)
        poly = hp.polyline(P(X), width=3.5, closed=True)
        self.add(fill, poly, *[hp.dot(p, r=0.06) for p in P(X)])
        self.add(hp.text("offset every edge by ε", size=32).move_to([CEN[0], 3.45, 0]))

        def curves():
            e = eps.get_value()
            g = VGroup()
            if abs(e) < 1e-4:
                return g
            if e < 0:
                edges, arcs, chords, q = pieces(e)
                for ed in edges:
                    g.add(hp.polyline(P(ed), color=hp.C["aux"], width=3))
                op = show_ab.get_value()
                for i in range(NV):
                    if op > 0.01:
                        g.add(hp.polyline(P(arcs[i]), color=hp.KIND["A"], width=5).set_stroke(opacity=op))
                        g.add(hp.polyline(P(chords[i]), color=hp.KIND["B"], width=5).set_stroke(opacity=op))
                    pm, pp = chords[i]
                    g.add(hp.polyline(P(np.array([pm, q[i], pp])), color=hp.KIND["C"], width=5))
            else:                                   # inward: only the mitered polygon (C)
                g.add(hp.polyline(P(pieces(e)[3]), color=hp.KIND["C"], width=5, closed=True))
            return g
        cv = always_redraw(curves)
        self.add(cv)
        live = always_redraw(lambda: hp.text(f"ε = {eps.get_value():+.2f}".replace("-", "−"), size=32)
                             .move_to([CEN[0], -2.6, 0]))
        self.add(live)

        # ---- right: added length against the outward distance -eps
        ax = PLOT.axes(xtick=(-0.4, 0, 0.3, 0.6), ytick=(-2, 0, 2, 4), size=30,
                       xlabels=["−0.4", "0", "0.3", "0.6"], ylabels=["−2", "0", "2", "4"])
        zero = hp.dashed([PLOT(-0.45, 0)[:2], PLOT(0.66, 0)[:2]], color=hp.C["aux"], width=1.5, closed=False)
        vzero = hp.dashed([PLOT(0, -3.2)[:2], PLOT(0, 4.8)[:2]], color=hp.C["aux"], width=1.5, closed=False)
        # axis direction labels inside the plot (above the bottom band that the paused control bar covers)
        xl1 = hp.text("outward →", size=30).move_to(PLOT(0.4, -3.2) + np.array([0, 0.32, 0]))
        xl2 = hp.text("← inward", size=30).move_to(PLOT(-0.235, 0.75))
        yl = hp.math(r"$L_X(\varepsilon) - L$", size=32).move_to(PLOT(-0.235, 3.75))
        self.add(ax, zero, vzero, xl1, xl2, yl)
        rows = [("A", "(A) arc"), ("B", "(B) chord"), ("C", "(C) extend")]
        leg = VGroup()
        for k, (key, name) in enumerate(rows):
            y = 3.5 - 0.5 * k
            leg.add(hp.seg([1.05, y, 0], [1.55, y, 0], color=hp.KIND[key], width=6),
                    hp.text(name, size=30, color=hp.KIND[key]).move_to([1.7, y, 0], aligned_edge=np.array([-1, 0, 0])))
        self.add(leg)
        seen = {"A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0]}     # visited range of -eps per curve

        def traces():
            e = eps.get_value()
            g = VGroup()
            for key in "ABC":
                active = key == "C" or e <= 0          # A and B are only drawn outward
                if active:
                    lo, hi = seen[key]
                    seen[key] = [min(lo, -e), max(hi, -e)]
                lo, hi = seen[key]
                if hi - lo > 1e-4:
                    xs = np.array([lo, hi])
                    g.add(PLOT.curve(xs, xs * SUM[key], color=hp.KIND[key], width=4))
                x_now = -e if active else hi
                g.add(hp.dot(PLOT(x_now, x_now * SUM[key]), color=hp.KIND[key], r=0.08))
            return g
        tr = always_redraw(traces)
        self.add(tr)
        self.wait(1.5)

        # ---- part 1: outward
        self.play(eps.animate.set_value(-0.35), run_time=4.0, rate_func=smooth)
        vals = [L0 + 0.35 * SUM[k] for k in "ABC"]          # exact values (32.3.6), (32.3.7)
        vlab = VGroup(*[hp.text(f"L = {v:.4f}", size=30, color=hp.KIND[k]).move_to([4.35, 3.5 - 0.5 * j, 0],
                                                                                 aligned_edge=np.array([-1, 0, 0]))
                        for j, (k, v) in enumerate(zip("ABC", vals))])
        self.play(FadeIn(vlab), run_time=0.6)
        self.wait(2.0)
        self.play(FadeOut(vlab), run_time=0.4)
        self.play(eps.animate.set_value(-0.6), run_time=3.0, rate_func=smooth)
        slopes = VGroup(*[hp.text(f"slope {SUM[k]:.3f}", size=30, color=hp.KIND[k])
                          .move_to([4.35, 3.5 - 0.5 * j, 0], aligned_edge=np.array([-1, 0, 0]))
                          for j, k in enumerate("ABC")])
        for m in slopes:
            self.play(FadeIn(m), run_time=0.6)
        self.wait(1.5)

        # ---- part 2: (C) also works inward
        self.play(show_ab.animate.set_value(0.0), run_time=0.6)
        self.play(eps.animate.set_value(EPS_IN), run_time=5.0, rate_func=smooth)
        fin = VGroup(hp.math(r"$L_X = L - \varepsilon\,\Sigma_i\,\kappa^X_i$", size=30),
                     hp.text("(exact)", size=30)).arrange(np.array([1.0, 0, 0]), buff=0.3)
        fin.move_to([CEN[0], -1.95, 0])         # under the pentagon, above the eps readout
        self.play(FadeIn(fin), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=14.5)
