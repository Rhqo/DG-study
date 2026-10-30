"""Animation 32.4.2: no free lunch -- four discrete curve shortening flows, and each keeps a different property
(Section 32.4, Definition 32.4.5, Propositions 32.4.7-32.4.9, Figure 32.4.3, the summary table).

The 12-vertex polygon on the unit circle at the angles 0, 18, 50, 80, 105, 160, 185, 200, 240, 275, 300, 335 degrees
(ddg32.uneven_circle, Figure 32.4.3). Forward Euler x_i <- x_i + h kappa^X_i (32.4.5) for X = A, B, C, D side by side,
with the matched time steps of Figure 32.4.3: h = dt (D), h = dt / mean(lbar_i) (A, B, C), dt = 1e-3, up to t = 0.3.
In every panel the grey dashed circle has radius sqrt(1 - 2t) (the smooth flow of a circle, Proposition 32.4.3(c)).
Below each polygon, live: the total curvature measured by the definition itself (sum kappa^X_i; for D sum kappa^D_i lbar_i),
the distance the vertex average has moved, and the roundness defect of Figure 32.4.3(b).
At the end the exact entries are boxed: A keeps the total 2 pi (TOTAL), B keeps the vertex average (DRIFT), D keeps
the circle (ROUND), C keeps none of the three. On this polygon the D total stays at 6.1847 only because the D step is a
similarity; it is not 2 pi, and on the jittered flower of Figure 32.4.2 it changes.
Self-check: sum kappa^A = 2 pi (1e-12) at every step, B drift < 1e-14, D defect < 1e-13 and the D radius follows
r_{m+1} = r_m - h / r_m, while A, B, C end with defect > 0.05 and A, C, D move the average by > 1e-4.

Render: ``bash tools/render_anims.sh chapters/32-ddg-overview/anim/anim-32-4-2-no-free-lunch.py``
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
from manim import (Create, FadeIn, Rectangle, ValueTracker, VGroup, always_redraw, linear)  # noqa: E402

U = ddg32.uneven_circle()
LB = dd.curve_dual_lengths(U, True)
DT, T_END = 1e-3, 0.3
STEPS = int(round(T_END / DT))
KINDS = "ABCD"
RUNS, READ = {}, {}
for k in KINDS:
    h = DT if k == "D" else DT / LB.mean()
    tr = np.array(ddg32.flow(U, k, h, STEPS))                 # every step: 301 polygons
    tot = np.array([ddg32.own_total(X, k) for X in tr])
    drift = np.linalg.norm(tr.mean(1) - U.mean(0), axis=1)
    rd = np.array([ddg32.roundness_defect(X) for X in tr])
    RUNS[k], READ[k] = tr, (tot, drift, rd)
TIMES = DT * np.arange(STEPS + 1)
assert np.all(np.abs(READ["A"][0] - 2 * np.pi) < 1e-12)                      # TOTAL for A (Prop 32.4.7)
assert np.all(READ["B"][1] < 1e-14)                                          # DRIFT for B (Prop 32.4.8)
assert np.all(READ["D"][2] < 1e-13)                                          # ROUND for D (Prop 32.4.9)
r_m = 1.0
for _ in range(STEPS):
    r_m -= DT / r_m
assert np.allclose(np.linalg.norm(RUNS["D"][-1], axis=1), r_m, atol=1e-12)
assert all(READ[k][2][-1] > 0.05 for k in "ABC")                             # Figure 32.4.3: 0.117, 0.115, 0.121
assert [round(READ[k][2][-1], 3) for k in "ABC"] == [0.117, 0.115, 0.121]
assert all(READ[k][1][-1] > 1e-4 for k in "ACD")
assert all(np.all(np.abs(READ[k][0] - 2 * np.pi) > 0.05) for k in "BCD")      # never the smooth total 2 pi
assert abs(r_m - 0.6328) < 1e-4 and abs(np.sqrt(1 - 2 * T_END) - 0.6325) < 1e-4

XC = {"A": -5.2, "B": -1.85, "C": 1.85, "D": 5.2}
YC, SC = 1.25, 1.28
NAMES = {"A": "A  turning", "B": "B  length", "C": "C  Steiner", "D": "D  circle"}
ROWS = (-0.55, -1.15, -1.75)
KEEP = {"A": (0, "keeps TOTAL"), "B": (1, "keeps DRIFT"), "C": (None, "keeps none"), "D": (2, "keeps ROUND")}


def P(k, X):
    return np.array([XC[k], YC, 0.0]) + SC * hp.to3(X)


def fmt(k, j):
    tot, drift, rd = READ[k]
    return [f"Σκ = {tot[j]:.4f}", f"drift {drift[j]:.4f}", f"defect {rd[j]:.3f}"]


class Anim(hp.Scene2D):
    def construct(self):
        step = ValueTracker(0)
        for k in KINDS:
            self.add(hp.text(NAMES[k], size=30, color=hp.KIND[k]).move_to([XC[k], 3.0, 0]))
        tl = always_redraw(lambda: hp.text(f"t = {TIMES[int(round(step.get_value()))]:.2f}", size=30)
                           .move_to([0, 3.5, 0]))
        self.add(tl)

        def panel(k):
            def f():
                j = int(round(step.get_value()))
                t = TIMES[j]
                X = RUNS[k][j]
                g = VGroup(hp.ring([XC[k], YC, 0], SC * np.sqrt(1 - 2 * t), color=hp.C["aux"], width=2, dashed_=True),
                           hp.polyline(P(k, X), color=hp.KIND[k], width=4, closed=True),
                           *[hp.dot(p, color=hp.KIND[k], r=0.05) for p in P(k, X)])
                for y, s in zip(ROWS, fmt(k, j)):
                    g.add(hp.text(s, size=30).move_to([XC[k], y, 0]))
                return g
            return always_redraw(f)
        panels = [panel(k) for k in KINDS]
        self.add(*panels)
        start = VGroup(*[hp.polyline(P(k, U), color=hp.C["main"], width=1.5, closed=True) for k in KINDS])
        self.add(start)
        self.wait(2.0)
        self.play(step.animate.set_value(STEPS), run_time=12.0, rate_func=linear)
        self.wait(1.5)

        # ---- the exact entries
        boxes = VGroup()
        for row in range(3):
            for k in KINDS:
                if KEEP[k][0] == row:
                    b = Rectangle(width=3.05, height=0.5, stroke_color=hp.C["main"], stroke_width=5)   # black: green is B
                    b.move_to([XC[k], ROWS[row], 0])
                    boxes.add(b)
                    self.play(Create(b), run_time=0.9)
                    self.wait(0.6)
        verdict = VGroup(*[hp.text(KEEP[k][1], size=30, color=hp.C["main"] if KEEP[k][0] is not None else hp.C["aux"])
                           .move_to([XC[k], -2.4, 0]) for k in KINDS])
        self.play(FadeIn(verdict), run_time=0.9)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=22.0)
