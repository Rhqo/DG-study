"""Animation 37.2.1: the turning number is the winding number of the velocity curve, and it can change only when
the velocity passes through 0 (Section 37.2, Proposition 37.2.6, Theorem 37.2.10, Non-example 37.2.9(a), Figure 37.2.2(b)).

Curves: the limacons gamma_a(t) = (a + cos t)(cos t, sin t), t in [0, 2 pi] (lib/ddg37.limacon_expr), so that
gamma_a'(t) = a(-sin t, cos t) + (-sin 2t, cos 2t): a circle of radius a run once plus the unit circle run twice.
|gamma_a'|^2 = 1 + a^2 + 2a cos t, minimum |1 - a| at t = pi (Exercise 37.2.4).
Part 1 (a = 0.5): a point runs once around the curve; its velocity, moved to the origin, runs around the velocity
curve and turns twice around 0: rot = wind(gamma', 0) = 2 (37.2.4).
Part 2: a goes from 0.5 to 1.5. The inner loop of the velocity curve shrinks onto 0, passes through 0 at a = 1
(cardioid, cusp at t = pi) and leaves; afterwards rot = 1.
Part 3 (a = 1.5): the run again, one turn.
Self-check: the velocity from sympy (ddg37) equals the closed form above; min speed = |1 - a|; the rotation index
from the angle lift of gamma' and the winding number of the sampled velocity polygon about 0 are 2 for a < 1 and
1 for a > 1.

Render: bash tools/render_anims.sh chapters/37-curves-smooth-discrete/anim/anim-37-2-1-velocity-curve.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg37  # noqa: E402
import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
TT = np.linspace(0, 2 * np.pi, 721)


def curve(a, t=TT):
    r = a + np.cos(t)
    return np.stack([r * np.cos(t), r * np.sin(t)], -1)


def vel(a, t=TT):
    return np.stack([-a * np.sin(t) - np.sin(2 * t), a * np.cos(t) + np.cos(2 * t)], -1)


for a in (0.5, 0.8, 1.2, 1.5):
    dexpr = ddg37.limacon_expr(sp.Rational(a).limit_denominator(10)).diff(ddg37.T_SYM)
    f = sp.lambdify(ddg37.T_SYM, list(dexpr), "numpy")
    assert np.allclose(np.stack(f(TT), -1), vel(a), atol=1e-12)
    sp_ = np.linalg.norm(vel(a), axis=1)
    assert abs(sp_.min() - abs(1 - a)) < 1e-6 and abs(TT[np.argmin(sp_)] - np.pi) < 1e-2
    lift = ddg37.angle_lift(vel(a))
    rot = (lift[-1] - lift[0]) / (2 * np.pi)
    wind = ddg37.winding_number_polygon(vel(a)[:-1], np.zeros(2))
    assert round(rot) == round(wind) == (2 if a < 1 else 1) and abs(rot - round(rot)) < 1e-9
assert np.allclose(vel(1.0, np.array([np.pi])), 0, atol=1e-15)

# ---------------------------------------------------------------- layout
CC, CS = np.array([-6.2, -0.35, 0.0]), 1.15              # polar origin of the curve -> CC
HC, HS = np.array([0.1, -1.2, 0.0]), 1.15                # origin of the velocity plane -> HC
RX = 2.85                                               # readout column


def cp(p):
    return CC + CS * np.array([p[0], p[1], 0.0])


def hp_(p):
    return HC + HS * np.array([p[0], p[1], 0.0])


def scene_state(a, t=None, turns=None, status=None, status_color=None):
    out = []
    X, W = curve(a), vel(a)
    out.append(hp.polyline([cp(p) for p in X], color=hp.BLACK, width=4))
    out.append(hp.polyline([hp_(p) for p in W], color=hp.C["tangent"], width=4))
    vmin = abs(1 - a)
    if vmin > 0.02:
        circ = np.stack([np.cos(TT), np.sin(TT)], 1) * vmin
        out.append(hp.DashedVMobject(hp.polyline([hp_(p) for p in circ], color=hp.HEX["aux"], width=2),
                                     num_dashes=max(8, int(40 * vmin))))
    cusp = abs(a - 1) < 1e-9
    out.append(hp.dot(HC, color=hp.C["normal"] if cusp else hp.BLACK, r=0.075))
    if cusp:
        out.append(hp.dot(cp(curve(a, np.array([np.pi]))[0]), color="#FFFFFF", r=0.09, stroke_color=hp.C["normal"],
                          stroke_width=3))
    if t is not None:
        p, v = curve(a, np.array([t]))[0], vel(a, np.array([t]))[0]
        out.append(hp.dot(cp(p), r=0.08))
        out.append(hp.arrow(cp(p), cp(p + 0.45 * v), color=hp.C["tangent"], width=5, tip=0.2))
        out.append(hp.arrow(HC, hp_(v), color=hp.C["tangent"], width=5, tip=0.2))
        out.append(hp.dot(hp_(v), color=hp.C["tangent"], r=0.07))
    rot = "2" if a < 1 - 1e-9 else ("1" if a > 1 + 1e-9 else "undefined")
    out.append(hp.text(f"a = {a:.2f}", 32).move_to([RX, 2.05, 0], aligned_edge=hp.LEFT))
    out.append(hp.text(f"min |γ′| = {vmin:.2f}", 32).move_to([RX, 1.35, 0], aligned_edge=hp.LEFT))
    out.append(hp.text(f"rot = {rot}", 32, color=hp.C["normal"] if cusp else hp.BLACK)
               .move_to([RX, 0.65, 0], aligned_edge=hp.LEFT))
    if turns is not None:
        out.append(hp.text(f"turns of γ′: {turns:.2f}", 32, color=hp.C["tangent"])
                   .move_to([RX, -0.3, 0], aligned_edge=hp.LEFT))
    if status:
        for i, line in enumerate(status):
            out.append(hp.text(line, 30, color=status_color or hp.BLACK)
                       .move_to([RX, -1.15 - 0.45 * i, 0], aligned_edge=hp.LEFT))
    return out


def run_once(a):
    """(t, accumulated turns of gamma') along one run."""
    lift = ddg37.angle_lift(vel(a))
    return TT, (lift - lift[0]) / (2 * np.pi)


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("turning number = turns of the velocity around 0", 32).move_to([0, 3.55, 0])
        lab_c = hp.mt(r"curve $\gamma_a$, $r = a + \cos t$", 30).move_to([-6.85, 2.85, 0], aligned_edge=hp.LEFT)
        lab_h = hp.mt(r"velocity $\gamma_a'(t)$, drawn from $0$", 30, color=hp.C["tangent"]).move_to([-2.1, 2.85, 0], aligned_edge=hp.LEFT)
        self.add(title, lab_c, lab_h)
        g = VGroup()
        self.add(g)
        S = hp.setter

        def run(a, dur, status):
            ts, turns = run_once(a)
            ev, n = [], 150
            for k in range(n + 1):
                i = int(round(k * (len(ts) - 1) / n))
                ev.append((dur * k / n, dur * k / n, (lambda st: (lambda u: S(g, st)(1)))(
                    scene_state(a, ts[i], turns[i], status))))
            return hp.timeline(g, ev, dur + 1.5)

        # part 1: one run at a = 0.5
        S(g, scene_state(0.5, 0.0, 0.0, ["one run of γ"]))(1)
        self.mark("start")
        self.wait(1.5)
        self.play(run(0.5, 5.0, ["one run of γ"]))
        self.mark("run 1 done")
        # part 2: a from 0.5 to 1.5, with a stop at the cusp
        ev, t = [], 0.0
        A1 = np.linspace(0.5, 1.0, 76)
        A2 = np.linspace(1.0, 1.5, 76)[1:]
        for a in A1[:-1]:
            ev.append((t, t, (lambda st: (lambda u: S(g, st)(1)))(scene_state(a, status=["regular: γ′ ≠ 0"]))))
            t += 1 / 15
        ev.append((t, t, (lambda st: (lambda u: S(g, st)(1)))(
            scene_state(1.0, status=["a = 1: γ′(π) = 0", "(a cusp)"], status_color=hp.C["normal"]))))
        t += 2.5
        for a in A2:
            ev.append((t, t, (lambda st: (lambda u: S(g, st)(1)))(scene_state(a, status=["regular again"]))))
            t += 1 / 15
        self.play(hp.timeline(g, ev, t + 1.0))
        self.mark("sweep done")
        # part 3: one run at a = 1.5
        self.play(run(1.5, 4.0, ["one run of γ"]))
        self.mark("run 2 done")
        self.wait(1.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=14.5)
