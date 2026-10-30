"""Animation 37.6.1: the same time step, explicit and semi-implicit (Section 37.6, Example 37.6.8, Proposition 37.6.5,
Proposition 37.6.7, Figure 37.6.3).

Data as in Figure 37.6.3: the flower curve r = 1 + 0.3 cos 5t sampled at 64 equal parameter steps, the lbar-weighted
length flow, h0 = 2 / lambda_max(M^{-1} K) = 0.004137 at t = 0, h = 0.9 h0, 54 steps (t = 0.201).
Left: explicit steps (ddg37.explicit_step); right: semi-implicit steps (37.6.5) (ddg37.semi_implicit_step); grey
dashed in both: the explicit reference with h0/40 at the same time (36 fine steps per coarse step).
Bottom: max_i |theta_i| (the zigzag measure of Figure 37.6.3(b)) against time.
Self-check (as the figure and Example 37.6.8): explicit max |theta| > 2 (2.159), the explicit length grows at exactly
5 steps, the semi-implicit length never grows, the reference stays smooth (max |theta| < 0.9 after t = 0).

Render: bash tools/render_anims.sh chapters/37-curves-smooth-discrete/anim/anim-37-6-1-explicit-implicit.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg37  # noqa: E402
import dganim as da  # noqa: E402
import dgdec as dd  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
X0, _ = ddg37.sample("flower", 64)
H0 = ddg37.explicit_critical_step(X0)
assert abs(H0 - 0.0041372) < 1e-6
NSTEP = 54
H = 0.9 * H0


def run(step, h, n, every=1):
    X = X0.copy()
    out = [X.copy()]
    for k in range(n):
        X = step(X, h)
        if (k + 1) % every == 0:
            out.append(X.copy())
    return out


EXP = run(ddg37.explicit_step, H, NSTEP)
SEMI = run(ddg37.semi_implicit_step, H, NSTEP)
REF = run(ddg37.explicit_step, H0 / 40, NSTEP * 36, every=36)
assert len(EXP) == len(SEMI) == len(REF) == NSTEP + 1


def stats(Xs):
    th = np.array([np.abs(dd.turning_angles(X, True)).max() for X in Xs])
    L = np.array([dd.curve_length(X, True) for X in Xs])
    return th, L


TH_E, L_E = stats(EXP)
TH_S, L_S = stats(SEMI)
TH_R, L_R = stats(REF)
UP_E = np.diff(L_E) > 0
assert TH_E.max() > 2.0 and round(TH_E.max(), 3) == 2.159 and UP_E.sum() == 5
assert (np.diff(L_S) <= 1e-14).all() and TH_S[1:].max() < 0.65 and TH_R[1:].max() < 0.9
# h0 is the limit of the initial polygon; one step later the shortest edges (in the waists between the petals) are
# shorter, 0.070 -> 0.052, and the limit is below h
assert ddg37.explicit_critical_step(EXP[1]) < H and round(ddg37.explicit_critical_step(EXP[1]), 4) == 0.0028
ELL = [np.linalg.norm(np.roll(X, -1, 0) - X, axis=1) for X in EXP[:2]]
assert round(ELL[0].min(), 3) == 0.070 and round(ELL[1].min(), 3) == 0.052
TIMES = H * np.arange(NSTEP + 1)
assert abs(TIMES[-1] - 0.201) < 1e-3

# ---------------------------------------------------------------- layout
CL, CR, SC = np.array([-3.0, 0.85, 0.0]), np.array([3.0, 0.85, 0.0]), 1.4    # the upper petal clears the labels
PLOT = hp.Plot(-4.6, -2.5, 10.8, 0.8, (0, 0.21), (0, 2.3))


def poly(X, c, color, width=3.5, dashed=False):
    P = [c + SC * np.array([x, y, 0.0]) for x, y in X]
    m = hp.polyline(P, color=color, width=width, closed=True)
    return hp.DashedVMobject(m, num_dashes=80) if dashed else m


def state(k, step_text=True):
    out = [poly(REF[k], CL, hp.HEX["aux"], 2.5, True), poly(REF[k], CR, hp.HEX["aux"], 2.5, True),
           poly(EXP[k], CL, hp.C["normal"]), poly(SEMI[k], CR, hp.C["tangent"])]
    out += [hp.dot(CL + SC * np.array([x, y, 0]), color=hp.C["normal"], r=0.03) for x, y in EXP[k]]
    out += [hp.dot(CR + SC * np.array([x, y, 0]), color=hp.C["tangent"], r=0.03) for x, y in SEMI[k]]
    for c, th, L, col in ((CL, TH_E[k], L_E[k], hp.C["normal"]), (CR, TH_S[k], L_S[k], hp.C["tangent"])):
        out.append(hp.text(f"L = {L:.3f}", 30, color=col).move_to(c + np.array([0, -2.2, 0])))
    if k > 0 and UP_E[k - 1]:
        out.append(hp.text("L went up!", 30, color=hp.C["normal"]).move_to(CL + np.array([2.4, -2.2, 0])))
    if step_text:
        out.append(hp.text(f"step {k}   t = {TIMES[k]:.3f}", 30).move_to([0, 2.9, 0]))
    if k > 0:
        out += [PLOT.curve(TIMES[:k + 1], TH_R[:k + 1], color=hp.HEX["aux"], width=3),
                PLOT.curve(TIMES[:k + 1], TH_E[:k + 1], color=hp.C["normal"], width=4),
                PLOT.curve(TIMES[:k + 1], TH_S[:k + 1], color=hp.C["tangent"], width=4)]
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.mt(r"explicit and semi-implicit steps with the same $h = 0.9\,h_0$", 32).move_to([0, 3.55, 0])
        lab_e = hp.text("explicit", 32, color=hp.C["normal"]).move_to([-6.0, 2.45, 0])
        lab_s = hp.text("semi-implicit", 32, color=hp.C["tangent"]).move_to([5.4, 2.45, 0])
        self.add(lab_e, lab_s)
        axes = PLOT.axes(yticks=[(0, "0"), (2, "2")], size=30)
        ylab = hp.text("max|\u03b8\u1d62|", 30).move_to([-5.85, -2.1, 0])
        xlab = hp.mt(r"$t$", 30).next_to(PLOT.p(0.21, 0), hp.RIGHT, buff=0.15)
        self.add(title, axes, ylab, xlab)
        g = VGroup()
        self.add(g)
        S = hp.setter
        S(g, state(0))(1)
        self.mark("start")
        self.wait(1.5)
        ev, t = [], 0.0
        for k in range(1, NSTEP + 1):
            ev.append((t, t, (lambda st: (lambda u: S(g, st)(1)))(state(k))))
            t += 0.3 if k not in (4, 21) else 1.2              # stop at the first jump and at the largest zigzag
        self.play(hp.timeline(g, ev, t))
        self.mark("flow done")
        end = hp.text("explicit: zigzag, L grew 5 times  |  semi-implicit: L only falls", 30).move_to([0, 2.9, 0])
        self.play(hp.timeline(g, [(0, 0, S(g, state(NSTEP, False) + [end])), (0, 0.6, hp.fade([end]))], 1.0))
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.0)
