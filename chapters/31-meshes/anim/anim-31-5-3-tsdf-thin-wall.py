"""Animation 31.5.3: TSDF fusion makes a wall thinner than the truncation distance thicker (1D model, Section 31.5).

Model (the insight box "truncation보다 얇은 벽은 두꺼워진다"): a wall [0, w] seen once from each side, truncation
tau = 0.02 (2DGS uses voxel 0.004, tau = 0.02 [Huang24]). Each camera updates only the voxels with signed distance
d > -tau along its ray and stores min(1, d / tau) with weight 1:
  left camera  (surface at x = 0): value -x / tau for x < tau,
  right camera (surface at x = w): value (x - w) / tau for x > w - tau;
the fused value is the average of the values stored at x. Its sign changes give the reconstructed wall.
The wall thickness w sweeps from 0.035 down to 0.003 (holds at 0.03, 0.02, 0.01, 0.005).
Left: the two cameras' values (blue, green), the average (black), the true wall (grey band) and the reconstructed
wall (orange bar). Right: reconstructed thickness against w; the dashed line is the true thickness.
Self-check: the sign changes of the average, found numerically on a fine grid, are at 0 and w for w >= tau and at
w - tau and tau for w < tau, so the reconstructed thickness is w (w >= tau) and 2 tau - w (w < tau): 0.03 -> 0.03,
0.01 -> 0.03, 0.005 -> 0.035, and w -> 0 gives 2 tau = 0.04 (w = 1e-4: 0.0399).

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-5-3-tsdf-thin-wall.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

TAU = 0.02
XG = np.linspace(-0.045, 0.075, 24001)


def cams(w, x=XG):
    left = np.where(x < TAU, np.minimum(1.0, -x / TAU), np.nan)
    right = np.where(x > w - TAU, np.minimum(1.0, (x - w) / TAU), np.nan)
    both = np.nanmean(np.stack([left, right]), axis=0)
    return left, right, both


def recon(w):
    """Reconstructed wall: the interval where the fused value is negative (numerically, on XG)."""
    _, _, avg = cams(w)
    neg = np.where(avg < 0)[0]
    return XG[neg[0]], XG[neg[-1]]


def thickness_model(w):
    return w if w >= TAU else 2 * TAU - w


import warnings  # noqa: E402

warnings.simplefilter("ignore", RuntimeWarning)          # nanmean of an all-NaN column (no camera) is NaN
for w, want in ((0.03, 0.03), (0.02, 0.02), (0.01, 0.03), (0.005, 0.035), (1e-4, 0.0399)):
    a, b = recon(w)
    assert abs((b - a) - want) < 2e-5, (w, a, b)
    assert abs((b - a) - thickness_model(w)) < 2e-5
    if w >= TAU:
        assert abs(a) < 1e-5 and abs(b - w) < 1e-5
    else:
        assert abs(a - (w - TAU)) < 1e-5 and abs(b - TAU) < 1e-5
WS = np.linspace(0.003, 0.035, 400)


class Anim(da.Scene2D):
    def construct(self):
        P = Plot(-6.2, -1.75, 8.0, 3.75, (-0.045, 0.075), (-1.15, 1.15))
        axes = P.axes(xticks=[(v, f"{v:g}") for v in (-0.04, -0.02, 0, 0.02, 0.04, 0.06)],
                      yticks=[(-1, "-1"), (0, "0"), (1, "1")])
        axes.add(mt(r"$x$", 32).next_to(P.p(0.075, -1.15), RIGHT, buff=0.15))
        zero = DashedLine(P.p(-0.045, 0), P.p(0.075, 0), stroke_width=1.2, color=C["aux"])
        Q = Plot(3.35, -1.75, 3.3, 3.3, (0, 0.036), (0, 0.045))
        qaxes = Q.axes(xticks=[(0, "0"), (0.02, "0.02")], yticks=[(0, "0"), (0.02, "0.02"), (0.04, "0.04")],
                       xlabel=mt(r"true thickness $w$", 30))
        qlab = da.label("reconstructed", 30).next_to(Q.p(0, 0.045), UP, buff=0.15).shift(RIGHT * 0.9)
        ident = DashedLine(Q.p(0, 0), Q.p(0.036, 0.036), stroke_width=1.8, color=C["aux"])
        tau_line = DashedLine(Q.p(TAU, 0), Q.p(TAU, 0.045), stroke_width=1.2, color=C["aux"])
        tau_lab = mt(r"$\tau$", 30).next_to(Q.p(TAU, 0.043), RIGHT, buff=0.1)
        model = Q.curve(WS, [thickness_model(w) for w in WS], color=C["accent"], width=4)
        ttl = mt(r"TSDF fusion of a thin wall (truncation $\tau = 0.02$)", 34).to_edge(UP, buff=0.3)
        leg = VGroup(mt(r"left camera: $-x/\tau$", 30, color=C["tangent"]),
                     mt(r"right camera: $(x - w)/\tau$", 30, color=C["third"]),
                     da.label("fused average", 30))
        leg.arrange(RIGHT, buff=0.45).move_to([-0.2, 2.95, 0])
        leg2 = VGroup(da.label("grey: the true wall [0, w]", 30),
                      da.label("orange: the reconstructed wall", 30, color=C["accent"]))
        leg2.arrange(RIGHT, buff=0.45).move_to([-0.2, 2.47, 0])
        wv = ValueTracker(0.035)

        def scene_state():
            w = wv.get_value()
            left, right, avg = cams(w)
            g = VGroup()
            g.add(Rectangle(width=P.X(w) - P.X(0), height=P.h, stroke_width=0, fill_color=GREY_B,
                            fill_opacity=0.55).move_to([(P.X(0) + P.X(w)) / 2, P.y0 + P.h / 2, 0]))
            for arr, col, wd in ((left, C["tangent"], 4), (right, C["third"], 4)):
                ok = ~np.isnan(arr)
                g.add(P.curve(XG[ok], arr[ok], color=col, width=wd))
            # the fused curve is drawn piecewise (it jumps where the second camera starts or stops)
            ok = ~np.isnan(avg)
            idx = np.where(ok)[0]
            jumps = np.where(np.abs(np.diff(avg[idx])) > 0.05)[0]
            pieces = np.split(idx, jumps + 1)
            for pc in pieces:
                if len(pc) > 1:
                    g.add(P.curve(XG[pc], avg[pc], color=BLACK, width=6))
            a, b = recon(w)
            g.add(Line(P.p(a, 0), P.p(b, 0), stroke_width=12, color=C["accent"]).set_z_index(3))
            g.add(Dot(Q.p(w, b - a), radius=0.09, color=C["accent"], stroke_color=BLACK, stroke_width=1.2))
            return g

        dyn = always_redraw(scene_state)

        def read():
            w = wv.get_value()
            a, b = recon(w)
            return f"w = {w:.3f}   reconstructed {b - a:.3f}"

        rd = readout(read, [-6.2, -2.75, 0], size=32)
        self.add(axes, zero, qaxes, qlab, ident, tau_line, tau_lab, model, ttl, leg, leg2, dyn, rd)
        self.wait(1.8)
        for w, rt in ((0.03, 1.0), (0.02, 1.6), (0.01, 1.8), (0.005, 1.0), (0.003, 0.6)):
            self.play(wv.animate.set_value(w), run_time=rt, rate_func=smooth)
            self.wait(1.8 if w != 0.003 else 2.2)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=10.0)
