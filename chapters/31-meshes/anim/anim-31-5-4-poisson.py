"""Animation 31.5.4: Poisson reconstruction integrates a normal field; gaps are bridged, flipped normals make false bumps.

Part 1 (1D, Figure 31.5.3(a)): chi = the indicator of [-0.5, 0.6] smoothed (tanh profile, width 0.04 * 0.8), V = chi'
(grey, scaled): a peak at x = -0.5 (inward normal +x) and a dip at x = 0.6 (inward normal -x). A cursor sweeps x from -1
to 1 and the running integral of V (orange) is drawn; it ends on chi.
Part 2 (2D, Figure 31.5.3(b), (c), same solver): the closed curve r(theta) = 1 + 0.15 cos 3 theta + 0.1 sin 2 theta, 160
samples equally spaced in theta, inward normals times the arc length per sample, smoothed by a Gaussian of 2.5 cells,
Delta chi = div V solved by FFT on the periodic 256 x 256 grid over [-1.6, 1.6)^2, surface = {chi = mean of chi at the
samples} [Kazhdan06].
  left:  samples with |theta - 0.8| < g removed, g = 0 ... 0.4 (g = 0.4 is the gap of the figure);
  right: normals with |theta - 2.85| < f flipped, f = 0 ... 0.45 (f = 0.45 flips the figure's 23 normals).
Background grey = chi (darker = larger), dashed = true curve, orange = reconstruction, arrows = every second normal
(vermillion = flipped). Readout: IoU with the true region.
Self-check: the running integral returns to chi (max error < 0.01); IoU 0.999 with all samples, 0.995 at g = 0.4 and
0.853 at f = 0.45 (as in the figure); 20 samples removed and 23 normals flipped at the ends of the sweeps.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-5-4-poisson.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

import contourpy  # noqa: E402

# ---------------------------------------------------------------- 1D
X1 = np.linspace(-1, 1, 2001)
H1 = X1[1] - X1[0]
S1 = 0.04
CHI1 = 0.5 * (np.tanh((X1 + 0.5) / (S1 * 0.8)) - np.tanh((X1 - 0.6) / (S1 * 0.8)))
V1 = np.gradient(CHI1, H1)
REC1 = np.cumsum(V1) * H1
assert np.abs(REC1 - CHI1).max() < 1e-2

# ---------------------------------------------------------------- 2D (as fig-31-5-3-poisson.py)
N, LB = 256, 1.6
XS = (np.arange(N) / N) * 2 * LB - LB
DX = XS[1] - XS[0]
GX, GY = np.meshgrid(XS, XS, indexing="ij")
TH = np.linspace(0, 2 * np.pi, 160, endpoint=False)


def curve(th):
    r = 1 + 0.15 * np.cos(3 * th) + 0.1 * np.sin(2 * th)
    dr = -0.45 * np.sin(3 * th) + 0.2 * np.cos(2 * th)
    p = np.stack([r * np.cos(th), r * np.sin(th)], -1)
    t = np.stack([dr * np.cos(th) - r * np.sin(th), dr * np.sin(th) + r * np.cos(th)], -1)
    return p, t


def poisson(th, flip=None, sigma=2.5 * DX):
    p, t = curve(th)
    ds = np.linalg.norm(t, axis=1) * (2 * np.pi / len(TH))
    tn = t / np.linalg.norm(t, axis=1, keepdims=True)
    n_in = np.stack([-tn[:, 1], tn[:, 0]], -1)
    if flip is not None:
        n_in[flip] *= -1
    Vx, Vy = np.zeros((N, N)), np.zeros((N, N))
    for q, n, w in zip(p, n_in, ds):
        g = np.exp(-((GX - q[0]) ** 2 + (GY - q[1]) ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        Vx += w * n[0] * g
        Vy += w * n[1] * g
    k = 2 * np.pi * np.fft.fftfreq(N, d=DX)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    chi_hat = -(1j * KX * np.fft.fft2(Vx) + 1j * KY * np.fft.fft2(Vy)) / K2
    chi_hat[0, 0] = 0.0
    chi = np.real(np.fft.ifft2(chi_hat))
    ii = np.clip(np.round((p[:, 0] + LB) / DX).astype(int), 0, N - 1)
    jj = np.clip(np.round((p[:, 1] + LB) / DX).astype(int), 0, N - 1)
    return chi, chi[ii, jj].mean(), p, n_in


ANG = np.arctan2(GY, GX)
TRUE_IN = np.hypot(GX, GY) < 1 + 0.15 * np.cos(3 * ANG) + 0.1 * np.sin(2 * ANG)


def iou(chi, iso):
    inside = chi > iso
    return (inside & TRUE_IN).sum() / (inside | TRUE_IN).sum()


def iso_lines(chi, iso):
    gen = contourpy.contour_generator(x=XS, y=XS, z=chi.T)
    return [ln for ln in gen.lines(iso) if len(ln) > 3]


GAPS = np.linspace(0, 0.4, 9)
FLIPS = np.linspace(0, 0.45, 10)
GAP_STATES, FLIP_STATES = [], []
for g in GAPS:
    keep = np.abs(TH - 0.8) >= g if g > 0 else np.ones(len(TH), bool)
    chi, iso, p, n = poisson(TH[keep])
    GAP_STATES.append((chi, iso, p, n, np.zeros(len(p), bool), iou(chi, iso), (~keep).sum()))
for f in FLIPS:
    fl = np.abs(TH - 2.85) < f if f > 0 else np.zeros(len(TH), bool)
    chi, iso, p, n = poisson(TH, fl)
    FLIP_STATES.append((chi, iso, p, n, fl, iou(chi, iso), fl.sum()))
assert round(GAP_STATES[0][5], 3) == 0.999 and round(FLIP_STATES[0][5], 3) == 0.999
assert round(GAP_STATES[-1][5], 3) == 0.995 and GAP_STATES[-1][6] == 20
assert round(FLIP_STATES[-1][5], 3) == 0.853 and FLIP_STATES[-1][6] == 23
fig_gap = (TH < 0.4) | (TH > 1.2)
fig_flip = (TH > 2.4) & (TH < 3.3)
assert np.array_equal(np.abs(TH - 0.8) >= 0.4, fig_gap) and np.array_equal(np.abs(TH - 2.85) < 0.45, fig_flip)
PT, _ = curve(np.linspace(0, 2 * np.pi, 400))


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        # ---------------------------------------------------------- part 1: 1D
        ttl = title("Poisson reconstruction: integrate the normals", 34)
        P = Plot(-5.8, -2.55, 11.6, 3.7, (-1, 1), (-1.15, 1.25))
        axes = P.axes(xticks=[(-1, "-1"), (-0.5, "-0.5"), (0, "0"), (0.6, "0.6"), (1, "1")], yticks=[(0, "0"), (1, "1")])
        axes.add(mt(r"$x$", 32).next_to(P.p(1, -1.15), RIGHT, buff=0.15))
        zero = Line(P.p(-1, 0), P.p(1, 0), stroke_width=1.2, color=C["aux"])
        vcurve = P.curve(X1, V1 / np.abs(V1).max() * 0.9, color=C["aux"], width=3.5)
        chi_curve = DashedVMobject(P.curve(X1, CHI1, color=BLACK, width=3), num_dashes=70)
        a1 = Arrow(P.p(-0.5, -0.45), P.p(-0.32, -0.45), buff=0, stroke_width=5, color=C["tangent"], tip_length=0.2)
        a2 = Arrow(P.p(0.6, -0.45), P.p(0.42, -0.45), buff=0, stroke_width=5, color=C["normal"], tip_length=0.2)
        l1 = mt(r"inward normal $+x$", 30, color=C["tangent"]).next_to(a1, DOWN, buff=0.12).shift(LEFT * 0.9)
        l2 = mt(r"inward normal $-x$", 30, color=C["normal"]).next_to(a2, DOWN, buff=0.12)
        l2.shift(RIGHT * (P.X(0.545) - l2.get_right()[0]))     # keep the label left of the dip of V at x = 0.6
        vlab = mt(r"grey: $V = \chi'$ (scaled), nonzero only on the boundary", 30).move_to([-5.8, 2.75, 0], aligned_edge=LEFT)
        self.add(ttl, axes, zero, vcurve, vlab)
        self.play(GrowArrow(a1), GrowArrow(a2), FadeIn(l1), FadeIn(l2), run_time=0.8)
        self.wait(1.4)
        xc = ValueTracker(-1.0)

        def run():
            x = xc.get_value()
            n = max(2, int(np.searchsorted(X1, x)) + 1)
            g = VGroup(P.curve(X1[:n], REC1[:n], color=C["accent"], width=6))
            g.add(DashedLine(P.p(x, -1.15), P.p(x, 1.25), stroke_width=1.5, color=C["aux"]))
            return g

        rr = always_redraw(run)
        rlab = mt(r"orange: the running integral of $V$ from $-1$ to $x$", 30, color=C["accent"]).move_to([-5.8, 2.27, 0], aligned_edge=LEFT)
        self.add(rr)
        self.play(FadeIn(rlab), run_time=0.4)
        self.play(xc.animate.set_value(1.0), run_time=4.5, rate_func=linear)
        clab = mt(r"dashed: the indicator $\chi$ of $[-0.5,\ 0.6]$ (smoothed)", 30).move_to([-5.8, 1.79, 0], aligned_edge=LEFT)
        self.play(Create(chi_curve), FadeIn(clab), run_time=0.9)
        self.wait(1.6)
        part1 = VGroup(axes, zero, vcurve, vlab, a1, a2, l1, l2, rr, rlab, chi_curve, clab)
        rr.clear_updaters()

        # ---------------------------------------------------------- part 2: 2D
        SCALE = 1.42
        CL, CR = np.array([-3.45, -0.3]), np.array([3.45, -0.3])

        def w2s(c, p):
            p = np.asarray(p, float)
            return np.array([c[0] + SCALE * p[0], c[1] + SCALE * p[1], 0.0])

        def panel(c, state):
            chi, iso, p, n, fl, io, cnt = state
            lo, hi = chi.min(), chi.max()
            gval = 1 - 0.45 * np.clip((chi.T[::-1] - lo) / (hi - lo), 0, 1)
            img = np.dstack([gval] * 3 + [np.ones_like(gval)])
            im = ImageMobject((img * 255).astype(np.uint8), scale_to_resolution=720)
            im.set_resampling_algorithm(RESAMPLING_ALGORITHMS["bilinear"])
            im.height = SCALE * 2 * LB
            im.move_to(w2s(c, (-DX / 2, -DX / 2)))
            vec = VGroup()
            for ln in iso_lines(chi, iso):
                vec.add(polyline([w2s(c, q) for q in np.vstack([ln, ln[:1]])], color=C["accent"], width=5))
            for k in range(0, len(p), 2):
                col = C["normal"] if fl[k] else C["tangent"]
                vec.add(Arrow(w2s(c, p[k]), w2s(c, p[k] + 0.18 * n[k]), buff=0, stroke_width=2.5, color=col,
                              tip_length=0.08, max_tip_length_to_length_ratio=0.4,
                              max_stroke_width_to_length_ratio=100))
            for q in p:
                vec.add(Dot(w2s(c, q), radius=0.025, color=BLACK))
            return Group(im, vec)

        def frame_for(c):
            # a white frame hides the part of the chi image outside the window [-1.45, 1.45]^2
            out = VGroup()
            half = SCALE * 1.45
            big = SCALE * LB + 0.3
            for dx, dy, w, h in ((0, half + (big - half) / 2, 2 * big, big - half), (0, -half - (big - half) / 2, 2 * big, big - half),
                                 (half + (big - half) / 2, 0, big - half, 2 * big), (-half - (big - half) / 2, 0, big - half, 2 * big)):
                out.add(Rectangle(width=w, height=h, stroke_width=0, fill_color=WHITE, fill_opacity=1).move_to([c[0] + dx, c[1] + dy, 0]))
            out.add(Square(2 * half, stroke_width=1.2, color=GREY_B).move_to([c[0], c[1], 0]))
            return out

        true_l = DashedVMobject(polyline([w2s(CL, q) for q in np.vstack([PT, PT[:1]])], color=BLACK, width=2.5), num_dashes=80)
        true_r = DashedVMobject(polyline([w2s(CR, q) for q in np.vstack([PT, PT[:1]])], color=BLACK, width=2.5), num_dashes=80)
        for t_ in (true_l, true_r):
            t_.set_z_index(4)
        fl_, fr_ = frame_for(CL), frame_for(CR)
        fl_.set_z_index(2)
        fr_.set_z_index(2)
        tl = da.label("(a) remove samples (a gap)", 30).move_to([CL[0], 2.15, 0]).set_z_index(6)
        tr = da.label("(b) flip some normals", 30).move_to([CR[0], 2.15, 0]).set_z_index(6)

        def io_text(c, state, what):
            chi, iso, p, n, fl, io, cnt = state
            return mt(rf"{what} $= {cnt}$:  IoU $= {io:.3f}$", 30).move_to([c[0], -2.72, 0]).set_z_index(5)

        pl = panel(CL, GAP_STATES[0])
        pr = panel(CR, FLIP_STATES[0])
        il = io_text(CL, GAP_STATES[0], "removed")
        ir = io_text(CR, FLIP_STATES[0], "flipped")
        ttl2 = title("2D: gaps are bridged, flipped normals are not", 34)
        leg = da.label("dashed: true curve   orange: reconstruction", 30).move_to([0, 2.75, 0])
        self.play(FadeOut(part1), FadeTransform(ttl, ttl2), run_time=0.8)
        self.play(FadeIn(pl), FadeIn(pr), FadeIn(VGroup(true_l, true_r, fl_, fr_, tl, tr, il, ir, leg)), run_time=1.0)
        self.wait(1.6)
        for st in GAP_STATES[1:]:
            new, ni = panel(CL, st), io_text(CL, st, "removed")
            self.remove(pl, il)
            pl, il = new, ni
            self.add(pl, il)
            self.wait(0.5)
        self.wait(1.5)
        for st in FLIP_STATES[1:]:
            new, ni = panel(CR, st), io_text(CR, st, "flipped")
            self.remove(pr, ir)
            pr, ir = new, ni
            self.add(pr, ir)
            self.wait(0.55)
        self.wait(2.4)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=24.0)
