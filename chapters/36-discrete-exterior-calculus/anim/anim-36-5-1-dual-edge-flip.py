"""Animation 36.5.1: the circumcentric dual edge shrinks to 0 and turns over when the Delaunay condition fails
(Section 36.5, Lemma 36.5.9, Theorem 36.5.10, Example 36.5.12, Figure 36.5.3(b)).

Two counterclockwise triangles on the shared edge e = [x_0, x_1], x_0 = (0, 0), x_1 = (2, 0): f = (0, 1, 2) above with
x_2 = (1, h_2), g = (1, 0, 3) below with x_3 = (1, -h_3). The dual edge e* runs from c_g to c_f (Remark 36.5.1); its
signed length is h_f + h_g with h = (|e|/2) cot(angle opposite e) (Lemma 36.5.9), so star_1 = |e*|/|e| =
(cot alpha + cot beta)/2 (Theorem 36.5.10), alpha at x_2 and beta at x_3.
Part 1: h_3 = 0.8 fixed, h_2 from 1.6 down to 0.3. star_1 passes 0 at h_2 = 1.25 exactly, when x_2 lies on the
circumcircle of g (center (1, 0.225), radius 1.025) and alpha + beta = 180 degrees; then c_f is below c_g.
Part 2: h_2 = 0.3 fixed, h_3 from 0.8 to 0.25: the pair of Example 36.5.12, star_1 = -1.696, |e*| = -3.392.
Self-check: circumcenters from dgdec.circumcenters lie on x = 1 at heights cot(alpha) and -cot(beta);
dgdec.hodge1 of the shared edge equals (cot alpha + cot beta)/2 along the whole path; zero exactly at h_2 = 1.25;
the end values are those of Example 36.5.12 (lib/ch36.non_delaunay_pair).

Render: bash tools/render_anims.sh chapters/36-discrete-exterior-calculus/anim/anim-36-5-1-dual-edge-flip.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ch36  # noqa: E402
import dganim as da  # noqa: E402
import dgdec as dd  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
FF = np.array([[0, 1, 2], [1, 0, 3]])
E, EIDX = dd.edge_list(FF)
E01 = EIDX[(0, 1)]


def config_of(h2, h3):
    V = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, h2], [1.0, -h3]])
    cot_a, cot_b = (h2 ** 2 - 1) / (2 * h2), (h3 ** 2 - 1) / (2 * h3)
    C = dd.circumcenters(V, FF)
    star1 = dd.hodge1(V, FF)[E01]
    assert np.allclose(C[:, 0], 1.0) and np.isclose(C[0, 1], cot_a) and np.isclose(C[1, 1], -cot_b)
    assert np.isclose(star1, 0.5 * (cot_a + cot_b))
    alpha, beta = 2 * np.arctan(1 / h2), 2 * np.arctan(1 / h3)
    return dict(V=V, cf=C[0], cg=C[1], star1=star1, alpha=alpha, beta=beta)


H2A, H2B, H3A, H3B = 1.6, 0.3, 0.8, 0.25
H2_ZERO = 1.25
z = config_of(H2_ZERO, H3A)
assert abs(z["star1"]) < 1e-12 and np.isclose(np.degrees(z["alpha"] + z["beta"]), 180.0)
CIRC_G = (np.array([1.0, 0.225]), 1.025)              # circumcircle of g in part 1
assert np.isclose(np.linalg.norm(np.array([1.0, H2_ZERO]) - CIRC_G[0]), CIRC_G[1])
start, mid, end = config_of(H2A, H3A), config_of(H2B, H3A), config_of(H2B, H3B)
Vn, Fn = ch36.non_delaunay_pair()
assert np.allclose(end["V"], Vn) and (Fn == FF).all()
assert np.isclose(end["star1"], -1.6958, atol=1e-4) and np.isclose(2 * end["star1"], -3.392, atol=1e-3)
assert start["star1"] > 0 and mid["star1"] < 0
# the path: part 1 (h2 down), part 2 (h3 up); h2 moves linearly in time
PATH1 = [config_of(h, H3A) for h in np.linspace(H2A, H2B, 241)]
PATH2 = [config_of(H2B, h) for h in np.linspace(H3A, H3B, 161)]
S1 = np.array([c["star1"] for c in PATH1 + PATH2])
assert (np.diff(S1) <= 1e-15).all()                    # star_1 decreases monotonically along the path
SUM = np.degrees([c["alpha"] + c["beta"] for c in PATH1 + PATH2])

# ---------------------------------------------------------------- layout
O, SC = np.array([-5.25, 0.3, 0.0]), 1.45              # x_0 -> O; scene = O + SC * (x, y)


def P(p):
    return O + SC * np.array([p[0], p[1], 0.0])


PX = 0.55
PLOT = hp.Plot(1.75, -2.3, 4.6, 1.75, (160, 300), (-1.8, 0.4))


def arc_pts(c, r, a0, a1, n=40):
    t = np.linspace(a0, a1, n)
    return np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)


def draw(cfg, show_circle):
    V = cfg["V"]
    out = []
    out += [hp.poly_fill([P(V[i]) for i in FF[0]], hp.C["region"], 0.3, 1.5, hp.HEX["aux"]),
            hp.poly_fill([P(V[i]) for i in FF[1]], hp.C["region"], 0.3, 1.5, hp.HEX["aux"])]
    out.append(hp.seg(P(V[0]), P(V[1]), color=hp.C["tangent"], width=6))
    if show_circle:
        c, r = CIRC_G
        pts = arc_pts(c, r, 0, 2 * np.pi, 120)
        out.append(hp.DashedVMobject(hp.polyline([P(q) for q in pts], color=hp.HEX["aux"], width=2.5),
                                     num_dashes=60))
    # angles alpha at x_2 and beta at x_3
    for k, name in ((2, r"$\alpha$"), (3, r"$\beta$")):
        p = V[k]
        a0, a1 = sorted([np.arctan2(*(V[0] - p)[::-1]), np.arctan2(*(V[1] - p)[::-1])])
        if a1 - a0 > np.pi:
            a0, a1 = a1, a0 + 2 * np.pi
        rr = min(0.3, 0.55 * abs(p[1]))                 # the two arcs must not cross on a flat pair
        out.append(hp.polyline([P(q) for q in arc_pts(p, rr, a0, a1)], color=hp.C["normal"], width=3))
        mdir = np.array([np.cos(0.5 * (a0 + a1)), np.sin(0.5 * (a0 + a1))])
        h = abs(p[1])
        # off the line x = 1 of the dual edge: alpha to the left (c_g's label is on the right), beta to the right
        side = np.array([-0.22 if k == 2 else 0.22, 0.0])
        pos = p + 0.62 * mdir + side if h >= 0.75 else p + np.array([-0.45, 0.3 if k == 2 else -0.3])
        out.append(hp.mt(name, 32, color=hp.C["normal"]).move_to(P(pos)))
    # reference direction J t and the dual edge c_g -> c_f
    m = np.array([1.0, 0.0])
    out.append(hp.DashedVMobject(hp.arrow(P(m + np.array([0.45, -0.55])), P(m + np.array([0.45, 0.55])),
                                          color=hp.HEX["aux"], width=3, tip=0.18), num_dashes=8))
    cf, cg = cfg["cf"], cfg["cg"]
    if abs(cf[1] - cg[1]) > 0.05:
        out.append(hp.arrow(P(cg), P(cf), color=hp.C["normal"], width=6, tip=0.22))
    out += [hp.dot(P(cf), color=hp.C["accent"], r=0.085), hp.dot(P(cg), color=hp.C["accent"], r=0.085)]
    out.append(hp.mt(r"$c_f$", 30, color=hp.C["accent"]).next_to(P(cf), hp.LEFT, buff=0.15))
    out.append(hp.mt(r"$c_g$", 30, color=hp.C["accent"]).next_to(P(cg), hp.RIGHT, buff=0.15))
    # x_2 and x_3 sit on the line x = 1 that the dual edge runs along: label them up-left / down-left of the vertex
    for k, name, d, b in ((0, r"$x_0$", hp.LEFT, 0.12), (1, r"$x_1$", hp.RIGHT, 0.12),
                          (2, r"$x_2$", hp.UP + hp.LEFT, 0.12), (3, r"$x_3$", hp.DOWN + hp.LEFT, 0.12)):
        out.append(hp.mt(name, 30).next_to(P(V[k]), d, buff=b))
    return out


EQ = hp.mt(r"$\star_1 = \frac{|e^\star|}{|e|} = \frac{1}{2}(\cot\alpha + \cot\beta)$", 34).move_to([PX, 2.05, 0], aligned_edge=hp.LEFT)
EQ2 = hp.mt(r"$\star_1\ =$", 34).move_to([PX, 1.35, 0], aligned_edge=hp.LEFT)
EQ_RIGHT = EQ2.get_right()


def readouts(cfg, k_path):
    s = cfg["star1"]
    ssum = np.degrees(cfg["alpha"] + cfg["beta"])
    col = hp.C["tangent"] if s > 1e-9 else (hp.C["aux"] if abs(s) < 1e-9 else hp.C["normal"])
    out = [hp.text(f"α + β = {ssum:.1f}°", 32).move_to([PX, 2.75, 0], aligned_edge=hp.LEFT),
           hp.text(hp.num(s, 3), 36, color=col).next_to(EQ_RIGHT, hp.RIGHT, buff=0.2)]
    xs, ys = SUM[:k_path + 1], S1[:k_path + 1]
    if len(xs) > 1:
        out.append(PLOT.curve(xs, ys, color=hp.C["normal"], width=5))
    out.append(hp.dot(PLOT.p(SUM[k_path], S1[k_path]), color=hp.C["normal"], r=0.08))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("non-Delaunay edge: the dual edge turns over", 32).move_to([0, 3.55, 0])
        axes = PLOT.axes(xticks=[(180, "180\u00b0"), (240, "240\u00b0"), (300, "300\u00b0")],
                         yticks=[(0, "0"), (-1.5, "\u22121.5")], size=30, zero_line=0.0)
        ylab = hp.mt(r"$\star_1$", 32).next_to(PLOT.p(160, 0.4), hp.RIGHT, buff=0.15)
        xlab = hp.mt(r"$\alpha + \beta$", 30).move_to(PLOT.p(285, 0.15))
        v180 = hp.seg(PLOT.p(180, -1.8), PLOT.p(180, 0.4), color=hp.HEX["aux"], width=1.5, dashed=True)
        key = VGroup(hp.text("red: e* from c_g to c_f", 30, color=hp.C["normal"]),
                     hp.text("dashed: Jt (e turned +90\u00b0)", 30, color=hp.C["aux"]))
        key.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.08).move_to([PX, 0.05, 0], aligned_edge=hp.LEFT)
        self.add(title, EQ, EQ2, axes, ylab, xlab, v180, key)
        geo, rd, note = VGroup(), VGroup(), VGroup()
        world = VGroup(geo, rd, note)
        self.add(world)
        S = hp.setter

        def show(k, cfg, circle):
            def fn(u):
                S(geo, draw(cfg, circle))(1)
                S(rd, readouts(cfg, k))(1)
            return fn

        show(0, PATH1[0], True)(1)
        n1 = hp.text("Delaunay: e* along Jt", 30, color=hp.C["tangent"])
        n1.move_to([PX, 0.75, 0], aligned_edge=hp.LEFT)
        S(note, [n1])(1)
        self.mark("start")
        self.wait(2.5)
        n3 = hp.text("non-Delaunay: e* against Jt", 30, color=hp.C["normal"])
        n3.move_to([PX, 0.75, 0], aligned_edge=hp.LEFT)
        # part 1: x_2 comes down; pause where it meets the circle
        k0 = int(np.argmin(np.abs(np.array([c["star1"] for c in PATH1]))))
        assert abs(PATH1[k0]["star1"]) < 2e-3
        ev, t, dt = [], 0.0, 1 / 30
        for k in range(len(PATH1)):
            ev.append((t, t, show(k, PATH1[k], True)))
            t += dt * 0.5
            if k == k0:
                n2 = hp.text("x\u2082 on the circle of g: \u22c6\u2081 = 0", 30, color=hp.HEX["aux"])
                n2.move_to([PX, 0.75, 0], aligned_edge=hp.LEFT)
                ev.append((t, t, show(k, z, True)))
                ev.append((t, t, S(note, [n2])))
                t += 2.0
                ev.append((t, t, S(note, [n3])))
        self.play(hp.timeline(world, ev, t + 1.0))
        self.mark("part 1 done")
        # part 2: x_3 comes up to the pair of Example 36.5.12
        ev, t = [], 0.0
        for k in range(len(PATH2)):
            ev.append((t, t, show(len(PATH1) + k, PATH2[k], False)))
            t += dt * 0.6
        n4 = hp.text("Example 36.5.12: ⋆₁ = −1.696", 30, color=hp.C["normal"])
        n4.move_to([PX, 0.75, 0], aligned_edge=hp.LEFT)
        ev.append((t, t, S(note, [n4])))
        self.play(hp.timeline(world, ev, t + 0.3))
        self.mark("part 2 done")
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=15.0)
