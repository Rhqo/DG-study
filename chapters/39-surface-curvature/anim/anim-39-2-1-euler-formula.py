"""Animation 39.2.1: turning the direction at the saddle point: the normal curvature follows Euler's formula
(Section 39.2, Proposition 39.2.5, (39.2.4), Example 39.2.8(d)).

Surface: the saddle z = x^2 - y^2 (graph parametrization, N up at the origin), drawn over the disk r <= 1. At the origin
the principal curvatures are kappa_1 = 2 (x direction) and kappa_2 = -2 (y direction) (Example 39.2.8(d)).
For the unit tangent X = (cos v, sin v, 0) the normal plane spanned by X and N cuts the surface in the curve
s -> (s cos v, s sin v, s^2 cos 2v), whose curvature at s = 0 (sign from N) is
kappa_n = 2 cos 2v = kappa_1 cos^2 v + kappa_2 sin^2 v (39.2.4). Drawn: the normal plane (light blue), the normal
section (orange), its osculating circle (grey dashed, radius 1/|kappa_n|, centre at height 1/kappa_n on N) when
|kappa_n| > 0.35, X (blue) and N (vermilion). v runs from 0 to 180 degrees with stops at 0, 45 and 90 degrees.
Right: kappa_n(v) traced, with kappa_1 and kappa_2 marked.
Self-check: ddg39.weingarten at the origin is diag(2, -2) (central differences, 1e-6); kappa_n from the Rayleigh quotient
of W equals 2 cos 2v; the section curve lies on the surface and its numerical curvature at 0 equals kappa_n.

Render: bash tools/render_anims.sh chapters/39-surface-curvature/anim/anim-39-2-1-euler-formula.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg39  # noqa: E402
import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
W0 = np.asarray(ddg39.weingarten(ddg39.saddle, 0.0, 0.0), float)
assert np.allclose(W0, np.diag([2.0, -2.0]), atol=1e-5)
for v in np.linspace(0, np.pi, 13):
    X = np.array([np.cos(v), np.sin(v)])
    kn = X @ W0 @ X                               # g = identity at the origin
    assert abs(kn - 2 * np.cos(2 * v)) < 1e-5
    s, h = 0.0, 1e-3
    c = lambda s: np.array([s * np.cos(v), s * np.sin(v), s * s * np.cos(2 * v)])  # noqa: E731
    assert abs(c(0.3)[2] - (c(0.3)[0] ** 2 - c(0.3)[1] ** 2)) < 1e-12              # on the surface
    k_num = (c(h)[2] - 2 * c(0)[2] + c(-h)[2]) / h ** 2                            # z'' at 0, |c'| = 1 there
    assert abs(k_num - 2 * np.cos(2 * v)) < 1e-6

# saddle mesh over the disk r <= 1 (polar grid), faces oriented with N up
NR, NA = 10, 48
pts = [[0.0, 0.0, 0.0]]
for i in range(1, NR + 1):
    r = i / NR
    for j in range(NA):
        a = 2 * np.pi * j / NA
        x, y = r * np.cos(a), r * np.sin(a)
        pts.append([x, y, x * x - y * y])
PSAD = np.array(pts)
FSAD = []
ring = lambda i, j: 1 + (i - 1) * NA + j % NA  # noqa: E731
for j in range(NA):
    FSAD.append([0, ring(1, j), ring(1, j + 1)])
for i in range(1, NR):
    for j in range(NA):
        a, b, c, d = ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j + 1)
        FSAD += [[a, b, c], [a, c, d]]
FSAD = np.array(FSAD)

# ---------------------------------------------------------------- drawing
CAM = hp.Cam(elev=24, azim=-58, scale=1.95, origin=(-3.55, -0.3))
PLOT = hp.Plot(1.5, -2.25, 4.9, 2.6, (0, 180), (-2.3, 2.3))
RX = 0.95
BASE = hp.mesh_mobs(PSAD, FSAD, CAM, color=hp.HEX["surface"], edge_color="#9a9a9a", edge_width=0.4, opacity=0.62,
                    ambient=0.6)


def overlay(v):
    X = np.array([np.cos(v), np.sin(v), 0.0])
    N = np.array([0.0, 0.0, 1.0])
    kn = 2 * np.cos(2 * v)
    out = []
    a, b = 1.25, 1.25
    corners = [a * X + b * N, -a * X + b * N, -a * X - 0.9 * N, a * X - 0.9 * N]
    out.append(hp.poly_fill([CAM(p) for p in corners], hp.C["region"], opacity=0.22, stroke_width=2,
                            stroke_color=hp.C["tangent"]))
    s = np.linspace(-1.0, 1.0, 81)
    curve = np.stack([s * np.cos(v), s * np.sin(v), s * s * np.cos(2 * v)], 1)
    out.append(hp.polyline([CAM(p) for p in curve], color=hp.C["accent"], width=6))
    if abs(kn) > 0.35:
        R = 1 / abs(kn)
        c = N / kn
        ang = np.linspace(0, 2 * np.pi, 240)
        P = [c + R * (np.cos(t) * X + np.sin(t) * N) for t in ang]
        inside = [abs(p @ X) <= a and -0.9 <= p @ N <= b for p in P]
        for k in range(len(P) - 1):
            if inside[k] and inside[k + 1] and k % 2 == 0:
                out.append(hp.seg(CAM(P[k]), CAM(P[k + 1]), color=hp.HEX["aux"], width=3))
    out.append(hp.arrow(CAM([0, 0, 0]), CAM(0.8 * X), color=hp.C["tangent"], width=6, tip=0.2))
    out.append(hp.arrow(CAM([0, 0, 0]), CAM(0.8 * N), color=hp.C["normal"], width=6, tip=0.2))
    out.append(hp.mt(r"$X$", 32, color=hp.C["tangent"]).move_to(CAM(0.95 * X) + np.array([0.0, 0.25, 0])))
    out.append(hp.mt(r"$\mathbf{N}$", 32, color=hp.C["normal"]).next_to(CAM(0.8 * N), hp.RIGHT, buff=0.1))
    return out


def readouts(v, note=None, note_color=None):
    deg = np.degrees(v)
    kn = 2 * np.cos(2 * v)
    kn_txt = 0.0 if abs(kn) < 5e-3 else kn          # no "−0.00" at 135 degrees
    out = [hp.text(f"ϑ = {deg:.0f}°", 32).move_to([RX, 2.75, 0], aligned_edge=hp.LEFT),
           hp.text(f"κₙ = {hp.num(kn_txt, 2)}", 32, color=hp.C["accent"]).move_to([4.0, 2.75, 0], aligned_edge=hp.LEFT)]
    vs = np.linspace(0, v, max(2, int(deg) + 1))
    out.append(PLOT.curve(np.degrees(vs), 2 * np.cos(2 * vs), color=hp.C["accent"], width=5))
    out.append(hp.dot(PLOT.p(deg, kn), color=hp.C["accent"], r=0.08))
    if note:
        out.append(hp.text(note, 30, color=note_color or hp.BLACK).move_to([RX, 0.55, 0], aligned_edge=hp.LEFT))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("Euler's formula: the normal curvature as X turns", 32).move_to([0, 3.55, 0])
        eq = VGroup(hp.mt(r"$\kappa_n = \kappa_1\cos^2\vartheta + \kappa_2\sin^2\vartheta$", 30),
                    hp.mt(r"$\quad\ = 2\cos^2\vartheta - 2\sin^2\vartheta$", 30))
        eq.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.12).move_to([RX, 1.85, 0], aligned_edge=hp.LEFT)
        sub = hp.text("z = x\u00b2 \u2212 y\u00b2, N up", 30, color=hp.HEX["aux"]).move_to([RX, 1.05, 0],
                                                                                              aligned_edge=hp.LEFT)
        axes = PLOT.axes(xticks=[(0, "0°"), (45, "45°"), (90, "90°"), (135, "135°"), (180, "180°")],
                         yticks=[(2, "2"), (0, "0"), (-2, "−2")], size=30, zero_line=0.0)
        k1 = hp.mt(r"$\kappa_1$", 30).next_to(PLOT.p(180, 2), hp.RIGHT, buff=0.12)
        k2 = hp.mt(r"$\kappa_2$", 30).next_to(PLOT.p(180, -2), hp.RIGHT, buff=0.12)
        guide = VGroup(hp.seg(PLOT.p(0, 2), PLOT.p(180, 2), color=hp.HEX["aux"], width=1.5, dashed=True),
                       hp.seg(PLOT.p(0, -2), PLOT.p(180, -2), color=hp.HEX["aux"], width=1.5, dashed=True))
        self.add(title, eq, sub, axes, k1, k2, guide, *BASE)
        g = VGroup()
        self.add(g)
        S = hp.setter

        def at(v, note=None, col=None):
            return lambda u: S(g, overlay(v) + readouts(v, note, col))(1)

        at(0.0, "the largest: κ₁ = 2")(1)
        self.mark("start")
        self.wait(2.0)
        ev, t = [], 0.0
        stops = {45: ("45°: a line, κₙ = 0", hp.C["tangent"]),
                 90: ("the smallest: κ₂ = −2", hp.C["normal"]),
                 135: ("135°: κₙ = 0 again", hp.C["tangent"])}
        for d in range(1, 181):
            note = stops.get(d)
            ev.append((t, t, at(np.radians(d), *(note if note else (None, None)))))
            t += 1 / 18 if d not in stops else 1.6
        self.play(hp.timeline(g, ev, t))
        self.mark("turn done")
        self.play(hp.timeline(g, [(0, 0, at(np.pi, "extremes 90° apart", hp.BLACK))], 0.3))
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.3)
