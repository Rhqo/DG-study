"""Animation 44.1.1: lowering the energy of a polygonal path on the sphere straightens a circle of latitude into a
great arc (Section 44.1, Propositions 44.1.8 and 44.1.9, Non-example 44.1.3, Exercise 44.1.6).

Data (as the solution of Exercise 44.1.6): unit sphere, the polygonal path x_0, ..., x_40 that starts as the arc of
the circle of latitude theta = pi/3 between p = x(pi/3, -1.1) and q = x(pi/3, 1.1) (equal steps in phi), ends fixed;
every iteration replaces all interior points at once by x_k <- (x_{k-1} + x_{k+1}) / |x_{k-1} + x_{k+1}|.
Discrete energy E_h = (1/2) sum |x_{k+1} - x_k|^2 / h, h = 1/40. Shown iterations: 0 and the 116 distinct integers
among 140 values spaced logarithmically from 1 to 20000 (the drawing jumps between computed iterates; it does not
interpolate).
Sphere drawn orthographically from the direction x(50 deg, 0): light gray disk and grid (every 30 degrees, front
halves only), gray: the starting circle of latitude, orange dashed: the great circle through p and q, black: the
current path with its 41 points. Right: E_h against the iteration count (log axis), orange dashed line: the limit.
Self-check (the printed output of Exercise 44.1.6): E_h = 1.81454, 1.72240, 1.55822, 1.55447 and the largest distance
|<x_k, n>| from the great-circle plane 3.7e-1, 3.1e-1, 4.7e-2, 2.8e-14 at iterations 0, 100, 1000, 20000; the
final length 1.76322 against the great arc 1.76336; every drawn point faces the viewer.

Render: bash tools/render_anims.sh chapters/44-discrete-geodesics/anim/anim-44-1-1-harmonic-iteration.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import *  # noqa: E402,F401,F403


# ---------------------------------------------------------------- data (Exercise 44.1.6)
def sphere_point(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def energy(P):
    return 0.5 * np.sum(np.diff(P, axis=0) ** 2) * (len(P) - 1)


n = 40
P = np.array([sphere_point(np.pi / 3, ph) for ph in np.linspace(-1.1, 1.1, n + 1)])
P0 = P.copy()
NRM = np.cross(P[0], P[-1]) / np.linalg.norm(np.cross(P[0], P[-1]))
SHOW = np.unique(np.r_[0, np.round(np.logspace(0, np.log10(20000), 140)).astype(int)])
assert len(SHOW) == 117
keep = {}
for it in range(20001):
    if it in SHOW or it in (100, 1000):
        keep[it] = P.copy()
    Q = P[:-2] + P[2:]
    P[1:-1] = Q / np.linalg.norm(Q, axis=1, keepdims=True)
ref = {0: (1.81454, 3.7e-1), 100: (1.72240, 3.1e-1), 1000: (1.55822, 4.7e-2), 20000: (1.55447, 2.8e-14)}
for it, (e, dist) in ref.items():
    assert round(energy(keep[it]), 5) == e and float(f"{np.abs(keep[it] @ NRM).max():.1e}") == dist, it
LEN = np.linalg.norm(np.diff(keep[20000], axis=0), axis=1).sum()
ARC = np.arccos(P0[0] @ P0[-1])
assert round(LEN, 5) == 1.76322 and round(ARC, 5) == 1.76336
ITS = list(SHOW)
EN = np.array([energy(keep[i]) for i in ITS])
DIST = np.array([np.abs(keep[i] @ NRM).max() for i in ITS])
LENS = np.array([np.linalg.norm(np.diff(keep[i], axis=0), axis=1).sum() for i in ITS])
assert np.all(np.diff(EN) <= 1e-12)                      # the energy never increases along the shown iterates

# ---------------------------------------------------------------- layout
VIEW = sphere_point(np.radians(50), 0.0)
CAM = Cam.from_view(VIEW, scale=3.1, center=[-3.35, -0.75, 0])
for it in ITS:
    assert (keep[it] @ VIEW).min() > 0.3                  # all drawn points are on the visible hemisphere
PLT = Plot(1.3, -0.75, 4.9, 2.0, (1, 20000), (1.5, 1.85), xlog=True)


def front_curve(X, color, width, dashed=False, nd=60):
    vis = X @ VIEW > 0
    g = VGroup()
    run = []
    for x, v in zip(X, vis):
        if v:
            run.append(x)
        elif len(run) > 1:
            g.add(polyline(CAM(np.array(run)), color=color, width=width))
            run = []
        else:
            run = []
    if len(run) > 1:
        g.add(polyline(CAM(np.array(run)), color=color, width=width))
    if dashed:
        return VGroup(*[DashedVMobject(m, num_dashes=nd) for m in g])
    return g


class Anim(Scene2D):
    def construct(self):
        disk = Circle(radius=CAM.scale, color=HEX["aux"], stroke_width=2, fill_color=HEX["surface"],
                      fill_opacity=0.35).move_to(CAM.center)
        grid = VGroup()
        tt = np.linspace(0, 2 * np.pi, 361)
        for th in np.radians([30, 90, 120, 150]):
            grid.add(front_curve(np.stack([np.sin(th) * np.cos(tt), np.sin(th) * np.sin(tt), np.cos(th) + 0 * tt], 1),
                                 "#BDBDBD", 1.6))
        for ph in np.radians(np.arange(0, 360, 30)):
            ts = np.linspace(0, np.pi, 181)
            grid.add(front_curve(np.stack([np.sin(ts) * np.cos(ph), np.sin(ts) * np.sin(ph), np.cos(ts)], 1),
                                 "#BDBDBD", 1.6))
        lat = front_curve(np.stack([np.sin(np.pi / 3) * np.cos(tt), np.sin(np.pi / 3) * np.sin(tt),
                                    np.cos(np.pi / 3) + 0 * tt], 1), HEX["aux"], 3)
        e1 = P0[0]
        e2 = np.cross(NRM, e1)
        gc = front_curve(np.cos(tt)[:, None] * e1 + np.sin(tt)[:, None] * e2, HEX["accent"], 4, dashed=True, nd=70)
        kv = ValueTracker(0)

        def path():
            X = keep[ITS[int(kv.get_value())]]
            g = VGroup(polyline(CAM(X), color=BLACK, width=4))
            for x in X:
                g.add(Dot(CAM(x), radius=0.035, color=BLACK))
            for x in (X[0], X[-1]):
                g.add(Dot(CAM(x), radius=0.08, color=BLACK))
            return g.set_z_index(3)

        pm = always_redraw(path)
        pl = mt(r"$p$", 32).next_to(CAM(P0[0]), DOWN + LEFT, buff=0.06)
        ql = mt(r"$q$", 32).next_to(CAM(P0[-1]), DOWN + RIGHT, buff=0.06)
        ttl = mt(r"$x_k \leftarrow (x_{k-1} + x_{k+1})\,/\,|x_{k-1} + x_{k+1}|$, all $k$ at once", 32)
        ttl.move_to([-6.85, 3.5, 0], aligned_edge=LEFT)
        sub = lab("start: circle of latitude, ends fixed", 30, color=HEX["aux"]).move_to([-6.85, 2.9, 0],
                                                                                        aligned_edge=LEFT)
        axes = PLT.axes(xticks=[(1, "1"), (10, "10"), (100, mt(r"$10^2$", 30)), (1000, mt(r"$10^3$", 30)),
                                (10000, mt(r"$10^4$", 30))], yticks=[(1.6, "1.6"), (1.7, "1.7"), (1.8, "1.8")],
                        xlabel="iteration")
        elab = mt(r"energy $E_h$", 30).next_to(PLT.p(1, 1.85), UP, buff=0.1).shift(RIGHT * 0.9)
        lim = DashedVMobject(Line(PLT.p(1, EN[-1]), PLT.p(20000, EN[-1]), color=C["accent"], stroke_width=2.5),
                             num_dashes=40)
        x1 = np.array([max(i, 1) for i in ITS], float)

        def ecurve():
            k = int(kv.get_value())
            g = VGroup()
            if k >= 2:
                g.add(PLT.curve(x1[1:k + 1], EN[1:k + 1], color=BLACK, width=4))
            g.add(Dot(PLT.p(x1[k], EN[k]), radius=0.07, color=BLACK).set_z_index(3))
            return g

        ec = always_redraw(ecurve)

        def info():
            k = int(kv.get_value())
            return VGroup(lab(f"iteration {ITS[k]}", 30),
                          lab(f"length {LENS[k]:.5f}", 30),
                          lab(f"off the plane: {DIST[k]:.1e}".replace("-", "\u2212"), 30)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([1.3, 3.15, 0], aligned_edge=UL)

        box = always_redraw(info)
        self.add(disk, grid, lat, gc, pm, pl, ql, ttl, sub, axes, elab, lim, ec, box)
        self.wait(2.5)
        self.play(kv.animate.set_value(len(ITS) - 1), run_time=10.0, rate_func=linear)
        fin = VGroup(lab("limit: a great arc", 30, color=C["accent"]),
                     mt(r"$E_h = L^2/2 = 1.55447$", 30)).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        fin.move_to([0.9, -2.3, 0], aligned_edge=LEFT)        # clear of the axis label, above the bottom 15%
        self.play(FadeIn(fin), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=15.0)
