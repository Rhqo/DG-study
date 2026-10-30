"""Animation 43.4.1: E_C = E_D - A while z_eps(w) = w + eps conj(w) squashes the unit disk and turns it over
(Section 43.4, Example 43.4.5(b), Theorem 43.4.6, Proposition 43.4.7, Figure 43.4.2).

Left: the image under z_eps of the unit disk with its polar grid (circles |w| = 0.25, 0.5, 0.75, radii every 30
degrees; gray) and its boundary circle (black). The blue arrow is the velocity d/dt z_eps(e^{it}) at t = pi/2 (length
0.45 of the real one), i.e. the direction in which the image is traversed when the circle is traversed
counterclockwise. z_eps is real-linear: u = (1 + eps) x, v = (1 - eps) y, so the image is the ellipse with semi-axes
1 + eps and |1 - eps|.
Right: E_D = pi (1 + eps^2) (blue), signed area A = pi (1 - eps^2) (green) and E_C = 2 pi eps^2 (vermilion) for
0 <= eps <= 1.5; the vermilion bar at the current eps joins the green and the blue dot, its length is E_D - A = E_C.
eps runs 0 -> 1 (the image becomes the segment [-2, 2], A = 0), pauses, then 1 -> 1.5 (clockwise, A < 0).
Self-check (as Figure 43.4.2): midpoint quadrature on a polar grid agrees with the three formulas to 1e-6 and the
boundary integral (1/2) Im oint conj(z) dz agrees with A; at eps = 0.5: E_D = 3.927, A = 2.356, E_C = 1.571;
the arrow at t = pi/2 points in -x for every eps (counterclockwise for eps < 1, clockwise for eps > 1).

Render: bash tools/render_anims.sh chapters/43-conformal-geometry/anim/anim-43-4-1-flip.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data (the quadrature of Figure 43.4.2)
nr, nt = 200, 400
r = (np.arange(nr) + 0.5) / nr
t = 2 * np.pi * (np.arange(nt) + 0.5) / nt
dA = (1 / nr) * (2 * np.pi / nt) * np.meshgrid(r, t, indexing="ij")[0]


def energies(eps):
    ux, uy, vx, vy = 1 + eps, 0.0, 0.0, 1 - eps          # dz = dw + eps dwbar
    ED = 0.5 * np.sum((ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2) * dA)
    A = np.sum((ux * vy - uy * vx) * dA)
    EC = 0.5 * np.sum(((ux - vy) ** 2 + (uy + vx) ** 2) * dA)
    tb = 2 * np.pi * np.arange(4000) / 4000
    zb = np.exp(1j * tb) + eps * np.exp(-1j * tb)
    dzb = 1j * np.exp(1j * tb) - 1j * eps * np.exp(-1j * tb)
    Ab = 0.5 * np.sum((np.conj(zb) * dzb).imag) * (2 * np.pi / 4000)
    return ED, A, EC, Ab


ED_f = lambda e: np.pi * (1 + e ** 2)  # noqa: E731
A_f = lambda e: np.pi * (1 - e ** 2)  # noqa: E731
EC_f = lambda e: 2 * np.pi * e ** 2  # noqa: E731
for e in np.linspace(0, 1.5, 7):
    ED, A, EC, Ab = energies(e)
    assert abs(ED - ED_f(e)) < 1e-6 and abs(A - A_f(e)) < 1e-6 and abs(EC - EC_f(e)) < 1e-6 and abs(Ab - A) < 1e-9
    assert abs(EC - (ED - A)) < 1e-9
assert [round(v, 3) for v in energies(0.5)[:3]] == [3.927, 2.356, 1.571] and abs(energies(1.0)[1]) < 1e-9


def zmap(w, eps):
    return w + eps * np.conj(w)


def velocity(eps, tt=np.pi / 2):
    return 1j * np.exp(1j * tt) - 1j * eps * np.exp(-1j * tt)


for e in (0.2, 0.9, 1.2):
    v = velocity(e)
    assert v.real < 0 and abs(v.imag) < 1e-12
EPS1 = 1.5

# ---------------------------------------------------------------- layout
DC, DS = np.array([-3.75, -0.25, 0.0]), 1.25           # disk image: centre and scale
P = Plot(1.6, -1.95, 4.4, 3.35, (0, EPS1), (-4.5, 14.5))     # tick labels above the bottom 15% (poster)


def to_scene(zz):
    zz = np.asarray(zz)
    return np.stack([DC[0] + DS * zz.real, DC[1] + DS * zz.imag, 0 * zz.real], -1)


class Anim(Scene2D):
    def construct(self):
        ev = ValueTracker(0.0)
        tt = np.linspace(0, 2 * np.pi, 241)
        circ = np.exp(1j * tt)

        def grid():
            e = ev.get_value()
            g = VGroup()
            fill = Polygon(*to_scene(zmap(circ[:-1], e)), stroke_width=0, fill_color=C["region"], fill_opacity=0.3)
            g.add(fill)
            for rr in (0.25, 0.5, 0.75):
                g.add(polyline(to_scene(zmap(rr * circ, e)), color=HEX["aux"], width=2.0))
            for k in range(12):
                ray = np.exp(1j * k * np.pi / 6) * np.linspace(0, 1, 2)
                g.add(polyline(to_scene(zmap(ray, e)), color="#B5B5B5", width=1.6))
            g.add(polyline(to_scene(zmap(circ, e)), color=BLACK, width=4.0))
            p0 = zmap(1j, e)
            v = 0.45 * velocity(e)
            g.add(arrow(to_scene(p0), to_scene(p0 + v), color=C["tangent"], width=6, tip=0.22).set_z_index(3))
            g.add(Dot(to_scene(p0), radius=0.06, color=C["tangent"]).set_z_index(3))
            return g

        img = always_redraw(grid)
        ttl = mt(r"$z_\varepsilon(w) = w + \varepsilon\,\bar w$ on the unit disk", 34).move_to([-6.85, 3.5, 0],
                                                                                              aligned_edge=LEFT)
        epsr = readout(lambda: f"ε = {ev.get_value():.2f}", [-6.85, 2.85, 0], size=32)
        orient = always_redraw(lambda: lab("counterclockwise" if ev.get_value() < 0.999 else
                                           ("a segment" if ev.get_value() < 1.001 else "clockwise"), 30,
                                           color=C["tangent"]).move_to([-6.85, -2.5, 0], aligned_edge=LEFT))

        # plot
        axes = P.axes(xticks=[(0, "0"), (0.5, "0.5"), (1, "1"), (1.5, "1.5")],
                      yticks=[(0, "0"), (5, "5"), (10, "10")])
        axes.add(mt(r"$\varepsilon$", 32).move_to(P.p(EPS1, -4.5) + np.array([0.62, -0.34, 0])))   # after "1.5"
        zero = DashedVMobject(Line(P.p(0, 0), P.p(EPS1, 0), stroke_width=1.5, color=HEX["aux"]), num_dashes=30)
        one = DashedVMobject(Line(P.p(1, -4.5), P.p(1, 14.5), stroke_width=1.5, color=HEX["aux"]), num_dashes=30)
        es = np.linspace(0, EPS1, 301)

        def curves():
            e = ev.get_value()
            m = es[es <= e + 1e-9]
            if len(m) < 2:
                m = np.array([0.0, max(e, 1e-4)])
            g = VGroup(P.curve(m, ED_f(m), color=C["tangent"], width=4), P.curve(m, A_f(m), color=C["third"], width=4),
                       P.curve(m, EC_f(m), color=C["normal"], width=4))
            g.add(Line(P.p(e, A_f(e)), P.p(e, ED_f(e)), color=C["normal"], stroke_width=9, stroke_opacity=0.55))
            for f, c in ((ED_f, C["tangent"]), (A_f, C["third"]), (EC_f, C["normal"])):
                g.add(Dot(P.p(e, f(e)), radius=0.08, color=c).set_z_index(3))
            return g

        cv = always_redraw(curves)
        names = VGroup(mt(r"$E_D = $", 32, color=C["tangent"]), mt(r"$\mathcal{A} = $", 32, color=C["third"]),
                       mt(r"$E_C = E_D - \mathcal{A} = $", 32, color=C["normal"]))
        names.arrange(DOWN, aligned_edge=LEFT, buff=0.2).move_to([1.45, 3.6, 0], aligned_edge=UL)
        vals = always_redraw(lambda: VGroup(*[
            lab(num(f(ev.get_value()), 3), 32, color=c).next_to(nm, RIGHT, buff=0.12)
            for f, c, nm in ((ED_f, C["tangent"], names[0]), (A_f, C["third"], names[1]), (EC_f, C["normal"], names[2]))]))
        self.add(img, ttl, epsr, orient, axes, zero, one, cv, names, vals)
        self.wait(2.0)
        self.play(ev.animate.set_value(1.0), run_time=7.0, rate_func=linear)
        seg = mt(r"$\varepsilon = 1$: the image is $[-2, 2]$, $\mathcal{A} = 0$", 30).move_to([-6.85, 2.25, 0],
                                                                                           aligned_edge=LEFT)
        self.play(FadeIn(seg), run_time=0.5)
        self.wait(1.5)
        self.play(FadeOut(seg), run_time=0.4)
        self.play(ev.animate.set_value(EPS1), run_time=4.5, rate_func=linear)
        fin = mt(r"turned over: $\mathcal{A} < 0$, so $E_C > E_D$", 30).move_to([-6.85, 2.25, 0], aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=18.5)
