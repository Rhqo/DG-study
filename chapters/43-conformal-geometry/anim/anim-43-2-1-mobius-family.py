"""Animation 43.2.1: the conformal automorphisms e^{i theta} phi_a of the unit disk form a family with three real
parameters; every member keeps circles circles and right angles right (Section 43.2, Example 43.2.12,
Proposition 43.2.11, Figure 43.2.3; Theorem 43.3.10(b)).

Left: the unit disk with the polar grid of Figure 43.2.3 (circles |z| = 0.2, 0.4, 0.6, 0.8 gray, 12 radii light gray,
unit circle black) and the point a (orange). Right: the image of the same grid under
    f(z) = e^{i theta} (z - a) / (1 - conj(a) z),
with f(a) = 0 (orange) and f(0) = -e^{i theta} a (vermilion). At the grid point w* = 0.6 e^{i pi/3} the blue arrow is
the tangent of the circle |z| = 0.6 and the green arrow the tangent of the radius, before and after the map
(the image arrows are f'(w*) times the original ones, scaled to a fixed length), with a right-angle mark.
Path of the parameters: a = 0 (identity) -> a = 0.5 along the real axis (this is Figure 43.2.3) -> once around
|a| = 0.5 -> theta from 0 to 90 degrees (a = 0.5).
Self-check: for 40 parameter values on the path, every image circle and every image radius lies on a circle or a line
(three-point circle through samples, residual < 1e-9), the unit circle goes onto itself, f(a) = 0, the angle between
the two image curves at f(w*) is 90 degrees (finite differences, 1e-6); at a = 0.5, theta = 0 the image of |z| = 0.4 is
the circle with centre -0.4375 and radius 0.3125 (Example 43.2.12).

Render: bash tools/render_anims.sh chapters/43-conformal-geometry/anim/anim-43-2-1-mobius-family.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers import *  # noqa: E402,F401,F403


# ---------------------------------------------------------------- data
def f_map(zz, a, th):
    return np.exp(1j * th) * (zz - a) / (1 - np.conj(a) * zz)


def fprime(zz, a, th):
    return np.exp(1j * th) * (1 - abs(a) ** 2) / (1 - np.conj(a) * zz) ** 2


def circle_residual(P):
    """distance of the points P (complex) from the circle (or line) through P[0], P[len/3], P[2 len/3]."""
    p, q, r = P[0], P[len(P) // 3], P[2 * len(P) // 3]
    w = (r - p) / (q - p)
    if abs(w.imag) < 1e-12:
        return float(np.abs(((P - p) / (q - p)).imag).max())
    c = (q - p) * (w - abs(w) ** 2) / (2j * w.imag) + p
    return float(np.abs(np.abs(P - c) - abs(p - c)).max())


RADII = (0.2, 0.4, 0.6, 0.8)
TT = np.linspace(0, 2 * np.pi, 361)
WSTAR = 0.6 * np.exp(1j * np.pi / 3)


def params(s):
    """the parameter path, s in [0, 3]: 0-1 a real 0 -> 0.5, 1-2 once around |a| = 0.5, 2-3 theta 0 -> pi/2."""
    if s <= 1:
        return 0.5 * s + 0j, 0.0
    if s <= 2:
        return 0.5 * np.exp(2j * np.pi * (s - 1)), 0.0
    return 0.5 + 0j, 0.5 * np.pi * (s - 2)


for s in np.linspace(0, 3, 40):
    a, th = params(s)
    for rr in RADII:
        assert circle_residual(f_map(rr * np.exp(1j * TT), a, th)) < 1e-9
    for k in range(12):
        assert circle_residual(f_map(np.linspace(0, 0.999, 50) * np.exp(1j * k * np.pi / 6), a, th)) < 1e-9
    assert np.abs(np.abs(f_map(np.exp(1j * TT), a, th)) - 1).max() < 1e-12 and abs(f_map(a, a, th)) < 1e-15
    h = 1e-6
    t1 = f_map(WSTAR * np.exp(1j * h), a, th) - f_map(WSTAR * np.exp(-1j * h), a, th)   # along the circle
    t2 = f_map(WSTAR * (1 + h), a, th) - f_map(WSTAR * (1 - h), a, th)                    # along the radius
    assert abs(np.degrees(abs(np.angle(t1 / t2))) - 90) < 1e-6
img4 = f_map(0.4 * np.exp(1j * TT), 0.5, 0.0)
assert abs((img4.max().real + img4.min().real) / 2 + 0.4375) < 1e-6 and abs((img4.real.max() - img4.real.min()) / 2
                                                                            - 0.3125) < 1e-6

# ---------------------------------------------------------------- layout
LC, RC, S = np.array([-3.75, -0.2, 0.0]), np.array([3.55, -0.2, 0.0]), 2.05


def sc(zz, c):
    zz = np.asarray(zz)
    return np.stack([c[0] + S * zz.real, c[1] + S * zz.imag, 0 * zz.real], -1)


def grid_group(c, a=0j, th=0.0, ident=False):
    g = VGroup()
    mp = (lambda zz: zz) if ident else (lambda zz: f_map(zz, a, th))
    for k in range(12):
        ray = np.linspace(0, 1, 120) * np.exp(1j * k * np.pi / 6)
        g.add(polyline(sc(mp(ray), c), color="#B9B9B9", width=1.8))
    for rr in RADII:
        g.add(polyline(sc(mp(rr * np.exp(1j * TT)), c), color=HEX["aux"], width=2.4))
    g.add(polyline(sc(np.exp(1j * TT), c), color=BLACK, width=4))
    return g


def right_angle(c, p, d1, d2, size=0.17):
    """the small square of a right angle at the scene point of p between the unit directions d1, d2 (complex)."""
    q = sc(p, c)
    u = np.array([d1.real, d1.imag, 0]) * size
    v = np.array([d2.real, d2.imag, 0]) * size
    return polyline([q + u, q + u + v, q + v], color=BLACK, width=2.2)


def tangent_marks(c, p, d1, d2):
    q = sc(p, c)
    g = VGroup(arrow(q, q + 0.62 * np.array([d1.real, d1.imag, 0]), color=C["tangent"], width=5, tip=0.18),
               arrow(q, q + 0.62 * np.array([d2.real, d2.imag, 0]), color=C["third"], width=5, tip=0.18),
               right_angle(c, p, d1, d2), Dot(q, radius=0.05, color=BLACK))
    return g.set_z_index(4)


class Anim(Scene2D):
    def construct(self):
        sv = ValueTracker(0.0)
        ttl = mt(r"$\varphi(z) = e^{i\vartheta}\,(z - a)/(1 - \bar a z)$: the unit disk onto itself", 34)
        ttl.move_to([-6.85, 3.45, 0], aligned_edge=LEFT)
        left = grid_group(LC, ident=True)
        d1, d2 = 1j * WSTAR / abs(WSTAR), WSTAR / abs(WSTAR)
        lmarks = tangent_marks(LC, WSTAR, d1, d2)
        adot = always_redraw(lambda: Dot(sc(params(sv.get_value())[0], LC), radius=0.09,
                                         color=C["accent"]).set_z_index(5))
        alab = always_redraw(lambda: mt(r"$a$", 32).next_to(adot, DOWN + LEFT, buff=0.02).set_z_index(5))

        def right_panel():
            a, th = params(sv.get_value())
            g = grid_group(RC, a, th)
            fp = fprime(WSTAR, a, th)
            e1, e2 = fp * d1, fp * d2
            g.add(tangent_marks(RC, f_map(WSTAR, a, th), e1 / abs(e1), e2 / abs(e2)))
            g.add(Dot(sc(0, RC), radius=0.09, color=C["accent"]).set_z_index(5))
            g.add(Dot(sc(f_map(0, a, th), RC), radius=0.08, color=C["normal"]).set_z_index(5))
            return g

        right = always_redraw(right_panel)
        odot = Dot(sc(0, LC), radius=0.08, color=C["normal"]).set_z_index(4)
        arr = arrow(np.array([-1.25, -0.2, 0]), np.array([1.05, -0.2, 0]), color=BLACK, width=4, tip=0.22)
        arrl = mt(r"$\varphi$", 34).next_to(arr, UP, buff=0.12)

        def ptext():
            a, th = params(sv.get_value())
            return f"|a| = {abs(a):.2f}   arg a = {np.degrees(np.angle(a)) % 360:.0f}°   ϑ = {np.degrees(th):.0f}°"

        prm = readout(ptext, [-6.85, 2.7, 0], size=30)
        ra = lab("right angles stay right", 30).move_to([3.55, -2.6, 0])
        self.add(left, lmarks, odot, adot, alab, right, ttl, arr, arrl, prm)
        self.wait(1.5)
        self.play(sv.animate.set_value(1.0), run_time=3.5, rate_func=smooth)
        fig = mt(r"$a = 0.5$: Figure 43.2.3", 30).move_to([-6.85, -2.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(fig), run_time=0.4)
        self.wait(1.5)
        self.play(FadeOut(fig), FadeIn(ra), run_time=0.4)
        self.play(sv.animate.set_value(2.0), run_time=7.0, rate_func=linear)
        self.play(sv.animate.set_value(3.0), run_time=2.5, rate_func=smooth)
        fin = mt(r"three real parameters: $a \in \mathbb{D}$ and $\vartheta$", 30).move_to([-6.85, -2.6, 0],
                                                                                     aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.0)
