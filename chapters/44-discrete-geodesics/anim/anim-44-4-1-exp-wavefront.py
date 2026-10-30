"""Animation 44.4.1: the wavefront of the discrete exponential map near a cone vertex and near a saddle vertex
(Section 44.4, Definition 44.4.10, Propositions 44.4.12 and 44.4.13, Figure 44.4.3).

Model (as Figure 44.4.3): an infinite polyhedral cone with one vertex v, (a) Theta = 5 pi/3 (delta = pi/3),
(b) Theta = 8 pi/3 (delta = -2 pi/3); p at distance r0 = 1 from v. Straightest geodesics from p are straight lines in
the unfolding around p (ch44.cone_ray); the one aimed at v (psi = pi) continues at alpha = Theta/2.
The wavefront at time L is the set of points exp_p(L u), |u| = 1: for psi in (0, pi) the upper branch, for psi in
(pi, 2 pi) the lower branch (720 directions each), and the single point of the ray through v (orange dot) once L > 1.
Drawing: polar coordinates (r, alpha) at v with alpha mapped to 2 pi alpha / Theta (schematic angles, exact radii).
Gray: the 23 rays psi = pi + k pi/12 (k != 0) of Figure 44.4.3, grown to length L; orange: the ray through v.
(a) red dot: where the two branches cross (on the ray alpha = Theta/2), at the distance s of ch44.cone_meeting_point
    with t = L. (b) orange sector (radius 2.7): pi < alpha < Theta - pi, shaded once the front has passed v; only the ray
    through v enters it.
L runs 0 -> 1 (the front reaches v), pauses, then 1 -> 2.4.
Self-check: the crossing of the two cone branches is found from the drawn polylines and agrees with
cone_meeting_point to 1e-3; for every L <= 1 the cone map is injective on |u| <= L (the two branches do not meet);
on the saddle no endpoint except the orange one has |alpha| >= pi (Proposition 44.4.13); at L = 1.216 the cone
crossing is at s = 0.243 (Figure 44.4.3, beta = 0.1).

Render: bash tools/render_anims.sh chapters/44-discrete-geodesics/anim/anim-44-4-1-exp-wavefront.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _helpers import *  # noqa: E402,F401,F403
import ch44  # noqa: E402

# ---------------------------------------------------------------- data
TH_CONE, TH_SAD = 5 * np.pi / 3, 8 * np.pi / 3
LMAX = 2.4
NPSI = 720
PSI_UP = np.linspace(1e-4, np.pi - 1e-4, NPSI)


def endpoints(psi, L):
    """(r, alpha) of exp_p(L e^{i psi}) on the cone (alpha in (-pi, pi), the cover angle; p at alpha = 0)."""
    P = np.stack([1 + L * np.cos(psi), L * np.sin(psi)], -1)
    return np.linalg.norm(P, axis=-1), np.arctan2(P[..., 1], P[..., 0])


def to_disp(r, alpha, Theta):
    ang = 2 * np.pi * alpha / Theta
    return np.stack([r * np.cos(ang), r * np.sin(ang)], -1)


def cone_crossing(L):
    """distance from v of the crossing of the two cone branches (on the display ray at angle pi), from the upper
    branch alone (symmetry): the point of the upper branch with display angle exactly pi."""
    r, al = endpoints(PSI_UP, L)
    ang = 2 * np.pi * al / TH_CONE
    k = np.where((ang[:-1] - np.pi) * (ang[1:] - np.pi) <= 0)[0]
    if len(k) == 0:
        return None
    k = k[0]
    w = (np.pi - ang[k]) / (ang[k + 1] - ang[k])
    return (1 - w) * r[k] + w * r[k + 1]


DELTA = 2 * np.pi - TH_CONE
for L in (1.05, 1.216, 1.6, 2.0, 2.4):
    s_num = cone_crossing(L)
    beta = DELTA / 2 - np.arcsin(np.sin(DELTA / 2) / L)            # t(beta) = L, (44.4.3)
    t_ref, s_ref = ch44.cone_meeting_point(beta, DELTA)
    assert abs(t_ref - L) < 1e-12 and abs(s_num - s_ref) < 1e-3, (L, s_num, s_ref)
for L in (0.3, 0.7, 0.999):
    assert cone_crossing(L) is None                                   # no self-intersection before L = r0 = 1
assert abs(cone_crossing(1.216) - 0.243) < 1e-3
for L in (0.5, 1.5, 2.4):
    _, al = endpoints(np.r_[PSI_UP, -PSI_UP], L)
    assert np.abs(al).max() < np.pi                                   # the saddle sector [pi, Theta - pi] is missed
    r, al = ch44.cone_ray(PSI_UP[100], TH_SAD, L)
    assert abs(r[-1] - endpoints(PSI_UP[100], L)[0]) < 1e-12
RAYS = [np.pi + k * np.pi / 12 for k in range(-11, 13) if k != 0]

# ---------------------------------------------------------------- layout
S = 0.88                         # the cone front at L = 2.4 stays below the labels at the top
VA, VB = np.array([-4.55, -0.45, 0.0]), np.array([2.55, -0.45, 0.0])


def sc(xy, c):
    xy = np.asarray(xy, float)
    return np.concatenate([c[:2] + S * xy, np.zeros(xy.shape[:-1] + (1,))], -1)


class Anim(Scene2D):
    def construct(self):
        Lv = ValueTracker(0.0)

        def panel(c, Theta):
            L = max(Lv.get_value(), 1e-3)
            g = VGroup()
            if Theta > 2 * np.pi and L > 1:
                aa = np.linspace(np.pi, Theta - np.pi, 40)
                R = 2.7                          # as Figure 44.4.3
                pts = np.vstack([[0, 0], to_disp(R + 0 * aa, aa, Theta), [0, 0]])
                g.add(Polygon(*sc(pts, c)[:-1], stroke_width=0, fill_color=C["accent"],
                              fill_opacity=0.25 * min(1.0, 3 * (L - 1))))
            for psi in RAYS:
                r, al = ch44.cone_ray(psi, Theta, L, n=80)
                g.add(polyline(sc(to_disp(r, al, Theta), c), color="#A9A9A9", width=1.8))
            r, al = ch44.cone_ray(np.pi, Theta, L, n=80)
            g.add(polyline(sc(to_disp(r, al, Theta), c), color=C["accent"], width=4.5))
            for sgn in (1, -1):
                r, al = endpoints(sgn * PSI_UP, L)
                g.add(polyline(sc(to_disp(r, al, Theta), c), color=BLACK, width=5))
            if L > 1:
                q = to_disp(np.array([L - 1]), np.array([Theta / 2]), Theta)[0]
                g.add(Dot(sc(q, c), radius=0.08, color=C["accent"]).set_z_index(4))
                if Theta < 2 * np.pi:
                    s_ = cone_crossing(L)
                    if s_ is not None:
                        g.add(Dot(sc(np.array([-s_, 0.0]), c), radius=0.09, color=C["normal"]).set_z_index(5))
            g.add(Dot(sc(np.zeros(2), c), radius=0.07, color=BLACK).set_z_index(6))
            g.add(Dot(sc(np.array([1.0, 0.0]), c), radius=0.08, color=C["accent"]).set_z_index(6))
            return g

        pa = always_redraw(lambda: panel(VA, TH_CONE))
        pb = always_redraw(lambda: panel(VB, TH_SAD))
        ttl = mt(r"wavefront of the discrete exp map: $\exp_p(L\,u)$, $|u| = 1$", 32)
        ttl.move_to([-6.85, 3.5, 0], aligned_edge=LEFT)
        la = mt(r"(a) cone, $\Theta = 5\pi/3$", 30).move_to([-6.85, 2.75, 0], aligned_edge=LEFT)
        lb = mt(r"(b) saddle, $\Theta = 8\pi/3$", 30).move_to([0.25, 2.75, 0], aligned_edge=LEFT)
        Lr = readout(lambda: f"L = {Lv.get_value():.2f}", [5.0, 2.75, 0], size=32)
        names = VGroup(mt(r"$v$", 32).next_to(sc(np.zeros(2), VA), UP, buff=0.1),
                       mt(r"$p$", 32).next_to(sc(np.array([1.0, 0]), VA), DOWN + RIGHT, buff=0.05),
                       mt(r"$v$", 32).next_to(sc(np.zeros(2), VB), UP, buff=0.1),
                       mt(r"$p$", 32).next_to(sc(np.array([1.0, 0]), VB), DOWN + RIGHT, buff=0.05)).set_z_index(7)
        self.add(pa, pb, ttl, la, lb, Lr, names)
        self.wait(1.5)
        self.play(Lv.animate.set_value(1.0), run_time=4.5, rate_func=linear)
        reach = mt(r"$L = d(p, v) = 1$: the front reaches $v$", 30).move_to([-6.85, 2.15, 0], aligned_edge=LEFT)
        self.play(FadeIn(reach), run_time=0.4)
        self.wait(1.2)
        self.play(FadeOut(reach), run_time=0.4)
        self.play(Lv.animate.set_value(LMAX), run_time=6.5, rate_func=linear)
        fa = lab("the two branches cross", 30, color=C["normal"]).move_to([-6.85, 2.15, 0], aligned_edge=LEFT)
        fb = VGroup(lab("orange sector: never reached", 30), lab("except along the orange ray", 30))
        fb.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([0.25, 2.15, 0], aligned_edge=UL).shift(UP * 0.17)
        self.play(FadeIn(fa), FadeIn(fb), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.0)
