"""Animation 31.6.1: the Jacobian of a UV triangle, J = U Sigma W^T, sends a small circle to an ellipse (Definition 31.6.1).

Data (Figure 31.6.1): surface triangle (in its own plane) x_0 = (0, 0), x_1 = (2, 0), x_2 = (0.6, 1.4); UV triangle
u_0 = (0.1, 0.1), u_1 = (0.6, 0.35), u_2 = (0.15, 0.9); J = [x_1 - x_0, x_2 - x_0][u_1 - u_0, u_2 - u_0]^{-1} ((31.6.1)).
The UV triangle, the part of the 8 x 8 texel grid inside it and the circle of radius eps = 0.06 at its centroid are
carried to the surface triangle in three exact steps of the singular value decomposition J = U Sigma W^T (U, W
rotations, Sigma = diag(sigma_1, sigma_2)): rotate by W^T, stretch by Sigma, rotate by U (each step interpolated in
its angle or its scale factors). UV and surface are drawn at the same scale.
Self-check: U and W are rotations (det +1) after fixing the signs, chosen so that the rotation angles are small
(W^T: -1.5 deg, U: -12.8 deg; the other choice, -U and -W, would turn the triangle by about 180 deg twice);
U Sigma W^T = J; the final triangle is exactly x_0, x_1, x_2; sigma_1 = 3.85, sigma_2 = 1.88, sigma_1 / sigma_2 = 2.05, sigma_1 sigma_2 = 7.23 = area ratio.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-6-1-jacobian.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

X = np.array([[0.0, 0.0], [2.0, 0.0], [0.6, 1.4]])
U = np.array([[0.1, 0.1], [0.6, 0.35], [0.15, 0.9]])
P_ = np.stack([X[1] - X[0], X[2] - X[0]], 1)
Q_ = np.stack([U[1] - U[0], U[2] - U[0]], 1)
J = P_ @ np.linalg.inv(Q_)
Ul, S, Vt = np.linalg.svd(J)
if np.linalg.det(Ul) < 0:                     # det J > 0: make both factors rotations
    Ul = Ul @ np.diag([1, -1])
    Vt = np.diag([1, -1]) @ Vt
if abs(np.arctan2(Vt[1, 0], Vt[0, 0])) > np.pi / 2:   # (U, W) and (-U, -W) are both SVDs: take the small rotations
    Ul, Vt = -Ul, -Vt
assert np.isclose(np.linalg.det(Ul), 1) and np.isclose(np.linalg.det(Vt), 1)
assert np.allclose(Ul @ np.diag(S) @ Vt, J)
TH_W = np.arctan2(Vt[1, 0], Vt[0, 0])          # W^T is the rotation by TH_W
TH_U = np.arctan2(Ul[1, 0], Ul[0, 0])
assert (round(np.degrees(TH_W), 1), round(np.degrees(TH_U), 1)) == (-1.5, -12.8)
S1, S2 = S
assert (round(S1, 2), round(S2, 2), round(S1 / S2, 2), round(S1 * S2, 2)) == (3.85, 1.88, 2.05, 7.23)
assert np.isclose(S1 * S2, abs(np.linalg.det(P_)) / abs(np.linalg.det(Q_)))
CU = U.mean(0)
CX = X.mean(0)
assert np.allclose(CX[:, None] + J @ (U - CU).T, X.T)
EPS = 0.06


def rot(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def stage_matrix(s):
    """s in [0, 3]: 0 = identity, 1 = W^T, 2 = Sigma W^T, 3 = U Sigma W^T = J."""
    if s <= 1:
        return rot(s * TH_W)
    if s <= 2:
        t = s - 1
        return np.diag([1 + t * (S1 - 1), 1 + t * (S2 - 1)]) @ Vt
    t = s - 2
    return rot(t * TH_U) @ np.diag(S) @ Vt


assert np.allclose(stage_matrix(3), J) and np.allclose(stage_matrix(1), Vt)


def clip_to_triangle(a, b, tri):
    """The part of segment a-b inside triangle tri (or None)."""
    A, B, Cc = tri
    M = np.linalg.inv(np.stack([B - A, Cc - A], 1))
    la, lb = M @ (a - A), M @ (b - A)
    bary = lambda l: np.array([1 - l[0] - l[1], l[0], l[1]])  # noqa: E731
    ba, bb = bary(la), bary(lb)
    lo, hi = 0.0, 1.0
    for i in range(3):
        d = bb[i] - ba[i]
        if abs(d) < 1e-14:
            if ba[i] < 0:
                return None
            continue
        t = -ba[i] / d
        if d > 0:
            lo = max(lo, t)
        else:
            hi = min(hi, t)
    if hi - lo < 1e-9:
        return None
    return a + lo * (b - a), a + hi * (b - a)


GRID_SEGS = []
for k in range(9):
    for a, b in (((k / 8, 0), (k / 8, 1)), ((0, k / 8), (1, k / 8))):
        seg = clip_to_triangle(np.array(a, float), np.array(b, float), U)
        if seg is not None:
            GRID_SEGS.append(seg)
assert len(GRID_SEGS) > 6


class Anim(da.Scene2D):
    def construct(self):
        SC = 2.5
        UV0 = np.array([-6.3, -0.8])                  # scene position of UV (0, 0)
        DEST = np.array([3.3, -0.3])                   # scene position of the surface triangle's centroid

        def uv2s(p):
            return np.array([UV0[0] + SC * p[0], UV0[1] + SC * p[1], 0.0])

        ttl = title("One face of a UV map: J sends circles to ellipses", 34)
        sq = Square(SC, stroke_width=2, color=BLACK).move_to(uv2s((0.5, 0.5)))
        texels = VGroup(*[Line(uv2s((k / 8, 0)), uv2s((k / 8, 1)), stroke_width=1, color=GREY_B) for k in range(1, 8)],
                        *[Line(uv2s((0, k / 8)), uv2s((1, k / 8)), stroke_width=1, color=GREY_B) for k in range(1, 8)])
        uvlab = da.label("UV square, 8 x 8 texels", 30).move_to([-6.85, 2.15, 0], aligned_edge=LEFT)
        self.add(ttl, sq, texels, uvlab)

        prog = ValueTracker(0.0)          # 0: in the UV square; 1: moved; 1..4: stages 0..3
        st = {"show": False}

        def place(p, s):
            """UV point p -> scene, for progress s."""
            if s <= 1:
                c = (1 - s) * uv2s(CU)[:2] + s * DEST
                M = np.eye(2)
            else:
                c = DEST
                M = stage_matrix(s - 1)
            q = c + SC * (M @ (np.asarray(p) - CU))
            return np.array([q[0], q[1], 0.0])

        def body():
            s = prog.get_value()
            g = VGroup()
            g.add(Polygon(*[place(p, s) for p in U], stroke_width=2.5, color=C["tangent"], fill_color=C["region"],
                          fill_opacity=0.35))
            for a, b in GRID_SEGS:
                g.add(Line(place(a, s), place(b, s), stroke_width=1.6, color=GREY_D))
            t = np.linspace(0, 2 * np.pi, 90)
            circ = [place(CU + EPS * np.array([np.cos(a_), np.sin(a_)]), s) for a_ in t]
            g.add(Polygon(*circ, stroke_width=3.5, color=C["normal"], fill_color=C["normal"], fill_opacity=0.15))
            return g

        static_uv = body()
        self.add(static_uv)
        ulabs = VGroup(*[mt(rf"$u_{i}$", 32).next_to(uv2s(U[i]), d, buff=0.08) for i, d in enumerate((DL, RIGHT, UP))])
        clab = mt(r"circle of radius $\varepsilon = 0.06$", 30, color=C["normal"]).move_to([-6.85, -1.55, 0], aligned_edge=LEFT)
        self.add(ulabs, clab)
        self.wait(1.8)
        moving = always_redraw(body)
        self.add(moving)
        self.play(prog.animate.set_value(1.0), run_time=1.3, rate_func=smooth)
        steps = [(r"1. rotate by $W^{\mathsf{T}}$ ($-1.5^\circ$)", 2.0), (r"2. stretch by $\sigma_1 = 3.85,\ \sigma_2 = 1.88$", 2.6),
                 (r"3. rotate by $U$ ($-12.8^\circ$)", 2.0)]
        lab = None
        for k, (txt, rt) in enumerate(steps):
            new = mt(txt, 32).move_to([-1.6, 2.85, 0], aligned_edge=LEFT)
            anims = [FadeIn(new)] if lab is None else [FadeTransform(lab, new)]
            self.play(*anims, run_time=0.4)
            lab = new
            self.play(prog.animate.set_value(2.0 + k), run_time=rt, rate_func=smooth)
            self.wait(1.0)
        fin = mt(r"$J = U\,\Sigma\,W^{\mathsf{T}}$: the surface triangle", 32).move_to([-1.6, 2.85, 0], aligned_edge=LEFT)
        self.play(FadeTransform(lab, fin), run_time=0.4)
        xlabs = VGroup(*[mt(rf"$x_{i}$", 32).next_to(place(U[i], 4.0), d, buff=0.08)
                         for i, d in enumerate((DL, DR, UP))])
        # semi-axes sigma_i eps along the left singular vectors
        c = place(CU, 4.0)
        arrows = VGroup()
        for i, col in ((0, C["accent"]), (1, C["third"])):
            v = Ul[:, i] * S[i] * EPS * SC
            arrows.add(Arrow(c, c + np.array([v[0], v[1], 0]), buff=0, stroke_width=5, color=col, tip_length=0.12,
                             max_tip_length_to_length_ratio=0.35).set_z_index(5))
        res_l = VGroup(mt(r"orange: $\sigma_1\varepsilon$, green: $\sigma_2\varepsilon$", 30),
                       mt(r"texels per unit length: $2.08$, $4.26$", 30))
        res_l.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([-6.85, -1.35, 0], aligned_edge=UL)
        res_r = VGroup(mt(r"angle distortion $\sigma_1/\sigma_2 = 2.05$", 30),
                       mt(r"area ratio $\sigma_1\sigma_2 = 7.23$", 30))
        res_r.arrange(DOWN, aligned_edge=LEFT, buff=0.1).move_to([0.75, -2.05, 0], aligned_edge=UL)
        res = VGroup(res_l, res_r)
        self.play(FadeIn(xlabs), GrowArrow(arrows[0]), GrowArrow(arrows[1]), FadeOut(clab), run_time=0.8)
        self.play(FadeIn(res), run_time=0.6)
        self.wait(2.8)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.5)
