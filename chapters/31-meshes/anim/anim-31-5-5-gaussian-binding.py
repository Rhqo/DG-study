"""Animation 31.5.5: a Gaussian bound to a deforming face, affine binding vs. similarity binding (Proposition 31.5.4).

Data (Figure 31.5.4): rest face x_0 = (0, 0), x_1 = (2, 0), x_2 = (0.6, 1.6) in the plane z = 0; a Gaussian centred at
barycentric coordinates (0.3, 0.45, 0.25) with in-plane axes 0.35, 0.12 at 25 degrees and normal thickness 0.02.
  affine (blue):      mu' = barycentric combination of the moved corners, Sigma' = A Sigma A^T, A the affine map of
                      the face with A N = N' ((31.5.3));
  similarity (orange): GaussianAvatars [Qian24]: R = [unit(x_1 - x_0), N, cross], T = centroid,
                      k = (|x_1 - x_0| + height of x_2 over that edge) / 2; mu = k R mu_loc + T, Sigma = k^2 R Sigma_loc R^T
                      ((31.5.4)), with mu_loc, Sigma_loc fixed at rest.
Three motions of the face, each from the rest face (the corners move continuously):
  (a) linear blend to the deformed face of the figure, x_1' = (2.4, 0.3), x_2' = (1.5, 1.2);
  (b) a similarity: rotate by 0.4 s rad about x_0, scale by 1 + 0.3 s, translate by s (0.5, -0.2);
  (c) a shear: x_2 -> x_2 + (0.8 s, 0) (Exercise 31.5.4).
Grey lines: lines of constant barycentric coordinate (spacing 0.2), painted on the face. Ellipses: 1-sigma level sets.
Self-check: (a) axis ratios 2.92 -> 7.14 (affine) and 2.92 (similarity), centres 0.052 apart, k'/k = 0.951; (b) the two
Gaussians coincide at every s (in-plane covariance and mean); (c) the similarity frame R and k do not change, its
Gaussian moves by 0.8/3 = 0.267 with the same shape, the affine ratio becomes 4.21.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-5-5-gaussian-binding.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data (as fig-31-5-4-gaussian-on-face.py)
X0 = np.array([[0, 0, 0], [2, 0, 0], [0.6, 1.6, 0]], float)
X1 = np.array([[0, 0, 0], [2.4, 0.3, 0], [1.5, 1.2, 0]], float)
BARY = np.array([0.3, 0.45, 0.25])
_a = np.radians(25)
R2 = np.array([[np.cos(_a), -np.sin(_a), 0], [np.sin(_a), np.cos(_a), 0], [0, 0, 1]])
SIG0 = R2 @ np.diag([0.35 ** 2, 0.12 ** 2, 0.02 ** 2]) @ R2.T


def unit(v):
    return v / np.linalg.norm(v)


def face_frame(X):
    """GaussianAvatars utils/graphics_utils.py compute_face_orientation (one face)."""
    a0 = unit(X[1] - X[0])
    a1 = unit(np.cross(a0, X[2] - X[0]))
    a2 = -unit(np.cross(a1, a0))
    Rf = np.stack([a0, a1, a2], 1)
    s0 = np.linalg.norm(X[1] - X[0])
    s1 = abs(a2 @ (X[2] - X[0]))
    return Rf, (s0 + s1) / 2, X.mean(0)


def affine_of(Xa, Xb):
    Ea = np.stack([Xa[1] - Xa[0], Xa[2] - Xa[0], unit(np.cross(Xa[1] - Xa[0], Xa[2] - Xa[0]))], 1)
    Eb = np.stack([Xb[1] - Xb[0], Xb[2] - Xb[0], unit(np.cross(Xb[1] - Xb[0], Xb[2] - Xb[0]))], 1)
    return Eb @ np.linalg.inv(Ea)


MU0 = BARY @ X0
R0, K0, T0 = face_frame(X0)
MU_LOC = R0.T @ (MU0 - T0) / K0
SIG_LOC = R0.T @ SIG0 @ R0 / K0 ** 2


def gaussians(X):
    A = affine_of(X0, X)
    mu_a, sig_a = BARY @ X, A @ SIG0 @ A.T
    Rf, k, T = face_frame(X)
    mu_s, sig_s = k * Rf @ MU_LOC + T, k ** 2 * Rf @ SIG_LOC @ Rf.T
    return mu_a, sig_a, mu_s, sig_s, k


def ratio(S):
    w = np.linalg.eigvalsh(S[:2, :2])
    return np.sqrt(w[1] / w[0])


def motion(kind, s):
    if kind == "a":
        return (1 - s) * X0 + s * X1
    if kind == "b":
        th, sc = 0.4 * s, 1 + 0.3 * s
        Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        return sc * X0 @ Rz.T + s * np.array([0.5, -0.2, 0])
    X = X0.copy()
    X[2, 0] += 0.8 * s
    return X


mu_a, sig_a, mu_s, sig_s, k1 = gaussians(X1)
assert (round(ratio(SIG0), 2), round(ratio(sig_a), 2), round(ratio(sig_s), 2)) == (2.92, 7.14, 2.92)
assert round(np.linalg.norm(mu_a - mu_s), 3) == 0.052 and round(k1 / K0, 3) == 0.951
for s in np.linspace(0, 1, 11):
    ma, sa, ms, ss, _ = gaussians(motion("b", s))
    assert np.allclose(ma, ms, atol=1e-12) and np.allclose(sa[:2, :2], ss[:2, :2], atol=1e-12)
Xc = motion("c", 1.0)
Rc, kc, Tc = face_frame(Xc)
ma, sa, ms, ss, _ = gaussians(Xc)
assert np.allclose(Rc, R0) and np.isclose(kc, K0) and np.isclose(np.linalg.norm(ms - MU0), 0.8 / 3)
assert np.allclose(ss, SIG0) and round(ratio(sa), 2) == 4.21


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        ORG, SC, WC = np.array([-3.35, -0.1]), 2.3, np.array([1.3, 1.0])

        def w2s(p):
            p = np.asarray(p, float)
            return np.array([ORG[0] + SC * (p[0] - WC[0]), ORG[1] + SC * (p[1] - WC[1]), 0.0])

        def ellipse(mu, S, n=120):
            w, U = np.linalg.eigh(S[:2, :2])
            t = np.linspace(0, 2 * np.pi, n)
            pts = mu[:2, None] + U @ (np.sqrt(w)[:, None] * np.stack([np.cos(t), np.sin(t)]))
            return [w2s(q) for q in pts.T]

        st = {"kind": "a"}
        sv = ValueTracker(0.0)

        def draw():
            X = motion(st["kind"], sv.get_value())
            ma, sa, ms, ss, _ = gaussians(X)
            g = VGroup()
            g.add(Polygon(*[w2s(p) for p in X], stroke_width=3, color=BLACK, fill_color=GREY_A, fill_opacity=0.6))
            for i in range(3):
                j, kk = (i + 1) % 3, (i + 2) % 3
                for c in np.arange(0.2, 1.0, 0.2):
                    p = c * X[i] + (1 - c) * X[j]
                    q = c * X[i] + (1 - c) * X[kk]
                    g.add(Line(w2s(p), w2s(q), stroke_width=1.6, color=GREY_B))
            ea = Polygon(*ellipse(ma, sa), stroke_width=3, color=C["tangent"], fill_color=C["region"], fill_opacity=0.6)
            es = DashedVMobject(polyline(ellipse(ms, ss), color=C["accent"], width=4.5), num_dashes=30)
            g.add(ea, es)
            g.add(Dot(w2s(ma), radius=0.06, color=C["tangent"]), Dot(w2s(ms), radius=0.06, color=C["accent"]))
            for i, p in enumerate(X):
                g.add(Dot(w2s(p), radius=0.06, color=BLACK))
            return g

        dyn = always_redraw(draw)
        labs = always_redraw(lambda: VGroup(*[
            mt(rf"$x_{i}$", 32).next_to(w2s(motion(st["kind"], sv.get_value())[i]),
                                        (DL, DR, UP)[i], buff=0.08) for i in range(3)]))

        def read():
            X = motion(st["kind"], sv.get_value())
            ma, sa, ms, ss, _ = gaussians(X)
            return ratio(sa), ratio(ss)

        ra = readout(lambda: f"affine:  {read()[0]:.2f}", [1.35, -0.4, 0], size=32, color=C["tangent"])
        rs = readout(lambda: f"similarity:  {read()[1]:.2f}", [1.35, -0.95, 0], size=32, color=C["accent"])
        rh = da.label("axis ratio of the ellipse", 30).move_to([1.35, 0.2, 0], aligned_edge=LEFT)
        leg = VGroup(da.label("blue: affine binding", 30, color=C["tangent"]),
                     da.label("orange dashed: similarity", 30, color=C["accent"]),
                     da.label("(GaussianAvatars)", 30, color=C["accent"]),
                     da.label("grey: paint on the face", 30))
        leg.arrange(DOWN, aligned_edge=LEFT, buff=0.14).move_to([1.35, 2.75, 0], aligned_edge=UL)
        ttl = title("A Gaussian bound to a moving face: two bindings", 34)
        self.add(dyn, labs, ttl, leg, rh, ra, rs)

        def phase(txt1, txt2):
            g = VGroup(da.label(txt1, 32), da.label(txt2, 30))
            return g.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([1.35, -1.6, 0], aligned_edge=UL)

        ph = phase("rest face", "the Gaussian at rest")
        self.add(ph)
        self.wait(1.6)
        steps = [("a", "(a) stretch and shear", "affine follows the paint"),
                 ("b", "(b) rotate and scale", "a similarity: they agree"),
                 ("c", "(c) shear only", "similarity just slides")]
        for kind, t1, t2 in steps:
            if sv.get_value() > 0:
                self.play(sv.animate.set_value(0.0), run_time=1.1, rate_func=smooth)
            st["kind"] = kind
            new = phase(t1, t2)
            self.play(FadeTransform(ph, new), run_time=0.5)
            ph = new
            self.play(sv.animate.set_value(1.0), run_time=3.5, rate_func=smooth)
            self.wait(2.0 if kind != "c" else 2.4)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.0)
