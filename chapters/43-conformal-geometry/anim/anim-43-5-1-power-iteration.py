"""Animation 43.5.1: the inverse power iteration of the spectral conformal parametrization, from a random start to
(nearly) the stereographic projection (Section 43.5, Remark 43.5.10, Implementation box, Example 43.5.11).

Data (as in the Implementation box and Figure 43.5.1): mesh ch43.hemisphere(3) (337 vertices, 624 faces, outward
normals), B = M = dgmesh.mass_barycentric, E_C = ch43.conformal_energy_matrix, E_b = E_C + b b^T. Start y_0 = the
random vector of the Implementation box (numpy default_rng(0), standard normal real and imaginary parts), then
y <- E_b^{-1} B y, y <- y - <y, 1>_B / <1, 1>_B, y <- y / |y|_B. Every iterate is drawn after the same rotation
e^{-i arg a}, where a is the best fit z ~ a conj(sigma) + c of the converged z (so a > 0 as in Figure 43.5.1).
Between two iterates the drawing interpolates linearly (the iterates themselves are exact).
Colours: a checkerboard on the hemisphere made of 12 longitude sectors and 4 rings |zeta| in [0, 1/4), ... [3/4, 1]
(zeta = conj(sigma) of the face centroid), the same colour for the same face in both panels.
Right: the residual |E_C y - lambda B y| (lambda = Rayleigh quotient y^H E_C y) on a log scale against the step k,
and a dashed gray line with ratio lambda_1/lambda_2 = 0.042 per step.
Self-check: y_0 has 307 flipped faces (of 624) and lambda = 90.8; y_1 already has none; lambda_1 = 2.1937e-3 and
lambda_1/lambda_2 = 0.0421 (ch43.pencil_eigs); the converged z has residual < 1e-10 and relative distance 0.0078 to its
best fit a conj(sigma) + c (Example 43.5.11).

Render: bash tools/render_anims.sh chapters/43-conformal-geometry/anim/anim-43-5-1-power-iteration.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _helpers import *  # noqa: E402,F401,F403
import ch43  # noqa: E402

# ---------------------------------------------------------------- data
V, F = ch43.hemisphere(3)
b = dm.mass_barycentric(V, F)
EC = ch43.conformal_energy_matrix(V, F)
Eb = EC + np.outer(b, b)


def project(y):
    y = y - (b @ y) / b.sum()
    return y / np.sqrt(b @ np.abs(y) ** 2)


y = project(np.random.default_rng(0).standard_normal((len(V), 2)) @ np.array([1, 1j]))
Y = [y]
for _ in range(60):
    y = project(np.linalg.solve(Eb, b * y))
    Y.append(y)
z = Y[-1]
sig = ch43.stereo_conj(V)
a, err = ch43.best_fit(z, sig, b)
rot = np.exp(-1j * np.angle(a))
Y = [rot * yy for yy in Y]
z = Y[-1]
a, err = ch43.best_fit(z, sig, b)
cfit = (b @ (z - a * sig)) / b.sum()
KMAX = 12
LAM = np.array([float((yy.conj() @ EC @ yy).real) for yy in Y[:KMAX + 1]])
RES = np.array([float(np.linalg.norm(EC @ yy - lam * b * yy)) for yy, lam in zip(Y[:KMAX + 1], LAM)])
FLIPS = np.array([int((ch43.face_signed_areas(F, yy) < 0).sum()) for yy in Y[:KMAX + 1]])
lams, _ = ch43.pencil_eigs(EC, b, 2)
RATIO = lams[0] / lams[1]
assert len(V) == 337 and len(F) == 624
assert FLIPS[0] == 307 and abs(LAM[0] - 90.8) < 0.05 and FLIPS[1:].max() == 0
assert abs(lams[0] - 2.1937e-3) < 1e-7 and abs(RATIO - 0.0421) < 1e-4
assert np.linalg.norm(EC @ z - LAM[-1] * b * z) < 1e-10 and abs(err - 0.0078) < 1e-4 and abs(abs(a) - 0.663) < 1e-3
assert np.all(np.diff(np.log(RES[:9])) < 0) and RES[KMAX] < 1e-14
assert abs(a.imag) < 1e-12 and a.real > 0

# checkerboard colours (same face, same colour in both panels)
cen = V[F].mean(1)
zc = ((cen[:, 0] + 1j * cen[:, 1]) / (np.linalg.norm(cen, axis=1) - cen[:, 2])).conj()   # conj(sigma) of the
ring = np.minimum((np.abs(zc) * 4).astype(int), 3)                                          # centroid direction
sector = ((np.angle(zc) % (2 * np.pi)) / (np.pi / 6)).astype(int) % 12
checker = (ring + sector) % 2
RGB = np.where(checker[:, None] == 1, hex_rgb(HEX["region"]), hex_rgb("#E9E9E9"))
loop = dm.boundary_loop(F)
south = int(np.argmin(V[:, 2]))
target = a * sig + cfit

# ---------------------------------------------------------------- layout
CAM = Cam(elev=-50, azim=-60, scale=1.6, center=[-5.2, 0.15, 0], pivot=(0, 0, -0.35))
IMG_C, IMG_S = np.array([0.1, -0.15, 0.0]), 3.05
assert np.abs(Y[0]).max() * IMG_S < 3.65


class Anim(Scene2D):
    def construct(self):
        ttl = mt(r"inverse power iteration: $y \leftarrow \mathsf{E}_b^{-1} B\,y$", 34).move_to([-6.85, 3.5, 0],
                                                                                              aligned_edge=LEFT)
        hemi = MeshView(F, stroke_width=0.3, stroke_rgb=(0.35, 0.35, 0.35), cull=False, ambient=0.55)
        hemi.set_state(V, RGB, CAM)
        hlab = lab("hemisphere", 30).move_to([-5.2, -2.15, 0])
        img = FlatMeshView(F, stroke_width=0.3, stroke_rgb=(0.35, 0.35, 0.35))
        kv = ValueTracker(0.0)

        def zk():
            s = kv.get_value()
            j = min(int(np.floor(s)), KMAX - 1)
            t = s - j
            return (1 - t) * Y[j] + t * Y[j + 1]

        def upd(m):
            w = zk()
            m.set_state(np.c_[w.real, w.imag], RGB, center=IMG_C, scale=IMG_S)

        img.add_updater(upd)
        upd(img)
        sdot = always_redraw(lambda: Dot(IMG_C + IMG_S * np.array([zk()[south].real, zk()[south].imag, 0]),
                                         radius=0.06, color=BLACK).set_z_index(3))
        ilab = mt(r"image $z = y_k$", 30).move_to([0.1, 3.0, 0])

        # residual plot
        P = Plot(3.6, -1.75, 3.0, 2.85, (0, KMAX), (1e-15, 1e1), ylog=True)
        axes = P.axes(xticks=[(0, "0"), (4, "4"), (8, "8"), (12, "12")],
                      yticks=[(1e0, mt(r"$10^{0}$", 30)), (1e-5, mt(r"$10^{-5}$", 30)),
                              (1e-10, mt(r"$10^{-10}$", 30)), (1e-15, mt(r"$10^{-15}$", 30))],
                      xlabel="step k")
        ptitle = mt(r"$|\mathsf{E}_C\,y - \lambda B\,y|$", 30).next_to(P.p(6, 1e1), UP, buff=0.12)
        ref = DashedVMobject(Line(P.p(2, RES[2]), P.p(8, RES[2] * RATIO ** 6), stroke_width=2.5, color=HEX["aux"]),
                             num_dashes=14)
        reflab = VGroup(mt(r"$\times\,\lambda_1/\lambda_2$", 30, color=HEX["aux"]),
                        mt(r"$= 0.042$", 30, color=HEX["aux"])).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        reflab.move_to(P.p(6.4, 1e-3), aligned_edge=UL)
        dots = [Dot(P.p(k, RES[k]), radius=0.06, color=C["tangent"]).set_z_index(2) for k in range(KMAX + 1)]

        def info():
            k = int(round(kv.get_value()))
            lam = f"{LAM[k]:.7f}" if LAM[k] < 1 else f"{LAM[k]:.1f}"
            return VGroup(lab(f"k = {k}", 30), lab(f"λ = {lam}", 30),
                          lab(f"flipped faces: {FLIPS[k]}", 30, color=C["normal"] if FLIPS[k] else BLACK)
                          ).arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([3.1, 3.4, 0], aligned_edge=UL)

        box = always_redraw(info)
        self.add(hemi, hlab, img, sdot, ilab, ttl, axes, ptitle, box, dots[0])
        self.wait(2.0)
        # step 1: the random parts disappear
        self.play(kv.animate.set_value(1.0), run_time=3.0, rate_func=smooth)
        self.add(dots[1])
        self.wait(1.5)
        self.play(kv.animate.set_value(2.0), run_time=1.5, rate_func=smooth)
        self.add(dots[2])
        self.play(FadeIn(ref), FadeIn(reflab), run_time=0.8)
        for k in range(3, KMAX + 1):
            self.play(kv.animate.set_value(float(k)), run_time=0.45, rate_func=linear)
            self.add(dots[k])
        img.remove_updater(upd)
        # the result is (nearly) the stereographic projection
        tb = IMG_C + IMG_S * np.stack([target[loop].real, target[loop].imag, 0 * target[loop].real], 1)
        poly = DashedVMobject(polyline(np.vstack([tb, tb[:1]]), color=BLACK, width=3.0), num_dashes=90)
        poly.set_z_index(4)
        res1 = mt(r"dashed: best fit $a\,\bar\sigma + c$", 30).move_to([-6.85, -2.62, 0], aligned_edge=LEFT)
        res2 = mt(rf"distance $0.0078$, $|a| = {abs(a):.3f}$", 30).move_to([-1.75, -2.62, 0], aligned_edge=LEFT)
        self.play(Create(poly), FadeIn(res1), FadeIn(res2), FadeOut(hlab), run_time=1.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.0)
