"""Animation 31.4.1: the spectrum of a sphere mesh comes in groups, and the groups split when the symmetry breaks.

Data (Section 31.4, Figures 31.4.1 and 31.4.2): the unit icosphere of level 3 (642 vertices), mixed Voronoi mass,
generalized eigenproblem -L phi = lambda M phi (31.4.1).
Part 1: phi_k for k = 0, ..., 15 (degrees l <= 3) on the sphere (colour phi_k / max|phi_k|, RdBu_r, red > 0), and the
        eigenvalues lambda_k appearing one by one next to the exact values l (l + 1), each repeated 2 l + 1 times.
Part 2: the same mesh stretched by (1 + s (1.6 - 1), 1, 1 + s (0.7 - 1)), s from 0 (sphere) to 1 (the ellipsoid of
        Figure 31.4.2); colour = phi_1 of the current shape, curves = lambda_1, ..., lambda_8 against s.
Self-check: lambda_0 = 0, lambda_1..3 = 2 (to 2e-5), lambda_4..8 = 5.966 (one value), lambda_9..15 split 3 + 4
(11.803, 11.851); at s = 1 lambda_1..3 = 1.13, 2.02, 2.28 and phi_1, phi_2, phi_3 follow the x, y, z axes
(|correlation| > 0.95).

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-4-1-sphere-spectrum.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data
V0, F = dm.icosphere(3)
assert len(V0) == 642 and len(F) == 1280
lam, Phi = dm.generalized_eigs(dm.cotan_laplacian(V0, F), dm.mass_voronoi(V0, F), 16)
assert abs(lam[0]) < 1e-9 and np.allclose(lam[1:4], 2.0, atol=2e-5)
assert np.allclose(lam[4:9], lam[4], atol=1e-6) and round(lam[4], 3) == 5.966
assert np.allclose(lam[9:12], 11.8029, atol=1e-3) and np.allclose(lam[12:16], 11.8509, atol=1e-3)
FV = []
for k in range(16):
    f = Phi[:, k] / np.abs(Phi[:, k]).max()
    FV.append(f[F].mean(1))
FV = np.array(FV)

AX = np.array([1.6, 1.0, 0.7])
S = np.linspace(0, 1, 101)
LAM, F1 = [], []
for s in S:
    W = V0 * (1 + s * (AX - 1))
    l_, P_ = dm.generalized_eigs(dm.cotan_laplacian(W, F), dm.mass_voronoi(W, F), 9)
    LAM.append(l_[1:9])
    p1 = P_[:, 1] * np.sign(np.corrcoef(P_[:, 1], W[:, 0])[0, 1] + 1e-12)
    F1.append((p1 / np.abs(p1).max())[F].mean(1))
    if s == 1.0:
        assert [round(x, 2) for x in l_[1:4]] == [1.13, 2.02, 2.28]
        for k, axis in ((1, 0), (2, 1), (3, 2)):
            assert abs(np.corrcoef(P_[:, k], W[:, axis])[0, 1]) > 0.95
LAM, F1 = np.array(LAM), np.array(F1)
assert np.allclose(LAM[0, :3], 2.0, atol=2e-5) and np.all(np.diff(LAM[-1]) > 0)


def stretch(s):
    return V0 * (1 + s * (AX - 1))


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=22, azim=-55, scale=1.85, center=[-3.45, 0.6, 0])
        mv = MeshView(F, stroke_width=0.25)
        kk = ValueTracker(0.0)
        ss = ValueTracker(0.0)
        mode = {"part": 1}

        def upd(m):
            if mode["part"] == 1:
                x = kk.get_value()
                k0 = int(np.floor(x + 1e-9))
                t = x - k0
                k1 = min(k0 + 1, 15)
                f = (1 - t) * FV[k0] + t * FV[k1]
                m.set_state(V0, cmap_rgb(f, -1, 1), cam)
            else:
                s = ss.get_value()
                j = int(np.clip(np.round(s * 100), 2, 100))
                f = F1[j]
                if mode.get("blend", 1.0) < 1.0:     # cross-fade from phi_15 of the sphere
                    b = mode["blend"]
                    f = (1 - b) * FV[15] + b * f
                m.set_state(stretch(s), cmap_rgb(f, -1, 1), cam)

        mv.add_updater(upd)
        upd(mv)

        ttl = title("Laplacian eigenfunctions of a sphere mesh (642 vertices)", 34)
        bar = colorbar(-3.9, -2.6, 2.9, 0.2, "RdBu_r", -1, 1,
                       ticks=[(-1, "-1"), (0, "0"), (1, "1")], size=30)
        bar_lab = mt(r"$\varphi/\max|\varphi|$", 30).next_to(bar, LEFT, buff=0.3).shift(UP * 0.12)

        P = Plot(1.0, -1.95, 4.9, 4.1, (-0.8, 15.8), (0, 14))   # above the paused control bar (bottom 12%)
        axes = P.axes(xticks=[(0, "0"), (4, "4"), (9, "9"), (15, "15")],
                      yticks=[(0, "0"), (2, "2"), (6, "6"), (12, "12")],
                      xlabel=mt(r"index $k$", 32))
        ylab = mt(r"$\lambda_k$", 34).next_to(P.p(-0.8, 14), LEFT, buff=0.15)
        legend = mt(r"black: $\ell(\ell+1)$, $2\ell+1$ times", 30).move_to([P.x0 + P.w / 2 + 0.3, 2.75, 0])

        def read_k(k):
            m = mt(rf"$\varphi_{{{k}}}:\ \ \lambda_{{{k}}} = {abs(lam[k]):.2f}$", 36)
            return m.move_to([-3.45, -1.85, 0])

        self.add(mv, ttl, bar, bar_lab, axes, ylab, legend)
        read = read_k(0)
        self.add(read)
        groups = {0: (0, 0), 3: (1, 2), 8: (2, 6), 15: (3, 12)}
        stairs = VGroup()
        dots = VGroup()

        def stair(l):
            lo, hi = l * l - 0.5, (l + 1) ** 2 - 0.5
            y = l * (l + 1)
            seg = Line(P.p(lo, y), P.p(hi, y), stroke_width=4, color=BLACK)
            lab = mt(rf"$\ell = {l}$", 30).next_to(P.p((lo + hi) / 2, y), UP, buff=0.18)
            if l == 0:
                lab.next_to(P.p(hi, y), RIGHT, buff=0.15).shift(UP * 0.25)
            return VGroup(seg, lab)

        def dot(k):
            return Dot(P.p(k, lam[k]), radius=0.075, color=C["accent"], stroke_width=1.2,
                       stroke_color=BLACK).set_z_index(3)

        d0 = dot(0)
        dots.add(d0)
        self.add(d0)
        self.wait(1.2)
        st = stair(0)
        stairs.add(st)
        self.play(Create(st[0]), FadeIn(st[1]), run_time=0.5)
        self.wait(0.6)
        for k in range(1, 16):
            hold = 0.8 if k < 9 else 0.45
            d = dot(k)
            dots.add(d)
            new = read_k(k)
            self.play(kk.animate.set_value(k), FadeIn(d, scale=0.4), FadeTransform(read, new),
                      run_time=0.35, rate_func=smooth)
            read = new
            self.wait(hold)
            if k in groups:
                st = stair(groups[k][0])
                stairs.add(st)
                self.play(Create(st[0]), FadeIn(st[1]), run_time=0.6)
                self.wait(1.0)

        # ---------------------------------------------------------- part 2: break the symmetry
        P2 = Plot(1.0, -1.95, 4.5, 4.1, (0, 1), (0, 8.2))
        axes2 = P2.axes(xticks=[(0, "sphere"), (1, "ellipsoid")],
                        yticks=[(0, "0"), (2, "2"), (6, "6")],
                        xlabel=mt(r"stretch $s$", 32))
        ylab2 = mt(r"$\lambda_1, \dots, \lambda_8$", 32).next_to(P2.p(0, 8.2), UP, buff=0.15).shift(RIGHT * 0.6)
        ttl2 = mt(r"Stretch to axes $(1.6,\ 1.0,\ 0.7)$: every group splits", 34).to_edge(UP, buff=0.3)
        read2 = mt(r"colour: $\varphi_1$ of this shape", 34).move_to([-3.45, -1.85, 0])
        mode["part"] = 2
        mode["blend"] = 0.0
        blend = ValueTracker(0.0)

        def set_blend(m):
            mode["blend"] = blend.get_value()

        mv.add_updater(set_blend, index=0)
        self.play(FadeOut(VGroup(axes, ylab, legend, dots, stairs)), FadeTransform(ttl, ttl2),
                  FadeTransform(read, read2), run_time=0.8)
        self.play(FadeIn(VGroup(axes2, ylab2)), blend.animate.set_value(1.0), run_time=0.8)
        mv.remove_updater(set_blend)
        mode["blend"] = 1.0
        cols = [C["accent"]] * 3 + [C["tangent"]] * 5

        def traced():
            s = ss.get_value()
            n = max(2, int(np.floor(s * 100)) + 1)
            g = VGroup()
            for k in range(8):
                g.add(P2.curve(S[:n], LAM[:n, k], color=cols[k], width=4.5))
            return g

        curves = always_redraw(traced)
        grp = VGroup(mt(r"$\ell = 1$: $2, 2, 2$", 30, color=C["accent"]).move_to(P2.p(0.06, 0.7), aligned_edge=LEFT),
                     mt(r"$\ell = 2$: $5.97$, 5 times", 30, color=C["tangent"]).move_to(P2.p(0.06, 7.75), aligned_edge=LEFT))
        self.add(curves)
        self.play(FadeIn(grp), run_time=0.5)
        self.play(ss.animate.set_value(1.0), run_time=6.5, rate_func=linear)
        ends = VGroup()
        for k, txt in ((0, "1.13"), (1, "2.02"), (2, "2.28")):
            m = da.label(txt, 30, color=C["accent"]).next_to(P2.p(1.0, LAM[-1, k]), RIGHT, buff=0.12)
            ends.add(m)
        ends[1].shift(DOWN * 0.1)
        ends[2].shift(UP * 0.17)
        self.play(FadeIn(ends), run_time=0.6)
        self.wait(2.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=21.5)   # the full staircase (end of part 1)
