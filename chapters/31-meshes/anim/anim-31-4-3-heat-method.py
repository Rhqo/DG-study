"""Animation 31.4.3: the three steps of the heat method on the unit sphere [Crane13].

Data (Figure 31.4.4): unit icosphere level 3 (642 vertices), barycentric mass, source = the vertex nearest the north
pole, h = mean edge length 0.151, t = h^2 = 0.0227.
  I.   (M - tau L) u = delta for tau from t/100 to t (one backward Euler step of size tau; the last one is step I of
       the method). Colour = log10 u on the fixed range of the final u (magma), values below it are black.
  II.  X = -grad u / |grad u| on 150 front faces (white arrows, length 0.11, the faces of the figure).
  III. -L phi = sum_f A_f G_f^T X_f with phi(source) = 0. A front phi = d sweeps d from 0 to pi (orange), leaving
       black level lines every pi/8; on the right the vertices with theta <= d appear: heat method phi (orange) and
       the magnitude-only estimate sqrt(-4 t log(u / u_source)) (blue) against the exact distance theta = arccos z.
Self-check (as the figure): max |phi - theta| = 0.050, mean 0.021; u spans more than 8 orders of magnitude; the
magnitude-only estimate is 1.34 at the far pole.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-4-3-heat-method.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data (same steps as fig-31-4-4-heat-method.py)
V, F = dm.icosphere(3)
L = dm.cotan_laplacian(V, F)
m = dm.mass_barycentric(V, F)
h = np.mean([np.linalg.norm(V[a] - V[b]) for a, b in dm.edges(F)])
T = h * h
SRC = int(np.argmax(V[:, 2]))
delta = np.zeros(len(V))
delta[SRC] = 1.0
TAUS = T * np.logspace(-2, 0, 61)
U = np.array([np.linalg.solve(np.diag(m) - tau * L, delta) for tau in TAUS])
u = U[-1]
N = dm.face_normals(V, F)
A = dm.face_areas(V, F)


def face_grad(f):
    g = np.zeros((len(F), 3))
    for c in range(3):
        e = V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]]
        g += f[F[:, c], None] * np.cross(N, e) / (2 * A[:, None])
    return g


G = face_grad(u)
X = -G / np.linalg.norm(G, axis=1, keepdims=True)
b = np.zeros(len(V))
for c in range(3):
    e = V[F[:, (c + 2) % 3]] - V[F[:, (c + 1) % 3]]
    np.add.at(b, F[:, c], 0.5 * (np.cross(N, e) * X).sum(1))
idx = np.array([i for i in range(len(V)) if i != SRC])
phi = np.zeros(len(V))
phi[idx] = np.linalg.solve(-L[np.ix_(idx, idx)], b[idx])
theta = np.arccos(np.clip(V[:, 2], -1, 1))
err = np.abs(phi - theta)
assert round(err.max(), 3) == 0.050 and round(err.mean(), 3) == 0.021
assert 1e8 < u.max() / u.min() < 1e9
naive = np.sqrt(np.maximum(-4 * T * np.log(u / u[SRC]), 0))
assert round(naive[np.argmin(V[:, 2])], 2) == 1.34
LU = np.log10(U)
LO, HI = LU[-1].min(), LU[-1].max()
assert LO > -7.6 and HI < 1.2 and np.all(U > 0)
cen = V[F].mean(1)
cam0 = Cam(elev=35, azim=-60)
front = (N @ cam0.view) > 0.25
rng = np.random.default_rng(0)
pick = np.where(front)[0]
PICK = pick[rng.permutation(len(pick))[:150]]
ORD = np.argsort(theta)


def level_segments(vals, c):
    """Segments of the piecewise-linear level set {vals = c} (one per crossed face): (k, 2, 3), face ids."""
    fv = vals[F]
    s = fv > c
    cross = s.any(1) & ~s.all(1)
    segs, ids = [], []
    for f in np.where(cross)[0]:
        pts = []
        for a, bb in ((0, 1), (1, 2), (2, 0)):
            va, vb = fv[f, a], fv[f, bb]
            if (va > c) != (vb > c):
                w = (c - va) / (vb - va)
                pts.append(V[F[f, a]] + w * (V[F[f, bb]] - V[F[f, a]]))
        if len(pts) == 2:
            segs.append(pts)
            ids.append(f)
    return np.array(segs).reshape(-1, 2, 3), np.array(ids, int)


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=35, azim=-60, scale=1.8, center=[-3.55, -0.05, 0])
        mv = MeshView(F, stroke_width=0.15, stroke_rgb=(0.25, 0.25, 0.25), ambient=0.55)
        state = {"heat": 0.0, "grey": 0.0}
        tauk = ValueTracker(0.0)
        greyv = ValueTracker(0.0)
        grey = hex_rgb("#E8E8E8")

        def upd(mob):
            j = tauk.get_value()
            j0 = int(np.floor(j))
            t_ = j - j0
            j1 = min(j0 + 1, len(TAUS) - 1)
            lu = (1 - t_) * LU[j0] + t_ * LU[j1]
            rgb = cmap_rgb(lu[F].mean(1), LO, HI, "magma")
            if state["heat"] < 1:
                rgb = (1 - state["heat"]) * grey + state["heat"] * rgb
            g = greyv.get_value()
            if g > 0:
                rgb = (1 - g) * rgb + g * grey
            mob.set_state(V, rgb, cam)

        mv.add_updater(upd)
        upd(mv)
        star = Star(n=5, outer_radius=0.17, inner_radius=0.075, color=C["accent"], fill_opacity=1,
                    stroke_color=BLACK, stroke_width=1.5).move_to(cam(V[SRC])).set_z_index(5)
        src_lab = da.label("source", 30).next_to(star, UP + RIGHT, buff=0.02).set_z_index(5)

        ttl = title("The heat method on the unit sphere (642 vertices)", 34)
        bar = colorbar(-5.0, -2.5, 3.6, 0.2, "magma", LO, HI,
                       ticks=[(-7, "-7"), (-3, "-3"), (1, "1")], size=30)
        bar_lab = mt(r"$\log_{10} u$", 30).next_to(bar, LEFT, buff=0.25).shift(UP * 0.1)
        stepI = mt(r"I. heat from one vertex: $(M - tL)\,u = \delta$", 32).move_to([-6.85, 2.85, 0], aligned_edge=LEFT)
        self.add(mv, star, src_lab, ttl, stepI)
        self.wait(1.3)
        tread = readout(lambda: f"t = {TAUS[int(round(tauk.get_value()))]:.4f}", [-6.85, 2.25, 0], size=30)
        self.add(tread)
        hv = ValueTracker(0.0)

        def set_heat(mob):
            state["heat"] = hv.get_value()

        mv.add_updater(set_heat, index=0)
        self.play(hv.animate.set_value(1.0), FadeIn(bar), FadeIn(bar_lab), run_time=0.8)
        self.play(tauk.animate.set_value(len(TAUS) - 1), run_time=5.0, rate_func=linear)
        mv.remove_updater(set_heat)
        state["heat"] = 1.0
        self.remove(tread)
        tfin = mt(r"$t = h^2 = 0.0227$:  $u$ spans 8 orders of magnitude", 30).move_to([-6.85, 2.25, 0],
                                                                                     aligned_edge=LEFT)
        self.add(tfin)
        self.wait(1.8)

        # ---------------------------------------------------------- II: directions only
        stepII = mt(r"II. keep only the direction  $X = -\nabla u\,/\,|\nabla u|$", 32).move_to([-6.85, 2.85, 0],
                                                                                         aligned_edge=LEFT)
        arrows = VGroup()
        for f in PICK:
            p0 = cam(cen[f])
            p1 = cam(cen[f] + 0.11 * X[f])
            arrows.add(Arrow(p0, p1, buff=0, stroke_width=2.2, color=WHITE, tip_length=0.07,
                             max_stroke_width_to_length_ratio=100, max_tip_length_to_length_ratio=0.45))
        arrows.set_z_index(4)
        self.play(FadeTransform(stepI, stepII), FadeOut(tfin), run_time=0.6)
        tII = da.label("all arrows: away from the source, along meridians", 30).move_to([-6.85, 2.25, 0],
                                                                                      aligned_edge=LEFT)
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows], lag_ratio=0.01), FadeIn(tII), run_time=1.8)
        self.wait(2.0)

        # ---------------------------------------------------------- III: Poisson -> distance
        stepIII = mt(r"III. solve $\Delta\varphi = \mathrm{div}\,X$: $\varphi$ = distance", 32).move_to(
            [-6.85, 2.85, 0], aligned_edge=LEFT)
        P = Plot(1.0, -1.9, 4.1, 4.1, (0, np.pi), (0, np.pi + 0.15))   # above the paused control bar
        axes = P.axes(xticks=[(0, "0"), (np.pi / 2, mt(r"$\pi/2$", 30)), (np.pi, mt(r"$\pi$", 30))],
                      yticks=[(0, "0"), (np.pi / 2, mt(r"$\pi/2$", 30)), (np.pi, mt(r"$\pi$", 30))],
                      xlabel=mt(r"exact distance $\theta = \arccos z$", 30))
        ylab = da.label("estimated distance", 30).next_to(P.p(0, np.pi + 0.15), UP, buff=0.15).shift(RIGHT * 2.05)
        diag = Line(P.p(0, 0), P.p(np.pi, np.pi), stroke_width=2.5, color=BLACK)
        # direct labels, clear of the data: orange above the diagonal, blue below the blue points
        leg_h = mt(r"heat method $\varphi$", 30, color=C["accent"]).move_to(P.p(0.08, np.pi + 0.05), aligned_edge=UL)
        leg_n = VGroup(mt(r"magnitude only:", 30, color=C["tangent"]),
                       mt(r"$\sqrt{-4t\,\log(u/u_{\mathrm{src}})}$", 30, color=C["tangent"]))
        leg_n.arrange(DOWN, aligned_edge=LEFT, buff=0.06).move_to(P.p(1.6, 0.95), aligned_edge=UL)
        assert leg_n.get_top()[1] < float(P.Y(naive[theta >= 1.6].min())) - 0.07       # below every blue point
        assert leg_n.get_right()[0] < 7.11 - 0.2 and leg_n.get_bottom()[1] > P.y0 + 0.05
        y_low = (leg_h.get_bottom()[1] - P.y0) / P.h * (np.pi + 0.15)    # data height of the label's bottom
        assert leg_h.get_right()[0] < float(P.X(y_low)) - 0.2              # left of the diagonal there
        leg = VGroup(leg_h, leg_n)
        dots_h = [Dot(P.p(theta[i], phi[i]), radius=0.032, color=C["accent"]).set_z_index(3) for i in ORD]
        dots_n = [Dot(P.p(theta[i], naive[i]), radius=0.032, color=C["tangent"]).set_z_index(2) for i in ORD]
        shown = VGroup()
        dd = ValueTracker(0.0)
        ths = theta[ORD]

        def upd_dots(mob):
            n = int(np.searchsorted(ths, dd.get_value(), side="right"))
            mob.submobjects = dots_h[:n] + dots_n[:n]

        shown.add_updater(upd_dots)
        arrows_dim = arrows.copy().set_color(GREY_B).set_opacity(0.8)
        self.play(FadeTransform(stepII, stepIII), FadeOut(tII), greyv.animate.set_value(1.0),
                  FadeOut(bar), FadeOut(bar_lab), Transform(arrows, arrows_dim),
                  FadeIn(VGroup(axes, ylab, diag, leg)), run_time=1.2)

        def iso_mob(c, color, width):
            segs, ids = level_segments(phi, c)
            vis = (N[ids] @ cam.view) > 0
            mob = VMobject(stroke_color=color, stroke_width=width)
            for sgm in segs[vis]:
                a, b2 = cam(sgm[0]), cam(sgm[1])
                mob.start_new_path(a)
                mob.add_line_to(b2)
            return mob

        levels = [k * np.pi / 8 for k in range(1, 8)]

        def fronts():
            d = dd.get_value()
            g = VGroup()
            for c in levels:
                if c <= d:
                    g.add(iso_mob(c, BLACK, 2.5))
            if 0.02 < d < np.pi - 0.02:
                g.add(iso_mob(d, C["accent"], 6))
            return g

        fr = always_redraw(fronts)
        fr.set_z_index(4)
        self.add(fr, shown)
        self.play(dd.animate.set_value(np.pi), run_time=8.0, rate_func=linear)
        res = VGroup(mt(r"heat method: max error $0.050$", 30, color=C["accent"]),
                     mt(r"magnitude only: $1.34$ at $\theta = \pi$", 30, color=C["tangent"]))
        res.arrange(DOWN, aligned_edge=LEFT, buff=0.08).move_to([-6.85, -1.97, 0], aligned_edge=UL)
        assert res.get_bottom()[1] > -2.95                      # above the paused control bar
        lines_lab = mt(r"black: level lines of $\varphi$, every $\pi/8$", 30).move_to([-6.85, 2.25, 0],
                                                                                    aligned_edge=LEFT)
        self.play(FadeIn(res), FadeIn(lines_lab), run_time=0.7)
        self.wait(2.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=24.3)   # the final state with the results
