"""Animation 31.4.4: the heat kernel signature HKS(x, t) moves from local to global as t grows ((31.4.5), [Sun09]).

Data: the bumpy shape of Figure 31.4.3 (642 vertices, mixed Voronoi M, all 642 eigenpairs; h^2 = 0.028).
HKS(x, t) = sum_k exp(-lambda_k t) phi_k(x)^2. Colour: the scaled signature
    H(x, t) = HKS(x, t) * area / sum_k exp(-lambda_k t)   (area-weighted mean 1),
shown as (H - 1) / q with RdBu_r, q = the 95th percentile of |H - 1| at that t, clipped to [-1, 1] (red: more heat
stays at x than on average). The colour range is rescaled at every t, because max |H - 1| shrinks from about 0.35 to
0.001 over the sweep. Two views: azim -50 (front, as in the figure) and azim 130 (back), elev 20.
Right: Pearson correlation over the vertices of HKS(., t) with the point curvature K_i = delta_i / A_i (angle defect
over mixed area; local) and with phi_1^2 (the slowest vibration, large at both ends of the long axis; global), for
t from 0.05 to 5 (log axis).
Self-check: corr(HKS, K) = 0.805 and corr(HKS, phi_1^2) = 0.302 at t = 0.1; 0.235 and 0.747 at t = 2; the two curves
cross once, at t = 0.60 (log-linear interpolation between grid points); the area-weighted mean of H is 1.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-4-4-hks.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403


def bumpy_blob():
    """The shape of Figure 31.4.3 (same construction as fig-31-4-3-lowpass.py)."""
    V0, F = dm.icosphere(3)
    rng = np.random.default_rng(7)
    x, y, z = V0.T
    r = 1 + 0.18 * (x * z) + 0.12 * (y * y - z * z) + 0.1 * x
    cent = rng.normal(size=(12, 3))
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    for c in cent:
        r += 0.22 * np.exp(-np.sum((V0 - c) ** 2, 1) / (2 * 0.22 ** 2))
    return V0 * r[:, None], F


# ---------------------------------------------------------------- data
V, F = bumpy_blob()
m = dm.mass_voronoi(V, F)
lam, Phi = dm.generalized_eigs(dm.cotan_laplacian(V, F), m)
AREA = m.sum()
KV = dm.angle_defect(V, F) / m
P2 = Phi ** 2
c0 = (m[:, None] * V).sum(0) / AREA


def hks(t):
    return P2 @ np.exp(-lam * t)


def scaled(t):
    return hks(t) * AREA / np.exp(-lam * t).sum()


TS = np.logspace(np.log10(0.05), np.log10(5.0), 181)
CK = np.array([np.corrcoef(hks(t), KV)[0, 1] for t in TS])
CG = np.array([np.corrcoef(hks(t), P2[:, 1])[0, 1] for t in TS])
for t in (0.05, 0.5, 5.0):
    assert abs((m * scaled(t)).sum() / AREA - 1) < 1e-9
assert round(np.corrcoef(hks(0.1), KV)[0, 1], 3) == 0.805 and round(np.corrcoef(hks(0.1), P2[:, 1])[0, 1], 3) == 0.302
assert round(np.corrcoef(hks(2.0), KV)[0, 1], 3) == 0.235 and round(np.corrcoef(hks(2.0), P2[:, 1])[0, 1], 3) == 0.747
sgn = np.sign(CK - CG)
cross = np.where(np.diff(sgn) != 0)[0]
assert len(cross) == 1 and 0.3 < TS[cross[0]] < 1.5
_i = cross[0]
_d0, _d1 = CK[_i] - CG[_i], CK[_i + 1] - CG[_i + 1]
T_CROSS = 10 ** (np.log10(TS[_i]) + (np.log10(TS[_i + 1]) - np.log10(TS[_i])) * _d0 / (_d0 - _d1))
assert round(T_CROSS, 2) == 0.60


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        cams = [Cam(elev=20, azim=-50, scale=1.2, center=[-5.0, -0.25, 0], pivot=c0),
                Cam(elev=20, azim=130, scale=1.2, center=[-1.75, -0.25, 0], pivot=c0)]
        lt = ValueTracker(np.log10(0.05))
        views = VGroup()
        for cam in cams:
            mv = MeshView(F, stroke_width=0.1, stroke_rgb=(0.35, 0.35, 0.35), ambient=0.5)

            def upd(mob, cam=cam):
                t = 10 ** lt.get_value()
                H = scaled(t) - 1
                c = np.clip(H / np.quantile(np.abs(H), 0.95), -1, 1)
                mob.set_state(V, cmap_rgb(c[F].mean(1), -1, 1, "RdBu_r"), cam)

            mv.add_updater(upd)
            upd(mv)
            views.add(mv)
        vlab = VGroup(da.label("front", 30).move_to([-5.0, -1.8, 0]), da.label("back", 30).move_to([-1.75, -1.8, 0]))
        ttl = title("Heat kernel signature: from local to global", 34)
        sub = mt(r"$\mathrm{HKS}(x, t) = \sum_k e^{-\lambda_k t}\,\varphi_k(x)^2$", 34).move_to([-6.85, 2.65, 0], aligned_edge=LEFT)
        bar = colorbar(-4.4, -2.5, 2.3, 0.2, "RdBu_r", -1, 1, ticks=[(-1, "less"), (1, "more")], size=30)
        bar_lab = da.label("heat kept", 30).move_to([-6.85, -2.4, 0], aligned_edge=LEFT)
        tread = readout(lambda: f"t = {10 ** lt.get_value():.2f}", [-1.35, -2.4, 0], size=32)

        P = Plot(1.45, -1.9, 4.6, 3.25, (0.05, 5.0), (0, 1.0), xlog=True)   # above the paused control bar
        axes = P.axes(xticks=[(0.05, "0.05"), (0.5, "0.5"), (5, "5")],
                      yticks=[(0, "0"), (0.5, "0.5"), (1, "1")], xlabel=mt(r"time $t$", 32))
        ylab = da.label("correlation of HKS with", 30).move_to([1.0, 2.75, 0], aligned_edge=LEFT)
        legK = mt(r"curvature $K$  (local)", 30, color=C["third"]).move_to([1.25, 2.3, 0], aligned_edge=LEFT)
        legG = mt(r"$\varphi_1^2$  (global)", 30, color=C["covector"])
        legG.move_to([1.25, 1.85, 0], aligned_edge=LEFT)

        def traced():
            lv = lt.get_value()
            n = max(2, int(np.searchsorted(np.log10(TS), lv, side="right")))
            g = VGroup(P.curve(TS[:n], CK[:n], color=C["third"], width=5),
                       P.curve(TS[:n], CG[:n], color=C["covector"], width=5))
            t = 10 ** lv
            for arr, col in ((CK, C["third"]), (CG, C["covector"])):
                y = np.interp(lv, np.log10(TS), arr)
                g.add(Dot(P.p(t, y), radius=0.08, color=col, stroke_color=BLACK, stroke_width=1.2))
            g.add(DashedLine(P.p(t, 0), P.p(t, 1.0), stroke_width=1.5, color=C["aux"]))
            return g

        tr = always_redraw(traced)
        step = da.label("small t: the bumps stand out", 30).move_to([-6.85, 1.7, 0], aligned_edge=LEFT)
        self.add(views, vlab, ttl, sub, bar, bar_lab, tread, axes, ylab, legK, legG, tr, step)
        self.wait(1.8)
        self.play(lt.animate.set_value(np.log10(0.1)), run_time=1.5)
        self.wait(1.8)
        step2 = da.label("larger t: heat spreads further", 30).move_to([-6.85, 1.7, 0], aligned_edge=LEFT)
        self.play(lt.animate.set_value(np.log10(T_CROSS)), FadeTransform(step, step2), run_time=4.0, rate_func=linear)
        self.wait(1.5)
        step3 = da.label("large t: ends of the long axis", 30).move_to([-6.85, 1.7, 0], aligned_edge=LEFT)
        self.play(lt.animate.set_value(np.log10(2.0)), FadeTransform(step2, step3), run_time=3.5, rate_func=linear)
        self.wait(1.8)
        self.play(lt.animate.set_value(np.log10(5.0)), run_time=2.0, rate_func=linear)
        self.wait(2.2)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.0)
