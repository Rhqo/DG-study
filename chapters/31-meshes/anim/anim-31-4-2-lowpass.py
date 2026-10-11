"""Animation 31.4.2: rebuilding a shape from its own eigenfunctions, K = 1 -> 642 (low-pass filter, (31.4.3)).

Data (Figure 31.4.3): the level-3 icosphere (642 vertices) with radius 1 + 0.18 x z + 0.12 (y^2 - z^2) + 0.1 x and
12 Gaussian bumps (height 0.22, width 0.22, seed 7); eigenfunctions of -L phi = lambda M phi of this shape (mixed
Voronoi M). x^(K) = sum_{k<K} phi_k (phi_k^T M x); between integers K the next term is faded in linearly.
Left: x^(K) (camera elev 40, azim -120: the plane of x^(3) is seen obliquely, its normal 43 deg from the view). Right: relative error ||x - x^(K)||_M / ||x - c||_M against K
(log-log) and the wavelength 2 pi / sqrt(lambda_{K-1}) of the last eigenfunction added.
Self-check: x^(1) is the centroid; x^(2) lies on a line, x^(3) in a plane; errors 0.0728 (K = 4), 0.0399 (25),
0.0059 (100), 0.0014 (200), 0 (642); the error decreases monotonically.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-4-2-lowpass.py
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
N0 = len(V)
m = dm.mass_voronoi(V, F)
lam, Phi = dm.generalized_eigs(dm.cotan_laplacian(V, F), m)
coef = Phi.T @ (m[:, None] * V)                       # (642, 3)
TERM = Phi.T[:, :, None] * coef[:, None, :]           # term k: phi_k (phi_k^T M x), (642, n0, 3)
CUM = np.cumsum(TERM, axis=0)                         # CUM[K-1] = x^(K)
c0 = (m[:, None] * V).sum(0) / m.sum()
den = np.sqrt((m[:, None] * (V - c0) ** 2).sum())


def recon(kappa):
    K = int(np.floor(kappa + 1e-9))
    K = min(max(K, 1), N0)
    t = kappa - K
    X = CUM[K - 1].copy()
    if K < N0 and t > 1e-9:
        X += t * TERM[K]
    return X


def err(X):
    return np.sqrt((m[:, None] * (V - X) ** 2).sum()) / den


assert np.allclose(recon(1), c0[None, :], atol=1e-10)
assert np.linalg.svd(recon(2) - c0, compute_uv=False)[1] < 1e-9          # a segment
assert np.linalg.svd(recon(3) - c0, compute_uv=False)[2] < 1e-9          # a planar patch
_n3 = np.linalg.svd(recon(3) - c0, full_matrices=False)[2][2]
assert 0.7 < abs(_n3 @ Cam(elev=40, azim=-120).view) < 0.8               # ... seen obliquely (about 43 deg)
ERR = np.array([err(CUM[K - 1]) for K in range(1, N0 + 1)])
assert [round(ERR[K - 1], 4) for K in (4, 25, 100, 200)] == [0.0728, 0.0399, 0.0059, 0.0014] and ERR[-1] < 1e-10
assert np.all(np.diff(ERR) <= 1e-12)
WAVE = 2 * np.pi / np.sqrt(np.maximum(lam, 1e-300))
assert round(WAVE[24], 2) == 1.49 and round(WAVE[99], 2) == 0.76

STEPS = [(1, "K = 1: the centroid"), (2, "K = 2: a segment (1 direction)"),
         (3, "K = 3: a flat patch (2 directions)"), (4, "K = 4: an egg (3 directions)"),
         (25, "K = 25: the large bumps appear"), (100, "K = 100: almost the shape"),
         (642, "K = 642 (all): the shape itself")]


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=40, azim=-120, scale=1.7, center=[-3.55, 0.3, 0], pivot=c0)
        grey = hex_rgb(HEX["surface"])
        mv = MeshView(F, stroke_width=0.22, stroke_rgb=(0.45, 0.45, 0.45), ambient=0.4)
        kap = ValueTracker(1.0)

        def upd(mob):
            k = kap.get_value()
            mob.cull = k >= 3.0 - 1e-9
            mob.set_state(recon(k), grey, cam)

        mv.add_updater(upd)
        upd(mv)

        def seg():
            k = kap.get_value()
            if k > 2.0 + 1e-6:
                return VGroup()
            X = recon(k)
            P = cam(X)
            d = cam(c0)
            if k <= 1.0 + 1e-6:
                return Dot(d, radius=0.07, color=BLACK)
            u = P[:, :2] - d[:2]
            ax = np.linalg.svd(u, full_matrices=False)[2][0]
            s = u @ ax
            return Line([*(d[:2] + s.min() * ax), 0], [*(d[:2] + s.max() * ax), 0], stroke_width=5, color=BLACK)

        segm = always_redraw(seg)

        ttl = title("Low-pass: rebuild a shape from its first K eigenfunctions", 34)
        formula = mt(r"$x^{(K)} = \sum_{k<K}\,\varphi_k\,(\varphi_k^{\mathsf{T}} M x)$", 34).move_to([3.7, 2.75, 0])
        P = Plot(1.35, -1.9, 4.9, 3.5, (1, 700), (1e-4, 1.6), xlog=True, ylog=True)   # above the paused control bar
        axes = P.axes(xticks=[(1, "1"), (10, "10"), (100, "100"), (642, "642")],
                      yticks=[(1, "1"), (0.1, "0.1"), (0.01, "0.01"), (1e-3, "0.001"), (1e-4, "0.0001")],
                      xlabel=mt(r"number of eigenfunctions $K$", 30))
        ylab = da.label("relative error", 30).next_to(P.p(1, 1.6), UP, buff=0.12).shift(RIGHT * 0.9)

        def trace():
            k = kap.get_value()
            n = int(np.floor(k + 1e-9))
            xs = list(range(1, n + 1))
            ys = list(ERR[:n])
            if n < N0 and k - n > 1e-6:
                xs.append(k)
                ys.append(err(recon(k)))
            ys = np.maximum(ys, 1.2e-4)
            g = VGroup()
            if len(xs) > 1:
                g.add(P.curve(xs, ys, color=C["tangent"], width=4.5))
            g.add(Dot(P.p(xs[-1], ys[-1]), radius=0.08, color=C["accent"], stroke_width=1.2,
                      stroke_color=BLACK))
            return g

        curve = always_redraw(trace)

        def read_text():
            k = kap.get_value()
            K = int(np.round(k)) if abs(k - np.round(k)) < 1e-6 else k
            e = err(recon(k))
            ks = f"{K}" if isinstance(K, (int, np.integer)) else f"{k:.0f}"
            return f"K = {ks}    error {e:.3f}"

        read = readout(read_text, [-6.85, -2.25, 0], size=32)

        def wave_text():
            k = kap.get_value()
            K = max(1, int(np.ceil(k - 1e-6)))
            if K == 1:
                return "last wavelength: (constant)"
            return f"last wavelength 2π/√λ = {WAVE[K - 1]:.2f}"

        wave = readout(wave_text, [-6.85, -2.75, 0], size=30)

        step = da.label(STEPS[0][1], 32).move_to([-6.85, 2.75, 0], aligned_edge=LEFT)
        self.add(mv, segm, ttl, formula, axes, ylab, curve, read, wave, step)
        self.wait(1.6)
        for K, text in STEPS[1:]:
            if K <= 4:
                rt = 0.9
            else:
                rt = 3.0
            new = da.label(text, 32).move_to([-6.85, 2.75, 0], aligned_edge=LEFT)
            if K <= 4:
                self.play(kap.animate.set_value(K), FadeTransform(step, new), run_time=rt, rate_func=smooth)
            else:
                lo = kap.get_value()
                f = lambda t, lo=lo, K=K: np.exp(np.log(lo) + t * (np.log(K) - np.log(lo)))  # noqa: E731
                self.play(UpdateFromAlphaFunc(kap, lambda mob, a, f=f: mob.set_value(f(smooth(a)))),
                          FadeTransform(step, new), run_time=rt)
                kap.set_value(K)
            step = new
            self.wait(2.0 if K == 642 else 1.6)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.5)
