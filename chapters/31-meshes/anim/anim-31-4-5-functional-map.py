"""Animation 31.4.5: the functional map C stays diagonal under an isometric bend and leaves the diagonal under stretching.

Data (Figure 31.4.5): source M = the planar rectangle [0, 2] x [0, 0.7], 41 x 15 vertices, alternating diagonals;
targets N with the same connectivity (the correspondence T is the vertex identity), mixed Voronoi mass, 25
eigenpairs, C = Phi_N^T M_N Phi_M, shown as |C_ij| for i, j < 20 (white 0, black 1.1).
Phases (each a continuous family, eigenproblems solved on a grid of the parameter and |C| interpolated between):
  (a) isometric bend with amplitude A from 0 to 0.25: x is rolled onto z = A sin(pi t) keeping arc length
      (A = 0.25 is the bend of the figure); then back to flat;
  (b) uniform stretch x -> s x, s from 1 to 1.1;
  (c) from 1.1 x to the non-uniform stretch x (1 + 0.3 x / 2) of the figure (linear blend of the two maps).
Left: the target plate coloured by its phi_3 (RdBu_r, sign kept continuous between grid steps). Row i = 3 of C
is outlined in orange.
Self-check: at A = 0.25 |C| = identity to 1e-3 and edge lengths change by < 1e-3; at s = 1.1 one entry ~ sqrt(1.1)
per row and column, rows 3 and 4 swapped (lambda_3 and lambda_4 cross near s = 1.05); at the end 56 entries > 0.1.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-4-5-functional-map.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data
A_, B_ = 2.0, 0.7
V, F = dm.grid_mesh(np.linspace(0, A_, 41), np.linspace(0, B_, 15), pattern="alternate")
E = dm.edges(F)
LEN0 = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)


def eig(W, k=25):
    mm = dm.mass_voronoi(W, F)
    lam, Phi = dm.generalized_eigs(dm.cotan_laplacian(W, F), mm, k)
    return lam, Phi, mm


lamM, PhiM, mM = eig(V)
TT = np.linspace(0, 3.0, 30001)


def bend(amp):
    if amp == 0:
        return V.copy()
    zz = amp * np.sin(np.pi * TT)
    sarc = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(TT) ** 2 + np.diff(zz) ** 2))])
    xa = np.interp(V[:, 0], sarc, TT)
    return np.stack([xa, V[:, 1], amp * np.sin(np.pi * xa)], 1)


def stretch(s):
    return np.c_[s * V[:, 0], V[:, 1], V[:, 2]]


def blend(u):
    x = V[:, 0]
    return np.c_[(1 - u) * 1.1 * x + u * x * (1 + 0.3 * x / A_), V[:, 1], V[:, 2]]


def family(shape_fn, params):
    """|C| (20 x 20) and the sign-continuous phi_3 of the target for each parameter value."""
    Cs, P3 = [], []
    prev = None
    for p in params:
        W = shape_fn(p)
        lam, Phi, mm = eig(W)
        Cs.append(np.abs((Phi.T @ (mm[:, None] * PhiM))[:20, :20]))
        f = Phi[:, 3].copy()
        ref = prev if prev is not None else PhiM[:, 3]
        if (f * mm) @ ref < 0:
            f = -f
        prev = f
        P3.append(f / np.abs(f).max())
    return np.array(Cs), np.array(P3)


AMPS = np.linspace(0, 0.25, 41)
SS = np.linspace(1.0, 1.1, 61)
US = np.linspace(0, 1, 41)
C_bend, P_bend = family(bend, AMPS)
C_str, P_str = family(stretch, SS)
C_mix, P_mix = family(blend, US)
Wb = bend(0.25)
assert np.abs(np.linalg.norm(Wb[E[:, 0]] - Wb[E[:, 1]], axis=1) / LEN0 - 1).max() < 1e-3
assert np.abs(C_bend[-1] - np.eye(20)).max() < 1e-3 and np.abs(C_bend - np.eye(20)).max() < 2e-3
Bm = C_str[-1]
assert ((Bm > 0.5).sum(0) == 1).all() and ((Bm > 0.5).sum(1) == 1).all() and np.allclose(Bm[Bm > 0.5], np.sqrt(1.1), atol=0.01)
assert Bm[3, 4] > 0.5 and Bm[4, 3] > 0.5 and Bm[3, 3] < 0.1
assert (C_mix[-1] > 0.1).sum() == 56
# row 3 still matches column 3 at s = 1.04 and has moved to column 4 at s = 1.06 (the eigenvalues cross between)
assert np.isclose(SS[24], 1.04) and np.isclose(SS[36], 1.06)
assert C_str[24][3].argmax() == 3 and C_str[36][3].argmax() == 4


def interp(arr, x):
    """Linear interpolation along the first axis at fractional index x."""
    i0 = int(np.clip(np.floor(x), 0, len(arr) - 1))
    i1 = min(i0 + 1, len(arr) - 1)
    t = x - i0
    return (1 - t) * arr[i0] + t * arr[i1]


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=48, azim=-72, scale=2.35, center=[-3.75, 0.35, 0], pivot=(1.25, 0.35, 0.0))
        mv = MeshView(F, stroke_width=0.15, stroke_rgb=(0.3, 0.3, 0.3), cull=False, ambient=0.55)
        st = {"phase": "bend", "x": 0.0}
        prm = ValueTracker(0.0)

        def current():
            x = prm.get_value()
            ph = st["phase"]
            if ph == "bend":
                amp = np.interp(x, np.arange(len(AMPS)), AMPS)
                return bend(amp), interp(C_bend, x), P_bend[int(round(x))]
            if ph == "stretch":
                s = np.interp(x, np.arange(len(SS)), SS)
                return stretch(s), interp(C_str, x), P_str[int(round(x))]
            u = np.interp(x, np.arange(len(US)), US)
            return blend(u), interp(C_mix, x), P_mix[int(round(x))]

        def upd(mob):
            W, _, f = current()
            mob.set_state(W, cmap_rgb(f[F].mean(1), -1, 1), cam)

        mv.add_updater(upd)
        upd(mv)

        # matrix |C|
        n, cell = 20, 0.22
        x0, y0 = 1.25, 2.55                       # top-left corner (labels stay above the paused control bar)
        cells = VGroup(*[Square(cell, stroke_width=0, fill_opacity=1, fill_color=WHITE) for _ in range(n * n)])
        for i in range(n):
            for j in range(n):
                cells[i * n + j].move_to([x0 + (j + 0.5) * cell, y0 - (i + 0.5) * cell, 0])

        def upd_cells(mob):
            _, Cm, _ = current()
            g = 1 - np.clip(Cm / 1.1, 0, 1)
            for k, sq in enumerate(mob):
                v = g.flat[k]
                sq.fill_rgbas = np.array([[v, v, v, 1.0]])

        cells.add_updater(upd_cells)
        upd_cells(cells)
        frame = Rectangle(width=n * cell, height=n * cell, stroke_width=1.5, color=BLACK).move_to(
            [x0 + n * cell / 2, y0 - n * cell / 2, 0])
        row3 = Rectangle(width=n * cell, height=cell, stroke_width=3.5, color=C["accent"]).move_to(
            [x0 + n * cell / 2, y0 - 3.5 * cell, 0]).set_z_index(3)
        ticks = VGroup()
        for v in (0, 10, 19):
            ticks.add(da.label(str(v), 30).next_to([x0 + (v + 0.5) * cell, y0 - n * cell, 0], DOWN, buff=0.1))
            ticks.add(da.label(str(v), 30).next_to([x0, y0 - (v + 0.5) * cell, 0], LEFT, buff=0.1))
        jl = mt(r"$j$: source $\varphi^M_j$", 30).next_to([x0 + n * cell / 2, y0 - n * cell, 0], DOWN, buff=0.5)
        il = mt(r"$i$: target $\varphi^N_i$", 30).next_to([x0 + n * cell / 2, y0, 0], UP, buff=0.12)
        r3 = mt(r"$3$", 30, color=C["accent"]).next_to([x0, y0 - 3.5 * cell, 0], LEFT, buff=0.1)
        bar = colorbar(6.25, -1.75, 0.22, 2.4, "Greys", 0, 1.1, ticks=[(0, "0"), (1.0, "1")], size=30, vertical=True)
        bar_lab = mt(r"$|C_{ij}|$", 30).next_to(bar, UP, buff=0.12)
        plate_lab = mt(r"colour: $\varphi_3$ of the target plate", 30).move_to([-6.85, 2.75, 0], aligned_edge=LEFT)

        ttl = title("Functional map between a plate and its deformed copy", 34)

        def phase_text(a, b):
            g = VGroup(da.label(a, 32), da.label(b, 32)).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
            return g.move_to([-6.85, -1.3, 0], aligned_edge=UL)
        step = phase_text("source: the flat plate", "C = identity")
        self.add(mv, cells, frame, row3, ticks, jl, il, r3, bar, bar_lab, plate_lab, ttl, step)
        self.wait(1.8)


        new = phase_text("(a) bend without stretching:", "C stays diagonal")
        self.play(FadeTransform(step, new), run_time=0.5)
        step = new
        self.play(prm.animate.set_value(len(AMPS) - 1), run_time=4.0, rate_func=smooth)
        self.wait(1.6)
        self.play(prm.animate.set_value(0), run_time=1.2, rate_func=smooth)
        st["phase"] = "stretch"
        prm.set_value(0)
        new = phase_text("(b) stretch x by 10%:", "entries leave the diagonal")
        self.play(FadeTransform(step, new), run_time=0.5)
        step = new
        self.play(prm.animate.set_value(len(SS) - 1), run_time=5.0, rate_func=linear)
        swap = mt(r"$\varphi_3 \leftrightarrow \varphi_4$: their eigenvalues crossed", 30, color=C["accent"])
        swap.move_to([-6.85, -2.62, 0], aligned_edge=LEFT)
        assert swap.get_bottom()[1] > -2.95 and jl.get_bottom()[1] > -2.95     # above the paused control bar
        self.play(FadeIn(swap), run_time=0.4)
        self.wait(1.6)
        st["phase"] = "mix"
        prm.set_value(0)
        new = phase_text("(c) stretch unevenly:", "C spreads out")
        self.play(FadeTransform(step, new), FadeOut(swap), run_time=0.5)
        step = new
        self.play(prm.animate.set_value(len(US) - 1), run_time=4.0, rate_func=smooth)
        self.wait(2.2)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.0)
