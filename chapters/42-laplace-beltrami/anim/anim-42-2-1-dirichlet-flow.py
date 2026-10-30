"""Animation 42.2.1: with the boundary values held fixed, heat flow slides down the Dirichlet energy to the harmonic
interpolant (Section 42.2, Theorem 42.2.4, Figure 42.2.1).

Data (as Figure 42.2.1): dgmesh.grid_mesh on [0, 1]^2 with 32 x 32 squares (alternating diagonals, 1089 vertices),
boundary values g = sin(pi x) on the top side y = 1 and 0 on the other three sides. Start: the blend w = y sin(pi x) of
Figure 42.2.1(b) at the vertices. Flow: the heat equation M du/dt = L u at the interior vertices with u = g kept on the
boundary (L = dgmesh.cotan_laplacian, M = mixed Voronoi areas), i.e. the L2 gradient flow of the discrete Dirichlet
energy E(u) = -1/2 u^T L u among the functions with the boundary values g. It is solved exactly in time with the
generalized eigenvectors of (-L_II, M_II):  u(t) - u* = sum_j c_j exp(-lambda_j t) phi_j,  u* = the discrete
harmonic interpolant (L_II u*_I = -L_IB g_B, Section 42.5).
The video time tau runs t = 0.3 (tau / 11 s)^2 (slow at first, where the flow changes most).
Left: the square coloured by u (Blues on [0, 1.15], white 0, as Figure 42.2.1), black level sets 0.1, 0.3, ..., 0.9
(contourpy on the vertex grid), the top side thick blue. Right: E(u(t)) against t (orange), the dashed line E(u*).
Self-check: u* is within 3e-4 of sin(pi x) sinh(pi y) / sinh(pi) at the vertices; E(w) = 1.0722, E(u*) = 0.7890 (the
values of Figure 42.2.1 on the mesh); E decreases along the flow and E(u(t)) = E(u*) + E(u(t) - u*) to 1e-10
((42.2.2) on the mesh); at t = 0.3 max |u - u*| < 1e-3.

Render: bash tools/render_anims.sh chapters/42-laplace-beltrami/anim/anim-42-2-1-dirichlet-flow.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import _helpers as hp  # noqa: E402
import contourpy  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import DashedVMobject, FadeIn, ValueTracker, VGroup, VMobject, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
N = 32
XS = np.linspace(0, 1, N + 1)
V, F = dm.grid_mesh(XS, XS)
L = dm.cotan_laplacian(V, F)
MASS = dm.mass_voronoi(V, F)
B = np.array(dm.boundary_vertices(F))
I = np.setdiff1d(np.arange(len(V)), B)
G = np.where(np.isclose(V[:, 1], 1.0), np.sin(np.pi * V[:, 0]), 0.0)
USTAR = G.copy()
USTAR[I] = np.linalg.solve(L[np.ix_(I, I)], -L[np.ix_(I, B)] @ G[B])
U_EXACT = np.sin(np.pi * V[:, 0]) * np.sinh(np.pi * V[:, 1]) / np.sinh(np.pi)
assert np.abs(USTAR - U_EXACT).max() < 3e-4
W0 = V[:, 1] * np.sin(np.pi * V[:, 0])
assert np.allclose(W0[B], G[B])


def energy(u):
    return float(-0.5 * u @ L @ u)


assert abs(energy(W0) - 1.0722) < 5e-5 and abs(energy(USTAR) - 0.7890) < 5e-5
S = 1 / np.sqrt(MASS[I])
LAM, Q = np.linalg.eigh(-L[np.ix_(I, I)] * np.outer(S, S))          # M^{-1/2} (-L_II) M^{-1/2}
C0 = Q.T @ ((W0 - USTAR)[I] / S)                                    # coefficients of M^{1/2} (w - u*)
assert LAM.min() > 19 and abs(LAM.min() - 2 * np.pi ** 2) < 0.1     # the slowest mode ~ 2 pi^2


def u_at(t):
    u = USTAR.copy()
    u[I] += S * (Q @ (C0 * np.exp(-LAM * t)))
    return u


T_END = 0.3
TS = np.linspace(0, T_END, 400)
ES = np.array([energy(u_at(t)) for t in TS])
assert np.all(np.diff(ES) < 1e-14)
for t in (0.0, 0.01, 0.05, 0.2):
    ut = u_at(t)
    assert abs(energy(ut) - energy(USTAR) - energy(ut - USTAR)) < 1e-10
assert np.abs(u_at(T_END) - USTAR).max() < 1e-3
assert abs(energy(u_at(0.1)) - 0.7938) < 5e-5                         # quoted in the caption
GRIDIDX = np.lexsort((V[:, 0], V[:, 1])).reshape(N + 1, N + 1)     # GRIDIDX[j, i]: vertex at (x_i, y_j)
assert np.allclose(V[GRIDIDX[:, 0], 1], XS) and np.allclose(V[GRIDIDX[0, :], 0], XS)
LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
TAU_END = 11.0


def t_of(tau):
    return T_END * (np.clip(tau, 0, TAU_END) / TAU_END) ** 2


class Anim(hp.Scene2D):
    def construct(self):
        tau = ValueTracker(0.0)
        SQ, O = 5.2, np.array([-6.15, -2.95])                         # (x, y) -> O + SQ (x, y)

        def sc(x, y):
            return np.array([O[0] + SQ * x, O[1] + SQ * y, 0.0])

        # colour cells: one slot per grid square, recoloured every frame (fixed submobject list)
        cells = VGroup()
        for j in range(N):
            for i in range(N):
                c = VMobject(stroke_width=0.8, fill_opacity=1)
                c.set_points_as_corners([sc(XS[i], XS[j]), sc(XS[i + 1], XS[j]), sc(XS[i + 1], XS[j + 1]),
                                         sc(XS[i], XS[j + 1]), sc(XS[i], XS[j])])
                cells.add(c)

        def recolor(m):
            u = u_at(t_of(tau.get_value()))[GRIDIDX]
            cellv = 0.25 * (u[:-1, :-1] + u[1:, :-1] + u[:-1, 1:] + u[1:, 1:])
            rgbs = hp.cmap_rgb(cellv.ravel(), 0, 1.15, "Blues")
            for c, col in zip(m.submobjects, rgbs):
                c.set_fill(hp.ManimColor(tuple(col)), opacity=1)
                c.set_stroke(hp.ManimColor(tuple(col)), width=0.8, opacity=1)
        cells.add_updater(recolor)
        cells.update()

        def contours():
            u = u_at(t_of(tau.get_value()))[GRIDIDX]
            gen = contourpy.contour_generator(XS, XS, u)
            g = VGroup()
            for lev in LEVELS:
                for line in gen.lines(lev):
                    if len(line) > 1:
                        g.add(hp.polyline(np.array([sc(x, y) for x, y in line]), color=hp.C["main"], width=2.5))
            return g
        cont = always_redraw(contours)
        frame = VGroup(hp.polyline([sc(0, 1), sc(0, 0), sc(1, 0), sc(1, 1)], color=hp.C["aux"], width=2),
                       hp.seg(sc(0, 1), sc(1, 1), color=hp.C["tangent"], width=7))
        gtop = hp.math(r"$g = \sin\pi x$", 30, color=hp.C["tangent"]).next_to(sc(0.5, 1), np.array([0, 1.0, 0]),
                                                                            buff=0.12)
        start = hp.math(r"start: the blend $w = y\,\sin\pi x$", 30).move_to([3.7, 3.4, 0])

        # ---- the energy plot
        P = hp.Plot(1.7, -2.1, 4.6, 3.35, (0, T_END), (0.75, 1.1))
        ax = P.axes(xticks=[(0, "0"), (0.1, "0.1"), (0.2, "0.2"), (0.3, "0.3")],
                    yticks=[(0.8, "0.8"), (0.9, "0.9"), (1.0, "1.0")])
        xl = hp.math(r"$t$", 32).next_to(P.p(T_END, 0.75), np.array([1.0, 0, 0]), buff=0.15) \
            .shift(np.array([0, 0.15, 0]))                              # clear of the tick label 0.3
        yl = hp.math(r"$E_D$", 32).next_to(P.p(0, 1.1), np.array([0, 1.0, 0]), buff=0.12)
        floor = DashedVMobject(hp.seg(P.p(0, energy(USTAR)), P.p(T_END, energy(USTAR)), color=hp.C["main"], width=2.5),
                               num_dashes=30)
        fl = hp.math(r"dashed: harmonic $u$, $0.7890$", 30).move_to([1.0, 2.2, 0], aligned_edge=np.array([-1.0, 0, 0]))

        def trace():
            t = t_of(tau.get_value())
            k = int(np.searchsorted(TS, t, side="right"))
            xs = np.r_[TS[:k], t]
            ys = np.r_[ES[:k], energy(u_at(t))]
            return VGroup(P.curve(xs, ys, color=hp.C["accent"], width=5),
                          hp.dot(P.p(t, ys[-1]), color=hp.C["accent"], r=0.07))
        tr = always_redraw(trace)
        el = hp.math(r"$E_D(u) =$", 32).move_to([1.0, 2.75, 0], aligned_edge=np.array([-1.0, 0, 0]))
        en = always_redraw(lambda: hp.text(f"{energy(u_at(t_of(tau.get_value()))):.4f}", 32, color=hp.C["accent"])
                           .next_to(el, np.array([1.0, 0, 0]), buff=0.18))
        tl = hp.math(r"$t =$", 32).move_to([4.7, 2.75, 0], aligned_edge=np.array([-1.0, 0, 0]))
        tn = always_redraw(lambda: hp.text(f"{t_of(tau.get_value()):.3f}", 32).next_to(tl, np.array([1.0, 0, 0]),
                                                                                       buff=0.15))

        self.add(cells, cont, frame, gtop, start, ax, xl, yl, floor, fl, tr, el, en, tl, tn)
        for m in (cont, tr, en, tn):
            m.update()
        self.wait(2.0)
        t2 = hp.text("heat flow, g held fixed", 30).move_to(start)
        self.play(*hp.swap(start, t2), run_time=0.5)
        self.play(tau.animate.set_value(TAU_END), run_time=TAU_END, rate_func=linear)
        t3 = hp.math(r"the harmonic $u$: smallest $E_D$", 30, color=hp.C["tangent"]).move_to(start)
        self.play(*hp.swap(t2, t3), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=15.5)
