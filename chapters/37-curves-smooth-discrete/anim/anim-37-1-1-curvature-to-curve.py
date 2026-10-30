"""Animation 37.1.1: a curve from its curvature, and the two closing conditions (Section 37.1, (37.1.4),
Proposition 37.1.7, Example 37.1.8, Example 37.1.9, Figure 37.1.2).

The three integrals (37.1.4) are run in time: phi(s) = phi_0 + int kappa, t = (cos phi, sin phi), gamma = p_0 + int t,
with the trapezoid rule of the Implementation box of Section 37.1 (curve_from_curvature, m = 800 steps).
Run 1: kappa_s of the ellipse (2 cos t, sin t) as a function of arc length (ddg37.kappa at t(s), L = 9.6884),
p_0 = (2, 0), phi_0 = pi/2 as in Figure 37.1.2(b). Total turning 2 pi and the end meets the start (gap < 1e-4 at
m = 800): both conditions of Proposition 37.1.7 hold.
Run 2: kappa(s) = 1 + 0.5 cos s on [0, 2 pi], phi_0 = 0 (Example 37.1.9). Total turning again 2 pi, so the end has
the start direction, but the end point misses the start by |int t ds| = 2 pi J_1(0.5) = 1.5222: condition (ii) fails.
Self-check: L = 9.6884; int kappa = 2 pi (both runs, 1e-6); ellipse gap < 1e-4 and the recovered points match the
ellipse up to a rigid motion (ddg37.rigid_align_error < 1e-3); run 2 gap = 1.5222 (1e-4).

Render: bash tools/render_anims.sh chapters/37-curves-smooth-discrete/anim/anim-37-1-1-curvature-to-curve.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg37  # noqa: E402
import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
M = 800


def curve_from_curvature(kappa, L, phi0=0.0, p0=(0.0, 0.0)):
    """the Implementation box of Section 37.1 (same formulas)."""
    kappa = np.asarray(kappa, float)
    m = len(kappa) - 1
    ds = L / m
    phi = phi0 + np.concatenate([[0.0], np.cumsum(0.5 * (kappa[1:] + kappa[:-1]) * ds)])
    T = np.stack([np.cos(phi), np.sin(phi)], 1)
    steps = 0.5 * (T[1:] + T[:-1]) * ds
    return phi, np.asarray(p0, float) + np.vstack([[0.0, 0.0], np.cumsum(steps, 0)])


L1 = ddg37.smooth_length("ellipse")
assert abs(L1 - 9.6884) < 1e-4
S1 = np.linspace(0, L1, M + 1)
K1 = ddg37.kappa("ellipse", ddg37.t_of_s("ellipse", S1, 200000))
PHI1, G1 = curve_from_curvature(K1, L1, np.pi / 2, (2.0, 0.0))
assert abs(PHI1[-1] - PHI1[0] - 2 * np.pi) < 1e-5 and np.linalg.norm(G1[-1] - G1[0]) < 1e-4
ref = ddg37.point("ellipse", ddg37.t_of_s("ellipse", S1, 200000))
assert ddg37.rigid_align_error(G1, ref) < 1e-3

L2 = 2 * np.pi
S2 = np.linspace(0, L2, M + 1)
K2 = 1 + 0.5 * np.cos(S2)
PHI2, G2 = curve_from_curvature(K2, L2, 0.0, (0.0, 0.0))
GAP2 = np.linalg.norm(G2[-1] - G2[0])
assert abs(PHI2[-1] - 2 * np.pi) < 1e-9 and abs(GAP2 - 1.5222) < 1e-4

# ---------------------------------------------------------------- layout
PK = hp.Plot(-6.0, 0.55, 5.2, 1.85, (0, 10), (0, 2.2))       # kappa(s)
PF = hp.Plot(-6.0, -2.25, 5.2, 1.7, (0, 10), (0, 1.1))       # (phi - phi0) / 2 pi
CC, CS = np.array([3.0, -0.6, 0.0]), 1.2                    # curve plane: point (x, y) -> CC + CS (x, y)


def cp(p):
    return CC + CS * np.array([p[0], p[1], 0.0])


def state(S, K, PHI, G, j, color, done=None, ghost=None):
    out = []
    if ghost is not None:
        out.append(hp.polyline([cp(p) for p in ghost], color=hp.HEX["aux"], width=2.5))
    # kappa curve, area so far, phi curve so far
    out.append(PK.curve(S, K, color=hp.BLACK, width=3))
    if j > 0:
        area = [PK.p(S[0], 0)] + list(PK.pts(S[:j + 1], K[:j + 1])) + [PK.p(S[j], 0)]
        out.append(hp.poly_fill(area, hp.C["region"], opacity=0.35))
        out.append(PF.curve(S[:j + 1], (PHI[:j + 1] - PHI[0]) / (2 * np.pi), color=hp.C["tangent"], width=4))
        out.append(hp.polyline([cp(p) for p in G[:j + 1]], color=color, width=4))
    out.append(hp.seg(PK.p(S[j], 0), PK.p(S[j], K[j]), color=hp.C["accent"], width=3))
    out.append(hp.dot(PF.p(S[j], (PHI[j] - PHI[0]) / (2 * np.pi)), color=hp.C["tangent"], r=0.06))
    p, t = G[j], np.array([np.cos(PHI[j]), np.sin(PHI[j])])
    out += [hp.dot(cp(G[0]), color=hp.BLACK, r=0.07), hp.dot(cp(p), color=color, r=0.07),
            hp.arrow(cp(p), cp(p + 0.6 * t), color=hp.C["tangent"], width=5, tip=0.2)]
    turn = (PHI[j] - PHI[0]) / (2 * np.pi)
    out.append(hp.text(f"s = {S[j]:.2f}", 30).move_to([0.25, 2.9, 0], aligned_edge=hp.LEFT))
    out.append(hp.mt(r"$\int\kappa\,ds\ /\ 2\pi$", 30).move_to([3.15, 2.9, 0], aligned_edge=hp.LEFT))
    out.append(hp.text(f"= {turn:.3f}", 30, color=hp.C["tangent"]).move_to([5.3, 2.9, 0], aligned_edge=hp.LEFT))
    if done:
        out += done
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("curvature → tangent angle → tangent → curve", 32).move_to([0, 3.55, 0])
        axk = PK.axes(xticks=[(0, "0"), (5, "5"), (10, "10")], yticks=[(0, "0"), (2, "2")], size=30)
        axf = PF.axes(xticks=[(0, "0"), (5, "5"), (10, "10")], yticks=[(0, "0"), (1, "1")], size=30, zero_line=1.0)
        lk = hp.mt(r"$\kappa(s)$", 30).move_to(PK.p(6.6, 1.75), aligned_edge=hp.LEFT)
        lf = hp.mt(r"$(\varphi - \varphi_0)/2\pi$", 30, color=hp.C["tangent"]).move_to(PF.p(6.4, 0.3), aligned_edge=hp.LEFT)
        lsk = hp.mt(r"$s$", 30).next_to(PK.p(10, 0), hp.RIGHT, buff=0.12)
        lsf = hp.mt(r"$s$", 30).next_to(PF.p(10, 0), hp.RIGHT, buff=0.12)
        self.add(title, axk, axf, lk, lf, lsk, lsf)
        g = VGroup()
        self.add(g)
        S = hp.setter

        def ev_state(t, st):
            return (t, t, lambda u: S(g, st)(1))

        def run(Sx, K, PHI, G, color, dur, extra, ghost=None):
            ev, n = [], int(dur * 30)
            for k in range(n + 1):
                j = int(round(k * M / n))
                ev.append(ev_state(dur * k / n, state(Sx, K, PHI, G, j, color, done=extra, ghost=ghost)))
            return ev

        lab1 = hp.mt(r"ellipse: $\kappa_s$ of $(2\cos t,\ \sin t)$", 30).move_to([0.25, 2.1, 0], aligned_edge=hp.LEFT)
        S(g, state(S1, K1, PHI1, G1, 0, hp.BLACK, done=[lab1]))(1)
        self.mark("start")
        self.wait(1.5)
        ok1 = hp.text("turned 2\u03c0 and closed", 30, color=hp.C["third"]).move_to([0.25, -2.45, 0], aligned_edge=hp.LEFT)
        ev = run(S1, K1, PHI1, G1, hp.BLACK, 7.0, [lab1])
        ev.append(ev_state(7.0, state(S1, K1, PHI1, G1, M, hp.BLACK, done=[lab1, ok1])))
        self.play(hp.timeline(g, ev, 9.0))
        self.mark("run 1 done")
        # run 2
        lab2 = hp.mt(r"$\kappa(s) = 1 + 0.5\cos s$, $0 \leq s \leq 2\pi$", 30).move_to([0.25, 2.1, 0], aligned_edge=hp.LEFT)
        ev = run(S2, K2, PHI2, G2, hp.C["accent"], 6.0, [lab2])
        bad = [lab2, hp.seg(cp(G2[0]), cp(G2[-1]), color=hp.C["normal"], width=4),
               hp.text(f"turned 2\u03c0, gap {GAP2:.4f}", 30, color=hp.C["normal"]).move_to([0.25, -2.45, 0],
                                                                                         aligned_edge=hp.LEFT)]
        ev.append(ev_state(6.0, state(S2, K2, PHI2, G2, M, hp.C["accent"], done=bad)))
        self.play(hp.timeline(g, ev, 6.5))
        self.mark("run 2 done")
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.0)
