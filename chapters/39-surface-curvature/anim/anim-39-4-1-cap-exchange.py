"""Animation 39.4.1: as a polar cap grows, the area term and the boundary term trade off and their sum stays 2 pi
(Section 39.4, Theorem 39.4.4, Example 39.4.5, (39.4.3), Proposition 39.4.6, Example 39.4.7, Figures 39.4.1, 39.4.2).

Part 1: the unit sphere (N outward), cap {theta <= theta_0}, theta_0 from 0 to pi: int K dA = 2 pi (1 - cos theta_0),
int kappa_g ds = 2 pi cos theta_0 (kappa_g = cot theta_0 on the boundary circle, run with phi increasing: positive
orientation). Stops at theta_0 = pi/3 (both pi) and pi/2 (the boundary is a geodesic, kappa_g = 0).
Part 2: the bumpy sphere rho = 1 + 0.2 cos 4t (lib/ddg39.bumpy_sphere, N outward), cap {t <= t_0}, coloured by K
(RdBu_r, white = 0, K from -5.01 to 9.34 as Figure 39.4.2); the two terms from ddg39.cap_terms (the Implementation
box of Section 39.4). The area term goes up and down with the bands of K, the boundary term mirrors it.
Bars: int K dA (vermilion), int kappa_g ds (green), their sum (black), axis from -2 pi to 4 pi.
Self-check: cap_terms of the round sphere profile equals the closed forms (1e-5); for the bumpy sphere the sum is
2 pi within 1e-4 for every t_0 shown and the whole surface gives 4 pi; K range [-5.01, 9.34] (2 decimals).

Render: bash tools/render_anims.sh chapters/39-surface-curvature/anim/anim-39-4-1-cap-exchange.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg39  # noqa: E402
import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
SPH = lambda t: (np.sin(t), np.cos(t))  # noqa: E731
for th in (0.4, 1.0, np.pi / 2, 2.3):
    A, B = ddg39.cap_terms(SPH, th, n=4000)
    assert abs(A - 2 * np.pi * (1 - np.cos(th))) < 1e-5 and abs(B - 2 * np.pi * np.cos(th)) < 1e-5
T1 = np.concatenate([np.linspace(0.02, np.pi / 3, 36), np.linspace(np.pi / 3, np.pi / 2, 17)[1:],
                     np.linspace(np.pi / 2, np.pi - 0.02, 54)[1:]])
TERMS1 = [(2 * np.pi * (1 - np.cos(t)), 2 * np.pi * np.cos(t)) for t in T1]
T2 = np.linspace(0.02, np.pi - 0.02, 106)
TERMS2 = [ddg39.cap_terms(ddg39.bumpy_profile, t, n=2000) for t in T2]
for A, B in TERMS2:
    assert abs(A + B - 2 * np.pi) < 1e-4
assert abs(ddg39.cap_terms(ddg39.bumpy_profile, np.pi, n=8000)[0] - 4 * np.pi) < 1e-3

# meshes: (t, v) grids, faces oriented by x_t x x_v (outward)
NT, NV = 30, 40
TT = np.linspace(1e-3, np.pi - 1e-3, NT + 1)
VV = np.linspace(0, 2 * np.pi, NV + 1)


def grid_mesh(fn):
    T, Vg = np.meshgrid(TT, VV, indexing="ij")
    P = np.moveaxis(fn(T, Vg), 0, -1).reshape(-1, 3)
    idx = np.arange((NT + 1) * (NV + 1)).reshape(NT + 1, NV + 1)
    F, tc = [], []
    for i in range(NT):
        for j in range(NV):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            F += [[a, b, c], [a, c, d]]
            tc += [0.5 * (TT[i] + TT[i + 1])] * 2
    return P, np.array(F), np.array(tc)


PS, FS, TCS = grid_mesh(lambda t, v: np.array([np.sin(t) * np.cos(v), np.sin(t) * np.sin(v), np.cos(t) + 0 * v]))
PB, FB, TCB = grid_mesh(ddg39.bumpy_sphere)
for P, F in ((PS, FS), (PB, FB)):
    tri = P[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ok = np.linalg.norm(n, axis=1) > 1e-9
    assert (np.einsum("ij,ij->i", n[ok], tri[ok].mean(1)) > 0).all()     # outward
KB = np.array([ddg39.revolution_K(ddg39.bumpy_profile, t) for t in TCB])
tt = np.linspace(1e-3, np.pi - 1e-3, 20001)
Kfine = ddg39.revolution_K(ddg39.bumpy_profile, tt)
assert round(Kfine.min(), 2) == -5.01 and round(Kfine.max(), 2) == 9.34
KMIN, KMAX = -5.01, 9.34


def k_colors(K):
    t = np.where(K >= 0, 0.5 + 0.5 * K / KMAX, 0.5 - 0.5 * K / KMIN)
    return hp.cmap_rgb(t, 0, 1, "RdBu_r")


COLB = k_colors(KB)

# ---------------------------------------------------------------- drawing
CAM = hp.Cam(elev=18, azim=-60, scale=2.15, origin=(-3.55, -0.35))
FADE = 0.72                                    # how much the faces outside the cap are whitened
BARX = [1.55, 3.15, 4.75]
BAR_Y0, BAR_S = -0.45, 0.17                    # y of the value 0, scene units per unit of value


def by(v):
    return BAR_Y0 + BAR_S * v


def base(P, F, colors):
    faded = colors * (1 - FADE) + FADE
    return hp.mesh_mobs(P, F, CAM, colors=faded, edge_color="#bdbdbd", edge_width=0.3, cull=True, ambient=0.75)


def overlay(P, F, tc, colors, t0, fn):
    mask = tc <= t0
    out = hp.mesh_mobs(P, F, CAM, colors=colors, edge_color="#6a6a6a", edge_width=0.4, cull=True, ambient=0.6,
                       hidden=~mask)
    # boundary circle t = t0 (front half solid on top, back half hidden by culling of the front faces' order)
    v = np.linspace(0, 2 * np.pi, 121)
    C = np.moveaxis(fn(np.full_like(v, t0), v), 0, -1)
    for k in range(120):
        a, b = C[k], C[k + 1]
        mid = 0.5 * (a + b)
        front = (mid @ CAM.view) > 0.0
        if front:
            out.append(hp.seg(CAM(a), CAM(b), color=hp.BLACK, width=4))
        elif k % 2 == 0:                                   # hidden half: grey, dashed
            out.append(hp.seg(CAM(a), CAM(b), color=hp.HEX["aux"], width=2))
    # t and N x t at three front points
    h = 1e-5
    for vv in (np.deg2rad(-75), np.deg2rad(-20), np.deg2rad(35)):
        p = fn(t0, vv)
        if p @ CAM.view <= 0.05:
            continue
        xt = (fn(t0 + h, vv) - fn(t0 - h, vv)) / (2 * h)
        xv = (fn(t0, vv + h) - fn(t0, vv - h)) / (2 * h)
        N = np.cross(xt, xv)
        N /= np.linalg.norm(N)
        tv = xv / np.linalg.norm(xv)
        nt = np.cross(N, tv)
        out.append(hp.arrow(CAM(p), CAM(p + 0.38 * tv), color=hp.C["tangent"], width=5, tip=0.16))
        out.append(hp.arrow(CAM(p), CAM(p + 0.38 * nt), color=hp.C["third"], width=5, tip=0.16))
    return out


def bars(A, B, label_t):
    out = []
    for x, val, col in ((BARX[0], A, hp.C["normal"]), (BARX[1], B, hp.C["third"]), (BARX[2], A + B, hp.BLACK)):
        y0, y1 = sorted([by(0), by(val)])
        if y1 - y0 > 1e-3:
            out.append(hp.poly_fill([[x - 0.42, y0, 0], [x + 0.42, y0, 0], [x + 0.42, y1, 0], [x - 0.42, y1, 0]],
                                    col, opacity=0.75))
        out.append(hp.text(hp.num(val, 2), 30, color=col).move_to([x, max(by(val), by(0)) + 0.28, 0]))
    out.append(hp.text(label_t, 32).move_to([1.1, 2.95, 0], aligned_edge=hp.LEFT))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("Gauss–Bonnet for a growing cap: the two terms trade off", 32).move_to([0, 3.55, 0])
        axis = VGroup(hp.seg([1.0, by(-2 * np.pi), 0], [1.0, by(4 * np.pi), 0], width=2.5),
                      hp.seg([1.0, by(0), 0], [5.35, by(0), 0], width=2),
                      hp.seg([1.0, by(2 * np.pi), 0], [5.35, by(2 * np.pi), 0], color=hp.HEX["aux"], width=2,
                             dashed=True))
        ticks = VGroup(*[hp.text(s, 30).move_to([0.86, by(v), 0], aligned_edge=hp.RIGHT) for v, s in
                         ((0, "0"), (2 * np.pi, "2π"), (4 * np.pi, "4π"), (-2 * np.pi, "−2π"))])
        labs = VGroup(hp.mt(r"$\int K\,dA$", 30, color=hp.C["normal"]).move_to([BARX[0], -2.3, 0]),
                      hp.mt(r"$\int\kappa_g\,ds$", 30, color=hp.C["third"]).move_to([BARX[1], -2.3, 0]),
                      hp.text("sum", 30).move_to([BARX[2], -2.3, 0]))
        key = hp.mt(r"blue $\mathbf{t}$, green $\mathbf{N}\times\mathbf{t}$ (into the cap)", 30).move_to([-6.85, 2.95, 0],
                                                                                                    aligned_edge=hp.LEFT)
        self.add(title, axis, ticks, labs, key)
        sph_base = VGroup(*base(PS, FS, np.tile(hp.rgb(hp.HEX["surface"]), (len(FS), 1))))
        self.add(sph_base)
        g = VGroup()
        self.add(g)
        S = hp.setter
        blue = np.tile(hp.rgb(hp.HEX["region"]), (len(FS), 1))
        sphere_fn = lambda t, v: np.array([np.sin(t) * np.cos(v), np.sin(t) * np.sin(v), np.cos(t) + 0 * v])  # noqa

        def sph_state(k, note=None):
            t0 = T1[k]
            A, B = TERMS1[k]
            out = overlay(PS, FS, TCS, blue, t0, sphere_fn) + bars(A, B, f"sphere:  θ₀ = {np.degrees(t0):.0f}°")
            if note:
                out.append(hp.text(note, 30, color=hp.C["third"]).move_to([1.1, 2.4, 0], aligned_edge=hp.LEFT))
            return out

        S(g, sph_state(0))(1)
        self.mark("start")
        self.wait(1.5)
        ev, t = [], 0.0
        k60, k90 = 35, 35 + 16
        for k in range(1, len(T1)):
            note = "equal: π and π" if k == k60 else ("boundary is a geodesic" if k == k90 else None)
            ev.append((t, t, (lambda kk, nn: (lambda u: S(g, sph_state(kk, nn))(1)))(k, note)))
            t += 1 / 15 if k not in (k60, k90) else 1.6
        self.play(hp.timeline(g, ev, t + 1.0))
        self.mark("sphere done")
        # part 2: bumpy sphere
        self.remove(sph_base)
        bump_base = VGroup(*base(PB, FB, COLB))
        self.add(bump_base)
        self.bring_to_front(g)

        def bump_state(k, note=None):
            t0 = T2[k]
            A, B = TERMS2[k]
            out = overlay(PB, FB, TCB, COLB, t0, ddg39.bumpy_sphere) + bars(A, B, f"bumpy sphere: t₀ = {np.degrees(t0):.0f}°")
            if note:
                out.append(hp.text(note, 30, color=hp.C["third"]).move_to([1.1, 2.4, 0], aligned_edge=hp.LEFT))
            else:
                out.append(hp.text("colour: K (red +, blue \u2212)", 30).move_to([1.1, 2.4, 0], aligned_edge=hp.LEFT))
            return out

        S(g, bump_state(0))(1)
        self.wait(1.5)
        ev, t = [], 0.0
        for k in range(1, len(T2)):
            ev.append((t, t, (lambda kk: (lambda u: S(g, bump_state(kk))(1)))(k)))
            t += 1 / 15
        ev.append((t, t, lambda u: S(g, bump_state(len(T2) - 1, "the sum never moves: 2π"))(1)))
        self.play(hp.timeline(g, ev, t + 0.3))
        self.mark("bumpy done")
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.0)
