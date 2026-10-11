"""Animation 31.2.4: pushing the cube outward by epsilon and splitting the new area (Section 31.2, Proposition 31.2.6(b),
(31.2.5), Example 31.2.7).

The cube [-1, 1]^3 with outward normals (side 2, as in Example 31.2.7) is replaced by the surface of points at
distance epsilon from it, epsilon from 0 to 0.6. Its area is exactly
    A(eps) = A + eps sum_e theta_e l_e + eps^2 sum_i delta_i = 24 + 12 pi eps + 4 pi eps^2:
the grey faces only move (area 24), the sky-blue quarter cylinders along the 12 edges (theta_e = pi/2, l_e = 2) add
eps theta_e l_e each, the orange sphere pieces at the 8 vertices (angle defect pi/2) add eps^2 delta_i each.
Right: the stacked bar of these three parts. At the end the coefficients are matched with the smooth Steiner formula
(31.2.4): -2 iint H dA <-> sum theta_e l_e and iint K dA <-> sum delta_i.
Self-check: the bending angles and defects come from dgmesh on the triangulated cube of Example 31.1.2; the area of the
offset mesh (quarter circles split into 8) agrees with the formula within 1 percent (3 percent for the
sphere pieces).

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-2-4-steiner-offset.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (LEFT, FadeIn, Rectangle, ValueTracker, VGroup, always_redraw, smooth)  # noqa: E402

# ---------------------------------------------------------------- the triangulated cube (Example 31.1.2) and its data
CV = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
CF = []
for a, b, c, d in quads:
    CF += [[a, b, c], [a, c, d]]
CF = np.array(CF)
if dm.signed_volume(CV, CF) < 0:
    CF = CF[:, ::-1]
assert dm.counts(CF) == (8, 18, 12) and dm.oriented_consistently(CF)
BEND = dm.edge_bending_angles(CV, CF)
SUM_THETA_L = sum(th * l for th, l in BEND.values())
SUM_DELTA = dm.angle_defect(CV, CF).sum()
A0 = dm.face_areas(CV, CF).sum()
assert abs(A0 - 24) < 1e-12 and abs(SUM_THETA_L - 12 * np.pi) < 1e-12 and abs(SUM_DELTA - 4 * np.pi) < 1e-12
assert sorted(round(th, 12) for th, _ in BEND.values()).count(round(np.pi / 2, 12)) == 12   # 12 real edges
EPS1 = 0.6


def area_parts(eps):
    return A0, eps * SUM_THETA_L, eps ** 2 * SUM_DELTA


for eps in (0.1, 0.3, 0.6):
    T, lab = hc.rounded_cube(eps)
    A = hc.tri_area(T)
    parts = area_parts(eps)
    for k, tol in ((0, 1e-12), (1, 0.01), (2, 0.03)):           # inscribed polygons are slightly short
        assert abs(A[lab == k].sum() - parts[k]) <= tol * parts[k]
    assert abs(A.sum() - (24 + 12 * np.pi * eps + 4 * np.pi * eps ** 2)) < 0.01 * A.sum()
COLORS = {0: hc.rgb(hc.HEX["surface"]), 1: hc.rgb(hc.HEX["region"]), 2: hc.rgb(hc.HEX["accent"])}


class Anim(hc.Scene2D):
    def construct(self):
        eps = ValueTracker(1e-4)
        cam = hc.Cam(elev=24, azim=-58, scale=0.95, origin=(-4.95, 0.25))
        T, lab = hc.rounded_cube(1e-4)
        F = np.arange(3 * len(T)).reshape(-1, 3)
        mv = hc.MeshView(T.reshape(-1, 3), F, cam, colors=np.array([COLORS[x] for x in lab]), edge_color="fill",
                         edge_width=0.6, cull=True, sort=False, ambient=0.5)
        mv.add_updater(lambda m: m.set_mesh(hc.rounded_cube(eps.get_value())[0].reshape(-1, 3)))
        self.add(mv)
        title = hc.text("push the cube out by ε", size=36).move_to([-3.95, 3.45, 0])
        self.add(title)
        live = always_redraw(lambda: hc.text(f"ε = {eps.get_value():.2f}", size=32).move_to([-4.95, -2.75, 0]))
        self.add(live)

        # ---- stacked bar with the three terms of (31.2.5) next to their segments
        BX, B0, SC = -1.95, -2.7, 5.0 / 52.0

        def bars():
            parts = area_parts(eps.get_value())
            g = VGroup()
            y = B0
            for v, c in zip(parts, (hc.C["surface"], hc.C["region"], hc.C["accent"])):
                h = max(1e-3, v * SC)
                g.add(Rectangle(width=0.9, height=h, fill_color=c, fill_opacity=0.95, stroke_color=hc.BLACK,
                                stroke_width=1.2).move_to([BX, y + h / 2, 0]))
                y += h
            return g
        self.add(always_redraw(bars), hc.seg([BX - 0.7, B0, 0], [BX + 0.7, B0, 0], width=3))
        terms = [hc.math(r"$A$", size=34), hc.math(r"$\varepsilon\sum_e \theta_e\ell_e$", size=34, color=hc.C["tangent"]),
                 hc.math(r"$\varepsilon^2\sum_i \delta_i$", size=34, color=hc.C["accent"])]
        cols = [hc.BLACK, hc.C["tangent"], hc.C["accent"]]
        XT = BX + 0.65

        def ys():
            p = area_parts(eps.get_value())
            y0 = B0 + p[0] * SC / 2
            y1 = max(B0 + p[0] * SC + p[1] * SC / 2, y0 + 1.0)
            y2 = max(B0 + (p[0] + p[1]) * SC + p[2] * SC / 2, y1 + 1.0)
            return [y0, y1, y2]
        for k, m in enumerate(terms):
            m.add_updater(lambda m, k=k: m.move_to([XT, ys()[k], 0], aligned_edge=LEFT))
            m.update()                       # a static wait() does not run updaters
        self.add(*terms)

        def values():
            p = area_parts(eps.get_value())
            return VGroup(*[hc.text(f"= {v:.2f}", size=30, color=c).next_to(m, np.array([1.0, 0, 0]), buff=0.15)
                            for v, c, m in zip(p, cols, terms)])
        self.add(always_redraw(values))
        tot = always_redraw(lambda: hc.text(f"A(ε) = {sum(area_parts(eps.get_value())):.2f}", size=32)
                            .move_to([BX + 0.6, 3.0, 0]))
        self.add(tot)
        self.wait(2.0)
        self.play(eps.animate.set_value(EPS1), run_time=9.0, rate_func=smooth)
        self.wait(1.5)

        # ---- cube values and the smooth Steiner formula (31.2.4)
        XR = 4.65
        c1 = hc.math(r"$A(\varepsilon) = 24 + 12\pi\varepsilon + 4\pi\varepsilon^2$", size=30).move_to([XR - 0.15, 2.3, 0])
        c0 = hc.math(r"$\theta_e = \frac{\pi}{2},\ \ell_e = 2,\ \delta_i = \frac{\pi}{2}$", size=34).move_to([XR, 3.05, 0])
        self.play(FadeIn(c0), FadeIn(c1), run_time=0.9)
        self.wait(1.8)
        s0 = hc.text("smooth, (31.2.4):", size=30, color=hc.C["third"]).move_to([XR, 0.55, 0])
        s1 = hc.math(r"$\sum_e \theta_e\ell_e \ \leftrightarrow\ -2\iint H\,dA$", size=32,
                     color=hc.C["tangent"]).move_to([XR, -0.35, 0])
        s2 = hc.math(r"$\sum_i \delta_i \ \leftrightarrow\ \iint K\,dA$", size=32,
                     color=hc.C["accent"]).move_to([XR, -1.3, 0])
        self.play(FadeIn(s0), run_time=0.6)
        self.play(FadeIn(s1), run_time=0.8)
        self.play(FadeIn(s2), run_time=0.8)
        s3 = hc.text("H lives on the edges,", size=30).move_to([XR, -2.2, 0])
        s4 = hc.text("K on the vertices", size=30).move_to([XR, -2.7, 0])    # above the control bar
        self.play(FadeIn(s3), FadeIn(s4), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=20.0)
