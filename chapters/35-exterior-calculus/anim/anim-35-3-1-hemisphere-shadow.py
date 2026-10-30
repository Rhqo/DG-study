"""Animation 35.3.1: integrating dx ^ dy over the upper hemisphere -- adding up the shadows of the patches
(Section 35.3, Definition 35.3.8, Example 35.3.10(b), Figure 35.3.2).

Unit sphere, upper hemisphere S+ (0 <= theta <= pi/2) with the theta, phi grid of Figure 35.3.2 (spacing pi/12), outward
N, orientation x_theta x x_phi. The 2-form dx ^ dy measures a small patch by the signed area of its vertical shadow
((dx ^ dy)(x_theta, x_phi) = sin theta cos theta > 0 on S+). The six theta-bands are added from the pole down: band k
(theta in [k pi/12, (k + 1) pi/12]) turns orange on the dome and its shadow, an annulus, is filled on the disk in the
plane z = -0.95. Right: for every band its surface area 2 pi (cos theta_k - cos theta_{k+1}) (grey) and its
shadow = integral of dx ^ dy = pi (sin^2 theta_{k+1} - sin^2 theta_k) (orange), with running totals.
The surface areas grow toward the equator (0.214 ... 1.626), the shadows do not (0.210, 0.575, 0.785, 0.785, 0.575,
0.210): near the equator N is almost horizontal and the patches are flat in the shadow. The shadows add up to pi,
the area of the disk; the surface areas to 2 pi.
Self-check: band integrals of the integrand sin theta cos theta (Gauss-Legendre) equal the closed forms; their sum is
pi and the surface areas sum to 2 pi; per cell (phi width pi/6) the bands 2, 4, 6 give 0.0479, 0.0654, 0.0175 as in
Figure 35.3.2; the shadow annuli tile the unit disk.

Render: ``bash tools/render_anims.sh chapters/35-exterior-calculus/anim/anim-35-3-1-hemisphere-shadow.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, Polygon, Rectangle, VGroup)  # noqa: E402

NB, NP = 6, 24                                     # theta bands and phi sectors (spacing pi/12)
TH = np.pi / 12 * np.arange(NB + 1)
PH = 2 * np.pi * np.arange(NP + 1) / NP
ZP = -0.95
AREA = 2 * np.pi * (np.cos(TH[:-1]) - np.cos(TH[1:]))
SHADOW = np.pi * (np.sin(TH[1:]) ** 2 - np.sin(TH[:-1]) ** 2)
g, w = np.polynomial.legendre.leggauss(12)
for k in range(NB):
    t = TH[k] + (TH[k + 1] - TH[k]) * (g + 1) / 2
    I = 2 * np.pi * np.sum(w * np.sin(t) * np.cos(t)) * (TH[k + 1] - TH[k]) / 2     # int dx^dy over band k
    A = 2 * np.pi * np.sum(w * np.sin(t)) * (TH[k + 1] - TH[k]) / 2                  # surface area of band k
    assert abs(I - SHADOW[k]) < 1e-12 and abs(A - AREA[k]) < 1e-12
assert abs(SHADOW.sum() - np.pi) < 1e-12 and abs(AREA.sum() - 2 * np.pi) < 1e-12
assert [round(SHADOW[k] / 12, 4) for k in (1, 3, 5)] == [0.0479, 0.0654, 0.0175]          # Figure 35.3.2 cells
assert np.all(np.diff(AREA) > 0) and np.allclose(SHADOW, SHADOW[::-1])


def X(theta, phi):
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1)


CAM = dict(elev=30, azim=-25, scale=2.05, origin=(-3.55, 0.35))
BX0, BW, BY0, BSC = 1.35, 0.92, -1.85, 1.4        # bar chart: left, group width, baseline, units per unit area
assert BY0 + BSC * AREA.max() < 0.5


class Anim(hp.Scene2D):
    def construct(self):
        cam = hp.Cam(**CAM)
        view = cam.view
        # the shadow plane: a unit disk at z = -0.95 (sky blue), then the shadows, then the dome (painter's order)
        ring = np.linspace(0, 2 * np.pi, 121)[:-1]
        disk = Polygon(*cam(np.stack([np.cos(ring), np.sin(ring), ZP + 0 * ring], 1)), fill_color=hp.C["region"],
                       fill_opacity=0.25, stroke_color=hp.C["tangent"], stroke_width=2)
        self.add(disk)
        shadows = VGroup()
        self.add(shadows)
        dome = []
        for k in range(NB):
            for j in range(NP):
                q = np.array([X(TH[k], PH[j]), X(TH[k + 1], PH[j]), X(TH[k + 1], PH[j + 1]), X(TH[k], PH[j + 1])])
                c = X((TH[k] + TH[k + 1]) / 2, (PH[j] + PH[j + 1]) / 2)
                front = c @ view > 0
                poly = Polygon(*cam(q), fill_color=hp.C["surface"], fill_opacity=0.55 if front else 0.18,
                               stroke_color="#8A8A8A", stroke_width=1.0 if front else 0.5)
                dome.append((cam.depth(c), k, front, poly))
        dome.sort(key=lambda t: t[0])
        self.add(*[t[3] for t in dome])
        self.add(hp.math(r"$\int_{S^+} dx\wedge dy$", size=34).move_to([-3.55, 3.05, 0]))

        # ---- right: bars of the band areas and their shadows
        self.add(hp.seg([BX0 - 0.25, BY0, 0], [BX0 + NB * BW, BY0, 0], width=2.5))
        for k in range(NB):
            self.add(hp.text(str(k + 1), size=30).move_to([BX0 + BW * k + 0.4, BY0 - 0.35, 0]))
        self.add(hp.text("band", size=30).move_to([BX0 + NB * BW + 0.15, BY0 - 0.35, 0], aligned_edge=np.array([-1, 0, 0]))
                 if BX0 + NB * BW + 1.0 < 6.9 else VGroup())
        leg = VGroup(VGroup(Rectangle(width=0.3, height=0.3, fill_color=hp.C["surface"], fill_opacity=1,
                                      stroke_color=hp.BLACK, stroke_width=1),
                            hp.text("area of the band", size=30)).arrange(np.array([1.0, 0, 0]), buff=0.2),
                     VGroup(Rectangle(width=0.3, height=0.3, fill_color=hp.C["accent"], fill_opacity=1,
                                      stroke_color=hp.BLACK, stroke_width=1),
                            hp.math(r"its shadow $= \int dx\wedge dy$", size=30)).arrange(np.array([1.0, 0, 0]), buff=0.2))
        leg.arrange(np.array([0, -1.0, 0]), buff=0.25, aligned_edge=np.array([-1, 0, 0])).move_to(
            [BX0 - 0.25, 2.45, 0], aligned_edge=np.array([-1, 0, 0]))
        self.add(leg)
        tot = VGroup()
        self.add(tot)
        self.wait(2.0)
        for k in range(NB):
            band = [t[3] for t in dome if t[1] == k]
            ann_out = np.stack([np.sin(TH[k + 1]) * np.cos(ring), np.sin(TH[k + 1]) * np.sin(ring), ZP + 0 * ring], 1)
            ann_in = np.stack([np.sin(TH[k]) * np.cos(ring), np.sin(TH[k]) * np.sin(ring), ZP + 0 * ring], 1)
            ann = Polygon(*cam(np.vstack([ann_out, ann_out[:1], ann_in[:1], ann_in[::-1]])), fill_color=hp.C["accent"],
                          fill_opacity=0.85, stroke_width=0)
            x0 = BX0 + BW * k
            bars = VGroup(Rectangle(width=0.36, height=AREA[k] * BSC, fill_color=hp.C["surface"], fill_opacity=1,
                                    stroke_color=hp.BLACK, stroke_width=1).move_to([x0 + 0.21, BY0 + AREA[k] * BSC / 2, 0]),
                          Rectangle(width=0.36, height=SHADOW[k] * BSC, fill_color=hp.C["accent"], fill_opacity=1,
                                    stroke_color=hp.BLACK, stroke_width=1)
                          .move_to([x0 + 0.59, BY0 + SHADOW[k] * BSC / 2, 0]))
            new_tot = VGroup(hp.math(r"$\Sigma$ shadow $= %.4f$" % SHADOW[:k + 1].sum(), size=30, color=hp.C["accent"]),
                             hp.math(r"$\Sigma$ area $= %.4f$" % AREA[:k + 1].sum(), size=30)) \
                .arrange(np.array([0, -1.0, 0]), buff=0.2, aligned_edge=np.array([-1, 0, 0])) \
                .move_to([BX0 - 0.25, 1.2, 0], aligned_edge=np.array([-1, 0, 0]))
            anims = [p.animate.set_fill(hp.C["accent"]) for p in band]
            shadows.add(ann)
            self.play(*anims, FadeIn(ann), FadeIn(bars), *hp.swap(tot, new_tot), run_time=1.0)
            tot = new_tot
            self.wait(1.0)
        fin = hp.math(r"$= \pi$", size=30, color=hp.C["accent"])
        fin.next_to(tot[0], np.array([1.0, 0, 0]), buff=0.25)
        fin2 = hp.math(r"$= 2\pi$", size=30).next_to(tot[1], np.array([1.0, 0, 0]), buff=0.25)
        self.play(FadeIn(fin), FadeIn(fin2), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.5)
