"""Animation 33.4.1: refining a closed surface -- the mean valence goes to 6, the total deviation from 6 stays 6 chi
(Section 33.4, Definition 33.4.20, Example 33.4.21, Proposition 33.4.23, Figure 33.4.3).

Left: icosphere(level) of dgmesh, level 0, 1, 2, 3 (n0 = 12, 42, 162, 642, chi = 2). Right: torus_mesh(2, 0.8, nu, nv)
with (nu, nv) = (4, 6), (6, 8), (8, 16), (16, 32) (n0 = 24, 48, 128, 512, chi = 0). Vertex dots: orange = valence 5,
grey = valence 6 (no other valences occur). Below each mesh, live: n0, the mean valence 2 n1 / n0 and
sum_i (6 - deg i). The sphere keeps exactly twelve orange vertices (the icosahedron's), so the mean valence
6 - 12/n0 climbs to 6 while the sum stays 12; the torus is regular everywhere (mean exactly 6, sum 0).
Drawing: orthographic projection with back-face culling (sphere) or the painter's algorithm (torus), as in
Chapter 31's MeshView; a dot is drawn only if its vertex normal faces the viewer.
Self-check: valences from dgmesh.neighbors; sum deg = 2 n1, mean = 6 - 6 chi / n0, sum (6 - deg) = 6 chi (33.4.2);
icosphere has exactly 12 vertices of valence 5 and the rest 6; the torus meshes are all valence 6.

Render: ``bash tools/render_anims.sh chapters/33-combinatorial-surfaces/anim/anim-33-4-1-valence-refinement.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers as hp  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, FadeOut, VGroup)  # noqa: E402

LEVELS = [0, 1, 2, 3]
TORI = [(4, 6), (6, 8), (8, 16), (16, 32)]


def stats(F):
    n0, n1, n2 = dm.counts(F)
    deg = np.array([len(s) for s in dm.neighbors(F)])
    chi = n0 - n1 + n2
    assert deg.sum() == 2 * n1                                                       # Proposition 33.4.23
    assert abs(deg.mean() - (6 - 6 * chi / n0)) < 1e-12 and (6 - deg).sum() == 6 * chi   # (33.4.2)
    return dict(n0=n0, n1=n1, n2=n2, chi=chi, deg=deg, mean=deg.mean(), excess=int((6 - deg).sum()))


SPH, TOR = [], []
for lev in LEVELS:
    V, F = dm.icosphere(lev)
    s = stats(F)
    assert s["chi"] == 2 and (s["deg"] == 5).sum() == 12 and set(s["deg"]) <= {5, 6}    # Example 33.4.21
    SPH.append(dict(V=V, F=F, **s))
for nu, nv in TORI:
    V, F = dm.torus_mesh(2, 0.8, nu, nv)
    s = stats(F)
    assert s["chi"] == 0 and np.all(s["deg"] == 6) and dm.is_manifold(F)
    TOR.append(dict(V=V, F=F, nu=nu, nv=nv, **s))
assert [d["n0"] for d in SPH] == [12, 42, 162, 642] and [d["n0"] for d in TOR] == [24, 48, 128, 512]

CAM_S = dict(elev=18, azim=-62, scale=1.75, origin=(-4.3, 0.85))
CAM_T = dict(elev=38, azim=-62, scale=0.6, origin=(4.3, 0.85))
ORANGE, GREY = hp.C["accent"], hp.ManimColor("#555555")


def vertex_normals(V, F):
    return dm.vertex_normals(V, F, kind="angle")


def sphere_view(d):
    cam = hp.Cam(**CAM_S)
    mv = hp.MeshView(d["V"], d["F"], cam, edge_width=1.0 if d["n0"] < 200 else 0.6, cull=True, sort=False,
                     ambient=0.55)
    dots = VGroup()
    vis = d["V"] @ cam.view > 0.05
    for i in np.nonzero(vis)[0]:
        five = d["deg"][i] == 5
        dots.add(hp.dot(cam(d["V"][i]), color=ORANGE if five else GREY, r=0.085 if five else 0.028))
    return VGroup(mv, dots)


def torus_view(d):
    cam = hp.Cam(**CAM_T)
    mv = hp.MeshView(d["V"], d["F"], cam, edge_width=1.0 if d["n0"] < 200 else 0.6, cull=False, sort=True,
                     ambient=0.55)
    N = vertex_normals(d["V"], d["F"])
    for i in range(len(d["V"])):
        if N[i] @ cam.view > 0.15:
            mv.extras.append([hp.dot(cam(d["V"][i]), color=GREY, r=0.03), cam.depth(d["V"][i]) + 0.03])
    mv.set_mesh()
    return mv


def readouts(d, x):
    rows = [hp.math(r"$n_0 = %d$" % d["n0"], size=30),
            hp.text(f"mean valence {d['mean']:.3f}", size=30),
            hp.math(r"$\Sigma\,(6 - \deg i) = %d$" % d["excess"], size=30)]
    return VGroup(*[m.move_to([x, y, 0]) for m, y in zip(rows, (-1.55, -2.05, -2.6))])


class Anim(hp.Scene2D):
    def construct(self):
        leg = VGroup(VGroup(hp.dot([0, 0, 0], color=ORANGE, r=0.085), hp.text("valence 5", size=30))
                     .arrange(np.array([1.0, 0, 0]), buff=0.18).move_to([0, 3.45, 0]),
                     VGroup(hp.dot([0, 0, 0], color=GREY, r=0.05), hp.text("valence 6", size=30))
                     .arrange(np.array([1.0, 0, 0]), buff=0.18).move_to([0, 2.95, 0]))
        self.add(leg)
        cur = None
        for k in range(len(LEVELS)):
            s, t = SPH[k], TOR[k]
            g = VGroup(sphere_view(s), torus_view(t),
                       hp.text(f"icosphere({LEVELS[k]}), χ = 2", size=30).move_to([-4.3, 3.45, 0]),
                       hp.text(f"torus {t['nu']}×{t['nv']}, χ = 0", size=30).move_to([4.3, 3.45, 0]),
                       readouts(s, -4.3), readouts(t, 4.3))
            if cur is None:
                self.add(g)
                self.wait(2.5)
            else:
                self.play(FadeOut(cur), FadeIn(g), run_time=0.8)
                self.wait(2.8)
            cur = g
        f1 = hp.math(r"$\Sigma\,(6 - \deg i) = 6\chi$", size=32).move_to([0, 0.95, 0])
        f2 = hp.math(r"mean $= 6 - 6\chi/n_0$", size=32).move_to([0, 0.2, 0])
        self.play(FadeIn(f1), run_time=0.8)
        self.play(FadeIn(f2), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.0)
