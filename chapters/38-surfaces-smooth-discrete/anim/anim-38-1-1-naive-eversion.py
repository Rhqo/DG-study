"""Animation 38.1.1: pushing the sphere through itself turns it inside out, but through a crease (Section 38.1,
Non-example 38.1.15, (38.1.3), (38.1.4), Figure 38.1.3).

Family: x_lambda(theta, phi) = (sin theta cos phi, sin theta sin phi, (1 - 2 lambda) cos theta) (lib/ddg38.eversion),
lambda from 0 to 1. Drawn: three quarters of the surface (phi in [0, 3 pi/2]) as a triangulated (theta, phi) grid
(18 x 27 cells) with an orthographic camera and painter's sorting (as Chapter 31's MeshView). Each triangle is
oriented like x_theta x x_phi, and is painted light blue when the camera sees the side that x_theta x x_phi points
to and orange when it sees the other side. Red arrows: the unit normal along the meridian phi = 0 (length 0.35).
Right: |x_theta x x_phi| on the equator theta = pi/2, which is |1 - 2 lambda| by (38.1.4).
lambda = 0: the round sphere, blue outside. lambda = 1/2: a flat disk, x_theta x x_phi = 0 on the whole equator
(a crease). lambda = 1: the same round sphere as a set, orange outside: the normal points inward.
Self-check: (38.1.4) against a numerical cross product; the equator value |1 - 2 lambda|; at lambda = 0 every
outward-facing triangle normal points away from the origin and at lambda = 1 toward it.

Render: bash tools/render_anims.sh chapters/38-surfaces-smooth-discrete/anim/anim-38-1-1-naive-eversion.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg38  # noqa: E402
import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
NT, NP = 18, 27
TH = np.linspace(1e-3, np.pi - 1e-3, NT + 1)
PH = np.linspace(0.0, 1.5 * np.pi, NP + 1)
TT, PP = np.meshgrid(TH, PH, indexing="ij")
IDX = np.arange((NT + 1) * (NP + 1)).reshape(NT + 1, NP + 1)
F = []
for i in range(NT):
    for j in range(NP):
        a, b, c, d = IDX[i, j], IDX[i + 1, j], IDX[i + 1, j + 1], IDX[i, j + 1]
        F += [[a, b, c], [a, c, d]]                    # (d theta) x (d phi): the orientation of x_theta x x_phi
F = np.array(F)


def verts(lam):
    return np.moveaxis(ddg38.eversion(TT, PP, lam), 0, -1).reshape(-1, 3)


def cross_formula(th, ph, lam):
    """(38.1.4)."""
    return np.array([(1 - 2 * lam) * np.sin(th) ** 2 * np.cos(ph), (1 - 2 * lam) * np.sin(th) ** 2 * np.sin(ph),
                     np.sin(th) * np.cos(th)])


for lam in (0.0, 0.3, 0.5, 0.8, 1.0):
    th, ph, h = 1.1, 0.7, 1e-6
    xt = (ddg38.eversion(th + h, ph, lam) - ddg38.eversion(th - h, ph, lam)) / (2 * h)
    xp = (ddg38.eversion(th, ph + h, lam) - ddg38.eversion(th, ph - h, lam)) / (2 * h)
    assert np.allclose(np.cross(xt, xp), cross_formula(th, ph, lam), atol=1e-8)
    assert np.isclose(np.linalg.norm(cross_formula(np.pi / 2, 0.3, lam)), abs(1 - 2 * lam))
for lam, sign in ((0.0, 1), (1.0, -1)):
    V = verts(lam)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ok = np.linalg.norm(n, axis=1) > 1e-6
    assert (sign * np.einsum("ij,ij->i", n[ok], tri[ok].mean(1)) > 0).all()

# ---------------------------------------------------------------- drawing
CAM = hp.Cam(elev=22, azim=-40, scale=2.05, origin=(-3.35, -0.25))
BLUE, ORANGE = hp.rgb(hp.HEX["region"]), hp.rgb(hp.HEX["accent"])
NTH = np.deg2rad([25, 50, 75, 90, 105, 130, 155])
PLOT = hp.Plot(1.4, -2.2, 4.9, 1.5, (0, 1), (0, 1.05))


def normal_arrows(lam):
    out = []
    for th in NTH:
        c = cross_formula(th, 0.0, lam)
        if np.linalg.norm(c) < 0.06:
            continue
        p = ddg38.eversion(th, 0.0, lam)
        q = p + 0.35 * c / np.linalg.norm(c)
        n = c / np.linalg.norm(c)
        d = 1e9 if n @ CAM.view > 0 else CAM.depth(0.5 * (p + q))   # toward the viewer: in front of the surface
        out.append([hp.arrow(CAM(p), CAM(q), color=hp.C["normal"], width=4, tip=0.14), d])
    return out


def state(lam, note=None, note_color=None):
    V = verts(lam)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    facing = n @ CAM.view > 0
    cols = np.where(facing[:, None], BLUE, ORANGE)
    out = hp.mesh_mobs(V, F, CAM, colors=cols, edge_color="#707070", edge_width=0.5, ambient=0.55,
                       extras=[(d, m) for m, d in normal_arrows(lam)])
    if abs(lam - 0.5) < 1e-9:
        eq = [ddg38.eversion(np.pi / 2, ph, lam) for ph in np.linspace(0, 1.5 * np.pi, 60)]
        out.append(hp.polyline([CAM(p) for p in eq], color=hp.C["normal"], width=6))
    out.append(hp.text(f"λ = {lam:.2f}", 32).move_to([1.4, 2.25, 0], aligned_edge=hp.LEFT))
    xs = np.linspace(0, lam, 50)
    if lam > 0:
        out.append(PLOT.curve(xs, np.abs(1 - 2 * xs), color=hp.C["normal"], width=4))
    out.append(hp.dot(PLOT.p(lam, abs(1 - 2 * lam)), color=hp.C["normal"], r=0.08))
    if note:
        for k, line in enumerate(note):
            out.append(hp.text(line, 30, color=note_color or hp.BLACK).move_to([1.4, 1.6 - 0.45 * k, 0],
                                                                               aligned_edge=hp.LEFT))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("squashing the sphere through itself: inside out, with a crease", 30).move_to([0, 3.55, 0])
        key = VGroup(hp.mt(r"blue: the side $\mathbf{x}_\theta\times\mathbf{x}_\varphi$ points to", 30),
                     hp.text("orange: the other side", 30, color=hp.C["accent"]))
        key.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.08).move_to([-6.85, 2.85, 0], aligned_edge=hp.LEFT)
        axes = PLOT.axes(xticks=[(0, "0"), (0.5, "½"), (1, "1")], yticks=[(0, "0"), (1, "1")], size=30)
        yl = hp.mt(r"$|\mathbf{x}_\theta\times\mathbf{x}_\varphi|$ on the equator", 30).next_to(PLOT.p(0, 1.05), hp.UP,
                                                                                            buff=0.12)
        yl.align_to(PLOT.p(0, 0), hp.LEFT)
        xl = hp.mt(r"$\lambda$", 30).next_to(PLOT.p(1, 0), hp.RIGHT, buff=0.15)
        self.add(title, key, axes, yl, xl)
        g = VGroup()
        self.add(g)
        S = hp.setter
        S(g, state(0.0, ["round sphere,", "blue outside"]))(1)
        self.mark("start")
        self.wait(2.0)
        ev, t = [], 0.0
        def at(lam, note, col=None):
            return lambda u: S(g, state(lam, note, col))(1)

        for lam in np.linspace(0, 0.5, 61)[1:-1]:
            ev.append((t, t, at(lam, ["squashing ..."])))
            t += 1 / 15
        ev.append((t, t, at(0.5, ["λ = ½: a flat disk;", "xθ × xφ = 0 on the",
                                   "whole equator (crease)"], hp.C["normal"])))
        t += 2.5
        for lam in np.linspace(0.5, 1.0, 61)[1:]:
            ev.append((t, t, at(lam, ["popping out ..."])))
            t += 1 / 15
        self.play(hp.timeline(g, ev, t))
        self.mark("sweep done")
        self.play(hp.timeline(g, [(0, 0, S(g, state(1.0, ["the same sphere, but", "orange outside: inside", "out, through a crease"],
                                                    hp.C["normal"])))], 0.5))
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=7.5)
