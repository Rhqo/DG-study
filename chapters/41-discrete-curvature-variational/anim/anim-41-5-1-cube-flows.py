"""Animation 41.5.1: on a cube, mean curvature flow moves the edges and Gauss curvature flow only the corners
(Section 41.5, Example 41.5.6, Example 41.5.9, Figure 41.5.2).

Data (as Figure 41.5.2): lib/ddg41.py cube_mesh(8, center 0, side 2) (each face an 8 x 8 grid, 768 triangles, faces
outward). Explicit steps x <- x + h v(x), h = 0.1 / 32 (about 0.2 h_c), 32 steps to t = 0.1, with
  left:  v = (L x)_i / A_i        (discrete mean curvature flow, Definition 41.5.3)
  right: v = -(K N)_i / A_i       (discrete Gauss curvature flow, Definition 41.5.7),
A_i the mixed area (ddg41.mcf_velocity, ddg41.gcf_velocity). Frames between two steps interpolate linearly.
Colour: |x_i(t) - x_i(0)| averaged over the face, viridis on [0, 0.7] (the colour bar of Figure 41.5.2), Lambert
shading; view elevation 22, azimuth 30. Dashed grey: the front edges of the cube at t = 0.
Readouts: t, the largest displacement of a corner and of the face centres.
Self-check (Example 41.5.6, 41.5.9): at t = 0 the face vertices have (L x)_i = 0 and (K N)_i = 0 except at the corners;
at t = 0.1 MCF: corner 0.705, edge 0.478, face centre 0.017, area 14.76, volume 5.14; GCF: corner 0.416, edge 0.290,
face centre about 5e-9, area 18.66, volume 6.91.

Render: bash tools/render_anims.sh chapters/41-discrete-curvature-variational/anim/anim-41-5-1-cube-flows.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg41  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, Rectangle, ValueTracker, VGroup, always_redraw, linear  # noqa: E402

# ---------------------------------------------------------------- data
V0, F = ddg41.cube_mesh(8, center=(0, 0, 0), side=2.0)
assert len(F) == 768
ON = np.isclose(np.abs(V0), 1).sum(1)                       # 1 face, 2 edge, 3 corner
CENTERS = [int(np.argmin(np.linalg.norm(V0 - c, axis=1))) for c in np.vstack([np.eye(3), -np.eye(3)])]
assert np.abs(dm.cotan_apply(V0, F, V0)[ON == 1]).max() < 1e-12
assert np.abs(ddg41.gauss_curvature_normal(V0, F)[ON < 3]).max() < 1e-13
H_STEP, STEPS = 0.1 / 32, 32
assert abs(H_STEP * ddg41.largest_eigenvalue(V0, F) / 2 - 0.2) < 0.01        # h ~ 0.2 h_c


def run(velocity):
    X = []
    ddg41.explicit_flow(V0, F, velocity, H_STEP, STEPS, record=lambda k, V: X.append(V.copy()))
    return np.array(X)


XM, XG = run(ddg41.mcf_velocity), run(ddg41.gcf_velocity)
assert len(XM) == STEPS + 1


def summary(X):
    d = np.linalg.norm(X[-1] - V0, axis=1)
    return [d[ON == 3].max(), d[ON == 2].max(), max(d[c] for c in CENTERS), dm.face_areas(X[-1], F).sum(),
            dm.signed_volume(X[-1], F)]


assert np.allclose(summary(XM), [0.705, 0.478, 0.017, 14.76, 5.14], atol=0.0051)
sg = summary(XG)
assert np.allclose([sg[k] for k in (0, 1, 3, 4)], [0.416, 0.290, 18.66, 6.91], atol=0.0051) and 3e-9 < sg[2] < 7e-9
VMAX = 0.7


def at(X, t):
    """positions at time t (linear between steps)"""
    s = np.clip(t / H_STEP, 0, STEPS)
    k = min(int(np.floor(s)), STEPS - 1)
    w = s - k
    return (1 - w) * X[k] + w * X[k + 1]


class Anim(hp.Scene2D):
    def construct(self):
        T = ValueTracker(0.0)
        cams = [hp.Cam(elev=22, azim=30, scale=1.4, origin=(x, -0.75)) for x in (-3.55, 3.55)]
        views = []
        for X, cam in ((XM, cams[0]), (XG, cams[1])):
            mv = hp.MeshView(V0, F, cam, colors=hp.cmap_rgb(np.zeros(len(F)), 0, VMAX, "viridis"), edge_width=0.4,
                             edge_color="#333333", cull=True, sort=True, ambient=0.62)

            def upd(m, X=X):
                V = at(X, T.get_value())
                d = np.linalg.norm(V - V0, axis=1)
                m.set_mesh(V, colors=hp.cmap_rgb(d[F].mean(1), 0, VMAX, "viridis"))
            mv.add_updater(upd)
            mv.update()
            views.append(mv)

        # dashed front edges of the cube at t = 0
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        ref = VGroup()
        for cam in cams:
            for a in range(8):
                for b in range(a + 1, 8):
                    if np.abs(corners[a] - corners[b]).sum() == 2:
                        mid = (corners[a] + corners[b]) / 2
                        nrm = [np.sign(mid[c]) * np.eye(3)[c] for c in range(3) if abs(mid[c]) == 1]
                        if any(n @ cam.view > 0 for n in nrm):
                            ref.add(hp.seg(cam(corners[a]), cam(corners[b]), color=hp.C["aux"], width=2, dashed=True))

        # ---- titles and readouts
        t1 = hp.math(r"mean curvature flow: $2H\mathbf{N}$", 32).move_to([-3.55, 3.4, 0])
        t2 = hp.math(r"Gauss curvature flow: $-K\mathbf{N}$", 32).move_to([3.55, 3.4, 0])
        f1, f2 = VGroup(), VGroup()
        tl = hp.math(r"$t =$", 32).move_to([-0.4, -2.4, 0])
        tn = always_redraw(lambda: hp.text(hp.num(T.get_value()), 32).next_to(tl, np.array([1.0, 0, 0]), buff=0.15))

        def disp(X):
            V = at(X, T.get_value())
            d = np.linalg.norm(V - V0, axis=1)
            return d[ON == 3].max(), max(d[c] for c in CENTERS)

        SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

        def fmt_small(v):
            if v >= 5e-4 or v < 1e-12:
                return f"{v:.3f}"
            m, ex = f"{v:.0e}".split("e")
            return f"{m}×10" + str(int(ex)).translate(SUP)

        r = []
        for k, X in enumerate((XM, XG)):
            x = (-3.55, 3.55)[k]
            r.append(always_redraw(lambda X=X, x=x: hp.text(f"corners moved {disp(X)[0]:.3f}", 30)
                                   .move_to([x, 2.8, 0])))
            r.append(always_redraw(lambda X=X, x=x: hp.text(f"face centres {fmt_small(disp(X)[1])}", 30)
                                   .move_to([x, 2.3, 0])))

        # ---- colour bar (between the panels)
        cb = VGroup()
        n = 48
        vals = np.linspace(0, VMAX, n)
        rgbs = hp.cmap_rgb(vals, 0, VMAX, "viridis")
        y0, h = -1.85, 2.5
        for k in range(n):
            cb.add(Rectangle(width=0.28, height=h / n + 0.005, stroke_width=0,
                             fill_color=hp.ManimColor(tuple(rgbs[k])), fill_opacity=1)
                   .move_to([0, y0 + (k + 0.5) * h / n, 0]))
        cb.add(Rectangle(width=0.28, height=h, stroke_width=1.5, stroke_color=hp.C["main"]).move_to([0, y0 + h / 2, 0]))
        for v in (0, 0.35, 0.7):
            cb.add(hp.text(f"{v:g}", 30).next_to([0.14, y0 + h * v / VMAX, 0], np.array([1.0, 0, 0]), buff=0.12))
        cb.add(hp.math(r"$|\mathbf{x}_i(t) - \mathbf{x}_i(0)|$", 30).move_to([0, y0 + h + 0.45, 0]))

        self.add(*views, ref, t1, f1, t2, f2, tl, tn, *r, cb)
        for m in (tn, *r):
            m.update()
        self.wait(2.0)
        self.play(T.animate.set_value(0.1), run_time=10.0, rate_func=linear)
        e1 = hp.text("pushed along every edge", 30, color=hp.C["tangent"]).move_to([-3.55, 1.8, 0])
        e2 = hp.text("pushed at the corners first", 30, color=hp.C["tangent"]).move_to([3.55, 1.8, 0])
        self.play(FadeIn(e1), FadeIn(e2), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=14.0)
