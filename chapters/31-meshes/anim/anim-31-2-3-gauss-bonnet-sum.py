"""Animation 31.2.3: adding up the angle defects (Section 31.2, Theorem 31.2.4, Figure 31.2.3).

(1) The icosphere of level 2 (162 vertices, every delta_i between 0.069 and 0.083) turns once about the vertical
    axis; every vertex is counted when it passes the front, and its angle defect is added to the bar. The total is
    4 pi = 2 pi chi.
(2) The vertices are moved radially, x_i -> x_i (1 - 0.38 s w_1(x_i) + 0.32 s w_2(x_i)) with two Gaussian weights
    w_1, w_2 (a dent and a bump on the front, s from 0 to 1): the individual defects change (blue = negative,
    saddle-like vertices around them) but the live sum stays 4 pi to 1e-11. No face flips over.
(3) The torus of revolution T_{R,r}, R = 2, r = 0.8, on the 24 x 48 grid of Figure 31.2.3(a). The rings u = const are
    added one at a time starting at the top ring (u = pi/2), down the outside and back up the inside: the outer half
    (delta > 0) brings the sum up to 12.46 (about 4 pi), the inner half (delta < 0) brings it back to 0 = 2 pi chi.
Faces are coloured by the mean defect of their three vertices (RdBu_r, white = 0) as in Figure 31.2.3(a).

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-2-3-gauss-bonnet-sum.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (LEFT, FadeIn, FadeOut, Rectangle, ValueTracker, VGroup, always_redraw, linear, smooth)  # noqa: E402

# ---------------------------------------------------------------- sphere
VS, FS = dm.icosphere(2)
DS = dm.angle_defect(VS, FS)
assert abs(DS.sum() - 4 * np.pi) < 1e-12 and len(VS) == 162
assert 0.069 < DS.min() and DS.max() < 0.083
ELEV, AZ0 = 18.0, -60.0
# a vertex is counted when the turning camera faces it: azimuth of the vertex = camera azimuth (mod 360)
AZV = np.degrees(np.arctan2(VS[:, 1], VS[:, 0]))
T_SWEEP = (AZV - AZ0) % 360 / 360          # fraction of the turn at which vertex i is counted


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


_c0 = hc.Cam(elev=ELEV, azim=AZ0)
C_DENT = _unit(_c0.view - 0.35 * _c0.right + 0.25 * _c0.up)     # pushed in
C_BUMP = _unit(_c0.view + 0.9 * _c0.right - 0.3 * _c0.up)       # pulled out


def deformed(s):
    w1 = np.exp(-np.sum((VS - C_DENT) ** 2, 1) / (2 * 0.33 ** 2))
    w2 = np.exp(-np.sum((VS - C_BUMP) ** 2, 1) / (2 * 0.30 ** 2))
    return VS * (1 - s * 0.38 * w1 + s * 0.32 * w2)[:, None]


for s in np.linspace(0, 1, 11):
    Vd = deformed(s)
    d = dm.angle_defect(Vd, FS)
    assert abs(d.sum() - 4 * np.pi) < 1e-11                                      # Theorem 31.2.4
    nrm = dm.face_normals(Vd, FS)
    assert (np.einsum("ij,ij->i", nrm, Vd[FS].mean(1)) > 0).all()                 # no face flips over
    assert dm.corner_angles(Vd, FS).min() > np.radians(35)
D1 = dm.angle_defect(deformed(1.0), FS)
assert D1.min() < -0.07 and D1.max() > 0.55      # saddle-like ring around the dent and the bump, peaks at their centres
SLIM = (-0.1, 0.6)                               # two-slope colour range of the sphere part (white = 0)

# ---------------------------------------------------------------- torus (Figure 31.2.3(a))
NU, NV = 24, 48
VT, FT = dm.torus_mesh(2.0, 0.8, NU, NV)
DT = dm.angle_defect(VT, FT)
assert abs(DT.sum()) < 1e-11
RING = np.repeat(np.arange(NU), NV)
# start at the top ring u = pi/2 (i = 6), go down the outside (u = 0) to the bottom (u = -pi/2), then up the inside
RING_ORDER = [(6 - j) % NU for j in range(NU)]
CUM = np.cumsum([DT[RING == i].sum() for i in RING_ORDER])
assert abs(CUM[0]) < 1e-12 and abs(CUM[-1]) < 1e-11
assert abs(CUM.max() - 12.459) < 1e-3 and CUM.argmax() in (11, 12)        # outer half: about 4 pi
assert (CUM[:12] >= -1e-12).all() and np.all(np.diff(CUM[:12]) >= -1e-12)  # rises on the outer half
DLIM_T = float(np.abs(DT).max())
# a face is revealed when all three of its vertices have been counted
FACE_STEP = np.array([max(RING_ORDER.index(RING[v]) for v in f) for f in FT])


class Anim(hc.Scene2D):
    def construct(self):
        # ---- the bar (right)
        BX, B0, B1 = 4.7, -2.5, 2.3                      # bar x, y of 0, y of 4 pi
        k = (B1 - B0) / (4 * np.pi)
        total = ValueTracker(0.0)

        def bar():
            v = total.get_value()
            h = max(1e-3, abs(v) * k)
            r = Rectangle(width=0.7, height=h, fill_color=hc.C["normal"] if v >= 0 else hc.C["tangent"],
                          fill_opacity=0.75, stroke_width=0)
            return r.move_to([BX, B0 + np.sign(v) * h / 2, 0])
        frame = VGroup(hc.seg([BX - 0.6, B0, 0], [BX + 0.6, B0, 0], width=3))
        for val_, lab in ((2 * np.pi, r"$2\pi$"), (4 * np.pi, r"$4\pi$")):
            y = B0 + k * val_
            frame.add(hc.seg([BX - 0.6, y, 0], [BX + 0.6, y, 0], color=hc.C["aux"], width=2, dashed=True))
            frame.add(hc.math(lab, size=34).move_to([BX + 1.05, y, 0]))
        frame.add(hc.math(r"$0$", size=34).move_to([BX + 1.05, B0, 0]))
        head = hc.math(r"$\sum_i \delta_i =$", size=36).move_to([BX - 0.5, 3.15, 0])
        def fmt(v):
            return "0.000" if abs(v) < 5e-4 else f"{v:.3f}".replace("-", "−")
        num = always_redraw(lambda: hc.text(fmt(total.get_value()), size=34)
                            .next_to(head, np.array([1.0, 0, 0]), buff=0.2))
        self.add(always_redraw(bar), frame, head, num)

        # ---- (1) sphere: turn once, count each vertex as it passes the front
        cam = hc.Cam(elev=ELEV, azim=AZ0, scale=2.05, origin=(-2.6, 0.25))
        turn = ValueTracker(0.0)
        counted = np.zeros(len(VS), bool)
        cols_s = lambda d: hc.cmap_rgb(d[FS].mean(1), SLIM[0], SLIM[1], center=0.0)  # noqa: E731
        mv = hc.MeshView(VS, FS, cam, colors=np.tile(hc.rgb(hc.HEX["surface"]), (len(FS), 1)), edge_width=0.7,
                         cull=True, sort=False, ambient=0.55)
        face_done = np.zeros(len(FS), bool)
        base = np.tile(hc.rgb(hc.HEX["surface"]), (len(FS), 1))
        colS = cols_s(DS)

        def upd_sphere(m):
            t = turn.get_value()
            cam.set(ELEV, AZ0 + 360 * t)
            newly = (~counted) & (T_SWEEP <= t)
            if newly.any():
                counted[newly] = True
                total.set_value(DS[counted].sum())
            face_done[:] = counted[FS].all(1)
            m.set_mesh(cam=cam, colors=np.where(face_done[:, None], colS, base))
        mv.add_updater(upd_sphere)

        def dots():
            gdot = VGroup()
            P = VS
            vis = (P @ cam.view) > 0.05
            for i in np.nonzero(vis & counted)[0]:
                gdot.add(hc.dot(cam(P[i]), r=0.045))
            return gdot
        dts = always_redraw(dots)
        self.add(mv, dts)
        cap = hc.text("sphere mesh: χ = 2", size=34).move_to([-2.6, -2.45, 0])
        title = hc.text("count every vertex once", size=34).move_to([-2.6, 3.3, 0])
        self.add(cap, title)
        self.wait(1.5)
        self.play(turn.animate.set_value(1.0), run_time=9.0, rate_func=linear)
        mv.clear_updaters()
        self.remove(dts)
        res = hc.math(r"$\sum_i \delta_i = 4\pi = 2\pi\chi$", size=36, color=hc.C["normal"]).move_to([-2.6, -2.5, 0])    # above the control bar
        self.play(*hc.swap(cap, res), run_time=0.8)
        self.wait(1.5)

        # ---- (2) move the vertices: each delta_i changes, the sum does not
        s = ValueTracker(0.0)
        cam.set(ELEV, AZ0)

        def upd_def(m):
            Vd = deformed(s.get_value())
            d = dm.angle_defect(Vd, FS)
            total.set_value(d.sum())
            m.set_mesh(Vd, cam=cam, colors=cols_s(d))
        mv.cull, mv.sort = True, True
        mv.add_updater(upd_def)
        t2 = hc.text("move the vertices: each δ changes", size=34).move_to([-2.6, 3.3, 0])
        self.play(*hc.swap(title, t2), run_time=0.6)
        self.play(s.animate.set_value(1.0), run_time=4.0, rate_func=smooth)
        t3 = hc.text("the sum does not", size=34, color=hc.C["normal"]).move_to([-2.6, 2.7, 0])
        self.play(FadeIn(t3), run_time=0.6)
        self.wait(2.0)
        mv.clear_updaters()

        # ---- (3) torus, ring by ring
        camT = hc.Cam(elev=40, azim=-60, scale=1.0, origin=(-2.6, 0.25))
        colT = hc.cmap_rgb(DT[FT].mean(1), -DLIM_T, DLIM_T, center=0.0)
        baseT = np.tile(hc.rgb(hc.HEX["surface"]), (len(FT), 1))
        mvT = hc.MeshView(VT, FT, camT, colors=baseT, edge_width=0.35, cull=True, sort=True, ambient=0.6)
        step = ValueTracker(-1.0)

        def upd_torus(m):
            j = int(np.floor(step.get_value() + 1e-9))
            total.set_value(CUM[j] if j >= 0 else 0.0)
            m.set_mesh(colors=np.where((FACE_STEP <= j)[:, None], colT, baseT))
        mvT.add_updater(upd_torus)
        t4 = hc.text("torus: add one ring u = const at a time", size=34).move_to([-1.6, 3.3, 0])
        cap2 = hc.text("torus mesh: χ = 0", size=34).move_to([-2.6, -2.45, 0])
        self.play(FadeOut(mv), FadeOut(t3), *hc.swap(t2, t4), *hc.swap(res, cap2), run_time=1.0)
        self.add(mvT)
        self.play(FadeIn(mvT), run_time=0.6)
        self.wait(0.5)
        o1 = hc.text("outer half: δ > 0", size=32, color=hc.C["normal"]).move_to([-2.6, 2.72, 0])
        self.play(FadeIn(o1), run_time=0.4)
        self.play(step.animate.set_value(12.0), run_time=5.0, rate_func=linear)
        o2 = hc.text("inner half: δ < 0", size=32, color=hc.C["tangent"]).move_to([-2.6, 2.72, 0])
        self.play(*hc.swap(o1, o2), run_time=0.4)
        self.play(step.animate.set_value(float(NU - 1)), run_time=5.0, rate_func=linear)
        res2 = hc.math(r"$\sum_i \delta_i = 0 = 2\pi\chi$", size=36, color=hc.C["tangent"]).move_to([-2.6, -2.5, 0])
        self.play(*hc.swap(cap2, res2), run_time=0.8)
        self.wait(2.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.0)
