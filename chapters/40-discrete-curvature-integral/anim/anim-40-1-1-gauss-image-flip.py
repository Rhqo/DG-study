"""Animation 40.1.1: the Gauss image of a vertex shrinks to a point and comes back turned around (Section 40.1,
Proposition 40.1.7, Remark 40.1.9, Figure 40.1.1).

The star of Figure 40.1.1 (lib/ddg40.py example_star: x_i = (0, 0, 0.55), link angles 10, 65, 130, 185, 250, 305 deg,
radii 1.0, 0.85, 1.05, 0.9, 1.1, 0.8, heights STAR_HEIGHTS) is deformed in two phases with the link angles and radii
kept:
  s in [0, 1]  all heights (apex and link) are multiplied by 1 - s: the convex star is flattened into the plane;
  s in [1, 2]  apex 0, link heights (s - 1) * 0.5 cos(2 phi_k): a saddle grows out of the plane (two link points up,
               four down).
Left: the star (elev 24, azim -62, as Figure 40.1.1(a)) with the face normals N_k at the face centroids (length 0.45).
Right: the unit sphere seen from above (+z); the vermillion dots are N_1..N_6, the arcs are the great circle arcs
N_k -> N_{k+1} (Proposition 38.5.2) with arrowheads in the order of the faces, filled orange while the polygon turns
counterclockwise and sky blue while it turns clockwise.
Readouts: the angle defect delta_i = 2 pi - Theta_i and the signed area of the spherical polygon (fan of signed
spherical triangles from e_3, ddg40.spherical_polygon_area).
Self-check: at s = 0 the data are those of Figure 40.1.1 (delta = 0.8363); for 81 values of s the signed area equals
delta to 1e-10 and the polygon turns counterclockwise for s < 1, clockwise for s > 1; at s = 1 every N_k = e_3; at
s = 2 delta = -0.7585; no face turns over (N_k . e_3 > 0.7).

Render: bash tools/render_anims.sh chapters/40-discrete-curvature-integral/anim/anim-40-1-1-gauss-image-flip.py
"""

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg40  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import FadeIn, ValueTracker, VGroup, always_redraw, smooth  # noqa: E402

# ---------------------------------------------------------------- data
ANG, RAD, HGT = ddg40.STAR_ANGLES, ddg40.STAR_RADII, ddg40.STAR_HEIGHTS
SADDLE = 0.5 * np.cos(2 * ANG)                    # +, -, -, +, -, -: an ordinary saddle
E3 = np.array([0.0, 0.0, 1.0])


def star(s):
    if s <= 1:
        apex, h = 0.55 * (1 - s), (1 - s) * HGT
    else:
        apex, h = 0.0, (s - 1) * SADDLE
    return ddg40.star_mesh(np.c_[RAD * np.cos(ANG), RAD * np.sin(ANG), h], (0.0, 0.0, apex))


V0, F0 = star(0.0)
FS = ddg40.faces_around(F0, 0)                    # faces f_k = (i, j_k, j_{k+1}) in link order


def data(s):
    V, F = star(s)
    N = dm.face_normals(V, F)[FS]
    delta = 2 * np.pi - dm.corner_angles(V, F)[FS, 0].sum()
    area = ddg40.spherical_polygon_area(N, center=E3)
    return V, F, N, float(delta), float(area)


Vex, Fex = ddg40.example_star()
assert np.allclose(V0, Vex) and np.array_equal(F0, Fex)
assert abs(data(0.0)[3] - 0.8362637747907984) < 1e-12
for s in np.linspace(0, 2, 81):
    V, F, N, d, a = data(s)
    assert abs(a - d) < 1e-10                                     # signed area of the Gauss image = delta_i
    assert (N @ E3).min() > 0.7                                   # no face turns over
    az = np.unwrap(np.arctan2(N[:, 1], N[:, 0]))
    turn = np.sum(np.angle(np.exp(1j * np.diff(np.r_[az, az[0]]))))
    if abs(s - 1) > 1e-9:
        assert np.sign(turn) == np.sign(d) and abs(abs(turn) - 2 * np.pi) < 1e-9   # once around e_3, sign of delta
        assert (d > 0) == (s < 1)
    else:
        assert np.allclose(N, E3) and abs(d) < 1e-12
assert ddg40.is_convex_vertex(*star(0.0), 0)
D_END = data(2.0)[3]
assert abs(D_END + 0.7585) < 5e-5


def arc(a, b, n=24):
    t = np.linspace(0, 1, n)
    P = (1 - t)[:, None] * a + t[:, None] * b
    return P / np.linalg.norm(P, axis=1, keepdims=True)


# ---------------------------------------------------------------- scene
class Anim(hp.Scene2D):
    def construct(self):
        s = ValueTracker(0.0)
        camL = hp.Cam(elev=24, azim=-62, scale=2.7, origin=(-3.55, -0.75))
        camS = hp.Cam(elev=90, azim=-90, scale=2.55, origin=(3.75, -0.45))   # top view: x right, y up

        # ---- left: the star with its face normals
        mv = hp.MeshView(V0, F0, camL, edge_width=1.6, edge_color="#3a3a3a", ambient=0.55, n_extra=len(FS))

        def upd_star(m):
            V, F, N, _, _ = data(s.get_value())
            m.extra_polys = []
            for k, f in enumerate(FS):                   # the normal arrows, sorted with the faces by depth
                c = V[F[f]].mean(0)
                q = c + 0.45 * N[k]
                m.extra_polys.append((hp.arrow_poly(camL(c), camL(q), width=0.03, head=0.22, head_width=0.1),
                                      hp.HEX["normal"], camL.depth(c + 0.5 * 0.45 * N[k]) + 0.02))
            m.set_mesh(V)
        mv.add_updater(upd_star)
        mv.update()
        apex = always_redraw(lambda: hp.dot(camL(data(s.get_value())[0][0]), r=0.075))
        xi = always_redraw(lambda: hp.math(r"$\mathbf{x}_i$", size=34)
                           .next_to(camL(data(s.get_value())[0][0]), np.array([-1.0, 0.6, 0]), buff=0.12))

        def n1_label():
            V, F, N, _, _ = data(s.get_value())
            q = V[F[FS[0]]].mean(0) + 0.45 * N[0]
            return hp.math(r"$\mathbf{N}_1$", size=32, color=hp.C["normal"]).next_to(camL(q), np.array([1.0, 0.3, 0]),
                                                                                       buff=0.08)
        n1L = always_redraw(n1_label)

        # ---- right: the unit sphere from above
        tt = np.linspace(0, 2 * np.pi, 121)
        ring = VGroup(hp.polyline(camS(np.c_[np.cos(tt), np.sin(tt), np.zeros_like(tt)]), color=hp.C["main"], width=2.5))
        for lat in (30, 60):                              # latitude circles 30 and 60 degrees above the equator
            r_ = np.cos(np.radians(lat))
            ring.add(hp.polyline(camS(np.c_[r_ * np.cos(tt), r_ * np.sin(tt), np.zeros_like(tt)]),
                                 color=hp.C["aux"], width=1.5).set_stroke(opacity=0.6))
        pole = hp.dot(camS(E3), color=hp.C["main"], r=0.05)
        s2l = hp.math(r"$S^2$", size=34).move_to(camS([0.78, 0.78, 0]) + np.array([0.35, 0.35, 0]))

        def gauss_image():
            _, _, N, d, _ = data(s.get_value())
            g = VGroup()
            m = len(N)
            P = np.vstack([arc(N[k], N[(k + 1) % m]) for k in range(m)])
            col = hp.C["accent"] if d > 0 else hp.C["region"]
            line = hp.C["accent"] if d > 0 else hp.C["tangent"]
            if abs(d) > 2e-4:
                g.add(hp.filled(camS(P), col, opacity=0.45))
                g.add(hp.polyline(np.vstack([camS(P), camS(P[:1])]), color=line, width=4))
            if abs(d) > 0.04:                                  # arrowheads in the order of the faces
                for k in range(m):
                    A = arc(N[k], N[(k + 1) % m], 9)
                    p, q = camS(A[3]), camS(A[5])
                    if np.linalg.norm(q - p) > 0.05:
                        u = (q - p) / np.linalg.norm(q - p)
                        g.add(hp.arrow(p - 0.12 * u, p + 0.2 * u, color=line, width=5, tip=0.26))
            for k in range(m):
                g.add(hp.dot(camS(N[k]), color=hp.C["normal"], r=0.07))
            if abs(d) > 0.04:
                g.add(hp.math(r"$\mathbf{N}_1$", size=30, color=hp.C["normal"])
                      .next_to(camS(N[0]), np.array([1.0, 0.4, 0]), buff=0.08))
            return g
        gi = always_redraw(gauss_image)

        # ---- readouts (top)
        hd = hp.math(r"$\delta_i = 2\pi - \Theta_i =$", size=34)
        hd_num = always_redraw(lambda: hp.text(hp.num(data(s.get_value())[3]), size=34)
                               .next_to(hd, np.array([1.0, 0, 0]), buff=0.2))
        hd.move_to([-3.55 - 0.6, 3.3, 0])
        ha = hp.text("signed area =", size=34).move_to([3.75 - 0.6, 3.3, 0])

        def area_num():
            d = data(s.get_value())[3]
            return hp.text(hp.num(data(s.get_value())[4]), size=34,
                           color=hp.C["accent"] if d > 2e-4 else (hp.C["tangent"] if d < -2e-4 else hp.C["main"])) \
                .next_to(ha, np.array([1.0, 0, 0]), buff=0.2)
        ha_num = always_redraw(area_num)
        title = hp.text("convex vertex: the image turns counterclockwise", size=32).move_to([0, 2.6, 0])
        self.add(mv, apex, xi, n1L, ring, pole, s2l, gi, hd, hd_num, ha, ha_num, title)
        for m in (mv, apex, xi, n1L, gi, hd_num, ha_num):
            m.update()
        self.wait(2.0)

        t2 = hp.text("flatten the star", size=32).move_to(title)
        self.play(*hp.swap(title, t2), run_time=0.5)
        self.play(s.animate.set_value(1.0), run_time=6.0, rate_func=smooth)
        t3 = hp.math(r"flat: every $\mathbf{N}_k = e_3$, the image is a point", size=32).move_to(title)
        self.play(*hp.swap(t2, t3), run_time=0.5)
        self.wait(1.8)

        t4 = hp.text("make a saddle: 2 link points up, 4 down", size=32).move_to(title)
        self.play(*hp.swap(t3, t4), run_time=0.5)
        self.play(s.animate.set_value(2.0), run_time=6.0, rate_func=smooth)
        t5 = hp.text("saddle vertex: clockwise, signed area = δ < 0", size=32, color=hp.C["tangent"]).move_to(title)
        self.play(*hp.swap(t4, t5), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=18.0)
