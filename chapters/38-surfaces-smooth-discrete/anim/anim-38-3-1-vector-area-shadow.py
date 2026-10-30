"""Animation 38.3.1: the vector area of a cap is its signed shadow, and it depends only on the boundary circle
(Section 38.3, Theorem 38.3.11, Corollary 38.3.12(a), Proposition 38.3.10(c), Figure 38.3.3).

Caps as in Figure 38.3.3: the part of the sphere through the unit circle {x^2 + y^2 = 1, z = 0} with apex (0, 0, h),
radius R_h = (1 + h^2)/(2h), centre (0, 0, h - R_h), normal up at the apex. h runs from 0.1 to 1.8.
Mesh: 14 rings x 64 sectors in the polar angle from the apex, boundary ring on the unit circle, faces oriented by
the upward normal. Shown: the cap (semi-transparent, orthographic view), its shadow on z = 0 (Proposition 38.3.10(c)):
the up-facing part covers the disk of radius max(1, R_h) with sign +, the down-facing part (only for h > 1, below the
equator of the big sphere) covers the annulus 1 <= rho <= R_h with sign -. Their sum is the unit disk, area pi.
Red arrow: the vector area (0, 0, pi), drawn at 1/4 scale as in Figure 38.3.3.
Self-check: A(h) = 2 pi R_h h = pi (1 + h^2); the mesh sum sum_f A_f N_f equals (0, 0, 32 sin(2 pi/64)) = (0, 0, 3.1365)
for every h (the vector area of the boundary 64-gon, Proposition 38.3.10(d)); the signed shadow areas
pi R_h^2 - pi (R_h^2 - 1) = pi.

Render: bash tools/render_anims.sh chapters/38-surfaces-smooth-discrete/anim/anim-38-3-1-vector-area-shadow.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _helpers as hp  # noqa: E402
import dganim as da  # noqa: E402
import dgmesh as dm  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
NR, NS = 14, 64
PHI = 2 * np.pi * np.arange(NS) / NS


def cap_mesh(h):
    R = (1 + h * h) / (2 * h)
    cz = h - R
    psi_max = np.arccos(np.clip(-cz / R, -1, 1))
    V = [[0.0, 0.0, h]]
    for k in range(1, NR + 1):
        psi = psi_max * k / NR
        for ph in PHI:
            V.append([R * np.sin(psi) * np.cos(ph), R * np.sin(psi) * np.sin(ph), cz + R * np.cos(psi)])
    V = np.array(V)
    V[-NS:, 2] = 0.0                                          # boundary ring exactly on the unit circle
    F = []
    ring = lambda k, j: 1 + (k - 1) * NS + j % NS             # noqa: E731
    for j in range(NS):
        F.append([0, ring(1, j), ring(1, j + 1)])
    for k in range(1, NR):
        for j in range(NS):
            a, b, c, d = ring(k, j), ring(k + 1, j), ring(k + 1, j + 1), ring(k, j + 1)
            F += [[a, b, c], [a, c, d]]
    return V, np.array(F), R


VA_POLY = np.array([0.0, 0.0, 0.5 * NS * np.sin(2 * np.pi / NS)])
assert abs(VA_POLY[2] - 3.1365) < 1e-4
for h in (0.1, 0.5, 1.0, 1.4, 1.8):
    V, F, R = cap_mesh(h)
    assert np.allclose(np.linalg.norm(V[-NS:, :2], axis=1), 1.0)
    cr = dm.face_cross(V, F)
    assert np.allclose(0.5 * cr.sum(0), VA_POLY, atol=1e-12)          # discrete: exactly the boundary polygon's
    assert (cr[:NS] @ np.array([0, 0, 1.0]) > 0).all()                # upward normal at the apex
    assert np.isclose(2 * np.pi * R * h, np.pi * (1 + h * h))
    if h > 1:
        assert np.isclose(np.pi * R ** 2 - np.pi * (R ** 2 - 1), np.pi)

# ---------------------------------------------------------------- drawing
CAM = hp.Cam(elev=24, azim=-58, scale=1.75, origin=(-3.35, -0.95))
RX = 0.55
PLOT = hp.Plot(1.25, -2.2, 4.8, 1.6, (0, 1.8), (0, 14))
BLUE = hp.rgb(hp.HEX["region"])


def disk_pts(r, n=120, z=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(t), r * np.sin(t), np.full(n, z)], 1)


def state(h, note=None, note_color=None):
    V, F, R = cap_mesh(h)
    out = []
    rmax = max(1.0, R) if h > 1 else 1.0
    out.append(hp.poly_fill([CAM(p) for p in disk_pts(rmax)], hp.C["region"], opacity=0.45))
    if h > 1:
        outer, inner = disk_pts(R), disk_pts(1.0)[::-1]
        ring = [CAM(p) for p in np.vstack([outer, outer[:1], inner, inner[:1]])]
        out.append(hp.poly_fill(ring, hp.C["normal"], opacity=0.45))
    out.append(hp.polyline([CAM(p) for p in disk_pts(1.0)], color=hp.C["accent"], width=5, closed=True))
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cols = np.where(((n @ CAM.view) > 0)[:, None], BLUE, hp.rgb("#F2B36B"))
    out += hp.mesh_mobs(V, F, CAM, colors=cols, edge_color="#8a8a8a", edge_width=0.3, opacity=0.5, ambient=0.6)
    out.append(hp.arrow(CAM([0, 0, 0]), CAM([0, 0, np.pi / 4]), color=hp.C["normal"], width=6, tip=0.2))
    area = np.pi * (1 + h * h)
    rows = [(f"h = {h:.2f}", hp.BLACK), (f"area = π(1 + h²) = {area:.2f}", hp.BLACK),
            None]
    if h > 1:
        rows.append((f"shadow: {np.pi * R * R:.2f} − {np.pi * (R * R - 1):.2f} = π", hp.C["tangent"]))
    else:
        rows.append(("shadow: the unit disk, π", hp.C["tangent"]))
    for k, row in enumerate(rows):
        if row is None:
            m = hp.mt(r"mesh: $\Sigma\, A_f\mathbf{N}_f = (0, 0, %.4f)$" % VA_POLY[2], 30, color=hp.C["normal"])
        else:
            m = hp.text(row[0], 30, color=row[1])
        out.append(m.move_to([RX, 2.8 - 0.55 * k, 0], aligned_edge=hp.LEFT))
    if note:
        for k, line in enumerate(note):
            out.append(hp.text(line, 30, color=note_color or hp.BLACK).move_to([RX, 0.5 - 0.45 * k, 0],
                                                                               aligned_edge=hp.LEFT))
    hs = np.linspace(0.05, h, 60)
    out.append(PLOT.curve(hs, np.pi * (1 + hs ** 2), color=hp.BLACK, width=4))
    out.append(PLOT.curve(hs, np.full_like(hs, np.pi), color=hp.C["normal"], width=4))
    out.append(hp.dot(PLOT.p(h, area), r=0.07))
    out.append(hp.dot(PLOT.p(h, np.pi), color=hp.C["normal"], r=0.07))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("vector area = signed shadow: only the boundary matters", 32).move_to([0, 3.55, 0])
        axes = PLOT.axes(xticks=[(0, "0"), (1, "1"), (1.8, "1.8")], yticks=[(0, "0"), (np.pi, "π"), (12, "12")],
                         size=30)
        xl = hp.mt(r"$h$", 30).next_to(PLOT.p(1.8, 0), hp.RIGHT, buff=0.15)
        la = hp.text("area", 30).move_to(PLOT.p(1.15, 12.6))
        lv = hp.text("|A| = \u03c0", 30, color=hp.C["normal"]).move_to(PLOT.p(1.58, 5.4))
        self.add(title, axes, xl, la, lv)
        g = VGroup()
        self.add(g)
        S = hp.setter

        def at(h, note, col=None):
            return lambda u: S(g, state(h, note, col))(1)

        S(g, state(0.1, ["a nearly flat cap"]))(1)
        self.mark("start")
        self.wait(1.5)
        ev, t = [], 0.0
        H1 = np.linspace(0.1, 1.0, 46)[1:]
        H2 = np.linspace(1.0, 1.8, 41)[1:]
        for h in H1[:-1]:
            ev.append((t, t, at(h, ["the cap rises: more area,", "same shadow"])))
            t += 1 / 10
        ev.append((t, t, at(1.0, ["hemisphere: the shadow", "is still the unit disk"], hp.C["tangent"])))
        t += 2.0
        for h in H2:
            ev.append((t, t, at(h, ["the overhang faces down:", "its shadow counts −"],
                                hp.C["normal"])))
            t += 1 / 10
        self.play(hp.timeline(g, ev, t))
        self.mark("sweep done")
        self.play(hp.timeline(g, [(0, 0, at(1.8, ["same boundary circle,", "same vector area"], hp.C["normal"]))],
                              0.3))
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=13.5)
