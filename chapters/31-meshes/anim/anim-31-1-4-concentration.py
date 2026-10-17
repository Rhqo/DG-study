"""Animation 31.1.4: rounding the cube with radius rho and letting rho -> 0 (Section 31.1, Figure 31.1.4).

Left: the surface of points at distance rho from the cube [-1, 1]^3 (as in Figure 31.1.4(b)): grey flat pieces
(K = H = 0), sky-blue quarter cylinders of radius rho along the 12 edges (K = 0, H = -1/(2 rho) for the outward N),
orange eighths of a sphere of radius rho at the 8 vertices (K = 1/rho^2). Black curve: its cross-section z = 0,
which is the rounded square of Figure 31.1.4(a).
Right: the curvature kappa of that cross-section near one corner against arc length s measured from the start of
the corner arc: a pulse of height 1/rho and width pi rho / 2, whose area is always pi/2. Grey ghosts: rho = 0.3
and 0.1 (the values of Figure 31.1.4(a)).
rho goes from 0.45 to 0.03 geometrically. The mesh pieces are built exactly; their areas are checked against
pi rho^2 / 2 per vertex and (pi rho / 2) * 2 per edge, and the face count stays a few hundred.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-1-4-concentration.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (FadeIn, Polygon, ValueTracker, VGroup, always_redraw, linear)  # noqa: E402

rounded_cube = hc.rounded_cube         # quarter circles split into 8 segments
tri_area = hc.tri_area

# self-check: the pieces have the areas of Figure 31.1.4(b) (polygonal approximation, k = 8)
for rho in (0.45, 0.1, 0.03):
    T, lab = rounded_cube(rho)
    A = tri_area(T)
    assert abs(A[lab == 0].sum() - 24) < 1e-12
    assert abs(A[lab == 1].sum() / 12 - np.pi * rho / 2 * 2) / (np.pi * rho) < 0.01        # per edge (length 2)
    assert abs(A[lab == 2].sum() / 8 - np.pi * rho ** 2 / 2) / (np.pi * rho ** 2 / 2) < 0.03  # per vertex
    # integral of K over a corner piece = K * area = pi / 2, independent of rho
    assert abs((1 / rho ** 2) * (np.pi * rho ** 2 / 2) - np.pi / 2) < 1e-12
N_TRI = len(rounded_cube(0.1)[0])
assert N_TRI < 1500
RHO0, RHO1 = 0.45, 0.03
COLORS = {0: hc.rgb(hc.HEX["surface"]), 1: hc.rgb(hc.HEX["region"]), 2: hc.rgb(hc.HEX["accent"])}


def cross_section(rho, n_arc=24):
    """The rounded square (z = 0 cross-section), counterclockwise."""
    pts = []
    for cx, cy, a0 in ((1, -1, -np.pi / 2), (1, 1, 0), (-1, 1, np.pi / 2), (-1, -1, np.pi)):
        for tt in np.linspace(a0, a0 + np.pi / 2, n_arc):
            pts.append([cx + rho * np.cos(tt), cy + rho * np.sin(tt), 0.0])
    return np.array(pts + [pts[0]])


class Anim(hc.Scene2D):
    def construct(self):
        cam = hc.Cam(elev=24, azim=-58, scale=1.12, origin=(-3.7, 0.6))
        tt = ValueTracker(0.0)            # rho = RHO0 (RHO1/RHO0)^tt: a geometric schedule

        class _R:
            def get_value(self):
                return float(np.exp(np.log(RHO0) + tt.get_value() * (np.log(RHO1) - np.log(RHO0))))
        rho = _R()
        T, lab = rounded_cube(RHO0)
        F = np.arange(3 * len(T)).reshape(-1, 3)
        cols = np.array([COLORS[x] for x in lab])
        mv = hc.MeshView(T.reshape(-1, 3), F, cam, colors=cols, edge_color="fill", edge_width=0.6, cull=True,
                         sort=False, ambient=0.5)

        def upd_mesh(m):
            Tn, _ = rounded_cube(rho.get_value())
            m.set_mesh(Tn.reshape(-1, 3))
        mv.add_updater(upd_mesh)

        def section():
            P = cross_section(rho.get_value())
            # front half only (outward normal toward the viewer)
            nrm = P - np.clip(P, -1, 1)
            vis = (nrm @ cam.view) > 0
            g = VGroup()
            run = []
            for p, v in zip(P, vis):
                if v:
                    run.append(p)
                elif len(run) > 1:
                    g.add(hc.polyline(cam(np.array(run)), width=4))
                    run = []
                else:
                    run = []
            if len(run) > 1:
                g.add(hc.polyline(cam(np.array(run)), width=4))
            return g
        sec = always_redraw(section)
        self.add(mv, sec)

        title = hc.text("round the cube with radius ρ, then let ρ → 0", size=34).move_to([0, 3.45, 0])
        self.add(title)

        # corner formula and live values (left)
        fK = hc.math(r"$\mathrm{corner:}\ \iint K\,dA = \frac{1}{\rho^2}\cdot\frac{\pi\rho^2}{2} = \frac{\pi}{2}$",
                     size=34, color=hc.C["accent"]).move_to([-3.45, -2.05, 0])
        live = always_redraw(lambda: hc.text(
            f"ρ = {rho.get_value():.3f}     K = 1/ρ² = {1 / rho.get_value() ** 2:.0f}", size=32)
            .move_to([-3.45, -2.75, 0]))     # above the paused control bar (GUIDELINES 12.8)
        self.add(fK, live)

        # plot (right)
        pl = hc.Plot((-0.3, 0.8), (0, 35), ((1.3, -1.15), (6.3, 2.5)))
        axes = pl.axes(xtick=(0, 0.5), ytick=(0, 10, 20, 30), size=30)
        xl = hc.math(r"$s$", size=36).move_to(pl(0.8, 0) + np.array([0.3, -0.05, 0]))
        yl = hc.math(r"$\kappa$", size=38).move_to(pl(-0.3, 35) + np.array([-0.55, 0.0, 0]))
        sub = hc.text("cross-section, one corner", size=30).move_to([3.8, 2.9, 0])
        self.add(axes, xl, yl)

        def pulse_pts(r):
            w, hgt = np.pi * r / 2, 1 / r
            return [(-0.3, 0), (0, 0), (0, hgt), (w, hgt), (w, 0), (0.8, 0)]

        def pulse(r, color=hc.BLACK, width=4):
            P = np.array(pulse_pts(r))
            return hc.polyline(pl(P[:, 0], P[:, 1]), color=color, width=width)

        def area():
            r = rho.get_value()
            w, hgt = np.pi * r / 2, 1 / r
            P = pl(np.array([0, w, w, 0]), np.array([0, 0, hgt, hgt]))
            return Polygon(*P, fill_color=hc.C["region"], fill_opacity=0.55, stroke_width=0)
        shade = always_redraw(area)
        curve = always_redraw(lambda: pulse(rho.get_value()))
        self.add(shade, curve)
        alab = hc.math(r"$\mathrm{area} = \frac{\pi}{2}$", size=34, color=hc.C["tangent"])
        alab.move_to(pl(0.42, 24))
        lead = always_redraw(lambda: hc.seg(alab.get_left() + np.array([-0.05, -0.05, 0]),
                                            pl(np.pi * rho.get_value() / 4, 0.55 / rho.get_value()),
                                            color=hc.C["tangent"], width=2))
        fk = hc.math(r"$\int\kappa\,ds = \frac{1}{\rho}\cdot\frac{\pi\rho}{2} = \frac{\pi}{2}$", size=34,
                     color=hc.C["tangent"]).move_to([3.8, -2.42, 0])
        self.add(alab, lead, fk, sub)
        self.wait(2.0)

        # rho -> 0 geometrically; leave grey ghosts at 0.3 and 0.1
        ghosts = {}

        def ghost_upd(m):
            r = rho.get_value()
            for g in (0.3, 0.1):
                if r <= g and g not in ghosts:
                    gh = pulse(g, color=hc.C["aux"], width=2.5)
                    ghosts[g] = gh
                    self.add(gh)
        gtrack = VGroup()
        gtrack.add_updater(ghost_upd)
        self.add(gtrack)
        self.play(tt.animate(rate_func=linear).set_value(1.0), run_time=11.0)
        self.wait(1.0)
        # the limit: point masses pi/2 at the vertices
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        front = [c for c in corners if not np.all(np.sign(c) == -np.sign(cam.view))]
        dots = VGroup(*[hc.dot(cam(c), color=hc.C["accent"], r=0.1) for c in front]).set_z_index(5)
        dl = hc.text("each vertex: π/2 = angle defect", size=30, color=hc.C["accent"])
        dl.move_to(fK)
        self.play(FadeIn(dots), *hc.swap(fK, dl), run_time=1.0)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.0)
