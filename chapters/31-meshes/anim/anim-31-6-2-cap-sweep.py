"""Animation 31.6.2: the larger the cap, the more conformal flattening pays in area (Figure 31.6.2).

Data: the unit sphere as the level-4 icosphere (2562 vertices, 5120 faces); the cap {polar angle < Theta} = the faces
whose centroid has z > cos Theta, for Theta = 30, 45, ..., 150, 160, 170 degrees. Each cap is flattened by LSCM with
its farthest pair of boundary vertices pinned to (0, 0) and (1, 0) [Levy02]. sigma_1 sigma_2 = surface area per unit
UV area for each face (dgmesh.uv_distortion).
Left: the sphere (elev 15, azim -60), cap faces coloured by log2(sigma_1 sigma_2 / median) (RdBu_r on [-2.5, 2.5], red =
a texel covers more surface), other faces grey. Middle: the UV of the cap, scaled to a fixed width, same colours.
Right: max / min of sigma_1 sigma_2 against Theta (log axis); black curve sec^4(Theta / 2), the same ratio for the
stereographic projection from the antipode.
Self-check (as the figure): no flipped face; median sigma_1 / sigma_2 < 1.08; the ratio follows sec^4(Theta / 2) within a
factor 1.5; ratio 1.9 at 60, 4.6 at 90, 20.4 at 120 and 14470 at 170 degrees.

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-6-2-cap-sweep.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

V, F = dm.icosphere(4)
CEN = V[F].mean(1)


def farthest_pair(Vc, loop):
    P = Vc[loop]
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    a, b = np.unravel_index(np.argmax(D), D.shape)
    return loop[a], loop[b]


THETAS = [30, 45, 60, 75, 90, 105, 120, 135, 150, 160, 170]
CAPS = {}
for th in THETAS:
    mask = CEN[:, 2] > np.cos(np.radians(th))
    Vc, Fc, _ = dm.submesh(V, F, mask)
    assert dm.euler_characteristic(Fc) == 1 and dm.is_manifold(Fc)
    loop = dm.boundary_loop(Fc)
    UV = dm.lscm(Vc, Fc, farthest_pair(Vc, loop), [(0.0, 0.0), (1.0, 0.0)])
    s, det = dm.uv_distortion(Vc, Fc, UV)
    assert (det > 0).all()
    a = s[:, 0] * s[:, 1]
    assert np.median(s[:, 0] / s[:, 1]) < 1.08
    sec4 = 1 / np.cos(np.radians(th) / 2) ** 4
    assert abs(np.log(a.max() / a.min() / sec4)) < np.log(1.5)
    W = UV - 0.5 * (UV.min(0) + UV.max(0))
    W /= np.ptp(W, axis=0).max()
    CAPS[th] = dict(mask=mask, Fc=Fc, UV=W, val=np.log2(a / np.median(a)), ratio=a.max() / a.min())
assert [round(CAPS[t]["ratio"], 1) for t in (60, 90, 120)] == [1.9, 4.6, 20.4] and round(CAPS[170]["ratio"]) == 14470
POOL = max(len(c["Fc"]) for c in CAPS.values())


class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=15, azim=-60, scale=1.45, center=[-5.15, 0.55, 0])
        sph = MeshView(F, stroke_width=0.5, ambient=0.5)
        grey = hex_rgb("#DADADA")
        UVC, UVS = np.array([-1.45, 0.55]), 3.0

        def colors_for(th):
            c = CAPS[th]
            rgb = np.tile(grey, (len(F), 1))
            rgb[np.where(c["mask"])[0]] = cmap_rgb(c["val"], -2.5, 2.5, "RdBu_r")
            return rgb

        uv_views = {}
        for th in THETAS:
            c = CAPS[th]
            fm = FlatMeshView(c["Fc"], stroke_width=0.5)
            fm.set_state(c["UV"], cmap_rgb(c["val"], -2.5, 2.5, "RdBu_r"), center=UVC, scale=UVS, stroke="seal")
            uv_views[th] = fm
        ttl = title("Sphere caps under LSCM: the price is area", 34)
        lab_s = da.label("on the sphere", 30).move_to([-5.15, 2.8, 0])
        lab_u = da.label("UV (LSCM)", 30).move_to([-1.45, 2.8, 0])
        bar = colorbar(-6.6, -2.6, 2.8, 0.2, "RdBu_r", -2.5, 2.5, ticks=[(-2, "-2"), (0, "0"), (2, "2")], size=30)
        bar_lab = mt(r"colour: $\log_2(\sigma_1\sigma_2/\mathrm{median})$", 30).next_to(bar, RIGHT, buff=0.3)
        bar_lab.shift(UP * 0.1)

        P = Plot(2.3, -1.7, 4.0, 3.8, (0, 180), (1, 1e5), ylog=True)
        axes = P.axes(xticks=[(0, "0"), (90, "90"), (180, "180")],
                      yticks=[(1, "1"), (10, "10"), (100, mt(r"$10^2$", 30)), (1e3, mt(r"$10^3$", 30)),
                              (1e4, mt(r"$10^4$", 30)), (1e5, mt(r"$10^5$", 30))],
                      xlabel=mt(r"cap size $\Theta$ (degrees)", 30))
        ylab = mt(r"$\max\sigma_1\sigma_2\,/\min\sigma_1\sigma_2$", 30).next_to(P.p(0, 1e5), UP, buff=0.15).shift(RIGHT * 1.6)
        tt = np.linspace(2, 172, 300)
        sec = P.curve(tt, 1 / np.cos(np.radians(tt) / 2) ** 4, color=BLACK, width=3)
        sec_lab = mt(r"$\sec^4(\Theta/2)$", 30).move_to(P.p(15, 2.5e3), aligned_edge=LEFT)
        self.add(ttl, lab_s, lab_u, bar, bar_lab, axes, ylab, sec, sec_lab)
        dots = VGroup()
        self.add(dots)
        cur = None
        read = None
        for k, th in enumerate(THETAS):
            sph.set_state(V, colors_for(th), cam, stroke="seal")
            if k == 0:
                self.add(sph)
            if cur is not None:
                self.remove(cur)
            cur = uv_views[th]
            self.add(cur)
            r = CAPS[th]["ratio"]
            dots.add(Dot(P.p(th, r), radius=0.085, color=C["normal"], stroke_color=BLACK, stroke_width=1.2))
            tot = 2 * np.pi * (1 - np.cos(np.radians(th)))
            rtxt = f"{r:.1f}" if r < 1000 else f"{r:.0f}"
            new = VGroup(mt(rf"$\Theta = {th}^\circ$:  area ratio ${rtxt}$", 32),
                         mt(rf"total curvature in the cap: ${tot:.2f}$", 30))
            new.arrange(DOWN, aligned_edge=LEFT, buff=0.12).move_to([-6.85, -1.05, 0], aligned_edge=UL)
            if read is not None:
                self.remove(read)
            read = new
            self.add(read)
            self.wait(2.2 if th in (90, 170) else 1.6)
        self.wait(1.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=12.0)
