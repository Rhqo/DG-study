"""Animation 31.5.2: coarsening the grid changes the topology of the extracted mesh (Figure 31.5.1).

Shape: the dumbbell of Figure 31.5.1 (balls of radius 0.5 at (+-0.8, 0, 0), rod of radius rho = 0.05 with axis from
(-0.8, 0, -0.07) to (0.8, 0, 0.07)); exact SDF sampled on a grid of spacing h whose y- and z-lines sit at +-h/2,
+-3h/2, ...; marching tetrahedra (dgmesh.dumbbell_mesh). h runs through 0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.09,
0.10, 0.12, 0.14 (h / sqrt 2 = rho at h = 0.0707). The rod also survives at h = 0.072 (h / sqrt 2 = 0.051 > rho,
but the tilted axis passes within rho of a sample in every sample plane), so h / sqrt 2 < rho is sufficient, not
necessary; 0.072 is checked here but not shown.
Left: the mesh seen from the side (-y direction, z up, as in the figure), with the components and chi = n0 - n1 + n2
counted on the mesh; the true rod between the balls is outlined by dashed lines.
Right: the rod's cross-section at x = 0 (disk of radius rho centred on the axis) and the (y, z) sample lattice; the
dashed circle has radius h / sqrt 2, the distance from the axis (a cell centre) to its four nearest samples.
Samples inside the rod are filled.
Self-check: every mesh is a closed manifold mesh; 1 component and chi = 2 for h <= 0.072 (shown up to 0.07), 2 components and chi = 4 for
h >= 0.075; at h = 0.08 the gap is x in (-0.113, 0.114) (as in the figure); for h >= 0.10 (h / 2 >= rho) no sample
lies in the rod and the mesh has no part of it (|x| > 0.25 on every component's inner end).

Render: bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-5-2-dumbbell-sweep.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _helpers_d import *  # noqa: E402,F401,F403

# ---------------------------------------------------------------- data
RHO = 0.05
HS = [0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.09, 0.10, 0.12, 0.14]
V72, F72 = dm.dumbbell_mesh(0.072)
assert len(dm.components(F72)) == 1 and dm.euler_characteristic(F72) == 2
MESHES = {}
for h in HS:
    V, F = dm.dumbbell_mesh(h)
    comps = dm.components(F)
    chi = dm.euler_characteristic(F)
    assert dm.is_manifold(F) and not dm.boundary_edges(F)
    assert (len(comps), chi) == ((1, 2) if h <= 0.072 else (2, 4)), (h, len(comps), chi)
    ext = sorted((V[np.unique(F[c])][:, 0].min(), V[np.unique(F[c])][:, 0].max()) for c in comps)
    if h == 0.08:
        assert abs(ext[0][1] + 0.113) < 0.002 and abs(ext[1][0] - 0.114) < 0.002
    if h >= 0.10:
        assert ext[0][1] < -0.25 and ext[1][0] > 0.25
    MESHES[h] = (V, F, len(comps), chi, ext)
POOL = max(len(MESHES[h][1]) for h in HS)


def lattice(h, lim=0.17):
    n = int(np.ceil(1.4 / h))
    n += n % 2
    ys = (np.arange(n) - (n - 1) / 2) * h
    ys = ys[np.abs(ys) < lim]
    Y, Z = np.meshgrid(ys, ys, indexing="ij")
    return np.stack([Y.ravel(), Z.ravel()], 1)


for h in HS:
    P = lattice(h)
    d = np.linalg.norm(P, axis=1).min()
    assert np.isclose(d, h / np.sqrt(2))              # the axis at x = 0 is a cell centre


# ---------------------------------------------------------------- scene
class Anim(da.Scene2D):
    def construct(self):
        cam = Cam(elev=8, azim=-90, scale=3.3, center=[-2.45, 0.95, 0])
        grey = hex_rgb("#C8C8C8")
        mv = MeshView(MESHES[HS[0]][1], stroke_width=0, opacity=1.0, ambient=0.35, pool=POOL,
                      light=(-0.4, 0.55, 0.75))

        def show(h):
            V, F, nc, chi, _ = MESHES[h]
            mv.set_state(V, grey, cam, F=F, stroke=False)

        show(HS[0])
        rod = VGroup(*[DashedLine(cam(np.array([-0.32, 0, -0.07 * 0.32 / 0.8 + s * RHO])),
                                  cam(np.array([0.32, 0, 0.07 * 0.32 / 0.8 + s * RHO])),
                                  stroke_width=1.6, color=C["aux"], dash_length=0.08) for s in (-1, 1)])
        rod.set_z_index(2)
        ttl = title("Dumbbell SDF: change the grid spacing h", 34)

        # cross-section panel
        PC, PS = np.array([4.35, 0.2]), 10.0

        def c2s(p):
            return np.array([PC[0] + PS * p[0], PC[1] + PS * p[1], 0.0])

        win = Square(PS * 0.34, stroke_width=1.5, color=BLACK).move_to(c2s((0, 0)))
        disk = Circle(radius=PS * RHO, stroke_width=2.5, color=C["tangent"], fill_color=C["region"],
                      fill_opacity=0.45).move_to(c2s((0, 0)))
        axis_dot = Dot(c2s((0, 0)), radius=0.05, color=C["tangent"])
        ptitle = VGroup(da.label("rod cross-section", 30), mt(r"at $x = 0$, radius $\rho = 0.05$", 30))
        ptitle.arrange(DOWN, buff=0.1).next_to(win, UP, buff=0.15)
        yl = mt(r"$y$", 30).next_to(win, DOWN, buff=0.08)
        zl = mt(r"$z$", 30).next_to(win, LEFT, buff=0.08)

        def panel(h):
            g = VGroup()
            P = lattice(h)
            for p in P:
                inside = np.linalg.norm(p) < RHO
                if inside:
                    g.add(Dot(c2s(p), radius=0.07, color=C["tangent"]).set_z_index(3))
                else:
                    g.add(Circle(radius=0.06, stroke_width=1.8, color=BLACK, fill_color=WHITE, fill_opacity=1)
                          .move_to(c2s(p)).set_z_index(3))
            g.add(DashedVMobject(Circle(radius=PS * h / np.sqrt(2), stroke_width=2.2, color=C["accent"]),
                                 num_dashes=28).move_to(c2s((0, 0))))
            return g

        def texts(h):
            V, F, nc, chi, _ = MESHES[h]
            word = "component" if nc == 1 else "components"
            top = mt(rf"$h = {h:g}$:   {nc} {word},  $\chi = {chi}$", 34)
            r = h / np.sqrt(2)
            rel = "<" if r < RHO else ">"
            crit = mt(rf"$h/\sqrt{{2}} = {r:.3f}$ {rel} $\rho = 0.05$", 32, color=C["accent"])
            g = VGroup(top, crit).arrange(DOWN, aligned_edge=LEFT, buff=0.18)
            return g.move_to([-6.85, -1.15, 0], aligned_edge=UL)

        def note(h):
            if h <= 0.07:
                s = "a sample in every cross-section: rod kept"
            elif h < 0.10:
                s = "no sample near the middle: rod breaks"
            else:
                s = "h/2 \u2265 \u03c1: no sample in the rod at all"
            return da.label(s, 30).move_to([-6.85, -2.65, 0], aligned_edge=LEFT)

        pan = panel(HS[0])
        txt = texts(HS[0])
        nt = note(HS[0])
        circ_lab = mt(r"dashed: radius $h/\sqrt{2}$", 30, color=C["accent"]).next_to(win, DOWN, buff=0.55)
        self.add(mv, rod, ttl, win, disk, axis_dot, ptitle, yl, zl, pan, txt, nt, circ_lab)
        self.wait(2.2)
        for h in HS[1:]:
            show(h)
            new_pan, new_txt, new_nt = panel(h), texts(h), note(h)
            self.remove(pan, txt, nt)
            pan, txt, nt = new_pan, new_txt, new_nt
            self.add(pan, txt, nt)
            hold = 2.6 if h in (0.07, 0.075, 0.08, 0.10) else 1.6
            self.wait(hold)
        self.wait(0.6)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=12.0)
