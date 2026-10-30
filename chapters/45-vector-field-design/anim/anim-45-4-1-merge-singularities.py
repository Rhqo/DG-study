"""Animation 45.4.1: moving one of two index +1 singularities on the icosphere toward the other; the smoothest field
follows, its energy grows, and the two merge into one singularity of index 2 (Section 45.4, Theorem 45.4.10,
Example 45.4.14, Figure 45.4.1, Exercise 45.4.6).

Mesh: dgmesh.icosphere(3). k = +1 at vertex 0 and at a moving vertex j (k = 2 at vertex 0 when j = 0); sum k = 2 = chi.
For every j: xi = ch45.trivial_connection (Theorem 45.4.10), psi = ch45.parallel_field (psi = 0 in face 0), energy
|xi|_star^2 = xi^T star1^{-1} xi. Path of j: the vertices nearest to the great circle from x_0 through the vertex at 90
degrees (vertex 81, as Figure 45.4.1(b)) to the antipode (vertex 3), sampled every 1.5 degrees, keeping a vertex only
if it is strictly closer to x_0 than the previous one; the animation runs from the vertex nearest 150 degrees to vertex 0. Between two vertices the arrows are blended
for 0.3 s (psi interpolated along the shorter arc), then the computed field is held for 0.3 s.
Drawing: orthographic view from the point of the great circle halfway between x_0 and the moving vertex, screen-up the
normal of the great circle (x_0 left, the moving singularity right; the camera turns with the midpoint), back faces
culled; blue arrows: unit field vectors at the centroids of the faces with index = 3 mod 4 that face the viewer (length
0.12, centred, as Figure 45.4.1); black dots: the singular vertices. Right: the energy against the angle between the
two singularities for every vertex of the path (gray dots), the current one black.
Self-check (as Figure 45.4.1 and Example 45.4.14): for every j the curvature of xi is 2 pi k exactly, psi is parallel for
xi, the field indices equal k; energies 43.081 at the antipode, 47.169 at vertex 81, 98.770 at k_0 = 2.

Render: bash tools/render_anims.sh chapters/45-vector-field-design/anim/anim-45-4-1-merge-singularities.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _helpers import *  # noqa: E402,F401,F403
import dgdec as dd  # noqa: E402
import ch45  # noqa: E402

# ---------------------------------------------------------------- data
V, F = dm.icosphere(3)
D0, s1 = dd.d0(F, len(V)), dd.hodge1(V, F)
rho, delta = ch45.transport_angles(V, F), dm.angle_defect(V, F)
X0 = V[0]
J90 = int(np.argmin(np.abs(V @ V[0] - np.cos(np.radians(90)))))   # as Figure 45.4.1 and Exercise 45.4.6
assert J90 == 81 and int(np.argmin(V @ X0)) == 3
U = V[J90] - (V[J90] @ X0) * X0
U /= np.linalg.norm(U)
NGC = np.cross(X0, U)
PATH = []
for a in np.radians(np.linspace(180, 0, 121)):
    j = int(np.argmax(V @ (np.cos(a) * X0 + np.sin(a) * U)))
    if not PATH or V[j] @ X0 > V[PATH[-1]] @ X0 + 1e-9:     # strictly closer to x_0 than the previous vertex
        PATH.append(j)
assert PATH[0] == 3 and PATH[-1] == 0 and J90 in PATH
ANG = np.degrees(np.arccos(np.clip(V[PATH] @ X0, -1, 1)))
PSI, EN = [], []
for j in PATH:
    k = np.zeros(len(V), int)
    k[0] += 1
    k[j] += 1
    xi = ch45.trivial_connection(D0, s1, delta, rho, k)
    assert np.allclose(delta - D0.T @ xi, 2 * np.pi * k)
    psi = ch45.parallel_field(F, rho, xi)
    assert np.abs(ch45.wrap(ch45.connection_from_field(F, rho, psi) - xi)).max() < 1e-9
    ind, _ = ch45.field_indices(D0, delta, rho, psi, F)
    assert np.array_equal(ind, k)
    PSI.append(psi)
    EN.append(float(xi @ (xi / s1)))
EN = np.array(EN)
assert abs(EN[0] - 43.081) < 5e-4 and abs(EN[PATH.index(J90)] - 47.169) < 5e-4 and abs(EN[-1] - 98.770) < 5e-4
assert EN[-1] > EN[PATH.index(J90)] > EN[0]
assert np.all(np.diff(EN) > 0)                   # along this path the energy grows at every step toward x_0
START = int(np.argmin(np.abs(ANG - 150)))
STEPS = list(range(START, len(PATH)))
R_, B_, N_ = ch45.face_frames3(V, F)
CEN = V[F].mean(1)
SEL = np.where(np.arange(len(F)) % 4 == 3)[0]


def gc_point(a_deg):
    a = np.radians(a_deg)
    return np.cos(a) * X0 + np.sin(a) * U


# ---------------------------------------------------------------- layout
SCALE, CENTER = 2.7, np.array([-3.6, -0.75, 0.0])
PL = Plot(1.45, -1.55, 4.9, 3.0, (0, 180), (40, 100))


def cam_at(a_deg):
    return Cam.from_view(gc_point(a_deg / 2), scale=SCALE, center=CENTER, up_hint=NGC)


def light_arrow(p, q, color, width=3.2, tip=0.09):
    d = q - p
    L = np.linalg.norm(d) + 1e-12
    e = d / L
    nrm = np.array([-e[1], e[0], 0.0])
    base = q - tip * e
    g = VGroup(Line(p, base, color=color, stroke_width=width),
               Polygon(q, base + 0.5 * tip * nrm, base - 0.5 * tip * nrm, color=color, fill_color=color,
                       fill_opacity=1, stroke_width=0.5))
    return g


class Anim(Scene2D):
    def construct(self):
        sv = ValueTracker(0.0)
        mv = MeshView(F, stroke_width=0.25, stroke_rgb=(0.62, 0.62, 0.62), ambient=0.62)

        def state():
            s = sv.get_value()
            i = min(int(np.floor(s)), len(STEPS) - 2)
            t = min(max((s - i) / 0.5, 0.0), 1.0)          # first half of each unit: blend, second half: hold
            j0, j1 = STEPS[i], STEPS[i + 1]
            a = (1 - t) * ANG[j0] + t * ANG[j1]
            psi = PSI[j0] + t * ch45.wrap(PSI[j1] - PSI[j0])
            pos = ch45.unit((1 - t) * V[PATH[j0]] + t * V[PATH[j1]]) if PATH[j1] != PATH[j0] else V[PATH[j0]]
            return a, psi, pos, (j1 if t >= 1 else j0), t

        def upd_mesh(m):
            a, psi, pos, j, t = state()
            m.set_state(V, np.array([0.9, 0.9, 0.9]), cam_at(max(a, 2.0)))

        mv.add_updater(upd_mesh)
        upd_mesh(mv)

        def field():
            a, psi, pos, j, t = state()
            cam = cam_at(max(a, 2.0))
            X = np.cos(psi)[:, None] * R_ + np.sin(psi)[:, None] * B_
            g = VGroup()
            for f in SEL:
                if N_[f] @ cam.view > 0.15:
                    g.add(light_arrow(cam(CEN[f] - 0.06 * X[f]), cam(CEN[f] + 0.06 * X[f]), C["tangent"]))
            for p in ((X0, pos) if PATH[j] != 0 or t < 1 else (X0,)):
                if p @ cam.view > 0:
                    g.add(Dot(cam(p), radius=0.09, color=BLACK))
            return g.set_z_index(3)

        fv = always_redraw(field)
        ttl = lab("the smoothest field with k = +1 at two vertices", 34).move_to([-6.85, 3.5, 0], aligned_edge=LEFT)
        axes = PL.axes(xticks=[(0, "0°"), (90, "90°"), (180, "180°")],
                       yticks=[(50, "50"), (75, "75"), (100, "100")], xlabel="angle between them")
        elab = mt(r"energy $\|\xi\|_\star^2$", 30).next_to(PL.p(0, 100), UP, buff=0.12).shift(RIGHT * 1.2)
        gray = VGroup(*[Dot(PL.p(ANG[i], EN[i]), radius=0.05, color="#9A9A9A") for i in range(len(PATH))])

        def cur():
            a, psi, pos, j, t = state()
            return Dot(PL.p(ANG[j], EN[j]), radius=0.09, color=BLACK).set_z_index(4)

        cd = always_redraw(cur)

        def info():
            a, psi, pos, j, t = state()
            return VGroup(lab(f"angle {ANG[j]:.0f}°", 30), lab(f"energy {EN[j]:.3f}", 30)).arrange(
                RIGHT, buff=0.6).move_to([-6.85, 2.85, 0], aligned_edge=LEFT)

        box = always_redraw(info)
        self.add(mv, fv, ttl, axes, elab, gray, cd, box)
        self.wait(2.0)
        self.play(sv.animate.set_value(len(STEPS) - 1.5), run_time=0.6 * (len(STEPS) - 1.5), rate_func=linear)
        mv.clear_updaters()
        fin = mt(r"merged: one vertex with $k = 2$", 30).move_to([-6.85, 2.3, 0],
                                                                                        aligned_edge=LEFT)
        self.play(FadeIn(fin), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=9.1)
