"""Animation 45.3.1: Levi-Civita transport around a latitude band of the icosphere; the holonomy equals the sum of
the angle defects inside, a staircase that follows 2 pi (1 - cos theta_0) (Section 45.3, Corollary 45.3.12,
Example 45.3.13, Figure 45.3.3).

Mesh: dgmesh.icosphere(3) (642 vertices, unit sphere, outward faces), rho = ch45.transport_angles, delta =
dgmesh.angle_defect. For a colatitude theta_0, S = vertices with colatitude < theta_0 (faces with all three vertices in S
dark gray) and c* = -d0 1_S, the dual cycle through the band of faces with vertices on both sides (light blue),
traversed with S on the left (longitude increasing).
Part 1 (theta_0 = pi/3): a unit vector starts in the band face nearest to longitude 0, pointing south (w_0, blue), and
is carried face by face by psi -> psi +- rho_e (Definition 45.3.2), leaving an arrow (length 0.12) in every front
band face; after one turn it is w_1 (orange) in the start face.
Part 2: theta_0 sweeps on from pi/3 to pi - 0.05 (the values below pi/3 appear at once). Right: the holonomy c*.rho
mod 2 pi (black dots, the 400 values of Figure 45.3.3(b)) and 2 pi (1 - cos theta_0) mod 2 pi (orange); the start face
shows w_0 (blue) and the returned w_1 (orange) for the current band.
View: orthographic, azim 0 (longitude 0 in front), back faces culled; elevation 35 deg in part 1, then
95 deg - theta_0 so that the start face stays in front (the camera tilts down with the band).
A step of the staircase is the sum of delta_i over the vertices that join S between two samples (the mesh is
symmetric, so vertices come in groups of 1, 2 or 4 at one colatitude; 32 on the equator).
Self-check (as Figure 45.3.3): for every theta_0, c*.rho = sum_{i in S} delta_i mod 2 pi to 1e-10; the transported
vector returns rotated by that angle; at theta_0 = pi/3: 82 band faces and holonomy 3.1594, within 0.03 of pi.

Render: bash tools/render_anims.sh chapters/45-vector-field-design/anim/anim-45-3-1-band-holonomy.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
from _helpers import *  # noqa: E402,F401,F403
import dgdec as dd  # noqa: E402
import ch45  # noqa: E402

TWO_PI = 2 * np.pi
# ---------------------------------------------------------------- data
V, F = dm.icosphere(3)
D0 = dd.d0(F, len(V))
rho = ch45.transport_angles(V, F)
delta = dm.angle_defect(V, F)
TH = np.arccos(np.clip(V[:, 2], -1, 1))
E, LEFTF, RIGHTF = ch45.left_right_faces(F)
R_, B_, N_ = ch45.face_frames3(V, F)
CEN = V[F].mean(1)


def band(t0):
    """(S mask, c*, ordered band faces, psi in each face, total turn) for colatitude t0."""
    S = (TH < t0).astype(float)
    cs = -(D0 @ S)
    if not cs.any():
        return S, cs, [], np.zeros(0), 0.0
    nxt = {}
    for e in np.where(cs != 0)[0]:
        a, b = (RIGHTF[e], LEFTF[e]) if cs[e] > 0 else (LEFTF[e], RIGHTF[e])
        nxt[int(a)] = (int(b), int(e), 1.0 if cs[e] > 0 else -1.0)
    start = min(nxt, key=lambda f: abs(np.arctan2(CEN[f, 1], CEN[f, 0])))
    south = ch45.unit(np.array([0, 0, -1.0]) - (-N_[start, 2]) * N_[start])
    psi0 = np.arctan2(south @ B_[start], south @ R_[start])
    seq, psis, f, p = [start], [psi0], start, psi0
    while True:
        g, e, sg = nxt[f]
        p = p + sg * rho[e]
        if g == start:
            break
        seq.append(g)
        psis.append(p)
        f = g
    return S, cs, seq, np.array(psis), p - psi0


S3, CS3, SEQ, PSIS, TURN = band(np.pi / 3)
HOL3 = np.mod(CS3 @ rho, TWO_PI)
assert len(SEQ) == 82 and abs(ch45.wrap(TURN - HOL3)) < 1e-10 and abs(HOL3 - np.pi) < 0.03
assert round(HOL3, 4) == 3.1594
THS = np.linspace(0.05, np.pi - 0.05, 400)
HOL = []
for t0 in THS:
    S = (TH < t0).astype(float)
    h = np.mod(-(D0 @ S) @ rho, TWO_PI)
    assert abs(ch45.wrap(h - np.mod(delta @ S, TWO_PI))) < 1e-10
    HOL.append(h)
HOL = np.array(HOL)
SMOOTH = np.mod(TWO_PI * (1 - np.cos(THS)), TWO_PI)
_NS = np.array([(TH < t0).sum() for t0 in THS])                      # the steps of the staircase (caption numbers)
_dn, _dh = np.diff(_NS), np.diff(np.array([delta[TH < t0].sum() for t0 in THS]))
assert 0.017 < delta.min() and delta.max() < 0.023                     # one delta_i: 0.017 .. 0.023
assert set(_dn[_dn > 0]) - {32} <= set(range(2, 17)) and _dn.max() == 32  # 2 .. 16 vertices a step, 32 at the equator
assert round(float(_dh[np.argmax(_dn)]), 2) == 0.62 and abs(THS[np.argmax(_dn) + 1] - np.pi / 2) < 0.01
_lab = (THS > 1.8) & (THS < np.pi)                 # where the label of the smooth curve sits (data 0.4 .. 1.1)
assert not np.any(_lab & (HOL > 0.35) & (HOL < 1.15)) and not np.any(_lab & (SMOOTH > 0.35) & (SMOOTH < 1.15))
for t0 in (0.4, 1.3, 2.2):                      # the transported vector returns rotated by the holonomy
    _, cs, _, _, turn = band(t0)
    assert abs(ch45.wrap(turn - cs @ rho)) < 1e-10


def vec(f, psi):
    return np.cos(psi) * R_[f] + np.sin(psi) * B_[f]


# ---------------------------------------------------------------- layout
CAM = Cam(elev=35, azim=0, scale=2.55, center=[-3.65, -0.65, 0])
PL = Plot(1.25, -1.75, 5.0, 3.1, (0, np.pi), (0, TWO_PI))      # axis label above the bottom 15% (poster)
GRAY, BLUE, PALE = np.array([0.6, 0.6, 0.6]), np.array([0.80, 0.90, 0.97]), np.array([0.93, 0.93, 0.93])


def face_colors(S, seq):
    inside = S[F].min(1) > 0
    col = np.tile(PALE, (len(F), 1))
    col[inside] = GRAY
    col[seq] = BLUE
    return col


def arrow3(f, w, length, color, width):
    p = CEN[f] * 1.01
    return arrow(CAM(p), CAM(p + length * w), color=color, width=width, tip=0.13 if length < 0.2 else 0.2)


class Anim(Scene2D):
    def construct(self):
        mv = MeshView(F, stroke_width=0.3, stroke_rgb=(0.55, 0.55, 0.55), ambient=0.6)
        mv.set_state(V, face_colors(S3, SEQ), CAM)
        ttl = lab("Levi-Civita transport around a latitude band", 34).move_to([-6.85, 3.5, 0], aligned_edge=LEFT)
        t0lab = mt(r"$\theta_0 = \pi/3$", 32).move_to([-6.85, 2.85, 0], aligned_edge=LEFT)
        w0 = arrow3(SEQ[0], vec(SEQ[0], PSIS[0]), 0.33, C["tangent"], 7).set_z_index(5)
        w0l = mt(r"$w_0$", 32, color=C["tangent"]).next_to(w0.get_end(), DOWN, buff=0.08).set_z_index(5)
        self.add(mv, ttl, t0lab, w0, w0l)
        self.wait(2.0)
        # part 1: carry the vector around the band
        kv = ValueTracker(0.0)
        front = [k for k, f in enumerate(SEQ) if N_[f] @ CAM.view > 0.1 and k > 0]

        def trail():
            k = int(kv.get_value())
            g = VGroup()
            for j in front:
                if j <= k:
                    g.add(arrow3(SEQ[j], vec(SEQ[j], PSIS[j]), 0.13, C["tangent"], 3.5))
            return g.set_z_index(4)

        tr = always_redraw(trail)
        cnt = readout(lambda: f"face {int(kv.get_value()) + 1} of {len(SEQ)}", [-6.85, 2.3, 0], size=30)
        self.add(tr, cnt)
        self.play(kv.animate.set_value(len(SEQ) - 1), run_time=6.0, rate_func=linear)
        w1 = arrow3(SEQ[0], vec(SEQ[0], PSIS[0] + TURN), 0.33, C["accent"], 7).set_z_index(6)
        w1l = mt(r"$w_1$", 32, color=C["accent"]).next_to(w1.get_end(), UP, buff=0.08).set_z_index(6)
        back = mt(r"back in the start face: turned by $3.159 \approx \pi$", 30).move_to([-6.85, 2.3, 0],
                                                                                     aligned_edge=LEFT)
        self.play(GrowArrow(w1), FadeIn(w1l), FadeOut(cnt), FadeIn(back), run_time=0.8)
        self.wait(2.0)
        # part 2: sweep theta_0
        axes = PL.axes(xticks=[(0, "0"), (np.pi / 2, mt(r"$\pi/2$", 30)), (np.pi, mt(r"$\pi$", 30))],
                       yticks=[(0, "0"), (np.pi, mt(r"$\pi$", 30)), (TWO_PI, mt(r"$2\pi$", 30))],
                       xlabel=mt(r"colatitude $\theta_0$", 30))
        ylab = lab("holonomy mod 2π", 30).next_to(PL.p(0, TWO_PI), UP, buff=0.15).shift(RIGHT * 1.7)
        sm = VGroup()
        jump = np.where(np.abs(np.diff(SMOOTH)) > np.pi)[0]
        cuts = np.r_[0, jump + 1, len(THS)]
        for a_, b_ in zip(cuts[:-1], cuts[1:]):
            sm.add(PL.curve(THS[a_:b_], SMOOTH[a_:b_], color=C["accent"], width=7, clip=False).set_stroke(opacity=0.55))
        smlab = mt(r"$2\pi(1 - \cos\theta_0)$", 30, color=C["accent"]).move_to(PL.p(1.85, 0.75), aligned_edge=LEFT)
        J0 = int(np.searchsorted(THS, np.pi / 3))
        iv = ValueTracker(float(J0))

        def cam_for(j):
            """part 2: the camera tilts with the band (elevation 95 deg - theta_0, as in part 1 at pi/3)."""
            return Cam(elev=95 - np.degrees(THS[j]), azim=0, scale=CAM.scale, center=CAM.center)

        def dots():
            j = int(iv.get_value())
            g = VGroup(*[Dot(PL.p(THS[i], HOL[i]), radius=0.035, color=BLACK) for i in range(0, j + 1)])
            g.add(Dot(PL.p(THS[j], HOL[j]), radius=0.08, color=BLACK))
            return g.set_z_index(3)

        def sphere(m):
            j = int(iv.get_value())
            S, _, seq, psis, turn = band(THS[j])
            m.set_state(V, face_colors(S, seq), cam_for(j))

        def vecs():
            j = int(iv.get_value())
            cam = cam_for(j)
            S, _, seq, psis, turn = band(THS[j])
            f = seq[0]
            p0 = CEN[f] * 1.01
            a0 = arrow(cam(p0), cam(p0 + 0.33 * vec(f, psis[0])), color=C["tangent"], width=7, tip=0.2)
            a1 = arrow(cam(p0), cam(p0 + 0.33 * vec(f, psis[0] + turn)), color=C["accent"], width=7, tip=0.2)
            return VGroup(a0, a1).set_z_index(6)

        def tlabel():
            j = int(iv.get_value())
            return lab(f"\u03b8\u2080 = {THS[j]:.2f}    holonomy = {HOL[j]:.3f}", 32).move_to([-6.85, 2.85, 0],
                                                                                       aligned_edge=LEFT)

        eqd = mt(r"holonomy $= \Sigma_{i \in S}\,\delta_i$ (mod $2\pi$)", 30).move_to([-6.85, 2.3, 0],
                                                                                    aligned_edge=LEFT)
        dd_ = always_redraw(dots)
        vv = always_redraw(vecs)
        tl = always_redraw(tlabel)
        self.add(vv)
        self.remove(w0, w1)
        self.play(FadeOut(tr), FadeOut(w0l), FadeOut(w1l), FadeOut(back), FadeOut(t0lab), FadeIn(tl), FadeIn(dd_),
                  FadeIn(axes), FadeIn(ylab), FadeIn(sm), FadeIn(smlab), FadeIn(eqd), run_time=1.0)
        mv.add_updater(sphere)
        self.play(iv.animate.set_value(len(THS) - 1), run_time=10.0, rate_func=linear)
        mv.remove_updater(sphere)
        fin = mt(r"each step: $\Sigma\,\delta_i$ of the vertices $i$ that join $S$", 30).move_to([-6.85, 2.3, 0],
                                                                                              aligned_edge=LEFT)
        self.play(FadeOut(eqd), FadeIn(fin), run_time=0.5)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=16.5)
