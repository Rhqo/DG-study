"""Animation 38.4.1: a vertex star folds over while its angle sum stays 2 pi (Section 38.4, Proposition 38.4.11,
Non-example 38.4.14, Corollary 38.4.10, Figure 38.4.3(a)).

Star: the regular hexagonal star (interior vertex 0 at the origin, link vertices 1..6 on the unit circle at
0, 60, ..., 300 degrees, faces (0, k, k + 1), lib/ddg38.planar_star). Link vertex 2 moves on the straight path in
(angle, radius) from (60 deg, 1) to (150 deg, 0.8), the folded star of Non-example 38.4.14 (lib/ddg38.folded_star).
theta_k = signed angle at vertex 0 of face (0, k, k + 1) (lib/ddg38.projected_star_angles with n = e_3).
At s = 2/3 vertex 2 reaches the ray of vertex 3 (120 deg, radius 0.867): face (0, 2, 3) is degenerate; after that
theta_2 < 0, the face turns clockwise and lies inside its neighbour (0, 3, 4) (it also meets part of (0, 1, 2)):
the overlap area (ddg38.max_star_overlap, the largest over pairs of faces) is the whole area of face (0, 2, 3).
Self-check: sum theta_k = 2 pi along the whole path; overlap = 0 exactly while theta_2 > 0, and after that it equals
the area of face (0, 2, 3), all of it inside face (0, 3, 4) (0.200 at the end); the end
state equals folded_star with theta = (150, -30, 60, 60, 60, 60) degrees; star_passes is True at s = 0 and False at 1.

Render: bash tools/render_anims.sh chapters/38-surfaces-smooth-discrete/anim/anim-38-4-1-folding-star.py
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
BASE_ANG = np.deg2rad([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
E3 = np.array([0.0, 0.0, 1.0])


def star(s):
    ang = BASE_ANG.copy()
    rad = np.ones(6)
    ang[1] = np.deg2rad(60.0 + 90.0 * s)
    rad[1] = 1.0 - 0.2 * s
    return ddg38.planar_star(ang, rad)


SS = np.linspace(0.0, 1.0, 181)
STATES = []
for s in SS:
    V, F = star(s)
    th = ddg38.projected_star_angles(V, F, 0, E3)
    assert abs(th.sum() - 2 * np.pi) < 1e-9
    ov = ddg38.max_star_overlap(V, F, 0)
    STATES.append((s, V, F, th, ov))
S_FOLD = 2.0 / 3.0
for s, V, F, th, ov in STATES:
    if s < S_FOLD - 1e-9:
        assert th[1] > 0 and ov < 1e-12
    elif s > S_FOLD + 1e-9:
        f023 = V[F[1]]
        a023 = 0.5 * abs(np.cross(f023[1, :2] - f023[0, :2], f023[2, :2] - f023[0, :2]))
        assert th[1] < 0 and ov > 0 and np.isclose(ov, a023, atol=1e-12)
        assert np.isclose(ddg38.planar_overlap(f023, V[F[2]]), a023, atol=1e-12)    # inside face (0, 3, 4)
Vf, Ff = ddg38.folded_star()
assert np.allclose(STATES[-1][1], Vf) and (STATES[-1][2] == Ff).all()
assert np.allclose(np.degrees(STATES[-1][3]), [150, -30, 60, 60, 60, 60])
assert ddg38.star_passes(STATES[0][1], STATES[0][2], 0, E3) and not ddg38.star_passes(Vf, Ff, 0, E3)
OV_END = STATES[-1][4]
assert round(OV_END, 3) == 0.200

# ---------------------------------------------------------------- layout
C0, SC = np.array([-3.65, -0.1, 0.0]), 2.3
RX = 0.75


def P(p):
    return C0 + SC * np.array([p[0], p[1], 0.0])


def draw(state, note=None, note_color=None):
    s, V, F, th, ov = state
    out = []
    # counterclockwise faces light blue and translucent; a clockwise face orange and on top, so that it reads orange
    for k in sorted(range(len(F)), key=lambda k: th[k] < -1e-9):
        cw = th[k] < -1e-9
        out.append(hp.poly_fill([P(V[i]) for i in F[k]], hp.C["accent"] if cw else hp.C["region"],
                                opacity=0.75 if cw else 0.4, stroke_width=2.5, stroke_color=hp.HEX["aux"]))
    for k in range(1, 7):
        out.append(hp.seg(P(V[0]), P(V[k]), color=hp.BLACK, width=2.5))
    for i in range(7):
        out.append(hp.dot(P(V[i]), r=0.06))
        d = V[i, :2] / (np.linalg.norm(V[i, :2]) + 1e-12)
        if i == 2:     # turn the label of the moving vertex clockwise while it nears the ray of vertex 3 (off dot 3)
            ph = np.degrees(np.arctan2(d[1], d[0]))
            r = np.radians(-50.0 * np.clip((150.0 - ph) / 30.0, 0.0, 1.0))
            d = np.array([np.cos(r) * d[0] - np.sin(r) * d[1], np.sin(r) * d[0] + np.cos(r) * d[1]])
        pos = P(V[i]) + (0.3 * np.array([d[0], d[1], 0]) if i else np.array([0.25, -0.25, 0]))
        out.append(hp.text(str(i), 30, color=hp.C["normal"] if i == 2 else hp.BLACK).move_to(pos))
    # readouts
    t2 = 0.0 if abs(th[1]) < 1e-9 else np.degrees(th[1])
    um = lambda x: f"{x:6.1f}".replace("-", "\u2212")  # noqa: E731  (Unicode minus, dganim note)
    rows = [(f"θ₁ = {um(np.degrees(th[0]))}°", hp.BLACK),
            (f"θ₂ = {um(t2)}°", hp.C["normal"] if t2 < 0 else hp.BLACK),
            ("θ₃ = … = θ₆ = 60.0°", hp.BLACK),
            (f"Σθ = {np.degrees(th.sum()):.1f}°", hp.C["tangent"]),
            (f"overlap area = {ov:.3f}", hp.C["normal"] if ov > 1e-9 else hp.BLACK)]
    for k, (txt, col) in enumerate(rows):
        out.append(hp.text(txt, 32, color=col).move_to([RX, 2.75 - 0.62 * k, 0], aligned_edge=hp.LEFT))
    if note:
        for k, line in enumerate(note):
            out.append(hp.text(line, 30, color=note_color or hp.BLACK).move_to([RX, -0.6 - 0.45 * k, 0],
                                                                               aligned_edge=hp.LEFT))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("a planar vertex star folds over; its angle sum stays 2π", 32).move_to([0, 3.55, 0])
        key = VGroup(hp.text("light blue: θₖ > 0", 30, color=hp.C["tangent"]),
                     hp.text("orange: θₖ < 0", 30, color=hp.C["accent"]))
        key.arrange(hp.DOWN, aligned_edge=hp.LEFT, buff=0.08).move_to([RX, -2.25, 0], aligned_edge=hp.LEFT)
        self.add(title, key)
        g = VGroup()
        self.add(g)
        S = hp.setter
        S(g, draw(STATES[0], ["an embedded star"]))(1)
        self.mark("start")
        self.wait(2.0)
        ev, t = [], 0.0
        k_fold = int(np.argmin(np.abs(SS - S_FOLD)))
        for k, st in enumerate(STATES):
            if k < k_fold:
                note, col = ["vertex 2 moves: embedded"], None
            elif k == k_fold:
                note, col = ["vertex 2 on the ray of 3:", "face (0,2,3) degenerate"], hp.C["aux"]
            else:
                note, col = ["face (0,2,3) now clockwise", "and inside face (0,3,4)"], hp.C["normal"]
            ev.append((t, t, (lambda stt: (lambda u: S(g, stt)(1)))(draw(st, note, col))))
            t += 1 / 30 if k != k_fold else 2.0
        self.play(hp.timeline(g, ev, t + 0.5))
        self.mark("fold done")
        end = draw(STATES[-1], ["Σθ = 2π is not enough;", "every θₖ must be > 0"],
                   hp.C["normal"])
        self.play(hp.timeline(g, [(0, 0, S(g, end))], 0.5))
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=12.0)
