"""Animation 39.5.1: a mesh rebuilt face by face from edge lengths and bending angles, and what one wrong angle does
(Section 39.5, Lemma 39.5.9, Theorem 39.5.10, Proposition 39.5.11, Figure 39.5.3).

Data as in Figure 39.5.3: icosphere(1) (42 vertices, 80 faces); edge lengths and bending angles from the mesh
(ddg39.lengths_angles); faces glued in depth-first order from face 0 (ddg39.reconstruct: the new face's normal is the
old one turned about the shared edge by theta, its third vertex is apex (39.5.4)).
Part 1: the 80 faces appear in that order (orange = the face just placed). The finished mesh equals the original up
to a proper rigid motion (error < 1e-12).
Part 2: the bending angle of the tree edge that placed face number 60 is increased by delta = 0 ... 0.3. Light blue:
the 20 faces placed after it. A vertex is placed once, so only the 3 vertices first placed after it move, and with
them the 12 faces that contain one (not a rigid block). Thick orange: edges whose rebuilt length differs from the
given one by more than 1e-6. At delta = 0.3: 14 such edges, 3 vertices moved, the changed edge keeps its length (as
Figure 39.5.3(b)); (39.5.5) fails at the two ends of the changed edge.
Drawn with an orthographic camera (elev 20, azim -50, as the figure), aligned on face 0.
Self-check: the rebuild of the unchanged data has rigid error < 1e-12 and uses a spanning tree of 79 edges; the
delta = 0.3 rebuild has 14 wrong edges, 3 moved vertices and the changed edge not among the wrong ones; the vertex
compatibility product (39.5.5) is the identity for the true data at every vertex.

Render: bash tools/render_anims.sh chapters/39-surface-curvature/anim/anim-39-5-1-rebuild-from-angles.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lib"))
import _helpers as hp  # noqa: E402
import ddg39  # noqa: E402
import dganim as da  # noqa: E402
import dgmesh as dm  # noqa: E402
import numpy as np  # noqa: E402
from manim import VGroup  # noqa: E402

# ---------------------------------------------------------------- data
V, F = dm.icosphere(1)
ELL, THETA = ddg39.lengths_angles(V, F)
Y, SEQ, TREE = ddg39.reconstruct(F, ELL, THETA)
assert ddg39.rigid_error(V, Y) < 1e-12 and len(SEQ) == len(F) and len(TREE) == len(F) - 1
for i in range(len(V)):
    assert np.allclose(ddg39.vertex_holonomy(F, ELL, THETA, i), np.eye(3), atol=1e-9)


def frame(P, f):
    e = P[f[1]] - P[f[0]]
    e /= np.linalg.norm(e)
    n = np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]])
    n /= np.linalg.norm(n)
    return np.column_stack([e, np.cross(n, e), n])


def align(P):
    """move P so that face 0 coincides with face 0 of the original mesh."""
    R = frame(V, F[0]) @ frame(P, F[0]).T
    return (P - P[F[0][0]]) @ R.T + V[F[0][0]]


YA = align(Y)
assert np.abs(YA - V).max() < 1e-10
NPLACE = 60
F_list = F.tolist()
f_new = SEQ[NPLACE]
placed_before = set(SEQ[:NPLACE])
EDGE = None
for e in TREE:
    if set(e) <= set(F_list[f_new]):
        others = [g for g, f in enumerate(F_list) if g != f_new and set(e) <= set(f)]
        if others and others[0] in placed_before:
            EDGE = e
assert EDGE is not None
LATE = set(SEQ[NPLACE:])


def perturbed(delta):
    th = dict(THETA)
    th[EDGE] += delta
    Y2, seq2, _ = ddg39.reconstruct(F, ELL, th)
    assert seq2 == SEQ
    Y2 = align(Y2)
    errs = {(i, j): abs(np.linalg.norm(Y2[j] - Y2[i]) - l) for (i, j), l in ELL.items()}
    return Y2, errs


DELTAS = np.linspace(0, 0.3, 46)
PERT = [perturbed(d) for d in DELTAS]
Yend, errs_end = PERT[-1]
bad_end = [e for e, v in errs_end.items() if v > 1e-6]
moved_end = int((np.linalg.norm(Yend - V, axis=1) > 1e-6).sum())
assert len(bad_end) == 14 and moved_end == 3 and EDGE not in bad_end
MOVED = set(np.nonzero(np.linalg.norm(Yend - V, axis=1) > 1e-6)[0].tolist())
assert sum(1 for f in F_list if MOVED & set(f)) == 12 and all(g in LATE for g, f in enumerate(F_list) if MOVED & set(f))
TH_END = dict(THETA)
TH_END[EDGE] += DELTAS[-1]
for i in EDGE:                                          # Proposition 39.5.11: no mesh has the perturbed data
    assert np.abs(ddg39.vertex_holonomy(F, ELL, TH_END, i) - np.eye(3)).max() > 0.1
MAXERR_END = max(errs_end.values())

# ---------------------------------------------------------------- drawing
CAM = hp.Cam(elev=20, azim=-50, scale=2.6, origin=(-3.3, -0.25))
GREY, BLUE, ORANGE = hp.rgb(hp.HEX["surface"]), hp.rgb(hp.HEX["region"]), hp.rgb(hp.HEX["accent"])
RX = 0.9


def mesh_state(P, faces, newest=None, late=False, bad=()):
    hidden = np.ones(len(F), bool)
    hidden[list(faces)] = False
    cols = np.tile(GREY, (len(F), 1))
    if late:
        cols[list(LATE)] = BLUE
    if newest is not None:
        cols[newest] = ORANGE
    extras = []
    for (i, j) in bad:
        a, b = P[i], P[j]
        extras.append((CAM.depth(0.5 * (a + b)) + 1e-3, hp.seg(CAM(a), CAM(b), color=hp.C["accent"], width=7)))
    return hp.mesh_mobs(P, F, CAM, colors=cols, edge_color="#505050", edge_width=1.0, ambient=0.68, hidden=hidden,
                        extras=extras)


def text_rows(rows):
    out = []
    for k, (txt, col) in enumerate(rows):
        out.append(hp.text(txt, 30, color=col).move_to([RX, 2.35 - 0.5 * k, 0], aligned_edge=hp.LEFT))
    return out


class Anim(hp.Scene2D):
    def construct(self):
        title = hp.text("a mesh from edge lengths and bending angles", 32).move_to([0, 3.55, 0])
        sub = hp.text("icosphere(1), 80 faces", 30, color=hp.HEX["aux"]).move_to([RX, 2.95, 0],
                                                                                          aligned_edge=hp.LEFT)
        self.add(title, sub)
        g = VGroup()
        self.add(g)
        S = hp.setter

        def at(fn):
            return lambda u: S(g, fn())(1)

        S(g, mesh_state(YA, [SEQ[0]], SEQ[0]) + text_rows([("face 1 of 80", hp.BLACK),
                                                          ("face 0 put in the xy-plane", hp.HEX["aux"])]))(1)
        self.mark("start")
        self.wait(1.5)
        ev, t = [], 0.0
        for k in range(1, len(SEQ)):
            rows = [(f"face {k + 1} of 80", hp.BLACK), ("normal: turn by θ", hp.HEX["aux"]),
                    ("vertex: from 3 lengths", hp.HEX["aux"])]
            ev.append((t, t, at((lambda kk, rr: (lambda: mesh_state(YA, SEQ[:kk + 1], SEQ[kk]) + text_rows(rr)))(k, rows))))
            t += 0.12
        done = [("all 80 faces placed", hp.BLACK), ("= the original mesh", hp.C["third"]),
                ("(rigid error < 10⁻¹²)", hp.C["third"])]
        ev.append((t, t, at(lambda: mesh_state(YA, SEQ) + text_rows(done))))
        self.play(hp.timeline(g, ev, t + 2.0))
        self.mark("built")
        # part 2: one bending angle off by delta
        ev, t = [], 0.0
        for k, d in enumerate(DELTAS):
            P, errs = PERT[k]
            bad = [e for e, v in errs.items() if v > 1e-6]
            rows = [(f"one θ changed by δ = {d:.2f}", hp.C["normal"]),
                    (f"wrong edge lengths: {len(bad)}", hp.C["accent"]),
                    (f"largest error: {max(errs.values()):.3f}", hp.C["accent"])]
            ev.append((t, t, at((lambda P_, b_, r_: (lambda: mesh_state(P_, SEQ, late=True, bad=b_) + text_rows(r_)))(
                P, bad, rows))))
            t += 1 / 15
        self.play(hp.timeline(g, ev, t + 0.5))
        self.mark("perturbed")
        end = [(f"one θ changed by δ = 0.30", hp.C["normal"]),
               (f"wrong edge lengths: {len(bad_end)}", hp.C["accent"]),
               (f"largest error: {MAXERR_END:.3f}", hp.C["accent"]),
               ("no mesh has these (ℓ, θ):", hp.BLACK), ("(39.5.5) fails", hp.BLACK)]
        self.play(hp.timeline(g, [(0, 0, at(lambda: mesh_state(Yend, SEQ, late=True, bad=bad_end) + text_rows(end)))],
                              0.3))
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=17.5)
