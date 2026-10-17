"""Animation 31.2.1: the same surface triangulated differently, three vertex normals (Section 31.2, Example 31.2.2,
Proposition 31.2.3).

The one-ring of Figure 31.2.1(a): the vertex at the north pole of the unit sphere and 7 neighbours on the sphere at
longitudes (0, 25, 50, 75, 160, 235, 300) degrees and colatitudes (0.40, 0.30, 0.38, 0.30, 0.62, 0.50, 0.55), seen
from above. Arrows: horizontal components of the uniform (blue), area-weighted (green) and angle-weighted (orange)
vertex normals, length proportional to the tilt from (0, 0, 1) (1 degree = 0.065 world units, as in the figure).
Then the same piecewise-flat surface is re-triangulated in the two ways of Example 31.2.2 (and of
verify/v-31-2-discrete-curvature.py):
  (1) split the face (0, 5, 6) through the vertex, at the midpoint of the opposite edge: the uniform normal moves
      2.61 degrees, the other two do not move;
  (2) cut the corner: new vertices halfway along the edges 0-5 and 0-6, every face containing those edges split:
      the area-weighted normal moves 5.91 degrees, the other two do not move.
All values are computed with dgmesh.vertex_normals.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-2-1-normal-retriangulation.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (LEFT, RIGHT, Create, FadeIn, FadeOut, Polygon, Transform, VGroup)  # noqa: E402

# ---------------------------------------------------------------- data (identical to the verify script)
lon = np.radians([0, 25, 50, 75, 160, 235, 300])
col = np.array([0.40, 0.30, 0.38, 0.30, 0.62, 0.50, 0.55])
P = np.stack([np.sin(col) * np.cos(lon), np.sin(col) * np.sin(lon), np.cos(col)], 1)
V = np.vstack([[0, 0, 1.0], P])
F = np.array([[0, 1 + k, 1 + (k + 1) % 7] for k in range(7)])
f = 4
a, b = F[f, 1], F[f, 2]
assert (a, b) == (5, 6)
m_ = (V[a] + V[b]) / 2
V2 = np.vstack([V, m_])
F2 = np.vstack([np.delete(F, f, 0), [[0, a, 8], [0, 8, b]]])
p_, q_ = V[0] + 0.5 * (V[a] - V[0]), V[0] + 0.5 * (V[b] - V[0])
V3 = np.vstack([V, p_, q_])
ip, iq = 8, 9
F3 = []
NEW3 = []                        # new edges of the corner cut
for tri in F:
    t = list(tri)
    if set(t) == {0, a, b}:
        F3 += [[0, ip, iq], [ip, a, b], [ip, b, iq]]
        NEW3 += [(ip, iq), (ip, b)]
    elif 0 in t and a in t:
        o = [x for x in t if x not in (0, a)][0]
        F3 += [[0, o, ip], [ip, o, a]]
        NEW3 += [(ip, o)]
    elif 0 in t and b in t:
        o = [x for x in t if x not in (0, b)][0]
        F3 += [[0, iq, o], [iq, b, o]]
        NEW3 += [(iq, o)]
    else:
        F3.append(t)
F3 = np.array(F3)
KINDS = [("uniform", "uniform", "tangent"), ("area", "area", "third"), ("angle", "angle", "accent")]


def normals(VV, FF):
    return {k: dm.vertex_normals(VV, FF, k)[0] for k, _, _ in KINDS}


def angle(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v)))


N0, N2, N3 = normals(V, F), normals(V2, F2), normals(V3, F3)
TILT = {k: angle(N0[k], [0, 0, 1]) for k in N0}
MOVE2 = {k: angle(N0[k], N2[k]) for k in N0}
MOVE3 = {k: angle(N0[k], N3[k]) for k in N0}
assert np.allclose([TILT["uniform"], TILT["area"], TILT["angle"]], [1.239, 6.501, 4.222], atol=2e-3)
assert round(MOVE2["uniform"], 2) == 2.61 and MOVE2["area"] < 1e-9 and MOVE2["angle"] < 1e-9
assert round(MOVE3["area"], 2) == 5.91 and MOVE3["uniform"] < 1e-9 and MOVE3["angle"] < 1e-9
assert np.isclose(dm.face_areas(V3, F3).sum(), dm.face_areas(V, F).sum())    # same surface
assert np.isclose(dm.face_areas(V2, F2).sum(), dm.face_areas(V, F).sum())
SCALE = 0.065                     # world units per degree of tilt (Figure 31.2.1(a))

cam = hc.Cam(elev=90, azim=-90, scale=5.6, origin=(-3.7, -0.4))


def tip(N):
    """World point of the arrow tip for a normal N: horizontal direction, length = tilt * SCALE."""
    h = np.array([N[0], N[1], 0.0])
    return h / np.linalg.norm(h) * angle(N, [0, 0, 1]) * SCALE


class Anim(hc.Scene2D):
    def construct(self):
        def faces(VV, FF, hl=()):
            g = VGroup()
            for k, t in enumerate(FF):
                g.add(Polygon(*cam(VV[t]), stroke_color=hc.BLACK, stroke_width=1.6,
                              fill_color=hc.C["accent"] if k in hl else hc.C["surface"],
                              fill_opacity=0.45 if k in hl else 0.75))
            return g

        def arrows(NN):
            g = VGroup()
            for k, _, c in KINDS:
                g.add(hc.arrow(cam([0, 0, 0]), cam(tip(NN[k])), color=hc.C[c], width=7, tip=0.3))
            return g

        title = hc.text("same surface, different triangles", size=36).move_to([0, 3.45, 0])
        mesh = faces(V, F)
        arr = arrows(N0).set_z_index(4)
        self.add(title, mesh, hc.dot(cam([0, 0, 0]), r=0.08).set_z_index(5), arr)

        # table: rows = normals, columns = tilt, change by split, change by cut
        X = [0.45, 3.2, 4.7, 6.2]
        Y0, DY = 2.15, 0.75
        hdr = VGroup(hc.text("tilt", size=30).move_to([X[1], Y0 + 0.75, 0]),
                     hc.text("split", size=30).move_to([X[2], Y0 + 0.75, 0]),
                     hc.text("cut", size=30).move_to([X[3], Y0 + 0.75, 0]))
        rows = VGroup()
        for r, (k, name, c) in enumerate(KINDS):
            rows.add(hc.text(name, size=32, color=hc.C[c]).move_to([X[0], Y0 - r * DY, 0], aligned_edge=LEFT))
            rows.add(hc.text(f"{TILT[k]:.2f}°", size=32).move_to([X[1], Y0 - r * DY, 0]))
        self.add(rows, hdr[0])
        sub = hc.text("tilt: angle from (0, 0, 1)", size=30).move_to([3.45, -0.15, 0])
        self.add(sub)
        self.wait(3.0)

        def column(moves, j):
            g = VGroup()
            for r, (k, _, c) in enumerate(KINDS):
                v = moves[k]
                g.add(hc.text(f"{v:.2f}°", size=32, color=hc.C["normal"] if v > 0.1 else hc.C["aux"])
                      .move_to([X[j], Y0 - r * DY, 0]))
            return g

        # ---- (1) split the face (0, 5, 6) through the vertex
        hl = faces(V, F, hl=(f,))
        s1 = hc.text("(1) split a face at the vertex", size=30).move_to([3.45, -1.0, 0])
        self.play(Transform(mesh, hl), FadeIn(s1), run_time=1.0)
        new_edge = hc.seg(cam(V[0]), cam(m_), color=hc.BLACK, width=4).set_z_index(3)
        ghosts = arrows(N0).set_opacity(0.25).set_z_index(3)
        self.add(ghosts)
        arr2 = arrows(N2).set_z_index(4)
        self.play(Create(new_edge), FadeIn(hdr[1]), run_time=1.0)
        self.play(Transform(arr, arr2), FadeIn(column(MOVE2, 2)), run_time=1.5)
        r1 = hc.text("only the uniform normal moves", size=30, color=hc.C["tangent"]).move_to([3.45, -1.6, 0])
        self.play(FadeIn(r1), run_time=0.6)
        self.wait(2.6)

        # ---- undo
        self.play(FadeOut(new_edge), Transform(mesh, faces(V, F)), Transform(arr, arrows(N0).set_z_index(4)),
                  FadeOut(s1), FadeOut(r1), run_time=1.0)

        # ---- (2) cut the corner
        s2 = hc.text("(2) cut the corner off", size=30).move_to([3.45, -1.0, 0])
        hl3 = faces(V3, F3, hl=tuple(k for k, t in enumerate(F3) if 0 in t))
        cut_edges = VGroup(*[hc.seg(cam(V3[i]), cam(V3[j]), color=hc.BLACK, width=4) for i, j in NEW3]).set_z_index(3)
        pq = VGroup(hc.dot(cam(p_), r=0.06), hc.dot(cam(q_), r=0.06)).set_z_index(5)
        self.play(FadeIn(s2), FadeIn(pq), Create(cut_edges), FadeIn(hdr[2]), run_time=1.4)
        self.play(Transform(mesh, hl3), run_time=0.8)
        self.play(Transform(arr, arrows(N3).set_z_index(4)), FadeIn(column(MOVE3, 3)), run_time=1.5)
        r2 = hc.text("only the area normal moves", size=30, color=hc.C["third"]).move_to([3.45, -1.6, 0])
        self.play(FadeIn(r2), run_time=0.6)
        self.wait(2.0)
        fin = hc.text("angle-weighted: unchanged", size=32, color=hc.C["accent"]).move_to([3.45, -2.35, 0])
        fin2 = hc.text("it sees only the surface", size=30, color=hc.C["accent"]).move_to([3.45, -2.85, 0])  # above the control bar
        self.play(FadeIn(fin), FadeIn(fin2), run_time=0.8)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=19.5)
