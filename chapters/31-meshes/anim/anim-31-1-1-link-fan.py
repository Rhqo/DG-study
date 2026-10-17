"""Animation 31.1.1: building the star of a vertex face by face (Section 31.1, Definition 31.1.3).

The vertex i and its six neighbours are those of Figure 31.1.1(a): i = (0, 0, 0.35), neighbours on the unit
circle at angles 2 pi k / 6 with heights 0.12 (1, -1, 0.5, -0.4, 0.8, -0.2). Faces (i, k, k+1) are added one at
a time. While the fan is open, the link of i (orange) is one path, so i is a boundary vertex; when the sixth face
closes the fan, the link becomes one closed loop and i is an interior vertex. Then:
  (1) a seventh face (i, 1, f) on the edge e = i1 with f = (0.5, 0, 0.95): e has three faces (non-manifold edge);
  (2) that face is removed and a second closed fan of five faces is attached at i only (an upward cone, ring of
      radius 0.75 at height 1.2): every edge has at most two faces, but the link of i is two loops
      (non-manifold vertex), as in Figure 31.1.1(c)(d).
Every status line is computed with dgmesh.link_type / manifold_report.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-1-1-link-fan.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (LEFT, UP, Create, FadeIn, FadeOut, UpdateFromAlphaFunc,  # noqa: E402
                   VGroup)

# ---------------------------------------------------------------- data (Figure 31.1.1(a), (c), (d))
ang = 2 * np.pi * np.arange(6) / 6
ring6 = np.stack([np.cos(ang), np.sin(ang), 0.12 * np.array([1, -1, 0.5, -0.4, 0.8, -0.2])], 1)
fin = np.array([[0.5, 0.0, 0.95]])                                   # vertex 7
a5 = 0.3 + 2 * np.pi * np.arange(5) / 5
ring5 = np.stack([0.75 * np.cos(a5), 0.75 * np.sin(a5), np.full(5, 1.2)], 1)   # vertices 8..12
V = np.vstack([[0, 0, 0.35], ring6, fin, ring5])
FAN = [[0, 1 + k, 1 + (k + 1) % 6] for k in range(6)]
FIN = [[0, 1, 7]]
TOP = [[0, 8 + (k + 1) % 5, 8 + k] for k in range(5)]
F = np.array(FAN + FIN + TOP)

# the status after each stage, computed (not typed in)
status = []
for k in range(1, 7):
    lt = dm.link_type(dm.vertex_link(F[:k], 0))
    status.append(lt)
    assert lt == ("path" if k < 6 else "cycle")
    if k < 6:
        assert sorted(dm.boundary_vertices(F[:k]))[0] == 0          # i is on the boundary
rep_fin = dm.manifold_report(F[:7])
assert rep_fin["nonmanifold_edges"] == [(0, 1)]
assert sorted(len(fs) for fs in dm.edge_faces(F[:7]).values())[-1] == 3
EDGE_LINK = sorted({x for f in F[:7] for x in f if 0 in f and 1 in f} - {0, 1})
assert EDGE_LINK == [2, 6, 7]
F_two = np.array(FAN + TOP)
assert dm.link_type(dm.vertex_link(F_two, 0)) == "other"
rep_two = dm.manifold_report(F_two)
assert rep_two["nonmanifold_vertices"] == [0] and rep_two["nonmanifold_edges"] == []
assert len(dm.components(F_two)) == 2          # two fans, joined only through the vertex i
# the link of i in the two-fan mesh: two separate cycles (hexagon and pentagon)
link_two = dm.vertex_link(F_two, 0)
assert len(link_two) == 11

cam = hc.Cam(elev=28, azim=-62, scale=2.9, origin=(-3.4, -1.25))


class Anim(hc.Scene2D):
    def construct(self):
        mv = hc.MeshView(V, F, cam, edge_width=1.2, ambient=0.55)
        mv.hidden[:] = True
        mv.face_opacity = np.ones(len(F)) * 0.95
        mv.set_mesh()
        self.add(mv)

        def face_depth(f):
            return cam.depth(V[F[f]].mean(0))

        def link_line(f):
            """Orange link edge opposite i in face f, drawn just in front of its own face."""
            a, b = F[f][1], F[f][2]
            ln = hc.seg(cam(V[a]), cam(V[b]), color=hc.C["accent"], width=9)
            mv.extras.append([ln, face_depth(f) + 1e-3])
            mv.set_mesh()
            return ln

        def set_opacity(fs, op):
            def upd(m, a):
                for f in fs:
                    m.hidden[f] = False
                    m.face_opacity[f] = 0.95 * (a if op else 1 - a)
                    if not op and a >= 1:
                        m.hidden[f] = True
                m.set_mesh()
            return upd

        X0 = 0.55
        iv = hc.dot(cam(V[0]), r=0.08).set_z_index(5)
        ilab = hc.math(r"$i$", size=40).next_to(iv, UP + LEFT, buff=0.08).set_z_index(5)
        title = hc.text("Faces around one vertex", size=40).to_corner(UP + LEFT, buff=0.4)
        self.add(title, iv, ilab)

        def line_at(txt, y, color=hc.BLACK, size=34):
            return hc.text(txt, size=size, color=color).move_to([X0, y, 0], aligned_edge=LEFT)

        nface = line_at("faces at i: 0", 2.2)
        self.add(nface)
        self.wait(1.5)

        st_link = st_kind = None
        dashed = []
        for k in range(6):
            ln = link_line(k)
            if status[k] == "path":
                nl = line_at("link: one path", 1.3, hc.C["accent"])
                nk = line_at("boundary vertex", 0.55, hc.C["tangent"])
            else:
                nl = line_at("link: one closed loop", 1.3, hc.C["accent"])
                nk = line_at("interior vertex (disk)", 0.55, hc.C["third"])
            new_n = line_at(f"faces at i: {k + 1}", 2.2)
            # the two boundary edges at i (blue dashed) while the fan is open
            new_dashed = [] if k == 5 else [hc.seg(cam(V[0]), cam(V[v]), color=hc.C["tangent"], width=7,
                                                   dashed=True).set_z_index(3) for v in (1, k + 2)]
            anims = [UpdateFromAlphaFunc(mv, set_opacity([k], True)), Create(ln), *hc.swap(nface, new_n)]
            anims += [FadeOut(d) for d in dashed] + [FadeIn(d) for d in new_dashed]
            if st_link is None:
                anims += [FadeIn(nl), FadeIn(nk)]
            else:
                anims += [*hc.swap(st_link, nl), *hc.swap(st_kind, nk)]
            self.play(*anims, run_time=1.0 if k < 5 else 1.4)
            nface, st_link, st_kind, dashed = new_n, nl, nk, new_dashed
            self.wait(0.7 if k < 5 else 2.6)

        # ---- (1) a third face on the edge e = i1
        e_line = hc.seg(cam(V[0]), cam(V[1]), color=hc.BLACK, width=10).set_z_index(4)
        e_lab = hc.math(r"$e$", size=40).move_to(cam(0.5 * (V[0] + V[1])) + np.array([0.2, -0.4, 0])).set_z_index(5)
        pts = VGroup(*[hc.dot(cam(V[v]), color=hc.C["accent"], r=0.11).set_z_index(6) for v in EDGE_LINK])
        nl = line_at("edge e: 3 faces", 1.3, hc.BLACK)
        nk = line_at("non-manifold edge", 0.55, hc.C["normal"])
        new_n = line_at("faces: 7", 2.2)
        self.play(UpdateFromAlphaFunc(mv, set_opacity([6], True)), Create(e_line), FadeIn(e_lab),
                  *hc.swap(st_link, nl), *hc.swap(st_kind, nk),
                  *hc.swap(nface, new_n), run_time=1.3)
        nface, st_link, st_kind = new_n, nl, nk
        note = line_at("link of e: 3 points", -0.2, hc.C["accent"], size=30)
        self.play(FadeIn(pts), FadeIn(note), run_time=0.7)
        self.wait(2.6)

        # ---- (2) remove it; attach a second fan only at i
        self.play(UpdateFromAlphaFunc(mv, set_opacity([6], False)), FadeOut(e_line), FadeOut(e_lab), FadeOut(pts),
                  FadeOut(note), run_time=0.8)
        top_lines = [link_line(f) for f in range(7, 12)]
        nl = line_at("link of i: two loops", 1.3, hc.C["accent"])
        nk = line_at("non-manifold vertex", 0.55, hc.C["normal"])
        new_n = line_at("faces at i: 11", 2.2)
        self.play(UpdateFromAlphaFunc(mv, set_opacity(range(7, 12), True)), *[Create(ln) for ln in top_lines],
                  *hc.swap(st_link, nl), *hc.swap(st_kind, nk),
                  *hc.swap(nface, new_n), run_time=2.0)
        note2 = line_at("every edge: at most 2 faces", -0.2, size=30)
        note3 = line_at("near i: two disks touching", -0.85, size=30)
        self.play(FadeIn(note2), run_time=0.6)
        self.play(FadeIn(note3), run_time=0.6)
        self.wait(3.0)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=13.0)
