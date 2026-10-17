"""Animation 31.1.2: propagating an orientation face by face (Section 31.1, Definition 31.1.5, Example 31.1.6).

Two strips with the same combinatorics along the centre circle (radius 1, n = 24 quads = 48 triangles, width 1):
(a) a cylinder band (x, y, z) = (cos t, sin t, s), s in [-0.5, 0.5], built like dgmesh.mobius_mesh but glued
    without the twist; (b) the Moebius strip dgmesh.mobius_mesh(n=24, w=0.5) of Figure 31.1.2(b).
Starting from the first triangle, the orientation is copied to the next triangle so that their shared edge is
traversed in opposite directions (Definition 31.1.5); the vermillion arrows are the resulting face normals.
After one turn the last triangle meets the first one across the seam: on the band the seam edge is traversed in
opposite directions (consistent, the normals agree), on the Moebius strip in the same direction (conflict, the
normals are opposite). dgmesh.orient confirms: the band is orientable, the Moebius strip is not.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-1-2-orientation-propagation.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (Create, FadeIn, FadeOut, GrowArrow, VGroup, linear)  # noqa: E402

N_QUADS, W = 24, 0.5


def band_mesh(n=N_QUADS, w=W, R=1.0):
    """Cylinder band with the vertex numbering and face order of dgmesh.mobius_mesh, glued without a twist."""
    V = []
    for side in (-1, 1):
        for k in range(n):
            t = 2 * np.pi * k / n
            V.append([R * np.cos(t), R * np.sin(t), side * w])
    F = []
    for k in range(n):
        a, b = k, n + k
        c, d = (k + 1) % n, n + (k + 1) % n
        F += [[a, c, b], [b, c, d]]
    return np.array(V), np.array(F, int)


def directed(f):
    return {(f[0], f[1]), (f[1], f[2]), (f[2], f[0])}


def propagate(F):
    """Copy the orientation of each triangle to the next one along the strip (consecutive faces share an edge)."""
    Fp = [F[0].copy()]
    for f in F[1:]:
        prev = Fp[-1]
        shared = [(a, b) for (a, b) in directed(f) if (a, b) in directed(prev) or (b, a) in directed(prev)]
        assert len(shared) == 1
        Fp.append(f[::-1].copy() if shared[0] in directed(prev) else f.copy())
    Fp = np.array(Fp)
    seam = [(a, b) for (a, b) in directed(Fp[-1]) if (a, b) in directed(Fp[0]) or (b, a) in directed(Fp[0])]
    assert len(seam) == 1
    consistent = seam[0] not in directed(Fp[0])          # opposite traversal = consistent
    return Fp, seam[0], consistent


STRIPS = {}
for key, (V, F) in (("band", band_mesh()), ("mobius", dm.mobius_mesh(n=N_QUADS, w=W))):
    Fp, seam, ok = propagate(F)
    N = dm.face_normals(V, Fp)
    STRIPS[key] = dict(V=V, F=Fp, seam=seam, ok=ok, N=N, c=V[Fp].mean(1))
    assert dm.is_manifold(F) and len(F) == 2 * N_QUADS
    assert dm.orient(F)[1] == ok                                  # global check agrees with the walk
    assert (np.dot(N[0], N[-1]) > 0.5) == ok                      # last normal vs first normal across the seam
assert STRIPS["band"]["ok"] and not STRIPS["mobius"]["ok"]
assert dm.euler_characteristic(STRIPS["mobius"]["F"]) == 0 and len(dm.boundary_edges(STRIPS["mobius"]["F"])) == 2 * N_QUADS
# the Moebius boundary is one closed polygon of 2n edges; the band has two loops of n edges
ARROW = 0.42


KEEP = set(range(0, 2 * N_QUADS - 4, 4)) | {2 * N_QUADS - 1}     # arrows left behind (as in Figure 31.1.2(b))
CAMS = {"band": dict(elev=14, azim=20, scale=2.3, origin=(-3.55, 0.35)),
        "mobius": dict(elev=14, azim=20, scale=2.3, origin=(3.45, 0.35))}
for key, S in STRIPS.items():                       # layout check: the strips stay inside their half of the frame
    cam = hc.Cam(**CAMS[key])
    P = cam(np.vstack([S["V"], S["c"] + ARROW * S["N"]]))
    assert P[:, 1].min() > -2.05 and P[:, 1].max() < 2.75, (key, P[:, 1].min(), P[:, 1].max())
    assert abs(P[:, 0] - cam.origin[0]).max() < 3.4


class Anim(hc.Scene2D):
    def construct(self):
        cams = {k: hc.Cam(**v) for k, v in CAMS.items()}
        titles = {"band": "(a) band", "mobius": "(b) Möbius strip"}
        views, arrows = {}, {}
        for key, S in STRIPS.items():
            cam = cams[key]
            mv = hc.MeshView(S["V"], S["F"], cam, edge_width=0.9, ambient=0.6, opacity=0.82)
            mv.colors = np.tile(hc.rgb(hc.HEX["surface"]), (len(S["F"]), 1))
            mv.set_mesh()
            views[key] = mv
            self.add(mv)
            self.add(hc.text(titles[key], size=36).move_to([cam.origin[0], 3.35, 0]))
            arrows[key] = []
            for k in range(len(S["F"])):
                p0, p1 = S["c"][k], S["c"][k] + ARROW * S["N"][k]
                bold = k in (0, 2 * N_QUADS - 1)
                arr = hc.arrow(cam(p0), cam(p1), color=hc.C["normal"], width=7 if bold else 4, tip=0.2 if bold else 0.15)
                arrows[key].append((arr, cam.depth(0.5 * (p0 + p1))))
        head = hc.text("copy the orientation to the next triangle", size=32).move_to([0, -2.75, 0])
        self.add(head)
        self.wait(1.5)

        blue = hc.rgb(hc.HEX["region"]) * 0.75 + 0.25
        nface = 2 * N_QUADS

        def reach(key, k):
            mv = views[key]
            mv.colors[k] = blue
            arr, d = arrows[key][k]
            mv.extras.append([arr, d])
            mv.set_mesh()
            return arr

        def drop(key, k):
            mv = views[key]
            arr = arrows[key][k][0]
            mv.extras = [e for e in mv.extras if e[0] is not arr]
            mv.set_mesh()

        def side_label(txt, key, k, dx):
            arr = arrows[key][k][0]
            tip = arr.get_end()
            lab = hc.text(txt, size=30, color=hc.C["normal"])
            pos = tip + np.array([dx, 0, 0])
            lab.move_to(pos + np.array([np.sign(dx) * lab.width / 2, 0, 0]))
            lead = hc.seg(tip, pos, color=hc.C["normal"], width=2)
            return VGroup(lab, lead).set_z_index(6)

        firsts = [reach(key, 0) for key in STRIPS]
        labs = [side_label("start", key, 0, -1.1) for key in STRIPS]
        self.play(*[GrowArrow(a) for a in firsts], *[FadeIn(l) for l in labs], run_time=1.0)
        self.wait(1.0)
        # walk along both strips at the same time; normals of every 4th triangle stay
        per = 0.26
        for k in range(1, nface):
            new = [reach(key, k) for key in STRIPS]
            self.play(*[GrowArrow(a) for a in new], run_time=per, rate_func=linear)
            if k not in KEEP:
                for key in STRIPS:
                    drop(key, k)
        ends = [side_label("end", key, nface - 1, 1.1) for key in STRIPS]
        self.play(*[FadeIn(l) for l in ends], FadeOut(head), run_time=0.6)
        self.wait(1.0)

        # seam check
        res = []
        for key, S in STRIPS.items():
            cam = cams[key]
            a, b = S["seam"]
            col = hc.C["third"] if S["ok"] else hc.C["normal"]
            ln = hc.seg(cam(S["V"][a]), cam(S["V"][b]), color=col, width=12).set_z_index(5)
            txt = ["seam: opposite directions", "consistent"] if S["ok"] else \
                ["seam: same direction", "conflict: not orientable"]
            t1 = hc.text(txt[0], size=30).move_to([cam.origin[0], -2.15, 0])
            t2 = hc.text(txt[1], size=34, color=col).move_to([cam.origin[0], -2.72, 0])     # above the paused control bar
            res.append((ln, t1, t2))
        self.play(*[Create(r[0]) for r in res], run_time=1.0)
        self.play(*[FadeIn(r[1]) for r in res], run_time=0.8)
        self.play(*[FadeIn(r[2]) for r in res], run_time=0.8)
        self.wait(3.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=22.0)
