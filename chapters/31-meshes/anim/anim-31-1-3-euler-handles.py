"""Animation 31.1.3: adding handles to a closed mesh and counting n0 - n1 + n2 (Section 31.1, (31.1.2), (31.1.3)).

Start: the icosphere of level 3 (642 vertices, 1920 edges, 1280 faces, chi = 2). A handle is attached in steps that
keep the mesh a manifold mesh at every moment:
  1. remove the 24 faces touching the closed one-ring of a valence-6 vertex (a disk: 7 vertices, 30 edges and
     24 faces disappear): a 12-gon hole, chi drops by 1;
  2. remove a second disk: chi = 0, two boundary loops;
  3. build a tube between the two 12-gons ring by ring (each new ring: +12 vertices, +36 edges, +24 faces, so chi
     does not change) and close it on the second 12-gon (+24 edges, +24 faces): the boundary loops disappear.
A second handle is added the same way. The table shows n0, n1, n2, chi, the number b of boundary loops and
g = (2 - b - chi)/2 from (31.1.3), all counted from the current face list (dgmesh.counts, boundary loops).
Every intermediate mesh is checked: manifold, consistently oriented, one component.
Faces turned away from the (fixed) camera are drawn darker, so that the inside of the sphere seen through a hole
shows the hole.

Render: ``bash tools/render_anims.sh chapters/31-meshes/anim/anim-31-1-3-euler-handles.py``
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import _helpers_c as hc  # noqa: E402
import dgmesh as dm  # noqa: E402

import dganim as da  # noqa: E402
import numpy as np  # noqa: E402
from manim import (LEFT, RIGHT, UpdateFromAlphaFunc, VGroup)  # noqa: E402

CAM = dict(elev=18, azim=-60, scale=1.95, origin=(-3.0, -0.1))
V0, F0 = dm.icosphere(3)
_cam = hc.Cam(**CAM)
assert dm.counts(F0) == (642, 1920, 1280)


def boundary_loops(F):
    """Number of closed polygons formed by the boundary edges (each boundary vertex has two boundary edges)."""
    adj = {}
    for a, b in dm.boundary_edges(F):
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    seen, loops = set(), 0
    for s in adj:
        if s in seen:
            continue
        loops += 1
        stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack += adj[x]
    return loops


def disk_faces(h):
    """Faces touching the closed one-ring of vertex h (a disk of 24 triangles around a valence-6 vertex)."""
    ring = set(dm.neighbors(F0)[h]) | {h}
    return [tuple(f) for f in F0 if ring & set(f)]


def disk_loop(faces):
    """Boundary of a set of faces as one ordered loop, in the direction the faces traverse it."""
    d = {(f[k], f[(k + 1) % 3]) for f in faces for k in range(3)}
    nxt = {a: b for (a, b) in d if (b, a) not in d}
    loop = [min(nxt)]
    while len(loop) < len(nxt):
        loop.append(nxt[loop[-1]])
    assert nxt[loop[-1]] == loop[0]
    return loop


nbrs = dm.neighbors(F0)
valence = np.array([len(n) for n in nbrs])


def pick_vertex(direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    ok = [i for i in range(len(V0)) if valence[i] == 6 and all(valence[j] == 6 for j in nbrs[i])]
    return max(ok, key=lambda i: V0[i] @ d)


r, u, w = _cam.right, _cam.up, _cam.view
holes = [pick_vertex(s * 0.78 * r + t * 0.56 * u + 0.5 * w) for s, t in ((-1, 1), (-1, -1), (1, 1), (1, -1))]
DISKS = [disk_faces(h) for h in holes]
LOOPS = [disk_loop(D) for D in DISKS]
assert all(len(D) == 24 for D in DISKS) and all(len(L) == 12 for L in LOOPS)
assert all(dm.euler_characteristic(np.array(D)) == 1 for D in DISKS)           # each removed piece is a disk
vs = [set(np.array(D).ravel()) for D in DISKS]
assert all(not (vs[a] & vs[b]) for a in range(4) for b in range(a + 1, 4))     # the four disks are disjoint
NL = 12


def bezier(p0, p1, p2, p3, s):
    return (1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * p1 + 3 * (1 - s) * s ** 2 * p2 + s ** 3 * p3


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tube_rings(A, B, m=12, h=1.15, n_fine=1200):
    """Rings of a tube from loop A (its order) to loop B (reversed order, best cyclic shift).

    Centre curve: the cubic Bezier from the centre of loop A along its normal to the centre of loop B. A
    rotation-minimizing frame (double reflection) is carried along it, and each loop vertex is written in polar
    coordinates (rho, phi) in the frame at its end; the rings interpolate rho and phi linearly. So every ring is
    star-shaped about the curve and turns the same way as the two loops: the tube is embedded, not twisted
    inside out (blending the 3D offsets of the two loops directly folds the rings in a U-turn)."""
    la = LOOPS[A]
    cA, cB = V0[la].mean(0), V0[LOOPS[B]].mean(0)
    nA, nB = V0[holes[A]], V0[holes[B]]
    P = [cA, cA + h * nA, cB + h * nB, cB]
    ss = np.linspace(0, 1, n_fine + 1)
    X = np.array([bezier(*P, t) for t in ss])
    T = np.array([3 * ((1 - t) ** 2 * (P[1] - P[0]) + 2 * (1 - t) * t * (P[2] - P[1]) + t ** 2 * (P[3] - P[2]))
                  for t in ss])
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    r = np.cross(T[0], [0.0, 0.0, 1.0] if abs(T[0][2]) < 0.9 else [1.0, 0.0, 0.0])
    R = [r / np.linalg.norm(r)]
    for k in range(n_fine):                       # double reflection (Wang et al. 2008)
        v1 = X[k + 1] - X[k]
        c1 = v1 @ v1
        rL = R[-1] - 2 / c1 * (v1 @ R[-1]) * v1
        tL = T[k] - 2 / c1 * (v1 @ T[k]) * v1
        v2 = T[k + 1] - tL
        c2 = v2 @ v2
        rn = rL - 2 / c2 * (v2 @ rL) * v2 if c2 > 1e-20 else rL
        R.append(rn / np.linalg.norm(rn))
    R = np.array(R)

    def polar(ids, c, k):
        e1, e2 = R[k], np.cross(T[k], R[k])
        o = V0[ids] - c
        return np.hypot(o @ e1, o @ e2), np.arctan2(o @ e2, o @ e1)
    rho_a, phi_a = polar(la, cA, 0)
    lb_rev = LOOPS[B][::-1]
    best = None
    for sh in range(NL):
        lb = lb_rev[sh:] + lb_rev[:sh]
        rho_b, phi_b = polar(lb, cB, n_fine)
        d = _wrap(phi_b - phi_a)
        if best is None or np.abs(d).sum() < best[0]:
            best = (np.abs(d).sum(), lb, rho_b, phi_a + d)
    _, lb, rho_b, phi_b = best
    assert np.all(_wrap(np.diff(np.r_[phi_a, phi_a[0]])) > 0)            # loop A turns positively about the curve
    assert np.all(_wrap(np.diff(np.r_[phi_b, phi_b[0]])) > 0)            # so does loop B (reversed)
    rings = []
    for k in range(1, m):
        s, kk = k / m, k * n_fine // m
        e1, e2 = R[kk], np.cross(T[kk], R[kk])
        rho, phi = (1 - s) * rho_a + s * rho_b, (1 - s) * phi_a + s * phi_b
        assert np.all(_wrap(np.diff(np.r_[phi, phi[0]])) > 0)            # star-shaped ring, same turning sense
        rings.append(X[kk] + rho[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
    return la, lb, rings


# ---------------------------------------------------------------- build the sequence of meshes
Vall = [V0]
n_next = len(V0)
STAGES = []          # (kind, faces added, faces removed)
F_cur = [tuple(f) for f in F0]


def counts_of(Fl):
    F = np.array(Fl)
    n0, n1, n2 = dm.counts(F)
    chi = n0 - n1 + n2
    b = boundary_loops(F)
    assert (2 - b - chi) % 2 == 0
    g = (2 - b - chi) // 2
    assert dm.is_manifold(F) and dm.oriented_consistently(F) and len(dm.components(F)) == 1
    return dict(n0=n0, n1=n1, n2=n2, chi=chi, b=b, g=g)


TABLES = [counts_of(F_cur)]
assert TABLES[0] == dict(n0=642, n1=1920, n2=1280, chi=2, b=0, g=0)
for A, B in ((0, 1), (2, 3)):
    for h in (A, B):
        star = DISKS[h]
        F_cur = [f for f in F_cur if f not in set(star)]
        STAGES.append(("hole", [], star))
        TABLES.append(counts_of(F_cur))
        assert TABLES[-1]["chi"] == TABLES[-2]["chi"] - 1 and TABLES[-1]["b"] == TABLES[-2]["b"] + 1
    la, lb, rings = tube_rings(A, B)
    ids = [list(la)]
    for ring in rings:
        Vall.append(ring)
        ids.append(list(range(n_next, n_next + NL)))
        n_next += NL
    ids.append(list(lb))
    for k in range(len(ids) - 1):
        R0, R1 = ids[k], ids[k + 1]
        new = []
        for j in range(NL):
            a, b2, c, d = R0[j], R0[(j + 1) % NL], R1[(j + 1) % NL], R1[j]
            new += [(a, b2, c), (a, c, d)]
        Vcur = np.vstack(Vall)
        ctr = Vcur[R0 + R1].mean(0)                               # centre of this tube segment
        nrm = dm.face_cross(Vcur, np.array(new))
        assert (np.einsum("ij,ij->i", nrm, Vcur[np.array(new)].mean(1) - ctr) > 0).all()   # normals point out of the tube
        assert (np.linalg.norm(Vcur[R1], axis=1) > 1.0).all() or k == len(ids) - 2         # the tube stays outside the sphere
        F_cur = F_cur + new
        STAGES.append(("ring", new, []))
        TABLES.append(counts_of(F_cur))
        assert TABLES[-1]["chi"] == TABLES[-2]["chi"]            # a ring never changes chi
    assert TABLES[-1]["b"] == 0 and TABLES[-1]["g"] == (1 if A == 0 else 2)
V_ALL = np.vstack(Vall)
# a disk of 24 faces removes 7 vertices, 30 edges, 24 faces (chi - 1); a tube of 11 rings and 12 segments adds
# 11 * 12 vertices, 11 * 12 + 12 * 24 edges and 12 * 24 faces (chi unchanged)
assert TABLES[-1] == dict(n0=642 - 4 * 7 + 2 * 132, n1=1920 - 4 * 30 + 2 * (132 + 288), n2=1280 - 4 * 24 + 2 * 288,
                          chi=-2, b=0, g=2)
assert dm.signed_volume(V_ALL, np.array(F_cur)) > 0            # still oriented outward
FACES = sorted(set(map(tuple, F0)) | {f for st in STAGES for f in st[1]})
FIDX = {f: k for k, f in enumerate(FACES)}
TUBE = set(f for st in STAGES for f in st[1])
NOTE_Y = -2.65                     # above the control bar of a paused video (bottom 12 %, GUIDELINES 12.8)


class Anim(hc.Scene2D):
    def construct(self):
        cam = hc.Cam(**CAM)
        F = np.array(FACES)
        cols = np.array([hc.rgb(hc.HEX["region"]) * 0.8 + 0.2 if f in TUBE else hc.rgb(hc.HEX["surface"])
                         for f in FACES])
        # the camera does not move: darken the faces seen from behind (the inside, visible only through a hole)
        cols[(dm.face_cross(V_ALL, F) @ cam.view) < 0] *= 0.45
        mv = hc.MeshView(V_ALL, F, cam, colors=cols, edge_width=0.45, ambient=0.5)
        mv.hidden[:] = True
        for f in F0:
            mv.hidden[FIDX[tuple(f)]] = False
        mv.face_opacity = np.ones(len(F))
        mv.set_mesh()
        self.add(mv)

        # ---- the table
        X0, Y0, DY = 0.95, 2.2, 0.62
        rows = ["n0", "n1", "n2", "chi", "b", "g"]
        names = {"n0": r"$n_0$", "n1": r"$n_1$", "n2": r"$n_2$", "chi": r"$\chi = n_0 - n_1 + n_2$",
                 "b": r"$b$", "g": r"$g = (2 - b - \chi)/2$"}
        heads = VGroup()
        for k, key in enumerate(rows):
            col = hc.C["normal"] if key == "chi" else (hc.C["tangent"] if key == "g" else hc.BLACK)
            m = hc.math(names[key], size=34, color=col).move_to([X0, Y0 - k * DY, 0], aligned_edge=LEFT)
            heads.add(m)
            if key == "b":
                heads.add(hc.text("(boundary loops)", size=30).next_to(m, RIGHT, buff=0.2))
        self.add(heads)
        XV = X0 + 4.35

        def val(key, t):
            col = hc.C["normal"] if key == "chi" else (hc.C["tangent"] if key == "g" else hc.BLACK)
            return hc.text(f"{t[key]}".replace("-", "\u2212"), size=34, color=col).move_to([XV, Y0 - rows.index(key) * DY, 0],
                                                                     aligned_edge=LEFT)
        cur = {key: val(key, TABLES[0]) for key in rows}
        shown = {key: TABLES[0][key] for key in rows}
        self.add(*cur.values())
        note = hc.text("closed sphere mesh", size=32).move_to([0, NOTE_Y, 0])
        self.add(note)
        self.wait(2.0)

        def update_table(t, extra=()):
            anims = list(extra)
            for key in rows:
                if t[key] != shown[key]:
                    new = val(key, t)
                    anims += hc.swap(cur[key], new)
                    cur[key], shown[key] = new, t[key]
            return anims

        def set_note(txt, color=hc.BLACK):
            nonlocal note
            new = hc.text(txt, size=32, color=color).move_to([0, NOTE_Y, 0])
            a = hc.swap(note, new)
            note = new
            return a

        orange = hc.rgb(hc.HEX["accent"])
        for s, (kind, add, rem) in enumerate(STAGES):
            t = TABLES[s + 1]
            if kind == "hole":
                idx = [FIDX[f] for f in rem]
                mv.colors[idx] = orange
                mv.set_mesh()
                self.play(*set_note("remove a disk: χ drops by 1", hc.C["accent"]), run_time=0.8)

                def vanish(m, a, idx=idx):
                    for k in idx:
                        m.face_opacity[k] = 1 - a
                        if a >= 1:
                            m.hidden[k] = True
                    m.set_mesh()
                self.play(UpdateFromAlphaFunc(mv, vanish), *update_table(t), run_time=0.8)
                self.wait(1.0 if t["g"] == 0 else 0.5)
            else:
                idx = [FIDX[f] for f in add]
                first = s == 0 or STAGES[s - 1][0] == "hole"
                last = s + 1 == len(STAGES) or STAGES[s + 1][0] == "hole"
                for k in idx:
                    mv.hidden[k] = False
                    mv.face_opacity[k] = 0.0

                def appear(m, a, idx=idx):
                    for k in idx:
                        m.face_opacity[k] = a
                    m.set_mesh()
                extra = set_note("add a tube ring: χ does not change", hc.C["tangent"]) if first else []
                fast = TABLES[s + 1]["g"] == 2 or (s > 0 and TABLES[s]["g"] >= 1)
                self.play(UpdateFromAlphaFunc(mv, appear), *update_table(t, extra), run_time=0.3 if fast else 0.55)
                if last:
                    self.play(*set_note(f"tube closed: one more handle, χ = {t['chi']}".replace("-", "\u2212"), hc.C["normal"]),
                              run_time=0.6)
                    self.wait(2.2)
        self.wait(1.5)


if __name__ == "__main__":
    da.main(Anim, __file__, poster_t=28.0)
