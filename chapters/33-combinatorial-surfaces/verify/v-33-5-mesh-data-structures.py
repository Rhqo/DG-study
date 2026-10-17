"""Verification for Section 33.5 (Mesh Data Structures).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/verify/v-33-5-mesh-data-structures.py``
"""

import contextlib
import io
import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch33  # noqa: E402

from dgcheck import check, summary  # noqa: E402

PAGE = HERE / "33-5-mesh-data-structures.html"
rng = np.random.default_rng(3305)


def run_printed(block, out_block, namespace=None):
    """Run <pre id=block> with dgcode.run_block; return (namespace, stdout == the text of <pre id=out_block>)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(PAGE, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(PAGE, out_block))


Vt, Ft = ch33.tetrahedron()
A0t, A1t = np.abs(dd.d0(Ft)).astype(int), np.abs(dd.d1(Ft)).astype(int)

# ---------------------------------------------------------------- Example 33.5.2
check("Example 33.5.2: tetrahedron rows (0,1,2), (0,3,1), (0,2,3), (1,3,2)", Ft.tolist() ==
      [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
check("Example 33.5.2: A0 and A1 as printed",
      A0t.tolist() == [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]]
      and A1t.tolist() == [[1, 1, 0, 1, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 1, 1]])

meshes = {"tetrahedron": Ft, "octahedron": ch33.octahedron()[1], "icosphere(2)": dm.icosphere(2)[1],
          "torus 6x8": dm.torus_mesh(2, 0.8, 6, 8)[1], "grid 5x4": dm.grid_mesh(np.arange(5.0), np.arange(4.0))[1],
          "mobius": dm.mobius_mesh(10)[1], "book": np.array([(0, 1, 2), (0, 1, 3), (0, 1, 4)])}

# ---------------------------------------------------------------- Propositions 33.5.3, 33.5.4
ok3 = ok4 = True
for name, F in meshes.items():
    A0, A1 = np.abs(dd.d0(F)).astype(int), np.abs(dd.d1(F)).astype(int)
    E, _ = dd.edge_list(F)
    n0 = A0.shape[1]
    W = np.zeros((n0, n0), int)
    W[E[:, 0], E[:, 1]] = W[E[:, 1], E[:, 0]] = 1
    deg = np.bincount(E.ravel(), minlength=n0)
    ok3 &= np.array_equal(A0.T @ A0, np.diag(deg) + W)
    ok4 &= (A0.sum(1) == 2).all() and (A1.sum(1) == 3).all() and np.array_equal(A0.sum(0), deg)
    ef = dm.edge_faces(F)
    ok4 &= all(A1[:, e].sum() == len(ef[tuple(E[e])]) for e in range(len(E)))
    inc = np.zeros((len(F), n0), int)
    inc[np.repeat(np.arange(len(F)), 3), F.ravel()] = 1
    ok4 &= np.array_equal(A1 @ A0, 2 * inc)
check("Proposition 33.5.3: A0^T A0 = D + W (7 complexes)", ok3)
check("Proposition 33.5.4: row sums 2 and 3, column sums valence and #triangles, A1 A0 = 2 * incidence", ok4)
check("Proposition 33.5.4(b): book has a column of A1 with sum 3; grid boundary edges have column sum 1",
      np.abs(dd.d1(meshes["book"])).sum(0).max() == 3
      and (np.abs(dd.d1(meshes["grid 5x4"])).sum(0) == 1).sum() == len(dm.boundary_edges(meshes["grid 5x4"])))
F3 = dm.icosphere(3)[1]
A03 = np.abs(dd.d0(F3))
check("Sparse storage: icosphere(3) A0 is 1920 x 642 = 1,232,640 entries with 3840 nonzeros (about 0.3%)",
      A03.shape == (1920, 642) and A03.size == 1232640 and int(A03.sum()) == 3840
      and abs(3840 / 1232640 - 0.003) < 0.0005)

# ---------------------------------------------------------------- Implementation 33.5a, Proposition 33.5.5
ns, same = run_printed("code-33-5-1", "out-33-5-1")
check("Implementation 33.5a: code-33-5-1 prints exactly out-33-5-1", same)
check("Implementation 33.5a: code-33-5-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-5-1") <= 30)
ok_inc = ok55 = True
for name, F in meshes.items():
    A0, A1 = ns["incidence"](F)
    ok_inc &= np.array_equal(A0, np.abs(dd.d0(F))) and np.array_equal(A1, np.abs(dd.d1(F)))
    K = dd.complex_from_faces(F)
    E, index = dd.edge_list(F)
    Kl = sorted(K)
    n0 = A0.shape[1]
    for _ in range(20):
        S = {Kl[i] for i in rng.choice(len(Kl), size=rng.integers(1, 5), replace=False)}
        s0 = np.zeros(n0, bool)
        s1 = np.zeros(len(E), bool)
        s2 = np.zeros(len(F), bool)
        rowof = {tuple(sorted(f)): t for t, f in enumerate(F)}
        for s in S:
            if len(s) == 1:
                s0[s[0]] = True
            elif len(s) == 2:
                s1[index[s]] = True
            else:
                s2[rowof[s]] = True

        def to_set(v0, v1, v2):
            return {(int(i),) for i in np.flatnonzero(v0)} | {tuple(int(x) for x in E[e]) for e in np.flatnonzero(v1)} \
                | {tuple(sorted(int(x) for x in F[t])) for t in np.flatnonzero(v2)}
        ok55 &= to_set(*ns["star"](A0, A1, s0, s1, s2)) == set(dd.star(K, S))
        ok55 &= to_set(*ns["closure"](A0, A1, s0, s1, s2)) == set(dd.closure(K, S))
        ok55 &= to_set(*ns["link"](A0, A1, (s0, s1, s2))) == set(dd.link(K, S))
check("Implementation 33.5a: incidence(F) equals |dgdec.d0|, |dgdec.d1| (7 complexes)", ok_inc)
check("Proposition 33.5.5: matrix star/closure/link = set algorithm (7 complexes x 20 random S)", ok55)

# ---------------------------------------------------------------- Definition 33.5.6, Example 33.5.7
D0, D1 = dd.d0(Ft).astype(int), dd.d1(Ft).astype(int)
check("Example 33.5.7: A0^pm as printed",
      D0.tolist() == [[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1], [0, -1, 1, 0], [0, -1, 0, 1], [0, 0, -1, 1]])
check("Example 33.5.7: A1^pm as printed",
      D1.tolist() == [[1, -1, 0, 1, 0, 0], [-1, 0, 1, 0, -1, 0], [0, 1, -1, 0, 0, 1], [0, 0, 0, -1, 1, -1]])
check("Example 33.5.7: A1^pm A0^pm = 0 and the column sums of A1^pm are 0 (tetrahedron)",
      not (D1 @ D0).any() and not D1.sum(0).any())


def signed_from_orientations(F_or, E_or, n0):
    """Signed incidence matrices for arbitrary orientations: rows of F_or for triangles, rows of E_or for edges,
    computed from Definition 33.3.6 via dgdec.induced_orientation (independently of dgdec.d0 / d1)."""
    S0 = np.zeros((len(E_or), n0), int)
    for e, (a, b) in enumerate(E_or):
        S0[e, a], S0[e, b] = dd.induced_orientation((a, b), (a,)), dd.induced_orientation((a, b), (b,))
    S1 = np.zeros((len(F_or), len(E_or)), int)
    pos = {frozenset(e): k for k, e in enumerate(E_or)}
    for f, tri in enumerate(F_or):
        for c in range(3):
            a, b = int(tri[c]), int(tri[(c + 1) % 3])
            k = pos[frozenset((a, b))]
            S1[f, k] = dd.induced_orientation(tuple(int(v) for v in tri), tuple(int(v) for v in E_or[k]))
    return S0, S1


ok6 = ok_dd = ok8 = True
for name, F in meshes.items():
    E, _ = dd.edge_list(F)
    n0 = int(F.max()) + 1
    S0, S1 = signed_from_orientations(F, E, n0)
    ok_dd &= np.array_equal(S0, dd.d0(F)) and np.array_equal(S1, dd.d1(F))
    ok6 &= not (S1 @ S0).any()
    p1 = rng.choice([1, -1], size=len(E))
    p2 = rng.choice([1, -1], size=len(F))
    E2 = np.array([e if s > 0 else e[::-1] for e, s in zip(E, p1)])
    F2 = np.array([f if s > 0 else f[::-1] for f, s in zip(F, p2)])
    T0, T1 = signed_from_orientations(F2, E2, n0)
    ok8 &= np.array_equal(T0, np.diag(p1) @ S0) and np.array_equal(T1, np.diag(p2) @ S1 @ np.diag(p1))
    ok8 &= np.array_equal(T1 @ T0, np.diag(p2) @ S1 @ S0)
check("Definition 33.5.6: the signed matrices from Definition 33.3.6 equal dgdec.d0, dgdec.d1 (7 complexes)", ok_dd)
check("Example 33.5.7: A1^pm A0^pm = 0 numerically (7 complexes, incl. icosphere, torus, grid)", ok6)
check("Proposition 33.5.8: changing orientations gives P1 A0^pm and P2 A1^pm P1 (random flips)", ok8)

ok9 = True
for name in ("tetrahedron", "octahedron", "icosphere(2)", "torus 6x8", "grid 5x4", "mobius"):
    F = meshes[name]
    for _ in range(10):
        flips = rng.random(len(F)) < 0.3 if _ else np.zeros(len(F), bool)
        F2 = np.array([f[::-1] if fl else f for f, fl in zip(F, flips)])
        D1 = dd.d1(F2)
        twocols = [e for e in range(D1.shape[1]) if np.count_nonzero(D1[:, e]) == 2]
        test = all(D1[:, e].sum() == 0 for e in twocols)
        ok9 &= test == dm.oriented_consistently(F2)
check("Proposition 33.5.9: consistent iff every two-entry column of A1^pm sums to 0 (random flips)", ok9)
check("Proposition 33.5.9: closed consistent surfaces have 1^T A1^pm = 0",
      all(not dd.d1(meshes[k]).sum(0).any() for k in ("tetrahedron", "octahedron", "icosphere(2)", "torus 6x8")))

# ---------------------------------------------------------------- halfedges: Prop 33.5.11, Example 33.5.12, Prop 33.5.13
ok11 = ok13 = True
for name in ("tetrahedron", "octahedron", "icosphere(2)", "torus 6x8", "grid 5x4"):
    F = meshes[name]
    he = dd.HalfedgeMesh(F)
    rot_ccw = he.twin[he.prev]                      # eta o rho^{-1}
    rot_cw = he.next[he.twin]                       # rho o eta
    ok11 &= np.array_equal(rot_ccw[rot_cw], np.arange(he.nh))
    ok11 &= sorted(map(sorted, dd.permutation_cycles(rot_ccw))) == sorted(map(sorted, dd.permutation_cycles(rot_cw)))
    if dm.boundary_edges(F):
        continue
    K = dd.complex_from_faces(F)
    directed = [(int(F[f, c]), int(F[f, (c + 1) % 3])) for f in range(len(F)) for c in range(3)]
    ok13 &= len(set(directed)) == len(directed)                                   # (a)
    ok13 &= all((b, a) in set(directed) for a, b in directed)                     # (b)
    ok13 &= np.all(he.vertex[he.twin] == he.head) and np.all(he.vertex[he.next] == he.head)
    ok13 &= np.all(he.face[he.next] == he.face) and np.all(he.next[he.next[he.next]] == np.arange(he.nh))
    faces = dd.permutation_cycles(he.next)
    edges = dd.permutation_cycles(he.twin)
    verts = dd.permutation_cycles(rot_cw)
    ok13 &= sorted(tuple(sorted(he.vertex[c])) for c in faces) == sorted(tuple(sorted(f)) for f in F)  # (c)
    ok13 &= sorted(tuple(sorted({int(he.vertex[h]) for h in c})) for c in edges) == [tuple(e) for e in dd.edge_list(F)[0]]
    ok13 &= all(len(set(he.vertex[c])) == 1 and len(c) == len(dd.k_simplices(dd.link(K, int(he.vertex[c[0]])), 0))
                for c in verts)                                                   # (e)
    for c in verts:                                   # consecutive heads are adjacent in the link cycle
        heads = [int(he.head[h]) for h in c]
        i = int(he.vertex[c[0]])
        Le = {tuple(sorted(s)) for s in dd.link(K, i) if len(s) == 2}
        ok13 &= all(tuple(sorted((heads[r], heads[(r + 1) % len(heads)]))) in Le for r in range(len(heads)))
    ok13 &= len(verts) - len(edges) + len(faces) == dm.euler_characteristic(F) == he.euler_characteristic()
check("Proposition 33.5.11: twin o prev = (next o twin)^{-1}, same orbits (5 meshes)", ok11)
ok_in = ok_cw = True
for name in ("tetrahedron", "octahedron", "icosphere(2)", "torus 6x8"):
    F = meshes[name]
    he = dd.HalfedgeMesh(F)
    V = {"tetrahedron": Vt, "octahedron": ch33.octahedron()[0], "icosphere(2)": dm.icosphere(2)[0],
         "torus 6x8": dm.torus_mesh(2, 0.8, 6, 8)[0]}[name]
    slide = he.twin[he.next]                                       # eta o rho (lecture slides)
    ok_in &= np.array_equal(slide, he.twin[(he.next[he.twin])[he.twin]])   # = eta (rho eta) eta
    ok_in &= all(len({int(he.head[h]) for h in c}) == 1 for c in dd.permutation_cycles(slide))
    ok_in &= len(dd.permutation_cycles(slide)) == len(dd.permutation_cycles(he.next[he.twin])) == int(F.max()) + 1
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # face normals (outward rows)
    for h in range(he.nh):                                         # one step of rho o eta at the tail of h
        g = int(he.next[he.twin[h]])
        i = int(he.vertex[h])
        n = fn[he.face[h]] + fn[he.face[g]]
        ok_cw &= np.dot(np.cross(V[he.head[h]] - V[i], V[he.head[g]] - V[i]), n) < 0
check("Conventions 33.5: eta o rho = eta (rho o eta) eta; its orbits are the halfedges into one vertex", ok_in)
check("Conventions 33.5: each step of rho o eta turns clockwise seen from the outward normal (4 closed meshes)", ok_cw)
check("Proposition 33.5.13(a)-(e): halfedge invariants and chi^H = chi (closed meshes)", ok13)
het = dd.HalfedgeMesh(Ft)
check("Example 33.5.12: tetrahedron: 12 halfedges, orbits 4 x 3 (faces), 6 (edges), 4 x 3 (vertices), chi 2",
      het.nh == 12 and sorted(map(len, dd.permutation_cycles(het.next))) == [3] * 4
      and len(dd.permutation_cycles(het.twin)) == 6
      and sorted(map(len, dd.permutation_cycles(het.next[het.twin]))) == [3] * 4 and het.euler_characteristic() == 2)

# ---------------------------------------------------------------- Example 33.5.15, Non-example 33.5.16, boundary
tw = np.array([1, 0])
for name, nx, expect in (("(a) next = id", np.array([0, 1]), (1, 1, 2)), ("(b) next = twin", tw, (2, 1, 1))):
    n0 = len(dd.permutation_cycles(nx[tw]))
    n1 = len(dd.permutation_cycles(tw))
    n2 = len(dd.permutation_cycles(nx))
    check(f"Example 33.5.15{name[:3]}: (n0, n1, n2) = {expect}, chi = 2", (n0, n1, n2) == expect and n0 - n1 + n2 == 2)
for name in ("book", "mobius"):
    try:
        dd.HalfedgeMesh(meshes[name])
        raised = False
    except ValueError:
        raised = True
    check(f"Non-example 33.5.16: dgdec.HalfedgeMesh rejects the {name}", raised)
heg = dd.HalfedgeMesh(dm.grid_mesh(np.arange(4.0), np.arange(4.0))[1])
check("Boundary halfedges: grid 4x4: 66 halfedges, one boundary loop of 12, chi = 1",
      heg.nh == 66 and [len(c) for c in heg.boundary_loops()] == [12] and heg.euler_characteristic() == 1)

# ---------------------------------------------------------------- Implementation 33.5b
ns, same = run_printed("code-33-5-2", "out-33-5-2")
check("Implementation 33.5b: code-33-5-2 prints exactly out-33-5-2", same)
check("Implementation 33.5b: code-33-5-2 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-5-2") <= 30)
ok = True
for name in ("tetrahedron", "octahedron", "icosphere(2)", "torus 6x8"):
    F = meshes[name]
    tail, twin, nxt = ns["halfedges"](F)
    he = dd.HalfedgeMesh(F)
    ok &= np.array_equal(tail, he.vertex) and np.array_equal(twin, he.twin) and np.array_equal(nxt, he.next)
check("Implementation 33.5b: tail, twin, next equal dgdec.HalfedgeMesh (closed meshes)", ok)
check("Implementation 33.5b: ring of vertex 0 = dgdec vertex_one_ring(0) = [11, 5, 1, 7, 10]",
      dd.HalfedgeMesh(dm.icosphere(0)[1]).vertex_one_ring(0) == [11, 5, 1, 7, 10])

# ---------------------------------------------------------------- Exercises
Fs = np.array([(0, 1, 2), (0, 2, 3)])
A0s, A1s = np.abs(dd.d0(Fs)).astype(int), np.abs(dd.d1(Fs)).astype(int)
check("Exercise 33.5.1: A0, A1, A0^T A0 as printed; boundary edges from column sums",
      A0s.tolist() == [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 1, 1]]
      and A1s.tolist() == [[1, 1, 0, 1, 0], [0, 1, 1, 0, 1]]
      and (A0s.T @ A0s).tolist() == [[3, 1, 1, 1], [1, 2, 1, 0], [1, 1, 3, 1], [1, 0, 1, 2]]
      and A1s.sum(0).tolist() == [1, 2, 1, 1, 1])
D0s, D1s = dd.d0(Fs).astype(int), dd.d1(Fs).astype(int)
check("Exercise 33.5.2: signed matrices as printed, product 0, column sums (1, 0, -1, 1, 1)",
      D0s.tolist() == [[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1], [0, -1, 1, 0], [0, 0, -1, 1]]
      and D1s.tolist() == [[1, -1, 0, 1, 0], [0, 1, -1, 0, 1]] and not (D1s @ D0s).any()
      and D1s.sum(0).tolist() == [1, 0, -1, 1, 1])
ok = True
for name in ("octahedron", "icosphere(2)", "torus 6x8", "grid 5x4"):
    F = meshes[name]
    A1 = np.abs(dd.d1(F)).astype(int)
    G = A1 @ A1.T - 3 * np.eye(len(F), dtype=int)
    adj = np.zeros_like(G)
    for fs in dm.edge_faces(F).values():
        if len(fs) == 2:
            adj[fs[0], fs[1]] = adj[fs[1], fs[0]] = 1
    ok &= np.array_equal(G, adj)
    EE = A1.T @ A1
    ef = dm.edge_faces(F)
    ok &= all(EE[e, e] == len(ef[tuple(dd.edge_list(F)[0][e])]) for e in range(len(EE)))
check("Exercise 33.5.3: A1 A1^T - 3I is the triangle adjacency of shared edges; diag(A1^T A1) = #triangles", ok)
twp = np.array([1, 0, 3, 2, 5, 4])
nxp = np.array([2, 5, 4, 1, 0, 3])
check("Exercise 33.5.4: pillow: faces {0,2,4},{1,5,3}; vertices {0,5},{1,2},{3,4}; chi = 2",
      sorted(map(sorted, dd.permutation_cycles(nxp))) == [[0, 2, 4], [1, 3, 5]]
      and sorted(map(sorted, dd.permutation_cycles(nxp[twp]))) == [[0, 5], [1, 2], [3, 4]]
      and len(dd.permutation_cycles(nxp[twp])) - 3 + 2 == 2)
h = [k for k in range(12) if het.vertex[k] == 0 and het.head[k] == 1][0]
orbit = [h]
for _ in range(2):
    orbit.append(int(het.next[het.twin[orbit[-1]]]))
check("Exercise 33.5.5: the next o twin orbit of 0 -> 1 is 0 -> 1, 0 -> 3, 0 -> 2",
      [(int(het.vertex[k]), int(het.head[k])) for k in orbit] == [(0, 1), (0, 3), (0, 2)]
      and int(het.next[het.twin[orbit[-1]]]) == h)
ns6, same6 = run_printed("code-ex-33-5-6", "out-ex-33-5-6")
check("Exercise 33.5.6 (coding): code-ex-33-5-6 prints exactly out-ex-33-5-6", same6)

summary()
