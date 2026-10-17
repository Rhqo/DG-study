"""Verification for Section 33.4 (Simplicial Manifolds).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/verify/v-33-4-simplicial-manifolds.py``
"""

import contextlib
import io
import itertools
import pathlib
import sys
from collections import Counter

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch33  # noqa: E402

from dgcheck import check, summary  # noqa: E402

PAGE = HERE / "33-4-simplicial-manifolds.html"
rng = np.random.default_rng(3304)


def run_printed(block, out_block, namespace=None):
    """Run <pre id=block> with dgcode.run_block; return (namespace, stdout == the text of <pre id=out_block>)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(PAGE, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(PAGE, out_block))


ns_impl, same_impl = run_printed("code-33-4-1", "out-33-4-1")
link_shape, link_edges = ns_impl["link_shape"], ns_impl["link_edges"]


def fan_connected(F, i):
    """Triangles containing i are connected through edges containing i (condition (V))."""
    T = [tuple(f) for f in F if i in f]
    if not T:
        return False
    seen, stack = {0}, [0]
    while stack:
        a = stack.pop()
        for b in range(len(T)):
            if b not in seen and len(set(T[a]) & set(T[b]) - {i}) >= 1:
                seen.add(b)
                stack.append(b)
    return len(seen) == len(T)


def edge_counts(F):
    return {e: len(fs) for e, fs in dm.edge_faces(np.asarray(F)).items()}


def criteria(F):
    """(E) and (V) of Corollary 33.4.5."""
    ec = edge_counts(F)
    E = all(c in (1, 2) for c in ec.values())
    V = all(fan_connected(F, i) for i in {int(v) for f in F for v in f})
    return E and V


# ---------------------------------------------------------------- Lemma 33.4.1 (random graphs of max degree 2)
ok = True
for trial in range(200):
    pieces, nxt, edges = rng.integers(1, 4), 0, []
    kinds = []
    for _ in range(pieces):
        kind = rng.choice(["cycle", "path"])
        m = int(rng.integers(3, 7)) if kind == "cycle" else int(rng.integers(1, 6))
        vs = list(range(nxt, nxt + (m if kind == "cycle" else m + 1)))
        nxt = vs[-1] + 1
        edges += [(vs[r], vs[(r + 1) % len(vs)]) for r in range(len(vs) if kind == "cycle" else len(vs) - 1)]
        kinds.append(kind)
    perm = rng.permutation(nxt)
    edges = [tuple(sorted((int(perm[a]), int(perm[b])))) for a, b in edges]
    rng.shuffle(edges)
    expect = kinds[0] if pieces == 1 else "other"
    ok &= link_shape(edges) == expect == dm.link_type(edges)
check("Lemma 33.4.1: unions of cycles/paths are classified correctly (200 random graphs)", ok)

# ---------------------------------------------------------------- Proposition 33.4.4, Corollary 33.4.5
book = np.array([(0, 1, 2), (0, 1, 3), (0, 1, 4)])
bowtie = np.array([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 5, 6), (0, 6, 7), (0, 7, 8)])
meshes = {"tetrahedron": ch33.tetrahedron()[1], "octahedron": ch33.octahedron()[1], "icosphere(1)": dm.icosphere(1)[1],
          "torus 3x3": dm.torus_mesh(2, 0.8, 3, 3)[1], "torus 6x8": dm.torus_mesh(2, 0.8, 6, 8)[1],
          "grid 5x5": dm.grid_mesh(np.arange(5.0), np.arange(5.0))[1], "band": ch33.cylinder_band(12)[1],
          "mobius": dm.mobius_mesh(12)[1], "klein": ch33.klein_bottle(5, 6), "book": book, "bowtie": bowtie}
Ftor = dm.torus_mesh(2, 0.8, 6, 8)[1]
for t in range(40):                                         # random pure subcomplexes of a torus mesh
    meshes[f"torus subset {t}"] = Ftor[rng.random(len(Ftor)) < rng.uniform(0.3, 0.95)]
ok_a = ok_bc = ok_cor = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    ec = edge_counts(F)
    for i in sorted({int(v) for f in F for v in f}):
        Lk = dd.link(K, i)
        Le = [s for s in Lk if len(s) == 2]
        ok_a &= not any(len(s) == 3 for s in Lk)
        for (j,) in [s for s in Lk if len(s) == 1]:
            ok_a &= sum(j in e for e in Le) == ec[(min(i, j), max(i, j))] >= 1
        counts_i = [c for e, c in ec.items() if i in e]
        shape = dd.vertex_link_type(K, i)
        cyc = all(c == 2 for c in counts_i) and fan_connected(F, i)
        path = all(c in (1, 2) for c in counts_i) and counts_i.count(1) == 2 and fan_connected(F, i)
        ok_bc &= (shape == "cycle") == cyc and (shape == "path") == path
    ok_cor &= dd.is_simplicial_surface(K) == criteria(F)
check("Proposition 33.4.4(a): link vertex degree = number of triangles on the edge (51 complexes)", ok_a)
check("Proposition 33.4.4(b),(c): cycle/path iff edge counts and connected fans (51 complexes)", ok_bc)
check("Corollary 33.4.5: simplicial surface with boundary iff (E) and (V) (51 complexes, incl. 40 random)", ok_cor)
nsurf = sum(dd.is_simplicial_surface(dd.complex_from_faces(F)) for k, F in meshes.items() if "subset" in k)
check(f"Corollary 33.4.5: the random subsets include both surfaces and non-surfaces ({nsurf} of 40 are surfaces)",
      0 < nsurf < 40)

# ---------------------------------------------------------------- Example 33.4.6, Non-example 33.4.7
closed = ["tetrahedron", "octahedron", "icosphere(1)", "torus 3x3", "torus 6x8", "klein"]
withb = ["grid 5x5", "band", "mobius"]
check("Example 33.4.6: closed examples: every vertex link is a cycle",
      all(all(dd.vertex_link_type(dd.complex_from_faces(meshes[k]), i) == "cycle"
              for i in range(meshes[k].max() + 1)) for k in closed))
check("Example 33.4.6: grid, band, Moebius are simplicial surfaces with boundary",
      all(dd.is_simplicial_surface(dd.complex_from_faces(meshes[k])) and dm.boundary_edges(meshes[k]) for k in withb))
Kb, Kw = dd.complex_from_faces(book), dd.complex_from_faces(bowtie)
check("Non-example 33.4.7(a): book: the edge endpoints have a link vertex of degree 3",
      dd.vertex_link_type(Kb, 0) == "other" and dd.vertex_link_type(Kb, 1) == "other")
check("Non-example 33.4.7(b): bowtie: (E) holds, (V) fails at the center",
      all(c <= 2 for c in edge_counts(bowtie).values()) and not fan_connected(bowtie, 0)
      and dd.vertex_link_type(Kw, 0) == "other")
Kn = dd.generate_complex([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (4, 6), (7,)])
check("Non-example 33.4.7(c): a non-pure complex is not a simplicial surface (isolated link vertex)",
      not dd.is_simplicial_surface(Kn) and dd.vertex_link_type(Kn, 4) == "other")

# ---------------------------------------------------------------- Proposition 33.4.8, Example 33.4.9


def loops_of(F):
    be = dm.boundary_edges(F)
    adj = {}
    for a, b in be:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    comps, seen = [], set()
    for s in adj:
        if s in seen:
            continue
        comp, stack = set(), [s]
        while stack:
            v = stack.pop()
            if v not in comp:
                comp.add(v)
                stack += adj[v]
        seen |= comp
        comps.append(comp)
    return be, adj, comps


ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    if not dd.is_simplicial_surface(K):
        continue
    be, adj, comps = loops_of(F)
    bverts = {i for i in range(int(F.max()) + 1) if any(i in f for f in F) and dd.vertex_link_type(K, i) == "path"}
    ok &= set(adj) == bverts and all(len(v) == 2 for v in adj.values())
    bd = dd.boundary_simplices(K) if be else frozenset()
    if be:
        edges_bd = [s for s in bd if len(s) == 2]
        ok &= sorted(edges_bd) == sorted(be) and dm.link_type(edges_bd) in ("cycle", "other")
        ok &= all(dm.link_type([e for e in be if e[0] in c]) == "cycle" for c in comps)
        ok &= dd.boundary_simplices(bd) == frozenset() if dd.is_pure(bd, 1) else False
check("Proposition 33.4.8: boundary vertices = path-link vertices, each on 2 boundary edges, loops, bd(bd) = 0", ok)
check("Example 33.4.9: grid 5x5: one loop of 16; band: two loops of 12; Moebius: one loop of 24",
      [len(c) for c in loops_of(meshes["grid 5x5"])[2]] == [16]
      and sorted(len(c) for c in loops_of(meshes["band"])[2]) == [12, 12]
      and [len(c) for c in loops_of(meshes["mobius"])[2]] == [24]
      and not dm.boundary_edges(meshes["icosphere(1)"]) and not dm.boundary_edges(meshes["torus 6x8"]))

# ---------------------------------------------------------------- Example 33.4.11, Non-example 33.4.12, Prop 33.4.13
C7 = dd.generate_complex([(k, (k + 1) % 7) for k in range(7)])
check("Example 33.4.11(a): every vertex link of a 7-cycle is two vertices",
      all(sorted(dd.link(C7, v)) == sorted([((v - 1) % 7,), ((v + 1) % 7,)]) for v in range(7)))
K4 = {s for m in range(1, 5) for s in itertools.combinations(range(5), m)}
ok = all(dd.link(K4, v) == {s for m in range(1, 4) for s in itertools.combinations([w for w in range(5) if w != v], m)}
         for v in range(5))
check("Example 33.4.11(b): in the boundary of the 4-simplex every vertex link is the boundary of a tetrahedron", ok)
Y = dd.generate_complex([(0, 1), (0, 2), (0, 3)])
check("Non-example 33.4.12(a): the Y graph: the center has a link of three vertices", dd.link(Y, 0) ==
      {(1,), (2,), (3,)})
TT = dd.generate_complex([(0, 1, 2, 3), (0, 4, 5, 6)])
check("Non-example 33.4.12(b): two tetrahedra at a vertex: link = two disjoint triangles",
      dd.link(TT, 0) == dd.generate_complex([(1, 2, 3), (4, 5, 6)]))
ok = True
for trial in range(100):
    pieces, nxt, edges, kinds = rng.integers(1, 4), 0, [], []
    for _ in range(pieces):
        cyc = rng.random() < 0.5
        m = int(rng.integers(3, 7))
        vs = list(range(nxt, nxt + m))
        nxt += m
        edges += [(vs[r], vs[(r + 1) % m]) for r in range(m if cyc else m - 1)]
        kinds.append("cycle" if cyc else "path")
    K1 = dd.generate_complex(edges)
    linksizes = {len(dd.link(K1, v)) for v in range(nxt)}
    ok &= (linksizes == {2}) == all(k == "cycle" for k in kinds) and linksizes <= {1, 2}
check("Proposition 33.4.13: 1-complexes with links of 1 or 2 vertices are cycles and paths (100 random)", ok)

# ---------------------------------------------------------------- Lemma 33.4.15, Corollary 33.4.16, Ex/Non-ex 33.4.17-18
ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    if not dd.is_simplicial_surface(K):
        continue
    G = {}
    for a, b in [s for s in K if len(s) == 2]:
        G.setdefault(a, set()).add(b)
        G.setdefault(b, set()).add(a)
    start = next(iter(G))
    seen, stack = set(), [start]
    while stack:
        v = stack.pop()
        if v not in seen:
            seen.add(v)
            stack += list(G[v])
    if len(seen) == len(G):
        ok &= len(dm.components(F)) == 1                     # edge-connected
check("Lemma 33.4.15: connected 1-skeleton implies edge-connected (all surfaces above)", ok)
check("Example 33.4.17: icosphere, torus, grid, band are orientable",
      all(dm.orient(meshes[k])[1] for k in ("icosphere(1)", "torus 6x8", "grid 5x5", "band")))
check("Non-example 33.4.18: Moebius band and Klein bottle are not orientable; Klein and torus 5x6 have equal counts",
      not dm.orient(meshes["mobius"])[1] and not dm.orient(meshes["klein"])[1]
      and dm.counts(meshes["klein"]) == dm.counts(dm.torus_mesh(2, 0.8, 5, 6)[1]))

# ---------------------------------------------------------------- Proposition 33.4.19 and the table
table = {"tetrahedron boundary": (ch33.tetrahedron()[1], (4, 6, 4), 2, 0, True),
         "octahedron": (ch33.octahedron()[1], (6, 12, 8), 2, 0, True),
         "icosphere(0)": (dm.icosphere(0)[1], (12, 30, 20), 2, 0, True),
         "icosphere(2)": (dm.icosphere(2)[1], (162, 480, 320), 2, 0, True),
         "torus 8x16": (dm.torus_mesh(2, 0.8, 8, 16)[1], (128, 384, 256), 0, 0, True),
         "grid 5x5": (meshes["grid 5x5"], (25, 56, 32), 1, 1, True),
         "band 12": (meshes["band"], (24, 48, 24), 0, 2, True),
         "mobius 12": (meshes["mobius"], (24, 48, 24), 0, 1, False),
         "genus2 0.16": (dm.genus2_mesh(h=0.16)[1], (4032, 12102, 8068), -2, 0, True)}
for name, (F, cnt, chi, b, orientable) in table.items():
    n0, n1, n2 = dm.counts(F)
    nb = len(dm.boundary_edges(F))
    ok = (n0, n1, n2) == cnt and n0 - n1 + n2 == chi and len(loops_of(F)[2]) == b and dm.orient(F)[1] == orientable
    ok &= 3 * n2 == 2 * n1 - nb and dm.is_manifold(F)
    check(f"Proposition 33.4.19 / table: {name}: counts {cnt}, chi {chi}, {b} loops, orientable {orientable}", ok)

# ---------------------------------------------------------------- valence: Example 33.4.21, Non-example 33.4.22, Prop 33.4.23
val = lambda F: np.bincount(dm.edges(F).ravel())  # noqa: E731
check("Example 33.4.21: valences 3 (tetrahedron), 4 (octahedron), 5 (icosahedron)",
      set(val(ch33.tetrahedron()[1])) == {3} and set(val(ch33.octahedron()[1])) == {4}
      and set(val(dm.icosphere(0)[1])) == {5})
ok = all(Counter(val(dm.icosphere(lv)[1]).tolist()) == ({5: 12} if lv == 0 else {5: 12, 6: 10 * 4 ** lv + 2 - 12})
         for lv in range(5))
check("Example 33.4.21: icosphere(level): 12 vertices of valence 5, the rest 6 (levels 0..4)", ok)
check("Example 33.4.21: torus mesh all regular; triangulated cube valences {6: 2, 4: 6}",
      set(val(dm.torus_mesh(2, 0.8, 8, 16)[1])) == {6}
      and Counter(val(ch33.cube_triangulated()[1]).tolist()) == {6: 2, 4: 6})
ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    for i in range(int(F.max()) + 1):
        if any(i in f for f in F) and dd.vertex_link_type(K, i) == "cycle":
            ok &= len(dd.k_simplices(dd.link(K, i), 0)) >= 3
            ok &= len(dd.k_simplices(dd.link(K, i), 0)) == sum(i in f for f in F)   # valence = #triangles
check("Non-example 33.4.22 / Definition 33.4.20: interior vertices have valence >= 3 = number of triangles", ok)
ok = True
for name, (F, cnt, chi, b, orientable) in table.items():
    n0, n1, n2 = dm.counts(F)
    d = val(F)
    ok &= d.sum() == 2 * n1
    if b == 0:
        ok &= int(np.sum(6 - d)) == 6 * chi and np.isclose(d.mean(), 6 - 6 * chi / n0)
        ok &= n1 == 3 * (n0 - chi) and n2 == 2 * (n0 - chi)
check("Proposition 33.4.23: sum deg = 2 n1; closed: sum (6 - deg) = 6 chi, n1 = 3(n0 - chi), n2 = 2(n0 - chi)", ok)
mv = 2 * dm.counts(dm.icosphere(5)[1])[1] / dm.counts(dm.icosphere(5)[1])[0]
check("Figure 33.4.3: icosphere(5) has n0 = 10242 and mean valence 5.9988", dm.counts(dm.icosphere(5)[1])[0] == 10242
      and abs(mv - 5.9988) < 5e-5)
g2 = dm.genus2_mesh(h=0.2)[1]
check("Figure 33.4.3: genus2_mesh(0.2) has n0 = 2524 and mean valence 6.0048",
      dm.counts(g2)[0] == 2524 and abs(2 * dm.counts(g2)[1] / 2524 - 6.0048) < 5e-5)

# ---------------------------------------------------------------- Implementation box
check("Implementation 33.4: code-33-4-1 prints exactly out-33-4-1", same_impl)
check("Implementation 33.4: code-33-4-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-4-1") <= 30)
ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    Fl = [tuple(int(v) for v in f) for f in F]
    for i in sorted({v for f in Fl for v in f}):
        ok &= link_shape(link_edges(Fl, i)) == dd.vertex_link_type(K, i)
check("Implementation 33.4: link_shape(link_edges) = dgdec.vertex_link_type on all 51 complexes", ok)

# ---------------------------------------------------------------- Exercises
Vo, Fo = ch33.octahedron()
two_v = np.vstack([Fo, Fo + 5])              # vertex 5 of the first = vertex 0 of the second (shared vertex 5)
K2 = dd.complex_from_faces(two_v)
check("Exercise 33.4.2(a): two octahedra at a vertex: (E) holds, the shared link is two 4-cycles",
      all(c == 2 for c in edge_counts(two_v).values()) and dd.vertex_link_type(K2, 5) == "other"
      and len(dd.k_simplices(dd.link(K2, 5), 0)) == 8)
# glue along the edge {0, 2} of both copies: relabel the second copy so that its 0 -> 0 and 2 -> 2
relabel = {v: v + 6 for v in range(6)}
relabel[0], relabel[2] = 0, 2
two_e = np.vstack([Fo, np.vectorize(relabel.get)(Fo)])
check("Exercise 33.4.2(b): two octahedra along an edge: that edge lies in 4 triangles",
      edge_counts(two_e)[(0, 2)] == 4 and dd.vertex_link_type(dd.complex_from_faces(two_e), 0) == "other")
tri4 = list(itertools.combinations(range(4), 3))
sols = [S for m in range(1, 5) for S in itertools.combinations(tri4, m)
        if dd.is_simplicial_surface(dd.generate_complex(S)) and not dm.boundary_edges(np.array(S))]
check("Exercise 33.4.3: the only closed simplicial surface on 4 vertices is the tetrahedron boundary",
      len(sols) == 1 and len(sols[0]) == 4)
check("Exercise 33.4.3: closed examples satisfy n0 >= 4, n1 >= 6, n2 >= 4",
      all(dm.counts(F)[0] >= 4 and dm.counts(F)[1] >= 6 and dm.counts(F)[2] >= 4
          for k, (F, c, chi, b, o) in table.items() if b == 0))
check("Exercise 33.4.4: genus2_mesh: sum (6 - deg) = -12", int(np.sum(6 - val(dm.genus2_mesh(h=0.16)[1]))) == -12)
ok = True
for name in ("icosphere(1)", "torus 6x8", "grid 5x5", "band", "mobius"):
    F = meshes[name]
    K = dd.complex_from_faces(F)
    ec = edge_counts(F)
    for e in dd.k_simplices(K, 1):
        Le = dd.link(K, [e])
        ok &= all(len(s) == 1 for s in Le) and len(Le) == ec[e] and len(Le) in (1, 2)
check("Exercise 33.4.5: edge links have 1 or 2 vertices (= number of triangles on the edge)", ok)
ns, same6 = run_printed("code-ex-33-4-6", "out-ex-33-4-6")
check("Exercise 33.4.6 (coding): code-ex-33-4-6 prints exactly out-ex-33-4-6", same6)
check("Exercise 33.4.6: band(12, True) = dgmesh.mobius_mesh(12), band(12, False) = lib cylinder_band(12)",
      np.array_equal(ns["band"](12, True), dm.mobius_mesh(12)[1])
      and np.array_equal(ns["band"](12, False), ch33.cylinder_band(12)[1]))

summary()
