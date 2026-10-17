"""Verification for Section 33.2 (Star, Closure, and Link).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/verify/v-33-2-star-closure-link.py``
"""

import contextlib
import io
import itertools
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

PAGE = HERE / "33-2-star-closure-link.html"
rng = np.random.default_rng(3302)


def run_printed(block, out_block, namespace=None):
    """Run <pre id=block> with dgcode.run_block; return (namespace, stdout == the text of <pre id=out_block>)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(PAGE, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(PAGE, out_block))

Vo, Fo = ch33.octahedron()
Ko = dd.complex_from_faces(Fo)
T = lambda *v: tuple(sorted(v))  # noqa: E731

# ---------------------------------------------------------------- Non-example 33.2.3(a)
Ktet = dd.generate_complex([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
rim = {(0, 1), (1, 2), (0, 2)}
check("Non-example 33.2.3(a): the three edges of {0,1,2} are not a subcomplex; with their vertices they are Lk(3)",
      not dd.is_complex(rim) and dd.is_complex(rim | {(0,), (1,), (2,)})
      and dd.link(Ktet, 3) == rim | {(0,), (1,), (2,)})

# ---------------------------------------------------------------- Examples 33.2.5, 33.2.8, 33.2.11, Non-examples
St4 = dd.star(Ko, 4)
check("Example 33.2.5: octahedron St(4) has 9 simplices (1 + 4 + 4)",
      len(St4) == 9 and [len(dd.k_simplices(St4, k)) for k in range(3)] == [1, 4, 4])
check("Example 33.2.5: St({0,4}) = {04, 024, 034}", dd.star(Ko, [(0, 4)]) == {T(0, 4), T(0, 2, 4), T(0, 3, 4)})
check("Example 33.2.5: the star of a triangle is itself", all(dd.star(Ko, [tuple(sorted(f))]) == {tuple(sorted(f))}
                                                             for f in Fo))
Cl = dd.closure(Ko, St4)
check("Example 33.2.8: Cl(St(4)) has 17 simplices (5 + 8 + 4)",
      len(Cl) == 17 and [len(dd.k_simplices(Cl, k)) for k in range(3)] == [5, 8, 4])
check("Example 33.2.8: Cl({04}) has 3 simplices", dd.closure(Ko, [(0, 4)]) == {T(0, 4), (0,), (4,)})
Ns = St4 | {(v,) for s in St4 for v in s}
check("Non-example 33.2.9: St(4) plus its vertices (13 simplices) is not a subcomplex",
      len(Ns) == 13 and not dd.is_complex(Ns))
Lk4 = dd.link(Ko, 4)
check("Example 33.2.11(a): Lk(4) = vertices 0,1,2,3 and the four equator edges (a 4-cycle)",
      Lk4 == {(0,), (1,), (2,), (3,), T(0, 2), T(0, 3), T(1, 2), T(1, 3)}
      and dd.vertex_link_type(Ko, 4) == "cycle")
Ki = dd.complex_from_faces(dm.icosphere(0)[1])
check("Example 33.2.11(a): every icosahedron vertex link is a 5-cycle",
      all(dd.vertex_link_type(Ki, v) == "cycle" and len(dd.k_simplices(dd.link(Ki, v), 0)) == 5 for v in range(12)))
check("Example 33.2.11(b): Lk({0,4}) = {2, 3}; Cl(St(e)) has 11 simplices",
      dd.link(Ko, [(0, 4)]) == {(2,), (3,)} and len(dd.closure(Ko, dd.star(Ko, [(0, 4)]))) == 11)
e = [(0, 4)]
wrong = dd.closure(Ko, dd.star(Ko, e)) - dd.star(Ko, e)
check("Non-example 33.2.12: Cl(St(e)) minus St(e) is the square 0-2-4-3 (contains the endpoints)",
      wrong == {(0,), (2,), (3,), (4,), T(0, 2), T(2, 4), T(3, 4), T(0, 3)})

# ---------------------------------------------------------------- Propositions 33.2.7, 33.2.13, 33.2.14 on random sets
meshes = {"octahedron": Fo, "icosphere(1)": dm.icosphere(1)[1], "torus 4x6": dm.torus_mesh(2, 0.8, 4, 6)[1],
          "grid 5x5": dm.grid_mesh(np.arange(5.0), np.arange(5.0))[1], "mobius": dm.mobius_mesh(8)[1]}
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    Kl = sorted(K)
    ok7 = ok13a = ok13b = ok13c = ok14 = True
    for trial in range(25):
        S = {Kl[i] for i in rng.choice(len(Kl), size=rng.integers(1, 5), replace=False)}
        C = dd.closure(K, S)
        ok7 &= dd.is_complex(C) and S <= C and dd.closure(K, C) == C
        St = dd.star(K, S)
        ok13a &= dd.is_complex(set(K) - St) and all(t in St for s in St for t in K if set(s) <= set(t))
        Lk = dd.link(K, S)
        ok13b &= dd.is_complex(Lk) and not (Lk & St)
    for s in Kl:
        formula = {t for t in K if not (set(t) & set(s)) and tuple(sorted(set(t) | set(s))) in K}
        ok13c &= dd.link(K, [s]) == formula
    for i in sorted({v for f in F for v in f}):
        Lk, St = dd.link(K, i), dd.star(K, i)
        image = {tuple(sorted(t + (i,))) for t in Lk}
        ok14 &= image == St - {(i,)} and dd.closure(K, St) == St | Lk and not (St & Lk)
        ok14 &= sum(len(t) == 1 for t in Lk) == sum(len(t) == 2 for t in St) \
            and sum(len(t) == 2 for t in Lk) == sum(len(t) == 3 for t in St)
    check(f"Proposition 33.2.7: Cl(S) is the smallest subcomplex, idempotent ({name}, 25 random S)", ok7)
    check(f"Proposition 33.2.13(a): K minus St(S) is a subcomplex, St(S) upward closed ({name})", ok13a)
    check(f"Proposition 33.2.13(b): Lk(S) is a subcomplex disjoint from St(S) ({name})", ok13b)
    check(f"Proposition 33.2.13(c): Lk(sigma) = {{tau : tau cap sigma = 0, tau cup sigma in K}} ({name})", ok13c)
    check(f"Proposition 33.2.14 / (33.2.3): star = cone over the link, counts match ({name})", ok14)

# ---------------------------------------------------------------- Example 33.2.16, Non-example 33.2.17
check("Example 33.2.16: complexes generated by face lists are pure 2-complexes",
      all(dd.is_pure(dd.complex_from_faces(F), 2) for F in meshes.values()))
Kn = dd.generate_complex([(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5), (4, 6), (7,)])
check("Non-example 33.2.17: Figure 33.1.2(a) is not pure", not dd.is_pure(Kn))

# ---------------------------------------------------------------- Example 33.2.20 (Figure 33.2.2(a))
xs = np.arange(5.0)
Vg, Fg = dm.grid_mesh(xs, xs, pattern="same")
sq = lambda f: (int(Vg[list(f)][:, 0].min()), int(Vg[list(f)][:, 1].min()))  # noqa: E731
Kp = dd.generate_complex([tuple(int(v) for v in f) for f in Fg if sq(f)[1] < 2 or sq(f)[0] < 2])
bd, it = dd.boundary_simplices(Kp), dd.interior(Kp)
check("Example 33.2.20: K' has (21, 44, 24), chi = 1", [len(dd.k_simplices(Kp, k)) for k in range(3)] == [21, 44, 24])
check("Example 33.2.20: bd(K') is one cycle of 16 vertices and 16 edges",
      [len(dd.k_simplices(bd, k)) for k in range(2)] == [16, 16] and dm.link_type([s for s in bd if len(s) == 2])
      == "cycle")
check("Example 33.2.20: int(K') has (5, 28, 24) and is not a subcomplex",
      [len(dd.k_simplices(it, k)) for k in range(3)] == [5, 28, 24] and not dd.is_complex(it))

# ---------------------------------------------------------------- remark after Definition 33.2.19: Crane's boundary


def bd_crane(Kp):
    """closure of the simplices that are proper faces of exactly one simplex of Kp [Cr §2.2]."""
    Kp = set(Kp)
    once = [t for t in Kp if sum(1 for s in Kp if len(s) > len(t) and set(t) <= set(s)) == 1]
    return set(dd.closure(Kp, once))


ok = bd_crane(Kp) == set(bd)
for trial in range(60):
    k = int(rng.integers(1, 4))
    tops = {tuple(sorted(rng.choice(8, size=k + 1, replace=False).tolist())) for _ in range(rng.integers(1, 6))}
    Kr = dd.generate_complex(tops)
    ok &= bd_crane(Kr) == set(dd.boundary_simplices(Kr))
check("Remark after Definition 33.2.19: Crane's boundary = closure of B (L-shape + 60 random pure complexes)", ok)

# ---------------------------------------------------------------- Non-example 33.2.21
check("Non-example 33.2.21: the octahedron has empty boundary", dd.boundary_simplices(Ko) == frozenset())
f0 = tuple(sorted(Fo[0]))
one = dd.closure(Ko, [f0])
check("Non-example 33.2.21: one triangle as a pure subcomplex has bd = its 3 edges and 3 vertices",
      dd.boundary_simplices(one) == one - {f0})

# ---------------------------------------------------------------- Proposition 33.2.22
ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    ef = dm.edge_faces(np.asarray(F))
    for i in sorted({v for f in F for v in f}):
        C = dd.closure(K, dd.star(K, i))
        lk_edges = [s for s in dd.link(K, i) if len(s) == 2]
        single = [e for e, fs in ef.items() if i in e and len(fs) == 1]
        ok &= dd.is_pure(C, 2) and dd.boundary_simplices(C) == dd.closure(K, lk_edges + single)
        if not single:
            ok &= dd.boundary_simplices(C) == dd.link(K, i)
check("Proposition 33.2.22: bd(Cl(St(i))) = Cl(link edges + edges ij in one triangle), = Lk(i) if none (all meshes)",
      ok)
check("Proposition 33.2.22 / Fig 33.2.2(b): octahedron bd(Cl(St(4))) = Lk(4)",
      dd.boundary_simplices(dd.closure(Ko, St4)) == Lk4)

# ---------------------------------------------------------------- Implementation box
check("Implementation 33.2: code-33-2-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-2-1") <= 30)
ns, same = run_printed("code-33-2-1", "out-33-2-1")
check("Implementation 33.2: code-33-2-1 prints exactly out-33-2-1", same)
ok = True
for name, F in meshes.items():
    K = set(dd.complex_from_faces(F))
    Kl = sorted(K)
    for trial in range(15):
        S = {Kl[i] for i in rng.choice(len(Kl), size=rng.integers(1, 4), replace=False)}
        ok &= ns["closure"](K, S) == set(dd.closure(K, S)) and ns["star"](K, S) == set(dd.star(K, S)) \
            and ns["link"](K, S) == set(dd.link(K, S))
check("Implementation 33.2: closure/star/link agree with dgdec on random S (5 meshes)", ok)

# ---------------------------------------------------------------- Exercises
Kt = dd.generate_complex([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
check("Exercise 33.2.1(a): S = {01}: St, Cl, Lk",
      dd.star(Kt, [(0, 1)]) == {(0, 1), (0, 1, 2), (0, 1, 3)} and dd.closure(Kt, [(0, 1)]) == {(0, 1), (0,), (1,)}
      and dd.link(Kt, [(0, 1)]) == {(2,), (3,)})
check("Exercise 33.2.1(b): S = {012}: St = itself, Cl has 7, Lk empty",
      dd.star(Kt, [(0, 1, 2)]) == {(0, 1, 2)} and len(dd.closure(Kt, [(0, 1, 2)])) == 7
      and dd.link(Kt, [(0, 1, 2)]) == frozenset())

Kg = dd.complex_from_faces(Fg)
idx = lambda a, b: a * 5 + b  # noqa: E731


def link_edges(v):
    return {tuple(sorted(s)) for s in dd.link(Kg, v) if len(s) == 2}


E = lambda p, q: tuple(sorted((idx(*p), idx(*q))))  # noqa: E731
check("Exercise 33.2.2(a): link of (0,0) is the path (1,0)-(1,1)-(0,1)",
      link_edges(idx(0, 0)) == {E((1, 0), (1, 1)), E((1, 1), (0, 1))} and dd.vertex_link_type(Kg, idx(0, 0)) == "path")
check("Exercise 33.2.2(a): link of (4,0) is the single edge (3,0)-(4,1)",
      link_edges(idx(4, 0)) == {E((3, 0), (4, 1))} and dd.vertex_link_type(Kg, idx(4, 0)) == "path")
check("Exercise 33.2.2(b): link of (2,0) is the path (1,0)-(2,1)-(3,1)-(3,0)",
      link_edges(idx(2, 0)) == {E((1, 0), (2, 1)), E((2, 1), (3, 1)), E((3, 1), (3, 0))})
cyc = [(3, 2), (3, 3), (2, 3), (1, 2), (1, 1), (2, 1)]
check("Exercise 33.2.2(c): link of (2,2) is the 6-cycle (3,2)-(3,3)-(2,3)-(1,2)-(1,1)-(2,1)",
      link_edges(idx(2, 2)) == {E(cyc[k], cyc[(k + 1) % 6]) for k in range(6)}
      and dd.vertex_link_type(Kg, idx(2, 2)) == "cycle")

LkU = dd.link(Kt, [(0,), (1,)])
check("Exercise 33.2.3: Lk({0},{1}) = {2, 3, 23} differs from Lk(0) cup Lk(1)",
      LkU == {(2,), (3,), (2, 3)} and LkU != dd.link(Kt, 0) | dd.link(Kt, 1))
ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    Kl = sorted(K)
    for trial in range(10):
        S = {Kl[i] for i in rng.choice(len(Kl), size=2, replace=False)}
        Tt = {Kl[i] for i in rng.choice(len(Kl), size=2, replace=False)}
        ok &= dd.star(K, S | Tt) == dd.star(K, S) | dd.star(K, Tt) \
            and dd.closure(K, S | Tt) == dd.closure(K, S) | dd.closure(K, Tt)
check("Exercise 33.2.3: St and Cl distribute over unions (random S, T)", ok)

ok = True
for name, F in meshes.items():
    K = dd.complex_from_faces(F)
    for (i, j) in dd.k_simplices(K, 1):
        for a, b in ((i, j), (j, i)):
            L = dd.link(K, a)
            ok &= dd.link(K, [(i, j)]) == dd.link(L, b)
check("Exercise 33.2.4: Lk({i,j}) = Lk_{Lk(i)}(j) for every edge and both orders (5 meshes)", ok)
check("Exercise 33.2.4: octahedron Lk_{Lk(4)}(0) = {2, 3}", dd.link(Lk4, 0) == {(2,), (3,)})

Kall = dd.complex_from_faces(Fo)
check("Exercise 33.2.5(a): St(all vertices) = K is a subcomplex",
      dd.star(Kall, [(v,) for v in range(6)]) == Kall)
Kiso = dd.generate_complex([(0, 1, 2), (3,)])
check("Exercise 33.2.5(a,b): an isolated vertex has St = {itself} (a subcomplex) and empty link",
      dd.star(Kiso, 3) == {(3,)} and dd.link(Kiso, 3) == frozenset())

ns, same = run_printed("code-ex-33-2-6", "out-ex-33-2-6")
check("Exercise 33.2.6 (coding): code-ex-33-2-6 prints exactly out-ex-33-2-6", same)

summary()
