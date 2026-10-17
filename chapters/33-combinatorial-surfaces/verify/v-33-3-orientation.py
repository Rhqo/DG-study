"""Verification for Section 33.3 (Orientation).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/verify/v-33-3-orientation.py``
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

PAGE = HERE / "33-3-orientation.html"
rng = np.random.default_rng(3303)


def run_printed(block, out_block, namespace=None):
    """Run <pre id=block> with dgcode.run_block; return (namespace, stdout == the text of <pre id=out_block>)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(PAGE, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(PAGE, out_block))


def inversions(p):
    return sum(1 for a in range(len(p)) for b in range(a + 1, len(p)) if p[a] > p[b])


def perm_matrix(p):
    n = len(p)
    P = np.zeros((n, n))
    P[list(p), range(n)] = 1.0          # column j is e_{p(j)}
    return P


def consistent(rows):
    """Every edge shared by two rows is traversed in opposite directions (Corollary 33.3.8)."""
    return dm.oriented_consistently(np.asarray(rows))


# ---------------------------------------------------------------- Lemma 33.3.1
ok = True
for n in range(1, 6):
    for p in itertools.permutations(range(n)):
        s = dd.permutation_sign(p)
        ok &= s == (-1) ** inversions(p) == round(np.linalg.det(perm_matrix(p)))
check("Lemma 33.3.1(c): sgn = (-1)^inv = det P_pi for all permutations of up to 5 symbols", ok)
ok = True
for n in (3, 4, 5):
    perms = list(itertools.permutations(range(n)))
    for _ in range(200):
        p, q = perms[rng.integers(len(perms))], perms[rng.integers(len(perms))]
        comp = tuple(p[q[i]] for i in range(n))
        ok &= dd.permutation_sign(comp) == dd.permutation_sign(p) * dd.permutation_sign(q)
check("Lemma 33.3.1(a): sgn is multiplicative (random pairs, n = 3, 4, 5)", ok)
check("Lemma 33.3.1(b): every transposition is odd",
      all(dd.permutation_sign(tuple(b if i == a else a if i == b else i for i in range(5))) == -1
          for a in range(5) for b in range(a + 1, 5)))

# ---------------------------------------------------------------- Example 33.3.3, misconception box
check("Example 33.3.3(b): cyclic shifts of a triangle are even, a swap is odd",
      dd.same_orientation((0, 1, 2), (1, 2, 0)) and dd.same_orientation((0, 1, 2), (2, 0, 1))
      and not dd.same_orientation((0, 1, 2), (0, 2, 1)))
check("Example 33.3.3(c): [1,0,3,2] = [0,1,2,3], [0,2,1,3] = -[0,1,2,3]",
      dd.same_orientation((0, 1, 2, 3), (1, 0, 3, 2)) and not dd.same_orientation((0, 1, 2, 3), (0, 2, 1, 3)))
check("Common misconception: a cyclic shift of a k-simplex has sign (-1)^k (k = 1..7)",
      all(dd.permutation_sign(tuple(range(1, k + 1)) + (0,)) == (-1) ** k for k in range(1, 8)))

# ---------------------------------------------------------------- Proposition 33.3.4, Example 33.3.5
D = lambda X: np.linalg.det((X[1:] - X[0]).T)  # noqa: E731
ok = True
for k in (1, 2, 3, 4):
    X = rng.normal(size=(k + 1, k))
    for p in itertools.permutations(range(k + 1)):
        ok &= np.isclose(D(X[list(p)]), dd.permutation_sign(p) * D(X))
check("Proposition 33.3.4: D(x_pi) = sgn(pi) D(x) for all permutations, k = 1..4 (random points)", ok)
X2 = np.array([[0, 0], [1, 0], [0, 1]], float)
check("Example 33.3.5(a): [x0,x1,x2] has D = 1, [x0,x2,x1] has D = -1",
      np.isclose(D(X2), 1) and np.isclose(D(X2[[0, 2, 1]]), -1))
X3 = np.vstack([np.zeros(3), np.eye(3)])
check("Example 33.3.5(b): [0, e1, e2, e3] has D = 1", np.isclose(D(X3), 1))

# ---------------------------------------------------------------- Proposition 33.3.7, Corollary 33.3.8
ok = True
for k in range(1, 6):
    s = tuple(rng.permutation(10)[:k + 1])
    for m in range(k + 1):
        tau = s[:m] + s[m + 1:]
        ok &= dd.induced_orientation(s, tau) == (-1) ** m
        ok &= dd.permutation_sign((s[m],) + tau, s) == (-1) ** m       # [v_m, tau_m] = (-1)^m sigma
check("Proposition 33.3.7: o(sigma, tau_m) = (-1)^m (k = 1..5)", ok)
check("Proposition 33.3.7: [i,j] induces +j and -i",
      dd.induced_orientation((3, 7), (7,)) == 1 and dd.induced_orientation((3, 7), (3,)) == -1)
check("Proposition 33.3.7: [i,j,k] induces [j,k], [k,i], [i,j]",
      all(dd.induced_orientation((0, 1, 2), f) == 1 for f in ((1, 2), (2, 0), (0, 1))))
ok = True
for a, b in itertools.permutations(range(4), 2):
    for c in set(range(4)) - {a, b}:
        for d in set(range(4)) - {a, b, c}:
            s, t = (a, b, c), (b, a, d)
            ok &= dd.relative_orientation(s, t) == 1
            ok &= dd.relative_orientation(s, (a, b, d)) == -1
check("Corollary 33.3.8: across a shared edge, o = +1 iff the edge is traversed in opposite directions", ok)

# ---------------------------------------------------------------- Example 33.3.9
check("Example 33.3.9(a): o([0,1],[1]) = +1, o([1,2],[1]) = -1, o([2,1],[1]) = +1",
      dd.induced_orientation((0, 1), (1,)) == 1 and dd.induced_orientation((1, 2), (1,)) == -1
      and dd.induced_orientation((2, 1), (1,)) == 1)
check("Example 33.3.9(a): o([0,1],[1,2]) = +1, o([0,1],[2,1]) = -1",
      dd.relative_orientation((0, 1), (1, 2)) == 1 and dd.relative_orientation((0, 1), (2, 1)) == -1)
Fm6 = dm.mobius_mesh(6)[1]
check("Example 33.3.9(b): rows t0 = [0,1,6], t1 = [6,1,7] (n = 6, band and Moebius alike)",
      [tuple(int(v) for v in r) for r in Fm6[:2]] == [(0, 1, 6), (6, 1, 7)]
      and [tuple(int(v) for v in r) for r in ch33.cylinder_band(6)[1][:2]] == [(0, 1, 6), (6, 1, 7)])
check("Example 33.3.9(b): o(t0, t1) = +1, o(t0, [6,7,1]) = -1",
      dd.relative_orientation((0, 1, 6), (6, 1, 7)) == 1 and dd.relative_orientation((0, 1, 6), (6, 7, 1)) == -1)
bd = dd.induced_boundary((0, 1, 2, 3))
faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
normals = [np.cross(X3[f[1]] - X3[f[0]], X3[f[2]] - X3[f[0]]) for f in faces]
check("Example 33.3.9(c): induced faces of [0,1,2,3] are [1,2,3], [0,3,2], [0,1,3], [0,2,1]",
      [s for s, _ in bd] == [1, -1, 1, -1] and all(dd.induced_orientation((0, 1, 2, 3), f) == 1 for f in faces))
check("Example 33.3.9(c): their normals are (1,1,1), -e1, -e2, -e3",
      all(np.allclose(n, e) for n, e in zip(normals, [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])))
check("Example 33.3.9(c): the four induced faces are pairwise consistent",
      all(dd.relative_orientation(f, g) == 1 for f, g in itertools.combinations(faces, 2)))

# ---------------------------------------------------------------- Example 33.3.12, 33.3.14
F = dm.icosphere(1)[1]
E, idx = dd.edge_list(F)
check("Example 33.3.12: edge_list is lexicographic with i < j", np.all(E[:, 0] < E[:, 1])
      and [tuple(e) for e in E] == sorted(tuple(e) for e in E))
Vt, Ft = ch33.tetrahedron()
Vo, Fo = ch33.octahedron()
for name, (V, F) in (("tetrahedron", (Vt, Ft)), ("octahedron", (Vo, Fo)), ("icosphere(2)", dm.icosphere(2)),
                     ("torus 8x16", dm.torus_mesh(2, 0.8, 8, 16))):
    check(f"Example 33.3.14(b): {name} face list is consistent and outward",
          consistent(F) and dm.signed_volume(V, F) > 0)
Vg, Fg = dm.grid_mesh(np.arange(5.0), np.arange(4.0))
area = np.cross(Vg[Fg[:, 1]] - Vg[Fg[:, 0]], Vg[Fg[:, 2]] - Vg[Fg[:, 0]])[:, 2]
check("Example 33.3.14(c): grid triangles all counterclockwise and consistent", np.all(area > 0) and consistent(Fg))

# ---------------------------------------------------------------- Non-example 33.3.15: brute force
book = [(0, 1, 2), (0, 1, 3), (0, 1, 4)]
cnt = sum(consistent([f if e > 0 else f[::-1] for f, e in zip(book, eps)])
          for eps in itertools.product((1, -1), repeat=3))
check("Non-example 33.3.15: none of the 8 orientations of the three-page book is consistent", cnt == 0)

# ---------------------------------------------------------------- Propositions 33.3.16, 33.3.17


def chain_product(F):
    F = [tuple(int(v) for v in f) for f in F]
    return int(np.prod([dd.relative_orientation(F[t], F[(t + 1) % len(F)]) for t in range(len(F))]))


ok16 = ok17 = True
for n in range(3, 13):
    Fm = dm.mobius_mesh(n)[1]
    Fb = ch33.cylinder_band(n)[1]
    signs = [dd.relative_orientation(tuple(Fm[t]), tuple(Fm[(t + 1) % len(Fm)])) for t in range(len(Fm))]
    ok17 &= signs.count(-1) == 1 and signs[-1] == -1 and chain_product(Fb) == 1 and consistent(Fb)
    ok17 &= not dm.orient(Fm)[1] and dm.orient(Fb)[1] and dd.is_simplicial_surface(dd.complex_from_faces(Fm))
    # every interior edge of the band appears exactly once in the chain
    ef = dm.edge_faces(Fm)
    interior = {e for e, fs in ef.items() if len(fs) == 2}
    chain_edges = {tuple(sorted(set(Fm[t]) & set(Fm[(t + 1) % len(Fm)]))) for t in range(len(Fm))}
    ok17 &= interior == chain_edges and len(interior) == 2 * n
    for _ in range(5):
        flips = rng.choice([1, -1], size=len(Fm))
        Fr = [f if e > 0 else f[::-1] for f, e in zip(Fm, flips)]
        ok16 &= chain_product(Fr) == -1
check("Proposition 33.3.16: the chain product is independent of the reference orientation (random flips)", ok16)
check("Proposition 33.3.17: Moebius n = 3..12: one -1 at the seam, not orientable; band consistent", ok17)
Fm3 = dm.mobius_mesh(3)[1]
cnt = sum(consistent([f if e > 0 else f[::-1] for f, e in zip(Fm3, eps)])
          for eps in itertools.product((1, -1), repeat=len(Fm3)))
Fb3 = ch33.cylinder_band(3)[1]
cntb = sum(consistent([f if e > 0 else f[::-1] for f, e in zip(Fb3, eps)])
           for eps in itertools.product((1, -1), repeat=len(Fb3)))
check("Proposition 33.3.17: brute force n = 3: Moebius has 0 consistent orientations, the band has 2",
      cnt == 0 and cntb == 2)

# ---------------------------------------------------------------- Proposition 33.3.18 (a): exactly two
for name, F in (("tetrahedron", Ft), ("octahedron", Fo)):
    cnt = sum(consistent([f if e > 0 else f[::-1] for f, e in zip(F, eps)])
              for eps in itertools.product((1, -1), repeat=len(F)))
    check(f"Proposition 33.3.18(a): {name} has exactly 2 consistent orientations (brute force)", cnt == 2)

# ---------------------------------------------------------------- Implementation box
check("Implementation 33.3: code-33-3-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-3-1") <= 30)
ns, same = run_printed("code-33-3-1", "out-33-3-1")
check("Implementation 33.3: code-33-3-1 prints exactly out-33-3-1", same)
orient_faces = ns["orient_faces"]
check("Remark after Proposition 33.3.18: orient_faces reports failure when an edge is in three triangles",
      not orient_faces([[0, 1, 2], [0, 1, 3], [0, 1, 4]])[1]
      and not orient_faces([[0, 1, 2], [2, 1, 3], [0, 3, 1], [1, 4, 0]])[1])
ok = True
for name, F in (("icosphere(2)", dm.icosphere(2)[1]), ("torus 6x10", dm.torus_mesh(2, 0.8, 6, 10)[1]),
                ("grid", Fg), ("band", ch33.cylinder_band(9)[1])):
    for _ in range(5):
        flips = rng.choice([1, -1], size=len(F))
        Fr = [list(f) if e > 0 else list(f)[::-1] for f, e in zip(F, flips)]
        G, okk = orient_faces(Fr)
        same = all(dd.same_orientation(g, f) for g, f in zip(G, F))
        opp = all(not dd.same_orientation(g, f) for g, f in zip(G, F))
        ok &= okk and consistent(G) and (same or opp)
check("Proposition 33.3.18(b): BFS recovers the original orientation or its reverse (random flips, 4 meshes)", ok)
check("Proposition 33.3.18(b): BFS reports failure on the Moebius band and the Klein bottle",
      not orient_faces(dm.mobius_mesh(7)[1].tolist())[1] and not orient_faces(ch33.klein_bottle(5, 6).tolist())[1])

# ---------------------------------------------------------------- Exercises
check("Exercise 33.3.1: signs of (1,0,3,2), (1,2,3,0), (3,2,1,0), (2,0,1,3) are +, -, +, +",
      [dd.permutation_sign(p) for p in ((1, 0, 3, 2), (1, 2, 3, 0), (3, 2, 1, 0), (2, 0, 1, 3))] == [1, -1, 1, 1]
      and [inversions(p) for p in ((1, 0, 3, 2), (1, 2, 3, 0), (3, 2, 1, 0), (2, 0, 1, 3))] == [2, 3, 6, 2])
rows = [(a, b, c) for a in (0, 1) for b in (2, 3) for c in (4, 5)]
neg = [r for r in rows if np.linalg.det(Vo[list(r)]) < 0]
fixed = [r if np.linalg.det(Vo[list(r)]) > 0 else (r[0], r[2], r[1]) for r in rows]
nr = [np.cross(Vo[r[1]] - Vo[r[0]], Vo[r[2]] - Vo[r[0]]) @ Vo[r[0]] for r in rows]
check("Exercise 33.3.2(a): <n, x_a> = det(x_a, x_b, x_c) for every row",
      np.allclose(nr, [np.linalg.det(Vo[list(r)]) for r in rows]))
check("Exercise 33.3.2(b): rows (0,2,5), (0,3,4), (1,2,4), (1,3,5) point inward; flipping them is consistent",
      neg == [(0, 2, 5), (0, 3, 4), (1, 2, 4), (1, 3, 5)] and consistent(fixed) and dm.signed_volume(Vo, np.array(fixed))
      > 0)
ok = True
for F in (dm.icosphere(1)[1], dm.torus_mesh(2, 0.8, 5, 7)[1], dm.mobius_mesh(9)[1]):
    K = dd.complex_from_faces(F)
    for i in range(F.max() + 1):
        if dd.vertex_link_type(K, i) != "cycle":
            continue
        L = [s for s in dd.link(K, i) if len(s) == 2]
        cyc = [L[0][0], L[0][1]]
        while len(cyc) < len(L):
            nxt = [b for a, b in L if a == cyc[-1] and b != cyc[-2]] + [a for a, b in L if b == cyc[-1] and a != cyc[-2]]
            cyc.append(nxt[0])
        fan = [(i, cyc[r], cyc[(r + 1) % len(cyc)]) for r in range(len(cyc))]
        for _ in range(3):
            flips = rng.choice([1, -1], size=len(fan))
            ok &= chain_product([f if e > 0 else f[::-1] for f, e in zip(fan, flips)]) == 1
check("Exercise 33.3.5: the sign product around every vertex fan is +1 (icosphere, torus, Moebius interior)", ok)

ns2, same = run_printed("code-ex-33-3-6", "out-ex-33-3-6", namespace=dict(ns))
check("Exercise 33.3.6 (coding): code-ex-33-3-6 (after code-33-3-1) prints exactly out-ex-33-3-6", same)
check("Exercise 33.3.6: the inline Klein bottle equals lib/ch33.klein_bottle",
      np.array_equal(ns2["klein_bottle"](5, 6), ch33.klein_bottle(5, 6)))

summary()
