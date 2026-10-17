"""Verification for Section 33.1 (Simplices and Simplicial Complexes).

Run from the project root:
``PYTHONPATH=tools python3 chapters/33-combinatorial-surfaces/verify/v-33-1-simplicial-complexes.py``
"""

import contextlib
import io
import itertools
import pathlib
import sys
from math import comb

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch33  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "33-1-simplicial-complexes.html"
rng = np.random.default_rng(3301)


def run_printed(block, out_block, namespace=None):
    """Run <pre id=block> with dgcode.run_block; return (namespace, stdout == the text of <pre id=out_block>)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = dgcode.run_block(PAGE, block, namespace)
    lines = lambda s: [ln.rstrip() for ln in s.strip().splitlines()]  # noqa: E731
    return ns, lines(buf.getvalue()) == lines(dgcode.code_block(PAGE, out_block))


def aff_indep(P):
    P = np.asarray(P, float)
    return np.linalg.matrix_rank(P[1:] - P[0]) == len(P) - 1


# ---------------------------------------------------------------- Example 33.1.3, Non-example 33.1.4
check("Example 33.1.3(b): det[(1,0),(0.35,0.9)] = 0.9", abs(np.linalg.det(np.array([[1, 0.35], [0, 0.9]])) - 0.9) < 1e-14)
check("Example 33.1.3(b): (0,0),(1,0),(0.35,0.9) affinely independent", aff_indep([[0, 0], [1, 0], [0.35, 0.9]]))
check("Example 33.1.3(c): 0, e1, e2, e3 affinely independent", aff_indep(np.vstack([np.zeros(3), np.eye(3)])))
P = np.array([[0, 0], [1, 1], [2, 2]], float)
mu = np.array([1, -2, 1.0])
check("Non-example 33.1.4(a): mu = (1,-2,1) satisfies (33.1.2)", np.allclose(mu @ P, 0) and mu.sum() == 0)
for n in (2, 3, 5):
    Q = rng.normal(size=(n + 2, n))
    check(f"Non-example 33.1.4(b): {n + 2} random points in R^{n} are affinely dependent", not aff_indep(Q))

# Lemma 33.1.2: the base point does not matter (random check)
for trial in range(5):
    k = rng.integers(1, 4)
    X = rng.normal(size=(k + 1, 3))
    ok = all(aff_indep(X[list(perm)]) == aff_indep(X) for perm in itertools.permutations(range(k + 1)))
    check(f"Lemma 33.1.2: affine independence is invariant under reordering (trial {trial})", ok)

# ---------------------------------------------------------------- Example 33.1.6: standard simplex
lam = rng.dirichlet(np.ones(3), size=20)
check("Example 33.1.6: points of Delta^2 have coordinates = barycentric coordinates",
      np.allclose(dd.barycentric_coordinates(lam, np.eye(3)), lam))

# ---------------------------------------------------------------- Proposition 33.1.8, Example 33.1.10, Non-example 33.1.11
X = np.array([[0.0, 0.0], [1.0, 0.0], [0.35, 0.9]])
p = np.array([0.25, 0.4, 0.35]) @ X
close("Example 33.1.10(b): p = (0.5225, 0.315)", p, [0.5225, 0.315], tol=1e-14)
close("Example 33.1.10(b): barycentric coordinates (0.25, 0.4, 0.35)", dd.barycentric_coordinates(p, X),
      [0.25, 0.4, 0.35], tol=1e-12)
close("Non-example 33.1.11: q = (1, 0.9) has coefficients (-0.65, 0.65, 1)",
      dd.barycentric_coordinates([1.0, 0.9], X), [-0.65, 0.65, 1.0], tol=1e-12)
# uniqueness: solving the linear system twice from random convex combinations recovers the weights
W = rng.dirichlet(np.ones(3), size=50)
check("Proposition 33.1.8: barycentric coordinates are unique (recovered from the point)",
      np.allclose(dd.barycentric_coordinates(W @ X, X), W))

# ---------------------------------------------------------------- Example 33.1.13: faces of a k-simplex
for k in range(6):
    fs = dd.faces(tuple(range(k + 1)))
    ok = len(fs) == 2 ** (k + 1) - 1 and all(sum(len(f) == m + 1 for f in fs) == comb(k + 1, m + 1)
                                             for m in range(k + 1))
    check(f"Example 33.1.13: a {k}-simplex has 2^{k + 1} - 1 = {2 ** (k + 1) - 1} faces, C(k+1, m+1) of dim m", ok)

# ---------------------------------------------------------------- Example 33.1.16: geometric complexes
V, F = ch33.tetrahedron()
check("Example 33.1.16(b): regular tetrahedron: 4 vertices, 6 edges, 4 triangles", ch33.counts(F) == (4, 6, 4))
check("Example 33.1.16(b): each pair of triangles shares exactly an edge",
      all(len(set(F[a]) & set(F[b])) == 2 for a in range(4) for b in range(a + 1, 4)))
check("Example 33.1.16(c): icosphere(0) has (12, 30, 20)", ch33.counts(dm.icosphere(0)[1]) == (12, 30, 20))
Vo, Fo = ch33.octahedron()
check("Example 33.1.16(d): octahedron has (6, 12, 8), faces outward",
      ch33.counts(Fo) == (6, 12, 8) and dm.signed_volume(Vo, Fo) > 0 and dm.oriented_consistently(Fo))
for name, (V_, F_) in (("tetrahedron", ch33.tetrahedron()), ("octahedron", ch33.octahedron()),
                       ("icosahedron", dm.icosphere(0))):
    check(f"Example 33.1.16: {name}: vertices affinely independent in every triangle",
          all(aff_indep(V_[list(f)]) for f in F_))

# Figure 33.1.2 and Non-example 33.1.17 (the figure script asserts the same; repeated here)
top = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.4]])
bl = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, -1.0]])
br = np.array([[1.0, 0.0], [2.0, 0.0], [1.5, -1.0]])
check("Non-example 33.1.17 (Fig 33.1.2(b)): T-junction breaks condition (2)",
      not ch33.meet_is_common_face(top, bl) and not ch33.meet_is_common_face(top, br))
t1 = np.array([[0.0, 0.0], [2.0, 0.0], [0.9, 1.4]])
t2 = np.array([[1.1, 0.6], [2.4, 1.6], [0.6, 1.9]])
check("Non-example 33.1.17 (Fig 33.1.2(c)): overlap has positive area",
      ch33.poly_area(ch33.clip(ch33.ccw(t1), ch33.ccw(t2))) > 0.05)

# ---------------------------------------------------------------- Example 33.1.20, Non-example 33.1.21
K = dd.generate_complex([(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)])
proper = {s for m in range(1, 4) for s in itertools.combinations(range(4), m)}
check("Example 33.1.20(a): boundary of the tetrahedron = 14 proper nonempty subsets", K == proper and len(K) == 14)
Fsh = F[rng.permutation(len(F))]
Fsh = np.array([np.roll(r, rng.integers(3))[::rng.choice([-1, 1])] for r in Fsh])
check("Example 33.1.20(b): the complex generated by F ignores row order and order within rows",
      dd.complex_from_faces(Fsh) == dd.complex_from_faces(F))
G = dd.generate_complex([(0, 1), (1, 2), (2, 0), (2, 3), (4,)])
check("Example 33.1.20(c): a graph is a 1-dimensional complex", dd.is_complex(G) and dd.dimension(G) == 1)
check("Non-example 33.1.21(a): {012, 01, 0, 1, 2} is not a complex",
      not dd.is_complex({(0, 1, 2), (0, 1), (0,), (1,), (2,)}))

# ---------------------------------------------------------------- Proposition 33.1.22: realization in R^{n0}
n0 = 4
simp = sorted(K, key=lambda s: (len(s), s))


def in_real(pt, s, tol=1e-12):
    """(ii) of the proof: coordinates >= 0, sum 1, zero outside s."""
    return np.all(pt >= -tol) and abs(pt.sum() - 1) < tol and all(abs(pt[m]) < tol for m in range(n0) if m not in s)


def sample(s, size):
    pts = np.zeros((size, n0))
    pts[:, list(s)] = rng.dirichlet(np.ones(len(s)), size=size)
    return pts


ok_ii = all(in_real(pt, s) for s in simp for pt in sample(s, 20))
check("Proposition 33.1.22 (ii): sampled points of |sigma| satisfy the coordinate description", ok_ii)
ok_iv = True
for s in simp:
    for t in simp:
        common = set(s) & set(t)
        pts = sample(s, 30)
        for pt in pts:
            in_both = in_real(pt, s) and in_real(pt, t)
            in_common = bool(common) and in_real(pt, tuple(common))
            ok_iv &= (in_both == in_common)
check("Proposition 33.1.22 (iv): |sigma| cap |tau| = |sigma cap tau| on sampled points (all pairs)", ok_iv)
ok_v = all(aff_indep(np.eye(n0)[list(s)]) for s in simp if len(s) > 1)
check("Proposition 33.1.22 (i): {e_m : v_m in sigma} affinely independent for every simplex", ok_v)

# ---------------------------------------------------------------- Implementation box
check("Implementation 33.1: code-33-1-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-33-1-1") <= 30)
ns, same = run_printed("code-33-1-1", "out-33-1-1")
check("Implementation 33.1: code-33-1-1 prints exactly out-33-1-1", same)
for name, F_ in (("icosphere(1)", dm.icosphere(1)[1]), ("torus 6x8", dm.torus_mesh(2, 0.8, 6, 8)[1]),
                 ("mobius", dm.mobius_mesh(12)[1])):
    mine = ns["generate_complex"]([tuple(int(v) for v in r) for r in F_])
    check(f"Implementation 33.1: generate_complex = dgdec.complex_from_faces ({name})",
          mine == set(dd.complex_from_faces(F_)) and ns["is_complex"](mine))
Kbad = set(dd.complex_from_faces(dm.icosphere(0)[1]))
Kbad.discard((0, 1))
check("Implementation 33.1: is_complex detects a removed edge", not ns["is_complex"](Kbad) and not dd.is_complex(Kbad))

# ---------------------------------------------------------------- Exercises
Pa = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
check("Exercise 33.1.1(a): the four points are affinely dependent, mu = (1,-1,-1,1)",
      not aff_indep(Pa) and np.allclose(np.array([1, -1, -1, 1.0]) @ Pa, 0))
Xb = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
close("Exercise 33.1.1(b): p = (0.5, 1) -> (0.5, 0.25, 0.25)", dd.barycentric_coordinates([0.5, 1.0], Xb),
      [0.5, 0.25, 0.25], tol=1e-12)
close("Exercise 33.1.1(b): q = (2, 2) -> (-0.5, 1, 0.5)", dd.barycentric_coordinates([2.0, 2.0], Xb),
      [-0.5, 1.0, 0.5], tol=1e-12)

K4 = {s for m in range(1, 5) for s in itertools.combinations(range(5), m)}
nk = [sum(len(s) == k + 1 for s in K4) for k in range(4)]
check("Exercise 33.1.2: boundary of the 4-simplex is a complex with (5, 10, 10, 5), alternating sum 0",
      dd.is_complex(K4) and nk == [5, 10, 10, 5] and nk[0] - nk[1] + nk[2] - nk[3] == 0)


def signed_area(a, b, c):
    return 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


ok = True
for _ in range(50):
    T = rng.normal(size=(3, 2))
    w = rng.normal(size=3)
    w = w / w.sum()
    q = w @ T
    A = signed_area(*T)
    ok &= np.allclose([signed_area(q, T[1], T[2]) / A, signed_area(T[0], q, T[2]) / A,
                       signed_area(T[0], T[1], q) / A], w)
check("Exercise 33.1.3: lambda_i = signed area ratio (50 random triangles and affine combinations)", ok)
Xf = np.array([[0.0, 0.0], [1.0, 0.0], [0.35, 0.9]])
pf = np.array([0.25, 0.4, 0.35]) @ Xf
Af = signed_area(*Xf)
close("Exercise 33.1.3 / Fig 33.1.1(b): piece areas are (0.25, 0.4, 0.35) A",
      [signed_area(pf, Xf[1], Xf[2]) / Af, signed_area(Xf[0], pf, Xf[2]) / Af, signed_area(Xf[0], Xf[1], pf) / Af],
      [0.25, 0.4, 0.35], tol=1e-12)

pts = {"a": (0, 0), "b": (1, 0), "c": (2, 0), "t": (1, 1.4), "u": (0.5, -1), "w": (1.5, -1)}
tris = [("a", "b", "t"), ("b", "c", "t"), ("a", "b", "u"), ("b", "c", "w")]
arr = {k: np.array(v, float) for k, v in pts.items()}
ok = all(ch33.meet_is_common_face(np.array([arr[v] for v in s]), np.array([arr[v] for v in t]))
         for s, t in itertools.combinations(tris, 2))
idx = {k: i for i, k in enumerate(pts)}
Kx = dd.generate_complex([tuple(idx[v] for v in s) for s in tris])
check("Exercise 33.1.4: the repaired T-junction is a simplicial complex with (6, 9, 4)",
      ok and [len(dd.k_simplices(Kx, k)) for k in range(3)] == [6, 9, 4])

ns, same = run_printed("code-ex-33-1-6", "out-ex-33-1-6")
check("Exercise 33.1.6 (coding): code-ex-33-1-6 prints exactly out-ex-33-1-6", same)

summary()
