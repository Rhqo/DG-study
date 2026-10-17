"""Section 34.1 (The Wedge Product and k-Vectors): verification of every computation (GUIDELINES 13).

Independent model: k-vectors as antisymmetric tensors with the determinant normalization (lib/ch34.py,
permutation signs from sympy). The Implementation box and the coding exercise are extracted from the HTML
and executed as printed.
Run from the project root: PYTHONPATH=tools python3 chapters/34-exterior-algebra/verify/v-34-1-wedge.py
"""
import contextlib
import io
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from dgcheck import check, close, summary  # noqa: E402
import ch34  # noqa: E402

PAGE = "34-1-wedge-k-vectors.html"
rng = np.random.default_rng(341)

# ---------------------------------------------------------------------------
# the independent model itself: components of u_1 ^ ... ^ u_k are the k x k minors
# ---------------------------------------------------------------------------
for n, k in [(3, 2), (4, 2), (4, 3), (5, 3), (5, 5)]:
    us = rng.normal(size=(k, n))
    T = ch34.vec_tensor(*us)
    U = np.column_stack(us)
    ok = all(abs(T[I] - np.linalg.det(U[list(I), :])) < 1e-10 for I in ch34.increasing(n, k))
    check(f"model: tensor components = minors (n={n}, k={k})", ok)

# symbolic version of (34.1.2): sum_sigma sgn(sigma) prod u_sigma(a)^{i_a} = det(U_I)
for n, k in [(3, 2), (4, 3)]:
    Us = sp.Matrix(n, k, lambda i, j: sp.Symbol(f"u{i}{j}"))
    for I in ch34.increasing(n, k):
        perm_sum = sum(ch34.perm_sign(p) * sp.prod([Us[I[a], p[a]] for a in range(k)])
                       for p in itertools.permutations(range(k)))
        check(f"(34.1.2) symbolic, n={n}, k={k}, I={tuple(i + 1 for i in I)}",
              sp.expand(perm_sum - Us.extract(list(I), list(range(k))).det()) == 0)

# ---------------------------------------------------------------------------
# Example 34.1.4 and Example 34.1.7, 34.1.10
# ---------------------------------------------------------------------------
b1, b2 = np.array([2.0, 1.0]), np.array([-1.0, 1.0])
close("Example 34.1.4(a): b1^b2 = 3 e12", ch34.vec_tensor(b1, b2)[0, 1], 3.0, 1e-12)
u, v, w = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 2.0]), np.array([1.0, 0.0, 1.0])
C = ch34.comps(ch34.vec_tensor(u, v))
close("Example 34.1.4(b): u^v = 2 e12 + 5 e13 + 4 e23", [C[(0, 1)], C[(0, 2)], C[(1, 2)]], [2, 5, 4], 1e-12)
close("Example 34.1.10: minors of U", [np.linalg.det(np.array([[1, -1], [2, 0]])),
                                        np.linalg.det(np.array([[1, -1], [3, 2]])),
                                        np.linalg.det(np.array([[2, 0], [3, 2]]))], [2, 5, 4], 1e-12)
close("Example 34.1.10: u^v^w = 6 e123", ch34.vec_tensor(u, v, w)[0, 1, 2], 6.0, 1e-12)
close("Example 34.1.10: u x v = (c23, -c13, c12) = (4, -5, 2)", np.cross(u, v), [4, -5, 2], 1e-12)
close("Example 34.1.10 / Fig 34.1.3: 2^2+5^2+4^2 = |u x v|^2 = 45", 4 + 25 + 16, np.dot(np.cross(u, v), np.cross(u, v)),
      1e-12)
# shadow interpretation: projection onto span(e_i, e_j) has signed area det U_(i,j)
for (i, j) in [(0, 1), (0, 2), (1, 2)]:
    pu, pv = np.array([u[i], u[j]]), np.array([v[i], v[j]])
    close(f"Example 34.1.10: shadow on plane ({i + 1},{j + 1}) has signed area = component",
          pu[0] * pv[1] - pu[1] * pv[0], C[(i, j)], 1e-12)
# (34.1.3) in R^2
x, y = rng.normal(size=(2, 2))
close("(34.1.3): u^v = (u1 v2 - u2 v1) e12", ch34.vec_tensor(x, y)[0, 1], x[0] * y[1] - x[1] * y[0], 1e-12)

# ---------------------------------------------------------------------------
# Proposition 34.1.3: antisymmetry, alternating, graded commutativity
# ---------------------------------------------------------------------------
us = rng.normal(size=(4, 6))
T = ch34.vec_tensor(*us)
T_swapped = ch34.vec_tensor(us[3], us[1], us[2], us[0])
check("Proposition 34.1.3(b): swapping factors 1 and 4 flips the sign", np.allclose(T_swapped, -T))
check("Proposition 34.1.3(b): repeated factor gives 0", np.allclose(ch34.vec_tensor(us[0], us[1], us[0]), 0))
for k, l in [(1, 1), (1, 2), (2, 2), (2, 3), (1, 3)]:
    S = ch34.vec_tensor(*rng.normal(size=(k, 6))) + ch34.vec_tensor(*rng.normal(size=(k, 6)))
    Tt = ch34.vec_tensor(*rng.normal(size=(l, 6)))
    check(f"Proposition 34.1.3(c): xi^eta = (-1)^(kl) eta^xi (k={k}, l={l})",
          np.allclose(ch34.wedge_t(S, Tt), (-1) ** (k * l) * ch34.wedge_t(Tt, S)))
# associativity of the tensor wedge and (W1) for vectors
a, b, c = rng.normal(size=(3, 5))
check("(W1): (a^b)^c = a^(b^c) = a^b^c (tensor model)",
      np.allclose(ch34.wedge_t(ch34.vec_tensor(a, b), c), ch34.wedge_t(a, ch34.vec_tensor(b, c)))
      and np.allclose(ch34.wedge_t(ch34.vec_tensor(a, b), c), ch34.vec_tensor(a, b, c)))

# ---------------------------------------------------------------------------
# Lemma 34.1.5: F(w) = sum_I det(A_I) F(b_I) for an alternating F into a vector space Z
# ---------------------------------------------------------------------------
for m, k in [(3, 2), (4, 2), (4, 3), (5, 4)]:
    M = rng.normal(size=(6, m))                     # F(w_1..w_k) = (M w_1) ^ ... ^ (M w_k) in Lambda^k R^6
    F = lambda *ws: ch34.vec_tensor(*[M @ q for q in ws])  # noqa: E731
    Bm = rng.normal(size=(m, m))                    # basis b_i = columns of Bm
    A = rng.normal(size=(m, k))
    ws = [Bm @ A[:, j] for j in range(k)]
    rhs = sum(np.linalg.det(A[list(I), :]) * F(*[Bm[:, i] for i in I]) for I in ch34.increasing(m, k))
    check(f"Lemma 34.1.5 (34.1.1): m={m}, k={k}", np.allclose(F(*ws), rhs))
m = 2
M = rng.normal(size=(6, m))
check("Lemma 34.1.5: k > m gives F = 0 (m=2, k=3)",
      np.allclose(ch34.vec_tensor(*[M @ q for q in rng.normal(size=(3, m))]), 0))

# ---------------------------------------------------------------------------
# Proposition 34.1.6, 34.1.8, 34.1.9
# ---------------------------------------------------------------------------
for k in (2, 3, 4):
    us = rng.normal(size=(k, 6))
    A = rng.normal(size=(k, k))
    ws = [sum(A[i, j] * us[i] for i in range(k)) for j in range(k)]
    check(f"Proposition 34.1.6(a): w-wedge = det(A) u-wedge (k={k})",
          np.allclose(ch34.vec_tensor(*ws), np.linalg.det(A) * ch34.vec_tensor(*us)))
Um = rng.normal(size=(4, 4))
close("Proposition 34.1.6(c): u1^..^u4 = det(U) e1234", ch34.vec_tensor(*Um.T)[0, 1, 2, 3], np.linalg.det(Um), 1e-10)

us = rng.normal(size=(3, 5))
dep = [us[0], us[1], 2 * us[0] - 3 * us[1]]
check("Proposition 34.1.8: dependent list gives 0", np.allclose(ch34.vec_tensor(*dep), 0))
check("Proposition 34.1.8: independent list gives nonzero", np.abs(ch34.vec_tensor(*us)).max() > 1e-6)
check("Proposition 34.1.8: k > n gives 0 (4 vectors in R^3)", np.allclose(ch34.vec_tensor(*rng.normal(size=(4, 3))), 0))

# Figure 34.1.2 / Proposition 34.1.9: the three pairs, the reversed pair, the degenerate pair
pairs = [(b1, b2), (b1, b1 + b2), (b2, -b1)]
dets = [p[0][0] * p[1][1] - p[0][1] * p[1][0] for p in pairs]
close("Figure 34.1.2(a): all three pairs give 3 e12", dets, [3, 3, 3], 1e-12)
close("Figure 34.1.2(b): (b2, b1) gives -3, (b1, 2 b1) gives 0",
      [b2[0] * b1[1] - b2[1] * b1[0], b1[0] * 2 * b1[1] - b1[1] * 2 * b1[0]], [-3, 0], 1e-12)
for A in ([[1, 1], [0, 1]], [[0, -1], [1, 0]]):
    close(f"Figure 34.1.2(a): change-of-basis matrix {A} has det 1", np.linalg.det(np.array(A, float)), 1.0, 1e-12)
# converse direction of 34.1.9: same span but det A = 2 gives a different 2-vector
p, q = rng.normal(size=(2, 4))
check("Proposition 34.1.9: same plane but det A = 2 gives a different 2-vector",
      not np.allclose(ch34.vec_tensor(p, q), ch34.vec_tensor(2 * p, q)))
r = rng.normal(size=4)
check("Proposition 34.1.9: different plane, equal area, different 2-vector",
      not np.allclose(ch34.vec_tensor(p, q), ch34.vec_tensor(p, r)))

# Figure 34.1.1 numbers
uf, vf, wf = np.array([1.0, 0.15, 0.0]), np.array([0.3, 0.9, 0.0]), np.array([0.15, 0.2, 0.85])
close("Figure 34.1.1: |u x v| = 0.855", np.linalg.norm(np.cross(uf, vf)), 0.855, 5e-4)
close("Figure 34.1.1: det(u, v, w) = 0.727", np.linalg.det(np.column_stack([uf, vf, wf])), 0.727, 5e-4)

# ---------------------------------------------------------------------------
# Theorem 34.1.11: dim Lambda^k R^n = C(n, k), independence via e_J ^ e_{J^c}
# ---------------------------------------------------------------------------
for n in range(1, 7):
    for k in range(0, n + 1):
        if k == 0:
            continue
        samples = [ch34.comps(ch34.vec_tensor(*rng.normal(size=(k, n)))) for _ in range(3 * math.comb(n, k) + 3)]
        mat = np.array([[s[I] for I in ch34.increasing(n, k)] for s in samples])
        check(f"Theorem 34.1.11: simple {k}-vectors span a space of dim C({n},{k}) = {math.comb(n, k)}",
              np.linalg.matrix_rank(mat) == math.comb(n, k))
n = 5
E = np.eye(n)
for J in ch34.increasing(n, 2):
    Jc = tuple(sorted(set(range(n)) - set(J)))
    eJ = ch34.vec_tensor(*[E[i] for i in J])
    zero_others = all(np.allclose(ch34.wedge_t(ch34.vec_tensor(*[E[i] for i in I]), ch34.vec_tensor(*[E[i] for i in Jc])), 0)
                      for I in ch34.increasing(n, 2) if I != J)
    top = ch34.wedge_t(eJ, ch34.vec_tensor(*[E[i] for i in Jc]))[tuple(range(n))]
    check(f"Theorem 34.1.11 proof: e_I ^ e_(J^c) = 0 for I != J and e_J ^ e_(J^c) = +-e12345, J={tuple(j + 1 for j in J)}",
          zero_others and abs(abs(top) - 1) < 1e-12)

# ---------------------------------------------------------------------------
# Non-example 34.1.13
# ---------------------------------------------------------------------------
E4 = np.eye(4)
xi = ch34.vec_tensor(E4[0], E4[1]) + ch34.vec_tensor(E4[2], E4[3])
close("Non-example 34.1.13: xi ^ xi = 2 e1234", ch34.wedge_t(xi, xi)[0, 1, 2, 3], 2.0, 1e-12)
s = ch34.vec_tensor(*rng.normal(size=(2, 4)))
check("Non-example 34.1.13: simple 2-vector has xi ^ xi = 0", np.allclose(ch34.wedge_t(s, s), 0))

# ---------------------------------------------------------------------------
# Smooth vs. Discrete box: relabeling a triangle
# ---------------------------------------------------------------------------
x0, x1, x2 = rng.normal(size=(3, 3))
base = ch34.vec_tensor(x1 - x0, x2 - x0)
check("Discrete box: cyclic relabeling (x1, x2, x0) gives the same 2-vector",
      np.allclose(ch34.vec_tensor(x2 - x1, x0 - x1), base))
check("Discrete box: swapping (x1, x0, x2) gives the negative",
      np.allclose(ch34.vec_tensor(x0 - x1, x2 - x1), -base))
close("Discrete box: |(x1-x0)^(x2-x0)| = 2 * triangle area",
      math.sqrt(ch34.inner_t(base, base)), 2 * 0.5 * np.linalg.norm(np.cross(x1 - x0, x2 - x0)), 1e-12)

# ---------------------------------------------------------------------------
# Implementation box (run as printed)
# ---------------------------------------------------------------------------
ns = ch34.run_block(PAGE, "code-34-1-1")
wv = ns["wedge_vectors"]
close("Implementation box: uv = {(0,1): 2, (0,2): 5, (1,2): 4}", [ns["uv"][(0, 1)], ns["uv"][(0, 2)], ns["uv"][(1, 2)]],
      [2, 5, 4], 1e-12)
close("Implementation box: uvw = {(0,1,2): 6}", ns["uvw"][(0, 1, 2)], 6.0, 1e-12)
for n, k in [(3, 2), (4, 2), (5, 3), (6, 4)]:
    us = rng.normal(size=(k, n))
    got = wv(*us)
    ref = ch34.comps(ch34.vec_tensor(*us))
    check(f"Implementation box agrees with the tensor model (n={n}, k={k})",
          set(got) == set(ref) and all(abs(got[I] - ref[I]) < 1e-10 for I in ref))
check("Implementation box: k > n gives {}", wv(*rng.normal(size=(4, 3))) == {})

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
e1, e2, e3 = np.eye(3)
Cx = ch34.comps(ch34.vec_tensor(e1 - e3, 2 * e2 + e3))
close("Exercise 34.1.1(a): 2 e12 + e13 + 2 e23", [Cx[(0, 1)], Cx[(0, 2)], Cx[(1, 2)]], [2, 1, 2], 1e-12)
close("Exercise 34.1.1(b): 2 e123", ch34.vec_tensor(e1 + e2, e2 + e3, e3 + e1)[0, 1, 2], 2.0, 1e-12)
close("Exercise 34.1.1(b): det check", np.linalg.det(np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1]], float)), 2.0, 1e-12)
uu, vv = rng.normal(size=(2, 3))
check("Exercise 34.1.2(c): (u+v)^(u-v) = -2 u^v",
      np.allclose(ch34.vec_tensor(uu + vv, uu - vv), -2 * ch34.vec_tensor(uu, vv)))
check("Exercise 34.1.2(d): three vectors in R^2 wedge to 0", np.allclose(ch34.vec_tensor(*rng.normal(size=(3, 2))), 0))
close("Exercise 34.1.3: det(u, v, (1,1,1)) = 1", np.linalg.det(np.column_stack([u, v, [1, 1, 1]])), 1.0, 1e-12)
close("Exercise 34.1.3: det(u, v, (1,4,8)) = 0", np.linalg.det(np.column_stack([u, v, [1, 4, 8]])), 0.0, 1e-12)
close("Exercise 34.1.3: (1,4,8) = 2u + v", 2 * u + v, [1, 4, 8], 1e-12)
# the printed cofactor expansion along the first row (reviewer fix: the middle term is 1*(16 - 12))
M2 = np.column_stack([u, v, [1, 4, 8]])
cof = [M2[0, 0] * (M2[1, 1] * M2[2, 2] - M2[1, 2] * M2[2, 1]),
       -M2[0, 1] * (M2[1, 0] * M2[2, 2] - M2[1, 2] * M2[2, 0]),
       M2[0, 2] * (M2[1, 0] * M2[2, 1] - M2[1, 1] * M2[2, 0])]
close("Exercise 34.1.3: printed terms 1*(0-8), 1*(16-12), 1*(4-0)", cof, [1 * (0 - 8), 1 * (16 - 12), 1 * (4 - 0)], 1e-12)

# Non-example 34.1.2(c), (d) (added in review) -------------------------------------------------------
E3 = np.eye(3)
left_d = ch34.wedge_t(ch34.vec_tensor(E3[0], E3[1]), E3[2])        # (e1^e2)^e3: Lambda^2 x Lambda^1 unchanged
right_d = -ch34.wedge_t(E3[0], ch34.vec_tensor(E3[1], E3[2]))      # e1^(e2^e3) with Lambda^1 x Lambda^2 negated
check("Non-example 34.1.2(d): twisted product gives (e1^e2)^e3 = e123 but e1^(e2^e3) = -e123 (W1 fails, W4 holds)",
      np.isclose(left_d[0, 1, 2], 1.0) and np.isclose(right_d[0, 1, 2], -1.0))
rows_c = [np.append([ch34.vec_tensor(*rng.normal(size=(2, 4)))[I] for I in ch34.increasing(4, 2)], 0.0)
          for _ in range(12)]
check("Non-example 34.1.2(c): simple 2-vectors span only 6 of the 7 dimensions of Lambda^2 R^4 + R (W3 fails)",
      np.linalg.matrix_rank(np.array(rows_c)) == 6)
aa, bb, cc, dd = sp.symbols("a b c d")
P_, Q_ = sp.symbols("p q")
# (au+bv)^(cu+dv) coefficient of u^v by the rules: ac*0 + ad - bc + bd*0
check("Exercise 34.1.4(b): coefficient is ad - bc",
      sp.expand(sp.Matrix([[aa, cc], [bb, dd]]).det() - (aa * dd - bb * cc)) == 0)
uu, vv = rng.normal(size=(2, 4))
check("Exercise 34.1.4(b): numeric", np.allclose(ch34.vec_tensor(1.5 * uu - 2 * vv, 0.5 * uu + 3 * vv),
                                                  (1.5 * 3 - (-2) * 0.5) * ch34.vec_tensor(uu, vv)))
ns5 = dict(ns)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch34.run_block(PAGE, "code-ex-34-1-5", ns5)
check("Exercise 34.1.5: solution output equals the printed block out-ex-34-1-5 (True True True True)",
      buf.getvalue().strip() == ch34.output_block(PAGE, "out-ex-34-1-5") == "True True True True",
      buf.getvalue())
X = sp.symbols("x1:4")
xi_sym = {(0, 1): 1, (0, 2): 1, (1, 2): 1}
# xi ^ x = (x1 - x2 + x3) e123 computed in the tensor model with numeric x
xt = rng.normal(size=3)
xi_t = ch34.from_comps(xi_sym, 3)
close("Exercise 34.1.6: xi ^ x = (x1 - x2 + x3) e123", ch34.wedge_t(xi_t, xt)[0, 1, 2], xt[0] - xt[1] + xt[2], 1e-12)
Cuv = ch34.comps(ch34.vec_tensor([1, 1, 0], [0, 1, 1]))
close("Exercise 34.1.6: (1,1,0)^(0,1,1) = e12 + e13 + e23", [Cuv[(0, 1)], Cuv[(0, 2)], Cuv[(1, 2)]], [1, 1, 1], 1e-12)
# every 2-vector in R^3 is simple: random xi has a 2-dim kernel of x -> xi ^ x and equals a multiple of u^v
xr = ch34.from_comps({I: rng.normal() for I in ch34.increasing(3, 2)}, 3)
Tm = np.array([[ch34.wedge_t(xr, ei)[0, 1, 2] for ei in np.eye(3)]])
ker = np.linalg.svd(Tm)[2][1:]
uk, vk = ker
wedge_k = ch34.vec_tensor(uk, vk)
ratio = xr[0, 1] / wedge_k[0, 1] if abs(wedge_k[0, 1]) > 1e-9 else xr[0, 2] / wedge_k[0, 2]
check("Exercise 34.1.6: random 2-vector in R^3 is a multiple of u^v for u, v spanning ker T",
      np.allclose(xr, ratio * wedge_k))

check("Implementation box code-34-1-1: at most 30 lines", ch34.code_lines(PAGE, "code-34-1-1") <= 30)
summary()
