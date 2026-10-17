"""Section 34.5 (Exterior Algebra in Coordinates): verification of every computation (GUIDELINES 13).

The multiplication table is checked against sympy permutation signs and against the antisymmetric-tensor model
of lib/ch34.py; the Implementation box and the coding exercise are extracted from the HTML and executed as printed.
The sphere example uses dgsym.EXAMPLES; the R^3 2-form representation is compared with dgdec.two_form_vector.
Run from the project root: PYTHONPATH=tools python3 chapters/34-exterior-algebra/verify/v-34-5-coordinates.py
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
from dgcheck import check, close, sym_equal, summary  # noqa: E402
from dgsym import EXAMPLES  # noqa: E402
import dgdec as dd  # noqa: E402
import ch34  # noqa: E402

PAGE = "34-5-coordinates.html"
rng = np.random.default_rng(345)


def N(I, J):
    return sum(1 for i in I for j in J if i > j)


def basis_t(I, n):
    E = np.eye(n)
    return ch34.vec_tensor(*[E[i] for i in I]) if I else np.array(1.0)


# ---------------------------------------------------------------------------
# Proposition 34.5.2: multiplication table and sgn(I, I^c)
# ---------------------------------------------------------------------------
for n in range(1, 6):
    ok_sign, ok_tensor, ok_c = True, True, True
    for k in range(n + 1):
        for l in range(n + 1 - k):
            for I in ch34.increasing(n, k):
                for J in ch34.increasing(n, l):
                    prod = ch34.wedge_t(basis_t(I, n), basis_t(J, n))
                    if set(I) & set(J):
                        ok_tensor &= np.allclose(prod, 0)
                    else:
                        s = (-1) ** N(I, J)
                        ok_sign &= s == ch34.perm_sign(I + J)
                        K = tuple(sorted(I + J))
                        ok_tensor &= np.allclose(prod, s * basis_t(K, n))
        for I in ch34.increasing(n, k):
            Ic = tuple(j for j in range(n) if j not in I)
            # (34.5.2) with 1-based indices: sum_a (i_a - a)
            ok_c &= (-1) ** sum((i + 1) - (a + 1) for a, i in enumerate(I)) == ch34.perm_sign(I + Ic)
    check(f"Proposition 34.5.2 (34.5.1): sgn(I,J) = (-1)^N(I,J) is the permutation sign (n={n})", ok_sign)
    check(f"Proposition 34.5.2: table agrees with the tensor model, overlap gives 0 (n={n})", ok_tensor)
    check(f"Proposition 34.5.2 (34.5.2): sgn(I, I^c) = (-1)^(sum (i_a - a)) (n={n})", ok_c)
close("Text: in R^4, star e13 = -e24", ch34.hodge_t(basis_t((0, 2), 4), 4)[1, 3], -1.0, 1e-12)

# ---------------------------------------------------------------------------
# Theorem 34.5.3: the formal model satisfies (W1)-(W4) and is isomorphic to the tensor model
# ---------------------------------------------------------------------------


def formal_wedge(x, y):
    out = {}
    for I, a in x.items():
        for J, b in y.items():
            if set(I) & set(J):
                continue
            K = tuple(sorted(I + J))
            out[K] = out.get(K, 0.0) + (-1) ** N(I, J) * a * b
    return out


def dict_close(x, y, tol=1e-10):
    return all(abs(x.get(K, 0) - y.get(K, 0)) < tol for K in set(x) | set(y))


n = 4
ok = True
for I, J, K in itertools.product([I for k in range(n + 1) for I in ch34.increasing(n, k)], repeat=3):
    lhs = formal_wedge(formal_wedge({I: 1.0}, {J: 1.0}), {K: 1.0})
    rhs = formal_wedge({I: 1.0}, formal_wedge({J: 1.0}, {K: 1.0}))
    ok &= dict_close(lhs, rhs)
check("Theorem 34.5.3(a): (W1) associativity on all basis triples (n=4)", ok)
check("Theorem 34.5.3(a): inversion count splits N(I u J, K) = N(I,K) + N(J,K)",
      all(N(tuple(sorted(I + J)), K) == N(I, K) + N(J, K) for I in ch34.increasing(5, 2) for J in ch34.increasing(5, 1)
          for K in ch34.increasing(5, 2)))
v = {(i,): float(c) for i, c in enumerate(rng.normal(size=5))}
check("Theorem 34.5.3(a): (W2) v ^ v = 0", dict_close(formal_wedge(v, v), {}))
w3 = formal_wedge(formal_wedge({(0,): 1.0}, {(2,): 1.0}), {(3,): 1.0})
check("Theorem 34.5.3(a): (W3) e_(i1) ^ e_(i2) ^ e_(i3) = e_I for increasing I", w3 == {(0, 2, 3): 1.0})
check("Theorem 34.5.3(a): (W4) e1 ^ ... ^ e5 != 0",
      formal_wedge(formal_wedge(formal_wedge(formal_wedge({(0,): 1.}, {(1,): 1.}), {(2,): 1.}), {(3,): 1.}), {(4,): 1.})
      == {(0, 1, 2, 3, 4): 1.0})
# (b): Phi(e_I) = tensor e_I preserves the product
for k, l in [(1, 2), (2, 2), (2, 3), (1, 1)]:
    X = {I: rng.normal() for I in ch34.increasing(5, k)}
    Y = {J: rng.normal() for J in ch34.increasing(5, l)}
    lhs = ch34.from_comps(formal_wedge(X, Y) or {tuple(range(k + l)): 0.0}, 5)
    rhs = ch34.wedge_t(ch34.from_comps(X, 5), ch34.from_comps(Y, 5))
    check(f"Theorem 34.5.3(b): Phi(xi ^ eta) = Phi(xi) ^' Phi(eta) (formal vs tensor, k={k}, l={l})", np.allclose(lhs, rhs))
check("Definition 34.5.1: dim Lambda V = 2^n (n = 1..6)",
      all(sum(math.comb(n, k) for k in range(n + 1)) == 2 ** n for n in range(1, 7)))

# ---------------------------------------------------------------------------
# Components table, Example 34.5.4
# ---------------------------------------------------------------------------
xi = ch34.vec_tensor(*rng.normal(size=(2, 4))) + ch34.vec_tensor(*rng.normal(size=(2, 4)))
om = {I: rng.normal() for I in ch34.increasing(4, 2)}
Wom = ch34.from_comps(om, 4)                    # function tensor of omega = sum omega_I e^I
us = rng.normal(size=(2, 4))
close("Components: omega(u1, u2) = sum_I omega_I det U_I", float(us[0] @ Wom @ us[1]),
      sum(c * np.linalg.det(np.column_stack(us)[list(I), :]) for I, c in om.items()), 1e-10)
star_formula = {}
for I, c in ch34.comps(xi).items():
    Ic = tuple(j for j in range(4) if j not in I)
    star_formula[Ic] = star_formula.get(Ic, 0) + (-1) ** sum(i - a for a, i in enumerate(I)) * c
check("Components: star xi = sum sgn(I, I^c) xi^I e_Ic agrees with the tensor star",
      np.allclose(ch34.from_comps(star_formula, 4), ch34.hodge_t(xi, 4)))
table3 = {(): ((0, 1, 2), 1), (0,): ((1, 2), 1), (1,): ((0, 2), -1), (2,): ((0, 1), 1), (0, 1): ((2,), 1),
          (0, 2): ((1,), -1), (1, 2): ((0,), 1), (0, 1, 2): ((), 1)}
check("Example 34.5.4: the R^3 table", all(np.allclose(ch34.hodge_t(basis_t(I, 3), 3), s * basis_t(J, 3))
                                           for I, (J, s) in table3.items()))
check("Example 34.5.4: sgn = -1 exactly for I = (2) and (1,3)",
      [I for I in table3 if table3[I][1] == -1] == [(1,), (0, 2)])

# ---------------------------------------------------------------------------
# Proposition 34.5.5 (R^3 dictionary) and the dgdec representation
# ---------------------------------------------------------------------------
a, b, c, wv = rng.normal(size=(4, 3))
x, y, z = rng.normal(size=(3, 3))
om2 = ch34.hodge_t(wv, 3)                       # star(w^flat) as a function tensor (Euclidean)
close("Proposition 34.5.5(b): star(w^flat) = w1 e^23 + w2 e^31 + w3 e^12", [om2[1, 2], om2[2, 0], om2[0, 1]], wv, 1e-12)
close("Proposition 34.5.5(b): star(w^flat)(u, v) = <w, u x v>", x @ om2 @ y, wv @ np.cross(x, y), 1e-12)
close("Proposition 34.5.5(b): dgdec.two_form_vector reads back w (w_yz, w_zx, w_xy)", dd.two_form_vector(om2), wv, 1e-12)
close("Proposition 34.5.5(c): c e^123 (u,v,w) = c det", 2.5 * ch34.vec_tensor(*np.eye(3))[0, 1, 2] * np.linalg.det(np.column_stack([x, y, z])),
      2.5 * np.linalg.det(np.column_stack([x, y, z])), 1e-12)
check("Proposition 34.5.5(d): a^flat ^ b^flat = star((a x b)^flat)",
      np.allclose(ch34.vec_tensor(a, b), ch34.hodge_t(np.cross(a, b), 3)))
close("Proposition 34.5.5(d): a^flat ^ star(b^flat) = <a,b> e^123", ch34.wedge_t(a, ch34.hodge_t(b, 3))[0, 1, 2], a @ b, 1e-12)
close("Proposition 34.5.5(d): a^flat ^ b^flat ^ c^flat = det(a,b,c) e^123", ch34.vec_tensor(a, b, c)[0, 1, 2],
      np.linalg.det(np.column_stack([a, b, c])), 1e-12)

# ---------------------------------------------------------------------------
# Propositions 34.5.6, 34.5.7
# ---------------------------------------------------------------------------
B = rng.normal(size=(4, 4))
if np.linalg.det(B) < 0:
    B[:, 0] *= -1
G = B.T @ B
for J in ch34.increasing(4, 2):
    # (b_J)^flat evaluated on (b_i1, b_i2) = det[g(b_j, b_i)] = det G_IJ, and these are its beta^I components
    vals = {I: np.linalg.det(np.array([[B[:, i] @ B[:, j] for j in J] for i in I])) for I in ch34.increasing(4, 2)}
    check(f"Proposition 34.5.6: (b_J)^flat has beta^I-components det G_IJ (J={tuple(j + 1 for j in J)})",
          all(abs(vals[I] - np.linalg.det(G[np.ix_(I, J)])) < 1e-9 for I in vals))
close("Proposition 34.5.6 (k = n): (b1^..^b4)^flat = det G beta^1234 (det G = det(B)^2)", np.linalg.det(G),
      np.linalg.det(B) ** 2, 1e-8)
close("Proposition 34.5.7: b1^..^b4 = sqrt(det G) vol", ch34.vec_tensor(*B.T)[0, 1, 2, 3], math.sqrt(np.linalg.det(G)), 1e-8)
beta = np.linalg.inv(B)                          # rows: dual basis beta^i
close("Proposition 34.5.7: sqrt(det G) beta^1^..^beta^4 = e^1234 = star 1",
      math.sqrt(np.linalg.det(G)) * ch34.vec_tensor(*beta)[0, 1, 2, 3], 1.0, 1e-8)

# ---------------------------------------------------------------------------
# Example 34.5.8 (sphere)
# ---------------------------------------------------------------------------
ex = EXAMPLES["sphere"]
th, ph = ex["coords"]
(r,) = ex["params"]
X = ex["expr"]
Xt, Xp = X.diff(th), X.diff(ph)
Gs = sp.Matrix([[Xt.dot(Xt), Xt.dot(Xp)], [Xp.dot(Xt), Xp.dot(Xp)]]).applyfunc(sp.simplify)
sym_equal("Example 34.5.8: G = diag(r^2, r^2 sin^2 th)", Gs, sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), ex["domain"])
sym_equal("Example 34.5.8: x_th x x_ph = r^2 sin th * (outward unit normal)", Xt.cross(Xp),
          r ** 2 * sp.sin(th) * ex["expected"]["normal"], ex["domain"])
s_ = sp.sin(th)
Fm = sp.diag(1 / r, 1 / (r * s_))              # f_1, f_2 in (x_th, x_ph) coordinates (columns)
sym_equal("Example 34.5.8: f orthonormal", (Fm.T * Gs * Fm).applyfunc(sp.simplify), sp.eye(2), ex["domain"])
Jm = sp.Matrix([[0, -1], [1, 0]])
Svec = (Fm * Jm * Fm.inv()).applyfunc(sp.simplify)
sym_equal("Example 34.5.8: star x_th = x_ph / sin th, star x_ph = -sin th x_th", Svec,
          sp.Matrix([[0, -s_], [1 / s_, 0]]), ex["domain"])
Mf = sp.diag(r, r * s_)                          # f^1 = r beta^1, f^2 = r sin th beta^2 (columns in beta coordinates)
Sform = (Mf * Jm * Mf.inv()).applyfunc(sp.simplify)
sym_equal("Example 34.5.8: star beta^1 = sin th beta^2, star beta^2 = -beta^1 / sin th", Sform,
          sp.Matrix([[0, -1 / s_], [s_, 0]]), ex["domain"])
sym_equal("Example 34.5.8: star 1 = f^1 ^ f^2 = r^2 sin th beta^1 ^ beta^2 = sqrt(det G)", Mf.det(),
          sp.sqrt(Gs.det()), ex["domain"])

# ---------------------------------------------------------------------------
# Implementation box
# ---------------------------------------------------------------------------
ns = ch34.run_block(PAGE, "code-34-5-1")
check("Implementation box: uv = {(0,1): 2, (0,2): 5, (1,2): 4}", dict_close(ns["uv"], {(0, 1): 2, (0, 2): 5, (1, 2): 4}))
check("Implementation box: star_uv = u x v = (4, -5, 2)", dict_close(ns["star_uv"], {(0,): 4, (1,): -5, (2,): 2}))
wedge, hodge = ns["wedge"], ns["hodge"]
for n_, k, l in [(4, 1, 2), (5, 2, 2), (5, 2, 3), (3, 1, 1)]:
    Xd = {I: rng.normal() for I in ch34.increasing(n_, k)}
    Yd = {J: rng.normal() for J in ch34.increasing(n_, l)}
    W_ = wedge(Xd, Yd)
    check(f"Implementation box: wedge agrees with the tensor model (n={n_}, k={k}, l={l})",
          np.allclose(ch34.from_comps(W_ or {tuple(range(k + l)): 0.0}, n_),
                      ch34.wedge_t(ch34.from_comps(Xd, n_), ch34.from_comps(Yd, n_))))
    check(f"Implementation box: hodge agrees with the tensor model (n={n_}, k={k})",
          np.allclose(ch34.from_comps(hodge(Xd, n_), n_), ch34.hodge_t(ch34.from_comps(Xd, n_), n_)))

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
u4, v4, w4 = np.array([1.0, 1, 0, 0]), np.array([0.0, 1, 1, 0]), np.array([0.0, 0, 1, 1])
cuv = ch34.comps(ch34.vec_tensor(u4, v4))
check("Exercise 34.5.1: u^v = e12 + e13 + e23",
      dict_close({I: c for I, c in cuv.items() if abs(c) > 1e-12}, {(0, 1): 1, (0, 2): 1, (1, 2): 1}))
cuvw = ch34.comps(ch34.vec_tensor(u4, v4, w4))
check("Exercise 34.5.1: u^v^w = e123 + e124 + e134 + e234", dict_close(cuvw, {I: 1.0 for I in ch34.increasing(4, 3)}))
close("Exercise 34.5.2: star e1 = e234, star e24 = -e13, star e234 = -e1",
      [ch34.hodge_t(basis_t((0,), 4), 4)[1, 2, 3], ch34.hodge_t(basis_t((1, 3), 4), 4)[0, 2],
       ch34.hodge_t(basis_t((1, 2, 3), 4), 4)[0]], [1, -1, -1], 1e-12)
a5, b5 = np.array([1.0, 0, 2]), np.array([0.0, 1, -1])
c5 = ch34.comps(ch34.vec_tensor(a5, b5))
close("Exercise 34.5.3: a^b = e^12 - e^13 - 2 e^23", [c5[(0, 1)], c5[(0, 2)], c5[(1, 2)]], [1, -1, -2], 1e-12)
close("Exercise 34.5.3: cyclic coefficients (e^23, e^31, e^12) = (-2, 1, 1) = a x b", [c5[(1, 2)], -c5[(0, 2)], c5[(0, 1)]],
      np.cross(a5, b5), 1e-12)
rr, tt = sp.symbols("r theta", positive=True)
Bp = sp.Matrix([[sp.cos(tt), -rr * sp.sin(tt)], [sp.sin(tt), rr * sp.cos(tt)]])
Gp = (Bp.T * Bp).applyfunc(sp.simplify)
sym_equal("Exercise 34.5.4: polar Gram matrix diag(1, r^2)", Gp, sp.diag(1, rr ** 2))
check("Exercise 34.5.4: det[b1 b2] = r > 0", sp.simplify(Bp.det() - rr) == 0)
Fp = sp.diag(1, 1 / rr)
Sp_vec = (Fp * Jm * Fp.inv()).applyfunc(sp.simplify)
sym_equal("Exercise 34.5.4: star b1 = b2/r, star b2 = -r b1", Sp_vec, sp.Matrix([[0, -rr], [1 / rr, 0]]))
Mp = sp.diag(1, rr)
sym_equal("Exercise 34.5.4: star beta^1 = r beta^2, star beta^2 = -beta^1/r", (Mp * Jm * Mp.inv()).applyfunc(sp.simplify),
          sp.Matrix([[0, -1 / rr], [rr, 0]]))
sym_equal("Exercise 34.5.4: star 1 = r beta^1 ^ beta^2 = sqrt(det G)", Mp.det(), sp.sqrt(Gp.det()))
# numeric polar check with Euclidean star in R^2: star b1 = b2 / r
tt0, rr0 = 0.7, 1.8
b1n = np.array([np.cos(tt0), np.sin(tt0)])
b2n = np.array([-rr0 * np.sin(tt0), rr0 * np.cos(tt0)])
close("Exercise 34.5.4 numeric: Euclidean star b1 = b2/r", ch34.hodge_t(b1n, 2), b2n / rr0, 1e-12)
xs = sp.symbols("x12 x13 x14 x23 x24 x34")
x12, x13, x14, x23, x24, x34 = xs
xi_sym = {(0, 1): x12, (0, 2): x13, (0, 3): x14, (1, 2): x23, (1, 3): x24, (2, 3): x34}
sq = {}
for I, cI in xi_sym.items():
    for J, cJ in xi_sym.items():
        if set(I) & set(J):
            continue
        sq[(0, 1, 2, 3)] = sq.get((0, 1, 2, 3), 0) + (-1) ** N(I, J) * cI * cJ
sym_equal("Exercise 34.5.5: xi ^ xi = 2 (x12 x34 - x13 x24 + x14 x23) e1234", sq[(0, 1, 2, 3)],
          2 * (x12 * x34 - x13 * x24 + x14 * x23))
uP = sp.Matrix([1, 0, -x23 / x12, -x24 / x12])
vP = sp.Matrix([0, x12, x13, x14])
UP = uP.row_join(vP)
minors = {I: sp.simplify(UP.extract(list(I), [0, 1]).det()) for I in ch34.increasing(4, 2)}
sym_equal("Exercise 34.5.5: u ^ v has the expansion in the solution",
          [minors[I] for I in ch34.increasing(4, 2)],
          [x12, x13, x14, x23, x24, (x13 * x24 - x14 * x23) / x12])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns6 = ch34.run_block(PAGE, "code-ex-34-5-6", dict(ns))
expected = ("True True True {(0, 1): 1.0, (0, 2): 1.0, (1, 2): 1.0} "
            "{(0, 1, 2): 1.0, (0, 1, 3): 1.0, (0, 2, 3): 1.0, (1, 2, 3): 1.0}")
check("Exercise 34.5.6: solution output equals out-ex-34-5-6",
      buf.getvalue().strip() == ch34.output_block(PAGE, "out-ex-34-5-6") == expected, buf.getvalue())

check("Implementation box code-34-5-1: at most 30 lines", ch34.code_lines(PAGE, "code-34-5-1") <= 30)
summary()
