"""Section 34.2 (The Hodge Star): verification of every computation (GUIDELINES 13).

Independent model: antisymmetric tensors; the Hodge star of the model contracts with the Levi-Civita symbol
(lib/ch34.py), the text uses the basis formula with sgn(I, I^c) -- here computed with sympy permutations.
Run from the project root: PYTHONPATH=tools python3 chapters/34-exterior-algebra/verify/v-34-2-hodge.py
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
import ch34  # noqa: E402

PAGE = "34-2-hodge-star.html"
rng = np.random.default_rng(342)


def sgn_IIc(I, n):
    Ic = tuple(j for j in range(n) if j not in I)
    return ch34.perm_sign(I + Ic), Ic


def basis_tensor(I, n):
    E = np.eye(n)
    return ch34.vec_tensor(*[E[i] for i in I]) if I else np.array(1.0)


def act(R, T):
    """Apply the linear map R of R^n to every slot of the tensor T (induced map on k-vectors)."""
    T = np.asarray(T, float)
    for ax in range(T.ndim):
        T = np.moveaxis(np.tensordot(R, T, axes=([1], [ax])), 0, ax)
    return T


# ---------------------------------------------------------------------------
# Proposition 34.2.1 and Example 34.2.2
# ---------------------------------------------------------------------------
for n, k in [(3, 2), (4, 2), (5, 3)]:
    us, ws = rng.normal(size=(2, k, n))
    lhs = ch34.inner_t(ch34.vec_tensor(*us), ch34.vec_tensor(*ws))
    close(f"(34.2.1): <u-wedge, w-wedge> = det[<u_i, w_j>] (n={n}, k={k})", lhs,
          np.linalg.det(np.array([[np.dot(a, b) for b in ws] for a in us])), 1e-10)
    close(f"Proposition 34.2.1(c): |u-wedge|^2 = det Gram (n={n}, k={k})",
          ch34.inner_t(ch34.vec_tensor(*us), ch34.vec_tensor(*us)), ch34.gram_det(*us), 1e-10)
    # volume: with an orthonormal basis q of the span, sqrt(det G) = |det A|
    Q, Rr = np.linalg.qr(np.column_stack(us))
    close(f"Proposition 34.2.1(c): sqrt(det G) = |det A| for u = Q A (n={n}, k={k})",
          math.sqrt(ch34.gram_det(*us)), abs(np.linalg.det(Rr)), 1e-10)
n = 4
ok = all(abs(ch34.inner_t(basis_tensor(I, n), basis_tensor(J, n)) - (1.0 if I == J else 0.0)) < 1e-12
         for I in ch34.increasing(n, 2) for J in ch34.increasing(n, 2))
check("Proposition 34.2.1(b): (e_I) orthonormal in Lambda^2 R^4", ok)
u, v = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 2.0])
close("Example 34.2.2: |u^v|^2 = 4 + 25 + 16 = 45", ch34.inner_t(ch34.vec_tensor(u, v), ch34.vec_tensor(u, v)), 45, 1e-12)
close("Example 34.2.2: |u|^2|v|^2 - <u,v>^2 = 14*5 - 25 = 45", np.dot(u, u) * np.dot(v, v) - np.dot(u, v) ** 2, 45, 1e-12)

# ---------------------------------------------------------------------------
# (34.2.2) orthogonal complement (definition sentence, added in review), Lemma 34.2.3, Proposition 34.2.4
# ---------------------------------------------------------------------------
Wxy = np.array([[1.0, 0, 0], [0, 1, 0]]).T
Wperp = np.linalg.svd(Wxy.T)[2][2:].T                       # null space of W^T = W^perp
check("(34.2.2): the orthogonal complement of the xy-plane is the z-axis",
      Wperp.shape == (3, 1) and np.allclose(np.abs(Wperp[:, 0]), [0, 0, 1]))
for n, k in [(3, 1), (4, 2), (5, 3)]:
    W = rng.normal(size=(n, k))
    Wp = np.linalg.svd(W.T)[2][k:].T
    check(f"(34.2.2): dim W^perp = n - k and W cap W^perp = 0 (n={n}, k={k})",
          Wp.shape[1] == n - k and np.allclose(W.T @ Wp, 0) and np.linalg.matrix_rank(np.column_stack([W, Wp])) == n)
for n, k in [(3, 1), (3, 2), (5, 2)]:
    W = rng.normal(size=(n, k))
    Q, _ = np.linalg.qr(np.column_stack([W, rng.normal(size=(n, n - k))]))
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    check(f"Lemma 34.2.3: span(q_1..q_k) = W and q_(k+1..n) is orthogonal to W (n={n}, k={k})",
          np.linalg.matrix_rank(np.column_stack([Q[:, :k], W])) == k and np.allclose(W.T @ Q[:, k:], 0)
          and np.linalg.det(Q) > 0)
for _ in range(3):
    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    close("Proposition 34.2.4: every positively oriented orthonormal basis gives vol = e1234",
          ch34.vec_tensor(*Q.T)[0, 1, 2, 3], 1.0, 1e-10)
B = rng.normal(size=(4, 4))
close("Proposition 34.2.4: b1^..^b4 = sign * sqrt(det G_b) vol",
      ch34.vec_tensor(*B.T)[0, 1, 2, 3], np.sign(np.linalg.det(B)) * math.sqrt(np.linalg.det(B.T @ B)), 1e-10)

# ---------------------------------------------------------------------------
# Theorem 34.2.6: basis formula, (c), independence of the basis, (H1)-(H3)
# ---------------------------------------------------------------------------
for n in range(1, 6):
    ok = True
    for k in range(n + 1):
        for I in ch34.increasing(n, k):
            s, Ic = sgn_IIc(I, n)
            ok &= np.allclose(ch34.hodge_t(basis_tensor(I, n), n), s * basis_tensor(Ic, n))
            # e_I ^ e_Ic = sgn vol
            top = ch34.wedge_t(basis_tensor(I, n), basis_tensor(Ic, n))
            ok &= abs(float(top[tuple(range(n))] if n else top) - s) < 1e-12
    check(f"Theorem 34.2.6(b) / (34.2.4)-(34.2.5): star e_I = sgn(I, I^c) e_Ic for all I (n={n})", ok)
for n, k in [(3, 1), (3, 2), (4, 2), (5, 2), (5, 3)]:
    Z = ch34.vec_tensor(*rng.normal(size=(k, n))) + ch34.vec_tensor(*rng.normal(size=(k, n)))
    X = ch34.vec_tensor(*rng.normal(size=(k, n))) + ch34.vec_tensor(*rng.normal(size=(k, n)))
    top = ch34.wedge_t(Z, ch34.hodge_t(X, n))[tuple(range(n))]
    close(f"Theorem 34.2.6(c): zeta ^ star xi = <zeta, xi> vol (n={n}, k={k})", top, ch34.inner_t(Z, X), 1e-10)
    Rm, _ = np.linalg.qr(rng.normal(size=(n, n)))
    if np.linalg.det(Rm) < 0:
        Rm[:, 0] *= -1
    check(f"Theorem 34.2.6, step 3: star commutes with rotations, i.e. no basis dependence (n={n}, k={k})",
          np.allclose(ch34.hodge_t(act(Rm, X), n), act(Rm, ch34.hodge_t(X, n))))
    # (H1)-(H3) for a simple xi
    us = rng.normal(size=(k, n))
    xi = ch34.vec_tensor(*us)
    sx = ch34.hodge_t(xi, n)
    Uc = np.linalg.svd(us)[2][k:]            # orthonormal basis of W-perp
    perp = ch34.vec_tensor(*Uc)
    c = ch34.inner_t(sx, perp)
    check(f"(H1): star xi is a multiple of the wedge of a basis of W-perp (n={n}, k={k})", np.allclose(sx, c * perp))
    close(f"(H2): |star xi| = |xi| (n={n}, k={k})", ch34.inner_t(sx, sx), ch34.inner_t(xi, xi), 1e-10)
    check(f"(H3): xi ^ star xi is a positive multiple of vol (n={n}, k={k})",
          ch34.wedge_t(xi, sx)[tuple(range(n))] > 0)

# ---------------------------------------------------------------------------
# Examples 34.2.7, 34.2.8, Proposition 34.2.9
# ---------------------------------------------------------------------------
e = np.eye(2)
close("Example 34.2.7: star e1 = e2, star e2 = -e1", [ch34.hodge_t(e[0], 2), ch34.hodge_t(e[1], 2)], [[0, 1], [-1, 0]], 1e-12)
close("Example 34.2.7: star 1 = e12", ch34.hodge_t(np.array(1.0), 2)[0, 1], 1.0, 1e-12)
close("Example 34.2.7: star b1 = (-1, 2)", ch34.hodge_t(np.array([2.0, 1.0]), 2), [-1, 2], 1e-12)
for _ in range(3):
    a, b = rng.normal(size=(2, 3))
    close("Proposition 34.2.8: star(u ^ v) = u x v (random)", ch34.hodge_t(ch34.vec_tensor(a, b), 3), np.cross(a, b), 1e-10)
E3 = np.eye(3)
table = {(): {(0, 1, 2): 1}, (0,): {(1, 2): 1}, (1,): {(0, 2): -1}, (2,): {(0, 1): 1},
         (0, 1): {(2,): 1}, (0, 2): {(1,): -1}, (1, 2): {(0,): 1}, (0, 1, 2): {(): 1}}
ok = all(np.allclose(ch34.hodge_t(basis_tensor(I, 3), 3), ch34.from_comps(out, 3)) for I, out in table.items())
check("(34.2.7): the R^3 table", ok)
xi = ch34.from_comps({(0, 1): 2, (0, 2): 5, (1, 2): 4}, 3)
close("Text after 34.2.8: star(2 e12 + 5 e13 + 4 e23) = (4, -5, 2)", ch34.hodge_t(xi, 3), [4, -5, 2], 1e-12)
for n in range(1, 7):
    ok = True
    for k in range(n + 1):
        for I in ch34.increasing(n, k):
            ok &= np.allclose(ch34.hodge_t(ch34.hodge_t(basis_tensor(I, n), n), n), (-1) ** (k * (n - k)) * basis_tensor(I, n))
    check(f"Proposition 34.2.9: star star = (-1)^(k(n-k)) on all basis k-vectors (n={n})", ok)
X = ch34.vec_tensor(*rng.normal(size=(2, 4))) + ch34.vec_tensor(*rng.normal(size=(2, 4)))
close("Proposition 34.2.9: star is an isometry (non-simple 2-vector in R^4)",
      ch34.inner_t(ch34.hodge_t(X, 4), ch34.hodge_t(X, 4)), ch34.inner_t(X, X), 1e-10)

# ---------------------------------------------------------------------------
# Example 34.2.10 and Figure 34.2.2
# ---------------------------------------------------------------------------
# (a) reversed orientation: vol -> -vol gives -star (Theorem 34.2.6(c) with -vol)
check("Example 34.2.10(a): reversed orientation gives -star (via basis (e2, e1))",
      np.allclose(-ch34.hodge_t(e[0], 2), [0, -1]))


def hodge_g_vector_2d(G, x):
    """star_g on vectors of R^2 for the inner product G (standard orientation), via a g-orthonormal positive
    basis F = L^{-T} (G = L L^T): star f1 = f2, star f2 = -f1."""
    L = np.linalg.cholesky(G)
    F = np.linalg.inv(L).T
    assert np.allclose(F.T @ G @ F, np.eye(2)) and np.linalg.det(F) > 0
    c = np.linalg.solve(F, x)
    return c[0] * F[:, 1] - c[1] * F[:, 0]


G = np.array([[2.0, 1.0], [1.0, 1.0]])
sg = hodge_g_vector_2d(G, e[0])
close("Example 34.2.10(b): star_g e1 = (-1, 2)", sg, [-1, 2], 1e-12)
close("Example 34.2.10(b): g(e1, star_g e1) = 0 and |star_g e1|_g^2 = |e1|_g^2 = 2",
      [e[0] @ G @ sg, sg @ G @ sg], [0, 2], 1e-12)
check("Example 34.2.10(b): G positive definite (det 1, G11 = 2)", np.all(np.linalg.eigvalsh(G) > 0))
check("Figure 34.2.2(b): tangent of {x^T G x = 2} at e1 is parallel to star_g e1",
      abs(-G[0, 1] * sg[1] - G[0, 0] * sg[0]) < 1e-12)
check("Figure 34.2.2(a): four quarter turns return u", np.allclose(
    ch34.hodge_t(ch34.hodge_t(ch34.hodge_t(ch34.hodge_t(np.array([2.0, 1.0]), 2), 2), 2), 2), [2, 1]))
uf, vf = np.array([1.0, 0.15, 0.25]), np.array([0.3, 0.9, 0.35])
close("Figure 34.2.1: u x v = (-0.1725, -0.275, 0.855)", np.cross(uf, vf), [-0.1725, -0.275, 0.855], 1e-12)
close("Figure 34.2.1: area |u ^ v| = 0.915", np.linalg.norm(np.cross(uf, vf)), 0.915, 5e-4)

# Smooth vs. Discrete box: star t is perpendicular to t with the same length; the dual edge vector is ratio * star t
t = rng.normal(size=2)
st = ch34.hodge_t(t, 2)
check("Discrete box: star t is perpendicular to t, same length", abs(t @ st) < 1e-12 and abs(t @ t - st @ st) < 1e-12)

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
close("Exercise 34.2.1(a): star(e12 - 3 e23) = (-3, 0, 1)",
      ch34.hodge_t(ch34.from_comps({(0, 1): 1, (1, 2): -3}, 3), 3), [-3, 0, 1], 1e-12)
su = ch34.hodge_t(np.array([1.0, -1.0, 2.0]), 3)
Cs = ch34.comps(su)
close("Exercise 34.2.1(b): star u = 2 e12 + e13 + e23", [Cs[(0, 1)], Cs[(0, 2)], Cs[(1, 2)]], [2, 1, 1], 1e-12)
close("Exercise 34.2.1(b): star star u = u, |star u|^2 = 6", list(ch34.hodge_t(su, 3)) + [ch34.inner_t(su, su)],
      [1, -1, 2, 6], 1e-12)
Cx = ch34.comps(ch34.vec_tensor([1, 0, 1], [0, 2, 1]))
close("Exercise 34.2.3: components 2, 1, -2 and sum of squares 9 = |u|^2|v|^2 - <u,v>^2",
      [Cx[(0, 1)], Cx[(0, 2)], Cx[(1, 2)], sum(c * c for c in Cx.values()), 2 * 5 - 1], [2, 1, -2, 9, 9], 1e-12)
lam = sp.symbols("lambda", positive=True)
Gs = sp.diag(lam ** 2, 1)
F = sp.diag(1 / lam, 1)                     # g-orthonormal positive basis f1 = e1/lam, f2 = e2
sym_equal("Exercise 34.2.4: F^T G F = I", F.T * Gs * F, sp.eye(2))
f1, f2 = F[:, 0], F[:, 1]
# star_g e1 = lam * star_g f1 = lam f2 ; star_g e2 = star_g f2 = -f1
sym_equal("Exercise 34.2.4: star_g e1 = lam e2", lam * f2, sp.Matrix([0, lam]))
sym_equal("Exercise 34.2.4: star_g e2 = -e1/lam", -f1, sp.Matrix([-1 / lam, 0]))
sym_equal("Exercise 34.2.4: star_g 1 = f1 ^ f2 = (1/lam) e12", f1[0] * f2[1] - f1[1] * f2[0], 1 / lam)
for lv in (0.5, 2.0, 3.0):
    Gn = np.diag([lv ** 2, 1.0])
    close(f"Exercise 34.2.4 numeric (lambda={lv}): star_g e1 = lambda e2", hodge_g_vector_2d(Gn, e[0]), [0, lv], 1e-12)
    close(f"Exercise 34.2.4 numeric (lambda={lv}): star_g star_g e1 = -e1",
          hodge_g_vector_2d(Gn, hodge_g_vector_2d(Gn, e[0])), [-1, 0], 1e-12)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = ch34.run_block(PAGE, "code-ex-34-2-5")
check("Exercise 34.2.5: solution output equals out-ex-34-2-5 (True True)",
      buf.getvalue().strip() == ch34.output_block(PAGE, "out-ex-34-2-5") == "True True", buf.getvalue())
hb = ns["hodge_basis"]
check("Exercise 34.2.5: hodge_basis agrees with sympy permutation signs (n <= 6)",
      all(hb(I, n)[0] == sgn_IIc(I, n)[0] for n in range(1, 7) for k in range(n + 1) for I in ch34.increasing(n, k)))
# Exercise 34.2.6: R^4 table, self-dual/anti-self-dual dimension 3 each
table4 = {(0, 1): ((2, 3), 1), (0, 2): ((1, 3), -1), (0, 3): ((1, 2), 1), (1, 2): ((0, 3), 1), (1, 3): ((0, 2), -1),
          (2, 3): ((0, 1), 1)}
check("Exercise 34.2.6(a): R^4 table for 2-vectors",
      all(np.allclose(ch34.hodge_t(basis_tensor(I, 4), 4), s * basis_tensor(J, 4)) for I, (J, s) in table4.items()))
Ms = np.array([[ch34.comps(ch34.hodge_t(basis_tensor(I, 4), 4))[J] for I in ch34.increasing(4, 2)]
               for J in ch34.increasing(4, 2)])
ev = np.round(np.linalg.eigvalsh(Ms), 10)
check("Exercise 34.2.6(b): star on Lambda^2 R^4 is symmetric with eigenvalues +1 (x3) and -1 (x3)",
      np.allclose(Ms, Ms.T) and list(ev) == [-1, -1, -1, 1, 1, 1])
sd = ch34.from_comps({(0, 1): 1.3, (2, 3): 1.3, (0, 3): -0.4, (1, 2): -0.4, (0, 2): 0.7, (1, 3): -0.7}, 4)
check("Exercise 34.2.6: e12 + e34, e14 + e23, e13 - e24 combinations are self-dual", np.allclose(ch34.hodge_t(sd, 4), sd))
close("Exercise 34.2.6(c): self-dual xi has xi ^ xi = |xi|^2 vol != 0", ch34.wedge_t(sd, sd)[0, 1, 2, 3],
      ch34.inner_t(sd, sd), 1e-10)

summary()
