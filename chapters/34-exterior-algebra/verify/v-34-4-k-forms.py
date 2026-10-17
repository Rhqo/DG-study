"""Section 34.4 (k-Forms as Measurement): verification of every computation (GUIDELINES 13).

A k-form is represented by its function tensor W[i_1..i_k] = omega(e_i1, ..., e_ik). The wedge of Theorem 34.4.6
(defined on the basis forms e^I by the determinant D[...]) is implemented separately from the tensor formula
((k+l)!/(k! l!)) Alt(omega (x) eta) of Chapter 17 / GUIDELINES 6.4, and the two are compared. The Alt convention
(the Alt normalization some texts use; see the Conventions box) is checked numerically and symbolically.
Run from the project root: PYTHONPATH=tools python3 chapters/34-exterior-algebra/verify/v-34-4-k-forms.py
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
import dgdec as dd  # noqa: E402
import ch34  # noqa: E402

PAGE = "34-4-k-forms.html"
rng = np.random.default_rng(344)


def D(alphas, us):
    """(34.4.2): det[alpha^i(u_j)]."""
    return float(np.linalg.det(np.array([[a @ u for u in us] for a in alphas])))


def D_tensor(alphas, n):
    """Function tensor of D[alpha^1..alpha^k]: entries D(alphas, (e_i1, ..., e_ik))."""
    k = len(alphas)
    E = np.eye(n)
    T = np.zeros((n,) * k)
    for idx in itertools.product(range(n), repeat=k):
        T[idx] = D(alphas, [E[i] for i in idx])
    return T


def wedge_basis_def(W1, W2, n):
    """Theorem 34.4.6: expand W1 = sum_I a_I e^I, W2 = sum_J b_J e^J (Lemma 34.4.5) and set e^I ^ e^J = D[e^I, e^J]."""
    k, l = W1.ndim, W2.ndim
    E = np.eye(n)
    out = np.zeros((n,) * (k + l))
    for I in ch34.increasing(n, k):
        for J in ch34.increasing(n, l):
            c = W1[I] * W2[J]
            if c:
                out += c * D_tensor([E[i] for i in I] + [E[j] for j in J], n)
    return out


def evalT(W, us):
    """omega(u_1..u_k) for a function tensor W."""
    out = W
    for u in us:
        out = np.tensordot(out, u, axes=([0], [0]))
    return float(out)


# ---------------------------------------------------------------------------
# Proposition 34.4.1 and Figure 34.4.1
# ---------------------------------------------------------------------------
Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
a, b = Q[:, 0], Q[:, 1]                         # orthonormal sharps of alpha, beta (Euclidean R^4)
u, v = rng.normal(size=(2, 4))
pu, pv = (a @ u) * a + (b @ u) * b, (a @ v) * a + (b @ v) * b
check("Proposition 34.4.1 (34.4.1): pi u ^ pi v = (alpha(u)beta(v) - alpha(v)beta(u)) a ^ b",
      np.allclose(ch34.vec_tensor(pu, pv), ((a @ u) * (b @ v) - (a @ v) * (b @ u)) * ch34.vec_tensor(a, b)))
uf, vf = np.array([1.1, 0.0, 0.25]), np.array([0.2, 0.9, 0.35])
t = np.deg2rad(40)
af, bf = np.array([np.cos(t), 0, np.sin(t)]), np.array([0.0, 1.0, 0.0])
close("Figure 34.4.1 / Example 34.4.10(b): alpha(u), alpha(v), beta(u), beta(v)",
      [af @ uf, af @ vf, bf @ uf, bf @ vf], [1.003, 0.378, 0.0, 0.9], 5e-4)
val = (af @ uf) * (bf @ vf) - (af @ vf) * (bf @ uf)
close("Figure 34.4.1: (alpha^beta)(u,v) = 0.903", val, 0.903, 5e-4)
close("Figure 34.4.1: |u ^ v| = 1.069", np.linalg.norm(np.cross(uf, vf)), 1.069, 5e-4)

# ---------------------------------------------------------------------------
# Example 34.4.3, Non-example 34.4.4
# ---------------------------------------------------------------------------
w = rng.normal(size=3)
flux = lambda x, y: w @ np.cross(x, y)  # noqa: E731
x, y, z = rng.normal(size=(3, 3))
check("Example 34.4.3(c): <w, u x v> is bilinear and alternating",
      np.isclose(flux(2 * x + z, y), 2 * flux(x, y) + flux(z, y)) and np.isclose(flux(x, y), -flux(y, x))
      and np.isclose(flux(x, x), 0))
check("Non-example 34.4.4(b): |u x v| is not linear (u -> -u keeps it)",
      np.isclose(np.linalg.norm(np.cross(-x, y)), np.linalg.norm(np.cross(x, y))) and np.linalg.norm(np.cross(x, y)) > 0)

# ---------------------------------------------------------------------------
# Lemma 34.4.5 and Theorem 34.4.6
# ---------------------------------------------------------------------------
n = 4
E = np.eye(n)
for k in (1, 2, 3):
    ok = all(abs(evalT(D_tensor([E[i] for i in I], n), [E[j] for j in J]) - (I == J)) < 1e-12
             for I in ch34.increasing(n, k) for J in ch34.increasing(n, k))
    check(f"Lemma 34.4.5(a): e^I(e_J) = delta_IJ (n=4, k={k})", ok)
alphas, betas = rng.normal(size=(2, 2, n))
W = D_tensor(list(alphas), n) - 0.7 * D_tensor(list(betas), n)      # some 2-form
recon = sum(W[I] * D_tensor([E[i] for i in I], n) for I in ch34.increasing(n, 2))
check("Lemma 34.4.5(b): omega = sum_I omega(e_I) e^I", np.allclose(W, recon))
for k, l in [(1, 1), (1, 2), (2, 2), (1, 3)]:
    al = list(rng.normal(size=(k, n)))
    be = list(rng.normal(size=(l, n)))
    lhs = wedge_basis_def(D_tensor(al, n), D_tensor(be, n), n)
    check(f"Theorem 34.4.6(a): D[alpha] ^ D[beta] = D[alpha, beta] (k={k}, l={l})", np.allclose(lhs, D_tensor(al + be, n)))
    tensor_wedge = ch34.wedge_t(D_tensor(al, n), D_tensor(be, n))
    check(f"Conventions box: the basis wedge = ((k+l)!/(k!l!)) Alt(omega (x) eta) of Chapter 17 (k={k}, l={l})",
          np.allclose(lhs, tensor_wedge))
# basis independence: define the product with another basis e'^i = rows of P and compare on random forms
P = rng.normal(size=(n, n))                    # new covector basis e'^i = P[i]
Pinv = np.linalg.inv(P)


def wedge_other_basis(W1, W2):
    """Theorem 34.4.6 with the basis (e'_i) dual to the covectors e'^i = P[i]: expand in e'^I, use D[e'^I, e'^J]."""
    k, l = W1.ndim, W2.ndim
    Eb = Pinv.T                                # e'_j = columns of P^{-1}; rows here
    out = np.zeros((n,) * (k + l))
    for I in ch34.increasing(n, k):
        for J in ch34.increasing(n, l):
            c = evalT(W1, [Pinv[:, i] for i in I]) * evalT(W2, [Pinv[:, j] for j in J])
            out += c * D_tensor([P[i] for i in I] + [P[j] for j in J], n)
    return out


W1, W2 = D_tensor(list(rng.normal(size=(2, n))), n), D_tensor(list(rng.normal(size=(1, n))), n)
check("Theorem 34.4.6(a): the wedge does not depend on the basis used to define it",
      np.allclose(wedge_other_basis(W1, W2), wedge_basis_def(W1, W2, n)))
al = list(rng.normal(size=(3, n)))
us = rng.normal(size=(3, n))
it = wedge_basis_def(wedge_basis_def(D_tensor(al[:1], n), D_tensor(al[1:2], n), n), D_tensor(al[2:], n), n)
close("Theorem 34.4.6(b) (34.4.3): (alpha^1 ^ alpha^2 ^ alpha^3)(u) = det[alpha^i(u_j)]", evalT(it, list(us)),
      D(al, list(us)), 1e-10)
A2, B2, C2 = [D_tensor(list(rng.normal(size=(k, n))), n) for k in (1, 2, 1)]
check("Theorem 34.4.6(c): associativity (W1)", np.allclose(wedge_basis_def(wedge_basis_def(A2, B2, n), C2, n),
                                                          wedge_basis_def(A2, wedge_basis_def(B2, C2, n), n)))
aa = rng.normal(size=n)
check("Theorem 34.4.6(c): alpha ^ alpha = 0 (W2)", np.allclose(wedge_basis_def(D_tensor([aa], n), D_tensor([aa], n), n), 0))
close("Theorem 34.4.6(c): (e^1 ^ ... ^ e^n)(e_1..e_n) = 1 (W4)", evalT(D_tensor(list(E), n), list(E)), 1.0, 1e-12)

# ---------------------------------------------------------------------------
# Conventions box and Exercise 34.4.4: the Alt normalization
# ---------------------------------------------------------------------------
al1, al2 = rng.normal(size=(2, 3))
altw = ch34.alt(np.multiply.outer(al1, al2))
check("Exercise 34.4.4(a): alpha ^_A beta = (1/2) alpha ^ beta", np.allclose(altw, 0.5 * D_tensor([al1, al2], 3)))
E3 = np.eye(3)
e12A = ch34.alt(np.multiply.outer(E3[0], E3[1]))
left = ch34.alt(np.multiply.outer(e12A, E3[2]))
right = ch34.alt(np.multiply.outer(E3[0], ch34.alt(np.multiply.outer(E3[1], E3[2]))))
close("Exercise 34.4.4(b): ((e^1 ^_A e^2) ^_A e^3)(e1,e2,e3) = 1/6 = e^1 ^_A (e^2 ^_A e^3)(...)",
      [left[0, 1, 2], right[0, 1, 2]], [1 / 6, 1 / 6], 1e-12)
for k, l in [(1, 1), (2, 1), (2, 2)]:
    S = D_tensor(list(rng.normal(size=(k, 4))), 4)
    T = D_tensor(list(rng.normal(size=(l, 4))), 4)
    check(f"Conventions box: omega ^_A eta = (k! l!/(k+l)!) omega ^ eta (k={k}, l={l})",
          np.allclose(ch34.alt(np.multiply.outer(S, T)),
                      math.factorial(k) * math.factorial(l) / math.factorial(k + l) * ch34.wedge_t(S, T)))
# symbolic 2x2: Alt(alpha (x) beta)(u, v) = (1/2) det
a1, a2, b1_, b2_, u1, u2, v1, v2 = sp.symbols("a1 a2 b1 b2 u1 u2 v1 v2")
alpha_s, beta_s = sp.Matrix([a1, a2]), sp.Matrix([b1_, b2_])
us_, vs_ = sp.Matrix([u1, u2]), sp.Matrix([v1, v2])
altv = sp.Rational(1, 2) * ((alpha_s.dot(us_)) * (beta_s.dot(vs_)) - (alpha_s.dot(vs_)) * (beta_s.dot(us_)))
detv = sp.Matrix([[alpha_s.dot(us_), alpha_s.dot(vs_)], [beta_s.dot(us_), beta_s.dot(vs_)]]).det()
sym_equal("Conventions box (symbolic): Alt convention = det/2 for two covectors", altv, detv / 2)

# ---------------------------------------------------------------------------
# Proposition 34.4.7, 34.4.8, Corollary 34.4.9
# ---------------------------------------------------------------------------
b1v, b2v = np.array([2.0, 1.0]), np.array([-1.0, 1.0])
om = D_tensor(list(rng.normal(size=(2, 2))), 2)
vals = [evalT(om, [b1v, b2v]), evalT(om, [b1v, b1v + b2v]), evalT(om, [b2v, -b1v])]
check("Proposition 34.4.7: same 2-vector (Figure 34.1.2(a)) gives the same value", np.allclose(vals, vals[0]))
# well-defined on sums: xi = p^q + r^s written differently
p_, q_, r_, s_ = rng.normal(size=(4, 4))
om4 = D_tensor(list(rng.normal(size=(2, 4))), 4) + D_tensor(list(rng.normal(size=(2, 4))), 4)
v1_ = evalT(om4, [p_, q_]) + evalT(om4, [r_, s_])
v2_ = evalT(om4, [p_ + 2 * q_, q_]) + evalT(om4, [r_, s_ - 3 * r_])
close("Proposition 34.4.7: omega(xi) is independent of how xi is written", v1_, v2_, 1e-10)
# omega(xi) = sum_I omega_I xi^I
xi = ch34.vec_tensor(p_, q_) + ch34.vec_tensor(r_, s_)
close("Proposition 34.4.7: omega(xi) = sum_I omega_I xi^I", v1_, sum(om4[I] * xi[I] for I in ch34.increasing(4, 2)), 1e-10)

Ar = rng.normal(size=(3, 3))
G = Ar.T @ Ar + np.eye(3)
Gi = np.linalg.inv(G)
uu, vv = rng.normal(size=(2, 3))
al3, be3 = rng.normal(size=(2, 3))
flat_uv = D_tensor([G @ uu, G @ vv], 3)
check("Proposition 34.4.8(a): (u^v)^flat = u^flat ^ v^flat (function tensor = G applied to every slot)",
      np.allclose(flat_uv, np.einsum("ia,jb,ab->ij", G, G, ch34.vec_tensor(uu, vv))))
lhs = D([al3, be3], [uu, vv])
sh_a, sh_b = Gi @ al3, Gi @ be3
rhs = np.linalg.det(np.array([[sh_a @ G @ uu, sh_a @ G @ vv], [sh_b @ G @ uu, sh_b @ G @ vv]]))
close("Proposition 34.4.8(b): omega(xi) = <omega^sharp, xi>_g", lhs, rhs, 1e-10)
ga, gb = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
ip_forms = np.linalg.det(np.array([[x @ Gi @ y for y in gb] for x in ga]))
ip_vecs = np.linalg.det(np.array([[(Gi @ x) @ G @ (Gi @ y) for y in gb] for x in ga]))
close("Proposition 34.4.8(c): sharp is an isometry", ip_forms, ip_vecs, 1e-10)
Q5, _ = np.linalg.qr(rng.normal(size=(5, 5)))
sharps = Q5[:, :3]
us5 = rng.normal(size=(3, 5))
proj = [sharps @ (sharps.T @ x) for x in us5]
valk = D([sharps[:, i] for i in range(3)], list(us5))
check("Corollary 34.4.9: pi u1 ^ pi u2 ^ pi u3 = value * (wedge of the sharps) (k=3, n=5)",
      np.allclose(ch34.vec_tensor(*proj), valk * ch34.vec_tensor(*sharps.T)))

# ---------------------------------------------------------------------------
# Example 34.4.10
# ---------------------------------------------------------------------------
u3, v3 = np.array([1.0, 2.0, 3.0]), np.array([-1.0, 0.0, 2.0])
close("Example 34.4.10(a): (e^12, e^13, e^23)(u, v) = (2, 5, 4)",
      [D([E3[0], E3[1]], [u3, v3]), D([E3[0], E3[2]], [u3, v3]), D([E3[1], E3[2]], [u3, v3])], [2, 5, 4], 1e-12)
Am, Um = rng.normal(size=(2, 3, 3))
close("Example 34.4.10(c): (alpha^beta^gamma)(u,v,w) = det A det U",
      D([Am[:, 0], Am[:, 1], Am[:, 2]], [Um[:, 0], Um[:, 1], Um[:, 2]]), np.linalg.det(Am) * np.linalg.det(Um), 1e-10)

# ---------------------------------------------------------------------------
# Proposition 34.4.11 and Example 34.4.12
# ---------------------------------------------------------------------------
ok = True
for k in range(4):
    for I in ch34.increasing(3, k):
        Ic = tuple(j for j in range(3) if j not in I)
        s = ch34.perm_sign(I + Ic)
        eI = D_tensor([E3[i] for i in I], 3) if I else np.array(1.0)
        eIc = D_tensor([E3[i] for i in Ic], 3) if Ic else np.array(1.0)
        ok &= np.allclose(ch34.hodge_t(eI, 3), s * eIc)
check("Proposition 34.4.11(a): star e^I = sgn(I, I^c) e^Ic (R^3, Euclidean)", ok)
fa, fb = D_tensor([rng.normal(size=3)], 3) + 0, D_tensor([rng.normal(size=3)], 3)
top = ch34.wedge_t(fa, ch34.hodge_t(fb, 3))[0, 1, 2]
close("Proposition 34.4.11(b): alpha ^ star beta = <alpha, beta> e^123", top, ch34.inner_t(fa, fb), 1e-10)

# (c) with a non-Euclidean metric: star (vectors, metric G) and star (forms, metric G^{-1}) commute with flat
L = np.linalg.cholesky(G)
Fm = np.linalg.inv(L).T                         # g-orthonormal basis f_j = columns, det > 0
assert np.allclose(Fm.T @ G @ Fm, np.eye(3)) and np.linalg.det(Fm) > 0


def act(R, T):
    T = np.asarray(T, float)
    for ax in range(T.ndim):
        T = np.moveaxis(np.tensordot(R, T, axes=([1], [ax])), 0, ax)
    return T


def star_vec_G(T):
    """Hodge star of k-vectors for g: transport to the g-orthonormal basis F, use the standard table, transport back."""
    return act(Fm, ch34.hodge_t(act(np.linalg.inv(Fm), T), 3))


def star_form_G(W):
    """Hodge star of k-forms (function tensors) for the dual inner product: components in the dual basis of F."""
    WF = act(Fm.T, W)                          # omega(f_i, ...)
    return act(np.linalg.inv(Fm).T, ch34.hodge_t(WF, 3))


for k in (0, 1, 2, 3):
    T = ch34.vec_tensor(*rng.normal(size=(k, 3))) if k else np.array(1.3)
    lhs = act(G, star_vec_G(T))               # (star xi)^flat
    rhs = star_form_G(act(G, T))              # star(xi^flat)
    check(f"Proposition 34.4.11(c): (star xi)^flat = star(xi^flat) for a non-Euclidean metric (k={k})", np.allclose(lhs, rhs))
check("Proposition 34.4.11(d): star star = 1 on forms in R^3 (metric G)",
      all(np.allclose(star_form_G(star_form_G(W)), W) for W in [D_tensor([rng.normal(size=3)], 3),
                                                                 D_tensor(list(rng.normal(size=(2, 3))), 3)]))
avec, bvec = rng.normal(size=(2, 3))
st = ch34.hodge_t(avec, 3)                      # star(a^flat) as a function tensor (Euclidean)
close("Example 34.4.12(b) (34.4.4): (star alpha)(u, v) = <a, u x v>", evalT(st, [x, y]), avec @ np.cross(x, y), 1e-10)
close("Example 34.4.12(b): star(alpha ^ beta) = (a x b)^flat", ch34.hodge_t(D_tensor([avec, bvec], 3), 3), np.cross(avec, bvec),
      1e-10)
close("Example 34.4.12(a): R^2 forms star e^1 = e^2, star e^2 = -e^1",
      [ch34.hodge_t(np.array([1.0, 0.0]), 2), ch34.hodge_t(np.array([0.0, 1.0]), 2)], [[0, 1], [-1, 0]], 1e-12)

# ---------------------------------------------------------------------------
# Smooth vs. Discrete box, Figure 34.4.2, Implementation box
# ---------------------------------------------------------------------------
ns = ch34.run_block(PAGE, "code-34-4-1")
close("Implementation box: edge values (2.5, -0.3, -2.2)", ns["edge_values"], [2.5, -0.3, -2.2], 1e-12)
close("Implementation box: face value 1.45", ns["face_value"], 1.45, 1e-12)
Xt = np.array(ns["tri"])
close("Implementation box vs dgdec.derham1 (constant 1-form, quadrature)",
      dd.derham1(lambda p: np.array([1.0, 1.0]), Xt, np.array([[0, 1], [1, 2], [2, 0]])), ns["edge_values"], 1e-12)
close("Implementation box vs dgdec.derham2 (constant density 1)", dd.derham2(lambda p: 1.0, Xt, np.array([[0, 1, 2]]))[0],
      ns["face_value"], 1e-12)
si = ns["simplex_integral"]
X3 = rng.normal(size=(3, 3))
wv = rng.normal(size=3)
close("Discrete box: constant 2-form <w, u x v> on a triangle in R^3 matches dgdec.derham2",
      si(lambda p, q: wv @ np.cross(p, q), X3), dd.derham2(lambda pnt: wv, X3, np.array([[0, 1, 2]]))[0], 1e-12)
X4 = rng.normal(size=(4, 3))
close("Discrete box: e^123 on a tetrahedron = signed volume det/6",
      si(lambda p, q, r: np.linalg.det(np.column_stack([p, q, r])), X4),
      np.linalg.det(np.column_stack([X4[1] - X4[0], X4[2] - X4[0], X4[3] - X4[0]])) / 6, 1e-12)
check("Discrete box: reversing the orientation flips the sign",
      np.isclose(si(ns["dxdy"], [Xt[0], Xt[2], Xt[1]]), -ns["face_value"]))
close("Figure 34.4.2: parallelogram det = 2.9", ns["dxdy"](Xt[1] - Xt[0], Xt[2] - Xt[0]), 2.9, 1e-12)

# ---------------------------------------------------------------------------
# Exercises
# ---------------------------------------------------------------------------
al1x, be1x = np.array([1.0, 2.0, 0.0]), np.array([0.0, 1.0, -1.0])
ux, vx = np.array([1.0, 0.0, 1.0]), np.array([0.0, 1.0, 1.0])
close("Exercise 34.4.1(a): (alpha^beta)(u,v) = 2", D([al1x, be1x], [ux, vx]), 2.0, 1e-12)
cw = ch34.comps(D_tensor([al1x, be1x], 3))
close("Exercise 34.4.1(b): alpha^beta = e^12 - e^13 - 2 e^23", [cw[(0, 1)], cw[(0, 2)], cw[(1, 2)]], [1, -1, -2], 1e-12)
cx = ch34.comps(ch34.vec_tensor(ux, vx))
close("Exercise 34.4.1(b): u^v components (1, 1, -1) and the sum 2", [cx[(0, 1)], cx[(0, 2)], cx[(1, 2)],
                                                                      sum(cw[I] * cx[I] for I in cw)], [1, 1, -1, 2], 1e-12)
E4 = np.eye(4)
om2 = D_tensor([E4[0], E4[1]], 4) + D_tensor([E4[2], E4[3]], 4)
close("Exercise 34.4.2(d): (e^12 + e^34)^2 = 2 e^1234", ch34.wedge_t(om2, om2)[0, 1, 2, 3], 2.0, 1e-12)
Ax = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]]).T
Ux = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0], [1.0, 0.0, 1.0]]).T
close("Exercise 34.4.3: det A = 2, det U = 6, value 12",
      [np.linalg.det(Ax), np.linalg.det(Ux), D([Ax[:, i] for i in range(3)], [Ux[:, j] for j in range(3)])], [2, 6, 12], 1e-12)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch34.run_block(PAGE, "code-ex-34-4-5")
check("Exercise 34.4.5: solution output equals out-ex-34-4-5 (True True)",
      buf.getvalue().strip() == ch34.output_block(PAGE, "out-ex-34-4-5") == "True True", buf.getvalue())
nrm = np.cross(af, bf)
uxv = np.cross(uf, vf)
close("Exercise 34.4.6: a# x b# = (-sin40, 0, cos40), u x v = (-0.225, -0.335, 0.99)", list(nrm) + list(uxv),
      [-np.sin(t), 0, np.cos(t), -0.225, -0.335, 0.99], 1e-12)
close("Exercise 34.4.6: <normals> = 0.903 = value", nrm @ uxv, val, 1e-12)
close("Exercise 34.4.6: cos(theta) = 0.903/1.069 = 0.845", val / np.linalg.norm(uxv), 0.845, 5e-4)

check("Implementation box code-34-4-1: at most 30 lines", ch34.code_lines(PAGE, "code-34-4-1") <= 30)
summary()
