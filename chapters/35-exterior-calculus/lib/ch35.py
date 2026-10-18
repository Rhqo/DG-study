"""Chapter 35 helpers (exterior calculus on open sets of R^n) for the verify and figure scripts.

Two independent symbolic models of differential forms live here, so that a verify script can cross-check the
code printed in an Implementation box (which sorts index tuples and counts inversions):

1. dict model: a k-form on R^n is {increasing 0-based tuple I: sympy coefficient a_I}, i.e. sum_I a_I dx^I.
   ``wedge``, ``hodge`` (Euclidean, standard orientation), ``d`` (the coordinate formula of Theorem 35.2.4).
2. tensor model: a k-form is its function tensor T[i_1, ..., i_k] = omega(e_i1, ..., e_ik) (a fully
   antisymmetric nested dict over all index tuples). Here the exterior derivative uses the invariant formula for
   the commuting coordinate fields,
       (d omega)(e_i0, ..., e_ik) = sum_m (-1)^m d_{i_m} [ omega(e_i0, ..., e_im-hat, ..., e_ik) ],
   and signs of permutations come from sympy.combinatorics (not from inversion counting).

Also: wrappers of tools/dgcode.py (run the printed code blocks), vector calculus in R^3, top-degree pullbacks
to a parameter domain (integrals over parametrized patches), the codifferential, and numeric quadrature.
Indices are 0-based in code (dx^0, ..., dx^{n-1}); the text uses 1-based indices.
"""

import itertools
import math
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
from sympy.combinatorics import Permutation  # noqa: E402

CHAPTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# code blocks in the HTML pages (GUIDELINES 2.5)
# ---------------------------------------------------------------------------

def page_path(page):
    return page if os.path.isabs(page) else os.path.join(CHAPTER_DIR, page)


def run_block(page, block_id, ns=None):
    """Execute <pre id="block_id"><code>...</code></pre> of ``page`` exactly as printed (tools/dgcode.py)."""
    import dgcode
    return dgcode.run_block(page_path(page), block_id, ns)


def output_block(page, out_id):
    """The printed output <pre id="out-ex-N-M-K"> of a coding exercise, stripped."""
    import dgcode
    return dgcode.code_block(page_path(page), out_id).strip()


def code_lines(page, block_id):
    import dgcode
    return dgcode.code_lines(page_path(page), block_id)


# ---------------------------------------------------------------------------
# permutations
# ---------------------------------------------------------------------------

def perm_sign(seq):
    """Sign of the permutation that sorts the distinct integers ``seq`` (sympy.combinatorics); 0 if repeated."""
    seq = list(seq)
    if len(set(seq)) < len(seq):
        return 0
    if len(seq) <= 1:
        return 1
    order = sorted(range(len(seq)), key=lambda a: seq[a])
    return int(Permutation(order).signature())


def increasing(n, k):
    return list(itertools.combinations(range(n), k))


# ---------------------------------------------------------------------------
# 1. dict model
# ---------------------------------------------------------------------------

def clean(form):
    """Simplify the coefficients and drop zeros."""
    out = {}
    for I, a in form.items():
        a = sp.simplify(a)
        if a != 0:
            out[tuple(I)] = a
    return out


def add(*forms):
    out = {}
    for f in forms:
        for I, a in f.items():
            out[I] = out.get(I, 0) + a
    return clean(out)


def scale(c, form):
    return clean({I: c * a for I, a in form.items()})


def wedge(a, b):
    out = {}
    for I, x in a.items():
        for J, y in b.items():
            s = perm_sign(I + J)
            if s:
                K = tuple(sorted(I + J))
                out[K] = out.get(K, 0) + s * x * y
    return clean(out)


def hodge(form, n):
    """Euclidean Hodge star on R^n with the standard orientation: star dx^I = sgn(I, I^c) dx^{I^c}."""
    out = {}
    for I, a in form.items():
        Ic = tuple(j for j in range(n) if j not in I)
        out[Ic] = out.get(Ic, 0) + perm_sign(I + Ic) * a
    return clean(out)


def d(form, xs):
    """Exterior derivative: d(sum_I a_I dx^I) = sum_I sum_j d_j a_I dx^j ^ dx^I (Theorem 35.2.4)."""
    out = {}
    for I, a in form.items():
        for j, x in enumerate(xs):
            s = perm_sign((j,) + tuple(I))
            if s:
                K = tuple(sorted((j,) + tuple(I)))
                out[K] = out.get(K, 0) + s * sp.diff(a, x)
    return clean(out)


def degree(form):
    ks = {len(I) for I in form}
    if len(ks) > 1:
        raise ValueError("mixed degrees")
    return ks.pop() if ks else None


def codifferential(form, n, xs, k=None):
    """d* = (-1)^(n(k+1)+1) star d star on k-forms of R^n (Theorem 35.5.4)."""
    k = degree(form) if k is None else k
    if k is None:
        return {}
    return scale((-1) ** (n * (k + 1) + 1), hodge(d(hodge(form, n), xs), n))


def equal(a, b):
    """Two forms in the dict model are equal (coefficients simplified)."""
    diff = add(a, scale(-1, b))
    return all(sp.simplify(c) == 0 for c in diff.values())


def evaluate(form, vectors):
    """omega(u_1, ..., u_k) = sum_I a_I det(U_I) (U has the vectors as columns)."""
    k = len(vectors)
    if k == 0:
        return form.get((), 0)
    U = sp.Matrix.hstack(*[sp.Matrix(v) for v in vectors])
    return sp.simplify(sum(a * U.extract(list(I), list(range(k))).det() for I, a in form.items()))


def subs(form, rule):
    return clean({I: sp.sympify(a).subs(rule) for I, a in form.items()})


def one_form(components):
    return clean({(i,): c for i, c in enumerate(components)})


def flat(X):
    return one_form(X)


def sharp(form, n):
    return [form.get((i,), 0) for i in range(n)]


def two_form_r3(w):
    """star(w^flat) = w1 dy^dz + w2 dz^dx + w3 dx^dy as a dict (dz^dx = -dx^dz)."""
    return clean({(1, 2): w[0], (0, 2): -w[1], (0, 1): w[2]})


def two_form_vector(form):
    """(w_yz, w_zx, w_xy) of a 2-form on R^3 (the representation of tools/dgdec.derham2)."""
    return [form.get((1, 2), 0), -form.get((0, 2), 0), form.get((0, 1), 0)]


# ---------------------------------------------------------------------------
# 2. tensor model (independent)
# ---------------------------------------------------------------------------

def to_tensor(form, n, k=None):
    """Function tensor {(i_1..i_k): omega(e_i1..e_ik)} over all k-tuples."""
    k = degree(form) if k is None else k
    T = {}
    for idx in itertools.product(range(n), repeat=k):
        s = perm_sign(idx)
        T[idx] = s * form.get(tuple(sorted(idx)), 0) if s else 0
    return T


def from_tensor(T, n, k):
    return clean({I: T[I] for I in increasing(n, k)})


def d_tensor(T, n, k, xs):
    """(d omega)(e_i0, ..., e_ik) = sum_m (-1)^m d_{i_m} omega(..., e_im-hat, ...)."""
    out = {}
    for idx in itertools.product(range(n), repeat=k + 1):
        out[idx] = sum((-1) ** m * sp.diff(T[idx[:m] + idx[m + 1:]], xs[idx[m]]) for m in range(k + 1))
    return out


def wedge_tensor(S, T, n, k, l):
    """(omega ^ eta)(u_1..u_{k+l}) = (1/(k! l!)) sum_sigma sgn(sigma) omega(u_s1..u_sk) eta(u_s(k+1)..)."""
    out = {}
    c = sp.Rational(1, math.factorial(k) * math.factorial(l))
    perms = list(itertools.permutations(range(k + l)))
    signs = [int(Permutation(list(p)).signature()) for p in perms]
    for idx in itertools.product(range(n), repeat=k + l):
        tot = 0
        for p, s in zip(perms, signs):
            q = tuple(idx[a] for a in p)
            tot += s * S[q[:k]] * T[q[k:]]
        out[idx] = c * tot
    return out


# ---------------------------------------------------------------------------
# vector calculus in R^3 (classical coordinate formulas, independent of forms)
# ---------------------------------------------------------------------------

def grad(f, xs):
    return [sp.diff(f, x) for x in xs]


def curl(X, xs):
    x, y, z = xs
    return [sp.diff(X[2], y) - sp.diff(X[1], z), sp.diff(X[0], z) - sp.diff(X[2], x),
            sp.diff(X[1], x) - sp.diff(X[0], y)]


def div(X, xs):
    return sum(sp.diff(Xi, x) for Xi, x in zip(X, xs))


def laplacian(f, xs):
    return sum(sp.diff(f, x, 2) for x in xs)


# ---------------------------------------------------------------------------
# integrals over parametrized patches (Definition 35.3.5)
# ---------------------------------------------------------------------------

def pullback_top(form, xs, phi, params):
    """Coefficient c(t) of phi^* omega = c dt^1 ^ ... ^ dt^k, i.e. omega_{phi(t)}(d_1 phi, ..., d_k phi)."""
    phi = sp.Matrix(phi)
    rule = dict(zip(xs, list(phi)))
    cols = [phi.diff(t) for t in params]
    return sp.simplify(evaluate(subs(form, rule), cols))


def integrate_patch(form, xs, phi, params, bounds):
    """sympy integral of a k-form over phi: box -> R^n, bounds = [(t1, a1, b1), ...] (innermost first)."""
    c = pullback_top(form, xs, phi, params)
    return sp.simplify(sp.integrate(c, *bounds))


def gauss01(m):
    x, w = np.polynomial.legendre.leggauss(m)
    return 0.5 * (x + 1.0), 0.5 * w


def line_integral_num(alpha, gamma, dgamma, a, b, m=40):
    """Numeric int_a^b alpha_{gamma(t)}(gamma'(t)) dt (Gauss-Legendre)."""
    t, w = gauss01(m)
    t = a + (b - a) * t
    return float((b - a) * sum(wi * np.dot(alpha(gamma(ti)), dgamma(ti)) for ti, wi in zip(t, w)))


def standard_simplex_volume(k):
    return sp.Rational(1, math.factorial(k))
