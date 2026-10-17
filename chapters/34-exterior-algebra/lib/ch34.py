"""Chapter 34 helpers for verify and figure scripts (exterior algebra in R^n).

Two independent implementations live here so that verify scripts can cross-check the code printed in the
Implementation boxes (which uses sorted index tuples and inversion counts):

1. ``run_block`` / ``output_block``: thin wrappers of tools/dgcode.py that execute the code printed in
   ``<pre id="code-...">`` blocks (and read the printed output ``<pre id="out-ex-...">``), so the verify script
   runs exactly the code the reader sees (GUIDELINES 2.5).
2. A *tensor* model: a k-vector in R^n is stored as a fully antisymmetric array of shape (n,)*k with the
   determinant normalization  u_1 ^ ... ^ u_k = sum_sigma sgn(sigma) u_sigma(1) (x) ... (x) u_sigma(k),
   so its entry at an increasing multi-index I is the component xi^I = det(U_I). The wedge of tensors is
   ((k+l)!/(k! l!)) Alt(S (x) T) and the Hodge star contracts with the Levi-Civita symbol. Signs of
   permutations come from sympy.combinatorics (not from inversion counting).

Indices are 0-based in code (e_0, ..., e_{n-1}); the text uses 1-based indices.
"""

import itertools
import math
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import numpy as np  # noqa: E402
from sympy.combinatorics import Permutation  # noqa: E402

CHAPTER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# code blocks in the HTML pages
# ---------------------------------------------------------------------------

def page_path(page):
    return page if os.path.isabs(page) else os.path.join(CHAPTER_DIR, page)


def run_block(page, block_id, ns=None):
    """Execute <pre id="block_id"><code>...</code></pre> of ``page`` as printed (tools/dgcode.py, GUIDELINES 2.5)."""
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
# tensor model of the exterior algebra (independent of the sorted-tuple code)
# ---------------------------------------------------------------------------

def perm_sign(seq):
    """Sign of the permutation that sorts the distinct integers ``seq`` (sympy)."""
    seq = list(seq)
    order = sorted(range(len(seq)), key=lambda a: seq[a])
    return int(Permutation(order).signature()) if len(seq) > 1 else 1


def increasing(n, k):
    """All increasing multi-indices I of length k in {0, ..., n-1}."""
    return list(itertools.combinations(range(n), k))


def alt(T):
    """Alt(T) = (1/k!) sum_sigma sgn(sigma) T permuted (k = T.ndim)."""
    k = T.ndim
    if k <= 1:
        return T.copy()
    out = np.zeros_like(T, dtype=float)
    for p in itertools.permutations(range(k)):
        out = out + perm_sign(p) * np.transpose(T, p)
    return out / math.factorial(k)


def vec_tensor(*us):
    """u_1 ^ ... ^ u_k as an antisymmetric tensor (determinant normalization)."""
    us = [np.asarray(u, float) for u in us]
    if not us:
        return np.array(1.0)
    T = us[0]
    for u in us[1:]:
        T = np.multiply.outer(T, u)
    return math.factorial(len(us)) * alt(T)


def wedge_t(S, T):
    """Wedge of antisymmetric tensors: ((k+l)!/(k! l!)) Alt(S (x) T)."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    k, l = S.ndim, T.ndim
    if k == 0:
        return float(S) * T
    if l == 0:
        return float(T) * S
    c = math.factorial(k + l) / (math.factorial(k) * math.factorial(l))
    return c * alt(np.multiply.outer(S, T))


def comps(T):
    """{I: T[I]} over increasing I (0-based), dropping nothing (zeros included)."""
    T = np.asarray(T, float)
    n = T.shape[0] if T.ndim else 0
    if T.ndim == 0:
        return {(): float(T)}
    return {I: float(T[I]) for I in increasing(n, T.ndim)}


def from_comps(C, n):
    """Antisymmetric tensor with components C = {I: c} (I increasing, all of one length k)."""
    ks = {len(I) for I in C}
    if not ks:
        return np.array(0.0)
    (k,) = ks
    if k == 0:
        return np.array(float(C[()]))
    T = np.zeros((n,) * k)
    for I, c in C.items():
        for p in itertools.permutations(range(k)):
            T[tuple(I[a] for a in p)] += perm_sign(p) * c
    return T


def levi_civita(n):
    E = np.zeros((n,) * n)
    for p in itertools.permutations(range(n)):
        E[p] = perm_sign(p)
    return E


def hodge_t(T, n):
    """Euclidean Hodge star of an antisymmetric k-tensor in R^n (standard orientation):
    (star T)_{j_1..j_{n-k}} = (1/k!) sum_{i} T_{i_1..i_k} eps_{i_1..i_k j_1..j_{n-k}}."""
    T = np.asarray(T, float)
    k = T.ndim
    E = levi_civita(n)
    if k == 0:
        return float(T) * E
    out = np.tensordot(T, E, axes=(list(range(k)), list(range(k))))
    return out / math.factorial(k)


def inner_t(S, T):
    """Inner product of k-vectors: sum over increasing I of S^I T^I (= full contraction / k!)."""
    S, T = np.asarray(S, float), np.asarray(T, float)
    if S.ndim == 0:
        return float(S) * float(T)
    return float(np.sum(S * T)) / math.factorial(S.ndim)


def gram_det(*us):
    U = np.column_stack(us)
    return float(np.linalg.det(U.T @ U))
