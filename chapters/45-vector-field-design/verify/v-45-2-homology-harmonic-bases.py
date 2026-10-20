"""Verification for Section 45.2 (Homology Generators and Harmonic Bases).

Run from the project root:
``PYTHONPATH=tools python3 chapters/45-vector-field-design/verify/v-45-2-homology-harmonic-bases.py``
"""

import pathlib
import sys
import warnings

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch45  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

warnings.simplefilter("ignore")
PAGE = HERE / "45-2-homology-harmonic-bases.html"


def rank(A):
    return int(np.linalg.matrix_rank(A))


# ---------------------------------------------------------------- Proposition 45.2.2 and Example 45.2.3
for name, (V, F), expect in (("icosphere(2)", dm.icosphere(2), 0), ("torus_mesh(2, 0.8, 12, 24)", dm.torus_mesh(2, 0.8, 12, 24), 2)):
    D0, D1 = dd.d0(F, len(V)), dd.d1(F)
    n1 = D0.shape[0]
    b1 = (n1 - rank(D0.T)) - rank(D1.T)                   # dim ker d1_partial - dim im partial2
    check(f"Proposition 45.2.2 ({name}): dim H_1 = n1 - rank d0 - rank d1 = {expect} = 2 - chi",
          b1 == expect == 2 - dm.euler_characteristic(F) and b1 == (n1 - rank(D1)) - rank(D0))
ft = ch45.FlatTorus(12, 12)
check("Example 45.2.3(c): flat torus has dim H_1 = 2", (432 - rank(dd.d0(ft.F))) - rank(dd.d1(ft.F)) == 2)
# annulus of Example 36.2.15: three circles of 12 vertices
rr, nt = (1.0, 1.5, 2.0), 12
Fa = []
for a in range(2):
    for k in range(nt):
        p, q = a * nt + k, a * nt + (k + 1) % nt
        p2, q2 = p + nt, q + nt
        Fa += [[p, q, q2], [p, q2, p2]]
Fa = np.array(Fa)
Da0, Da1 = dd.d0(Fa, 3 * nt), dd.d1(Fa)
check("Example 45.2.3(b): the annulus has 48 faces and dim H_1 = 1",
      len(Fa) == 48 and (Da0.shape[0] - rank(Da0)) - rank(Da1) == 1)
V, F = dm.icosphere(2)
E, idx = dd.edge_list(F)
hm = dd.HalfedgeMesh(F)
for i in (0, 37):
    ring = hm.vertex_one_ring(i)
    ell = np.zeros(len(E))
    for a, b in zip(ring, ring[1:] + ring[:1]):
        ell[idx[(min(a, b), max(a, b))]] += 1.0 if a < b else -1.0
    s = np.zeros(len(F))
    s[hm.vertex_faces(i)] = 1.0
    check(f"Example 45.2.3(d) / Exercise 45.2.2: the link loop of vertex {i} is the boundary of its star",
          np.allclose(dd.d1(F).T @ s, ell))

# ---------------------------------------------------------------- Example 45.2.6, Non-example 45.2.7
nu, nv = 12, 24
V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
E, idx = dd.edge_list(F)
D0, D1 = dd.d0(F, len(V)), dd.d1(F)
u = 2 * np.pi * (np.arange(len(V)) // nv) / nu
v = 2 * np.pi * (np.arange(len(V)) % nv) / nv
du, dv = ch45.wrap(D0 @ u), ch45.wrap(D0 @ v)
check("Example 45.2.6: R_1(du) and R_1(dv) are closed", np.abs(D1 @ du).max() < 1e-12 and np.abs(D1 @ dv).max() < 1e-12)


def loop(vs):
    c = np.zeros(len(E))
    for a, b in zip(vs, vs[1:] + vs[:1]):
        c[idx[(min(a, b), max(a, b))]] += 1.0 if a < b else -1.0
    return c


cu = loop([i * nv + 0 for i in range(nu)])
cv = loop([0 * nv + j for j in range(nv)])
cu5 = loop([i * nv + 5 for i in range(nu)])
check("Example 45.2.6: c_u and c_v are cycles", np.abs(D0.T @ cu).max() == 0 and np.abs(D0.T @ cv).max() == 0)
Pm = np.array([[cu @ du, cu @ dv], [cv @ du, cv @ dv]])
check("Example 45.2.6: the period matrix of (du, dv) on (c_u, c_v) is 2 pi I", np.allclose(Pm, 2 * np.pi * np.eye(2)))
band = np.zeros(len(F))
for f in range(len(F)):
    q = f // 2
    if q % nv < 5:
        band[f] = 1.0
bd = D1.T @ band
check("Non-example 45.2.7: the band 0 <= j < 5 has boundary +-(c_u' - c_u)",
      np.allclose(bd, cu5 - cu) or np.allclose(bd, cu - cu5))
check("Non-example 45.2.7: the two meridians have equal periods", np.isclose(cu @ du, cu5 @ du) and np.isclose(cu @ dv, cu5 @ dv))

# ---------------------------------------------------------------- Proposition 45.2.8
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("torus", dm.torus_mesh(2.0, 0.8, 8, 16))):
    D0, D1 = dd.d0(F, len(V)), dd.d1(F)
    E, left, right = ch45.left_right_faces(F)
    # dual boundary operator from the definition: e* runs right -> left, so its boundary is [left] - [right]
    Dt1 = np.zeros((len(F), len(E)))
    Dt1[left, np.arange(len(E))] += 1.0
    Dt1[right, np.arange(len(E))] -= 1.0
    check(f"Proposition 45.2.8 ({name}): the dual boundary of dual 1-chains is d1", np.allclose(Dt1, D1))
    # boundary of the dual cell of vertex i: dual edges of the edges at i, counterclockwise = -d0 e_i
    hm = dd.HalfedgeMesh(F)
    ok = True
    for i in range(len(V)):
        ring_f = hm.vertex_faces(i)
        c = np.zeros(len(E))
        for f, g in zip(ring_f, ring_f[1:] + ring_f[:1]):       # cross from f to g (counterclockwise)
            shared = set(F[f]) & set(F[g]) - {i}
            j = shared.pop()
            e = list(map(tuple, E)).index((min(i, j), max(i, j)))
            c[e] += 1.0 if (right[e] == f and left[e] == g) else -1.0
        ok = ok and np.allclose(c, -(D0[:, i]))
    check(f"Proposition 45.2.8 ({name}): the counterclockwise boundary of every dual cell i* is -d0 e_i", ok)

# ---------------------------------------------------------------- Lemma 45.2.10, Theorem 45.2.11, Example 45.2.12
rng = np.random.default_rng(4)
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("torus_mesh(2, 0.8, 8, 16)", dm.torus_mesh(2.0, 0.8, 8, 16)),
                     ("torus_mesh(2, 0.8, 12, 24)", dm.torus_mesh(2.0, 0.8, 12, 24)), ("genus2_mesh(0.2)", dm.genus2_mesh(0.2)),
                     ("flat torus", (None, ch45.FlatTorus(12, 12).F))):
    n0, n1, n2 = dm.counts(F)
    for roots in ((0, 0), (int(rng.integers(n0)), int(rng.integers(n2)))):
        tc, C, Cs = ch45.generators(F, root_vertex=roots[0], root_face=roots[1])
        D0, D1 = dd.d0(F, n0), dd.d1(F)
        m = len(tc["rest"])
        ok = (len(tc["T"]) == n0 - 1 and len(tc["Ts"]) == n2 - 1 and m == 2 - (n0 - n1 + n2)
              and not (tc["T"] & tc["Ts"]))
        if m:
            ok = ok and np.abs(D0.T @ C.T).max() == 0 and np.abs(D1 @ Cs.T).max() == 0 and np.array_equal(Cs @ C.T, np.eye(m))
        check(f"Theorem 45.2.11 ({name}, roots {roots}): |T| = n0-1, |T*| = n2-1, |R| = 2 - chi = {m}, "
              "loops are (dual) cycles, c*_a . c_b = delta_ab", ok)
# Lemma 45.2.10: a chain supported on a spanning tree with zero boundary is zero
V, F = dm.torus_mesh(2.0, 0.8, 8, 16)
tc = ch45.tree_cotree(F)
Tl = sorted(tc["T"])
D0 = dd.d0(F, len(V))
check("Lemma 45.2.10: d0^T restricted to the tree edges has trivial kernel", rank(D0[Tl, :].T) == len(Tl))
# Implementation box
ns, same, out = ch45.run_printed(PAGE, "code-45-2-1", "out-45-2-1")
check("code-45-2-1 runs and prints exactly out-45-2-1", same)
check("code-45-2-1 has at most 30 lines", dgcode.code_lines(PAGE, "code-45-2-1") <= 30)
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("torus_mesh(2, 0.8, 12, 24)", dm.torus_mesh(2, 0.8, 12, 24))):
    T, Ts, rest = ns["tree_cotree"](F)
    tc = ch45.tree_cotree(F)
    check(f"code-45-2-1 and lib/ch45.tree_cotree give the same decomposition ({name})",
          T == tc["T"] and Ts == tc["Ts"] and rest == tc["rest"])
# Example 45.2.12 / Figure 45.2.1: periods of the generators of torus_mesh(2, 0.8, 8, 16)
for nu, nv in ((8, 16), (24, 48)):
    V, F = dm.torus_mesh(2.0, 0.8, nu, nv)
    tc, C, Cs = ch45.generators(F)
    D0 = dd.d0(F, len(V))
    uu = 2 * np.pi * (np.arange(len(V)) // nv) / nu
    vv = 2 * np.pi * (np.arange(len(V)) % nv) / nv
    per = np.array([[c @ ch45.wrap(D0 @ uu), c @ ch45.wrap(D0 @ vv)] for c in C]) / (2 * np.pi)
    check(f"Example 45.2.12 / 45.2.15 (torus_mesh {nu} x {nv}): the (du, dv) periods of c_1, c_2 are (-1, 0), (-1, 1) times 2 pi",
          np.allclose(per, [[-1, 0], [-1, 1]]))

# ---------------------------------------------------------------- Proposition 45.2.14, Example 45.2.15
R, r = 2.0, 0.8
V, F = dm.torus_mesh(R, r, 24, 48)
D0, D1, s1 = dd.d0(F, len(V)), dd.d1(F), dd.hodge1(V, F)
tc, C, Cs = ch45.generators(F)
G = ch45.harmonic_basis(D0, s1, Cs)
check("Proposition 45.2.14(a): c_b . gamma_a = delta_ab", np.allclose(C @ G.T, np.eye(2), atol=1e-10))
check("Proposition 45.2.14(b): gamma_a are closed, co-closed, independent",
      np.abs(D1 @ G.T).max() < 1e-12 and np.abs(D0.T @ (s1[:, None] * G.T)).max() < 1e-12 and rank(G) == 2)
gram = G @ (s1[:, None] * G.T)
P = Cs @ (s1[:, None] * G.T)
check("Proposition 45.2.14(c): P = c*_a . star1 gamma_b equals the Gram matrix and is SPD",
      np.allclose(P, gram, atol=1e-10) and np.linalg.eigvalsh((P + P.T) / 2).min() > 0)
uu = np.arctan2(V[:, 2], np.hypot(V[:, 0], V[:, 1]) - R)
vv = np.arctan2(V[:, 1], V[:, 0])
H2 = ch45.harmonic_basis(D0, s1, np.array([ch45.wrap(D0 @ uu), ch45.wrap(D0 @ vv)]))
check("Example 45.2.15: gamma_1 = -(harm du + harm dv)/2 pi, gamma_2 = harm dv / 2 pi",
      np.allclose(G[0], -(H2[0] + H2[1]) / (2 * np.pi), atol=1e-12) and np.allclose(G[1], H2[1] / (2 * np.pi), atol=1e-12))
check("Example 45.2.15: Gram matrix (2.7382, -0.4359; -0.4359, 0.4359)",
      np.allclose(np.round(gram, 4), [[2.7382, -0.4359], [-0.4359, 0.4359]]))
c = np.sqrt(R * R - r * r)
sm = np.array([[c / r + r / c, -r / c], [-r / c, r / c]])
check("Example 45.2.15: smooth Gram matrix (2.7277, -0.4364; -0.4364, 0.4364)",
      np.allclose(np.round(sm, 4), [[2.7277, -0.4364], [-0.4364, 0.4364]]))
# smooth norms: ||gamma_du||^2 = 4 pi^2 c / r, ||dv||^2 = 4 pi^2 r / c (quadrature on the torus of revolution)
n = 400
U_, W_ = np.meshgrid(2 * np.pi * np.arange(n) / n, 2 * np.pi * np.arange(n) / n, indexing="ij")
rho = R + r * np.cos(U_)
dA = r * rho * (2 * np.pi / n) ** 2
close("Example 45.2.15: ||gamma_du||^2 = 4 pi^2 c / r", np.sum((c / rho) ** 2 / r ** 2 * dA), 4 * np.pi ** 2 * c / r, 1e-9)
close("Example 45.2.15: ||dv||^2 = 4 pi^2 r / c", np.sum(1 / rho ** 2 * dA), 4 * np.pi ** 2 * r / c, 1e-9)
# Remark 45.2.16: harmonic forms are determined by their periods on the generators (flat torus)
ft = ch45.FlatTorus(12, 12)
s0, s1f, s2 = ch45.stars_from_corners(ft.F, ft.C)
tcf, Cf, Csf = ch45.generators(ft.F)
Gf = ch45.harmonic_basis(dd.d0(ft.F), s1f, Csf)
h = ft.edge_vec @ np.array([0.3, 0.1])
check("Remark 45.2.16 (flat torus): a harmonic form equals sum_a (c_a . h) gamma_a", np.allclose(Gf.T @ (Cf @ h), h, atol=1e-12))

# ---------------------------------------------------------------- Exercises
for name, F in (("octahedron", np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])),):
    n0, n1, n2 = dm.counts(F)
    tc = ch45.tree_cotree(F)
    check("Exercise 45.2.1: octahedron counts (5, 7, 0)", (len(tc["T"]), len(tc["Ts"]), len(tc["rest"])) == (5, 7, 0)
          and (n0, n1, n2) == (6, 12, 8) and dm.oriented_consistently(F))
ns, same, out = ch45.run_printed(PAGE, "code-ex-45-2-5", "out-ex-45-2-5")
check("code-ex-45-2-5 runs and prints exactly out-ex-45-2-5", same)
# Exercise 45.2.4: the reversed order (cotree first) also works
V, F = dm.genus2_mesh(0.2)
E, left, right = ch45.left_right_faces(F)
n0, n2 = len(V), len(F)
from collections import deque  # noqa: E402
fnbr = [[] for _ in range(n2)]
for e in range(len(E)):
    fnbr[left[e]].append((int(right[e]), e))
    fnbr[right[e]].append((int(left[e]), e))
seen, Ts, bag = {0}, set(), deque([0])
while bag:
    f = bag.popleft()
    for g, e in fnbr[f]:
        if g not in seen:
            seen.add(g), Ts.add(e), bag.append(g)
nbr = [[] for _ in range(n0)]
for e, (i, j) in enumerate(E):
    if e not in Ts:
        nbr[i].append((int(j), e))
        nbr[j].append((int(i), e))
seen2, T, bag = {0}, set(), deque([0])
while bag:
    a = bag.popleft()
    for b, e in nbr[a]:
        if b not in seen2:
            seen2.add(b), T.add(e), bag.append(b)
check("Exercise 45.2.4 (genus2_mesh(0.2)): cotree first, then a primal tree avoiding it, leaves 4 edges",
      len(seen2) == n0 and len(E) - len(T) - len(Ts) == 4)

# Exercise 45.2.6: a dual cycle homologous to c*_a (c*_a - d0 y) is not exact and has the same harmonic part
V, F = dm.torus_mesh(2.0, 0.8, 12, 24)
D0, s1 = dd.d0(F, len(V)), dd.hodge1(V, F)
tc, C, Cs = ch45.generators(F)
G = ch45.harmonic_basis(D0, s1, Cs)
yv = np.random.default_rng(26).integers(-2, 3, len(V)).astype(float)
cmod = Cs[0] - D0 @ yv
Gm = ch45.harmonic_basis(D0, s1, cmod[None, :])[0]
check("Exercise 45.2.6: c_a . (c*_a - d0 y) = 1 and the harmonic part is still gamma_a",
      np.isclose(C[0] @ cmod, 1.0) and np.abs(Gm - G[0]).max() < 1e-12)

summary()
