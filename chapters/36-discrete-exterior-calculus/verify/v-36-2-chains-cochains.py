"""Verification for Section 36.2 (Chains and Cochains).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/verify/v-36-2-chains-cochains.py``
"""

import itertools
import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch36  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "36-2-chains-cochains.html"
rng = np.random.default_rng(3602)
bd = ch36.boundary_of_simplex
bdc = ch36.boundary_of_chain

# ---------------------------------------------------------------- Lemma 36.2.5: well defined
ok = True
for k in range(1, 5):
    base = tuple(range(k + 1))
    ref = bd(base)
    for perm in itertools.permutations(base):
        sgn = dd.permutation_sign(perm, base)
        ok &= bd(perm) == {f: sgn * v for f, v in ref.items()}
check("Lemma 36.2.5: boundary of a reordered simplex = sgn(pi) x boundary (k = 1..4, all permutations)", ok)

# ---------------------------------------------------------------- Theorem 36.2.8
ok = all(bdc(bd(tuple(range(k + 1)))) == {} for k in range(2, 7))
check("Theorem 36.2.8: dd = 0 on the standard k-simplex, k = 2..6", ok)
for name, F in (("icosphere(2)", dm.icosphere(2)[1]), ("torus 8x12", dm.torus_mesh(2, 0.8, 8, 12)[1]),
                ("grid 4x4", ch36.grid(4)[1]), ("annulus", ch36.annulus(3, 12)[1])):
    n0 = int(F.max()) + 1
    check(f"Theorem 36.2.8: boundary1 boundary2 = 0 on {name}", not np.any(dd.boundary1(F, n0) @ dd.boundary2(F)))

# ---------------------------------------------------------------- Examples 36.2.2, 36.2.6
V, F = ch36.grid(2)
E, index = dd.edge_list(F)
check("Example 36.2.6(a): boundary of [i, j] is [j] - [i]", bd((3, 7)) == {(7,): 1, (3,): -1})
check("Example 36.2.6(a): boundary of the path [0,3] + [3,4] + [4,8] is [8] - [0]",
      bdc({(0, 3): 1, (3, 4): 1, (4, 8): 1}) == {(8,): 1, (0,): -1})
check("Example 36.2.2(b): [0,4] - [8,4] = [0,4] + [4,8] (same boundary)",
      bdc({(0, 4): 1, (4, 8): 1}) == {(8,): 1, (0,): -1})
check("Example 36.2.2(c): c - c' = [0,3] + [3,4] - [0,4] is the boundary of [0,3,4]",
      bd((0, 3, 4)) == {(3, 4): 1, (0, 4): -1, (0, 3): 1})
check("Example 36.2.6(b): boundary of [0,1,2] is [1,2] - [0,2] + [0,1] and its boundary is 0",
      bd((0, 1, 2)) == {(1, 2): 1, (0, 2): -1, (0, 1): 1} and bdc(bd((0, 1, 2))) == {})
check("Example 36.2.6(c): boundary of [0,1,2,3] = [1,2,3] - [0,2,3] + [0,1,3] - [0,1,2]",
      bd((0, 1, 2, 3)) == {(1, 2, 3): 1, (0, 2, 3): -1, (0, 1, 3): 1, (0, 1, 2): -1})
Vt, Ft = ch36.tetrahedron()
outward = {}
for tri in Ft.tolist():
    key = tuple(sorted(tri))
    outward[key] = dd.permutation_sign(tri, key)
det = np.linalg.det(np.stack([Vt[1] - Vt[0], Vt[2] - Vt[0], Vt[3] - Vt[0]], 1))
check(f"Example 36.2.6(c): det(x1-x0, x2-x0, x3-x0) = {det:.2f} < 0; boundary of [0,1,2,3] = -(outward faces), "
      "boundary of [1,0,2,3] = outward faces (Example 33.5.2)",
      det < 0 and abs(det + 3.08) < 0.01 and bd((0, 1, 2, 3)) == {f: -v for f, v in outward.items()}
      and bd((1, 0, 2, 3)) == outward
      and np.linalg.det(np.stack([Vt[0] - Vt[1], Vt[2] - Vt[1], Vt[3] - Vt[1]], 1)) > 0)

# ---------------------------------------------------------------- Non-example 36.2.7
A0 = np.abs(dd.d0(F, len(V)))
A1 = np.abs(dd.d1(F))
check("Non-example 36.2.7: unsigned incidence A1 A0 = 2 x (face-vertex incidence) != 0",
      np.array_equal(A1 @ A0, 2 * np.array([[1.0 if v in tri else 0.0 for v in range(len(V))] for tri in F])))

# ---------------------------------------------------------------- Proposition 36.2.9
for name, F_ in (("grid 2x2", F), ("icosphere(1)", dm.icosphere(1)[1]), ("kite", ch36.kite()[1])):
    n0 = int(F_.max()) + 1
    E_, idx_ = dd.edge_list(F_)
    B1 = np.zeros((n0, len(E_)))
    B2 = np.zeros((len(E_), len(F_)))
    for e, (i, j) in enumerate(E_):
        for face, c in bd((i, j)).items():
            B1[face[0], e] += c
    for f, tri in enumerate(F_.tolist()):
        for face, c in bd(tuple(tri)).items():
            B2[idx_[face], f] += c
    check(f"Proposition 36.2.9: matrices of (36.2.1) = (A0^+-)^T, (A1^+-)^T = dgdec.boundary1/2 on {name}",
          np.array_equal(B1, dd.d0(F_, n0).T) and np.array_equal(B2, dd.d1(F_).T)
          and np.array_equal(B1, dd.boundary1(F_, n0)) and np.array_equal(B2, dd.boundary2(F_)))

# ---------------------------------------------------------------- Implementation 36.2
ns, same, printed = ch36.run_printed(PAGE, "code-36-2-1", "out-36-2-1")
check("Implementation 36.2: code-36-2-1 prints exactly out-36-2-1", same, printed)
check("Implementation 36.2: code-36-2-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-36-2-1") <= 30)
c = ns["c"]
bmask = np.abs(c) > 0
E4 = dd.edge_list(ns["F"])[0]
V4 = ns["V"]
on_bd = [(min(V4[i, 0], V4[j, 0]) == max(V4[i, 0], V4[j, 0]) and V4[i, 0] in (0, 1))
         or (min(V4[i, 1], V4[j, 1]) == max(V4[i, 1], V4[j, 1]) and V4[i, 1] in (0, 1)) for i, j in E4]
check("Implementation 36.2: the boundary of all faces is supported exactly on the 12 edges of the square's border",
      np.array_equal(bmask, np.array(on_bd)) and bmask.sum() == 12)

# ---------------------------------------------------------------- Example 36.2.11, Proposition 36.2.13
xdy = dd.derham1(lambda p: np.array([0.0, p[0]]), V, E)
fv = dd.derham0(lambda p: p[0] ** 2 - p[1] ** 2, V)
dfh = dd.derham1(lambda p: np.array([2 * p[0], -2 * p[1]]), V, E)


def chain_vec(ch):
    out = np.zeros(len(E))
    for (i, j), v in ch.items():
        out[index[(min(i, j), max(i, j))]] += v if i < j else -v
    return out


cpath = chain_vec({(0, 3): 1, (3, 4): 1, (4, 8): 1})
cdiag = chain_vec({(0, 4): 1, (4, 8): 1})
close("Example 36.2.11(a): <x dy, c> = 0.625, <x dy, c'> = 0.5", [xdy @ cpath, xdy @ cdiag], [0.625, 0.5], 1e-15)
fi = F.tolist().index([0, 3, 4])
close("Example 36.2.11(a): difference 0.125 = integral of dx^dy over [0,3,4] (area 1/8)",
      [xdy @ (cpath - cdiag), dd.derham2(lambda p: 1.0, V, F)[fi]], [0.125, 0.125], tol=1e-15)
close("Example 36.2.11(b): <df_hat, c> = <df_hat, c'> = f(1,1) - f(0,0) = 0", [dfh @ cpath, dfh @ cdiag], [0, 0], 1e-15)
# (36.2.3) for a random chain and a polynomial form; reversed orientation
coef = rng.integers(-3, 4, size=len(E)).astype(float)
al = lambda p: np.array([p[0] * p[1] ** 2 - 1, p[0] ** 3 + p[1]])  # noqa: E731
lhs = dd.derham1(al, V, E) @ coef
rhs = sum(cf * dd.derham1(al, V, np.array([E[e] if cf >= 0 else E[e][::-1]]))[0] * (1 if cf >= 0 else -1)
          for e, cf in enumerate(coef))
close("Proposition 36.2.13(b): <alpha_hat, c> = sum of c_r times the integrals over sigma_r (any orientation)", lhs, rhs,
      tol=1e-13)

# ---------------------------------------------------------------- Definition 36.2.14: coboundary = transpose
u = rng.normal(size=len(V))
a1 = rng.normal(size=len(E))
B1, B2 = dd.boundary1(F, len(V)), dd.boundary2(F)
ok = True
for _ in range(5):
    c1, c2 = rng.normal(size=len(E)), rng.normal(size=len(F))
    ok &= abs((B1.T @ u) @ c1 - u @ (B1 @ c1)) < 1e-12 and abs((B2.T @ a1) @ c2 - a1 @ (B2 @ c2)) < 1e-12
check("Definition 36.2.14: <d_k alpha, c> = <alpha, boundary c> with d_k = (boundary_{k+1})^T", ok)
close("Definition 36.2.14: (d0 u)_ij = u_j - u_i", B1.T @ u, u[E[:, 1]] - u[E[:, 0]], tol=1e-14)
ok = True
for f, (i, j, k) in enumerate(F):
    val = lambda a, b: a1[index[(min(a, b), max(a, b))]] * (1 if a < b else -1)  # noqa: E731
    ok &= abs((B2.T @ a1)[f] - (val(i, j) + val(j, k) + val(k, i))) < 1e-14
check("Definition 36.2.14: (d1 alpha)_ijk = alpha_ij + alpha_jk + alpha_ki", ok)

# ---------------------------------------------------------------- Example 36.2.15 and Figure 36.2.2
Va, Fa = ch36.annulus(3, 12)
Ea, idxa = dd.edge_list(Fa)
check("Example 36.2.15(b): annulus mesh has 36 vertices, 84 edges, 48 counterclockwise faces",
      (len(Va), len(Ea), len(Fa)) == (36, 84, 48)
      and all(ch36.cross2(Va[Fa[:, 1]] - Va[Fa[:, 0]], Va[Fa[:, 2]] - Va[Fa[:, 0]]) > 0))
dth = dd.derham1(lambda p: np.array([-p[1], p[0]]) / (p @ p), Va, Ea, order=12)
check("Example 36.2.15(b): <d theta_hat, boundary f> = 0 for every face", np.abs(dd.d1(Fa) @ dth).max() < 1e-12)
z = np.zeros(len(Ea))
for j in range(12):
    a_, b_ = j, (j + 1) % 12
    z[idxa[(min(a_, b_), max(a_, b_))]] += 1 if a_ < b_ else -1
close("Example 36.2.15(b): each inner edge subtends pi/6, <d theta_hat, z> = 2 pi", [dth @ z], [2 * np.pi], tol=1e-12)
check("Example 36.2.15(b): z is a cycle", not np.any(dd.boundary1(Fa, len(Va)) @ z))

# ---------------------------------------------------------------- Exercises
check("Exercise 36.2.1: boundary of [4,5,2] + [4,8,5] is the loop 4 -> 8 -> 5 -> 2 -> 4, whose boundary is 0",
      bdc({(2, 4, 5): dd.permutation_sign((4, 5, 2)), (4, 5, 8): dd.permutation_sign((4, 8, 5))})
      == {(4, 8): 1, (5, 8): -1, (2, 5): -1, (2, 4): 1}
      and bdc({(4, 8): 1, (5, 8): -1, (2, 5): -1, (2, 4): 1}) == {})
terms = [(c1 * c2, f2) for f1, c1 in bd((0, 1, 2, 3)).items() for f2, c2 in bd(f1).items()]
check("Exercise 36.2.2: 12 terms, each of the six edges twice with opposite signs",
      len(terms) == 12 and all(sorted(c for c, f in terms if f == e) == [-1, 1]
                               for e in itertools.combinations(range(4), 2)))
ok = True
for F_ in (dm.icosphere(1)[1], ch36.grid(3)[1], ch36.annulus(3, 8)[1]):
    n0 = int(F_.max()) + 1
    E_, _ = dd.edge_list(F_)
    for _ in range(5):
        cc = rng.normal(size=len(E_))
        cc -= dd.boundary1(F_, n0).T @ np.linalg.lstsq(dd.boundary1(F_, n0).T, cc, rcond=None)[0]   # project to a cycle
        heads = np.zeros(n0)
        tails = np.zeros(n0)
        np.add.at(heads, E_[:, 1], cc)
        np.add.at(tails, E_[:, 0], cc)
        ok &= np.allclose(heads, tails)
check("Exercise 36.2.3: on cycles, incoming = outgoing coefficient sums at every vertex (Kirchhoff)", ok)
ok = True
for F_ in (ch36.grid(3)[1], ch36.annulus(3, 8)[1], dm.icosphere(1)[1]):
    E_, idx_ = dd.edge_list(F_)
    s = dd.boundary2(F_) @ np.ones(len(F_))
    ef = dm.edge_faces(F_)
    for (i, j), fs in ef.items():
        e = idx_[(i, j)]
        ok &= (s[e] == 0) if len(fs) == 2 else (s[e] == dd.d1(F_)[fs[0], e])
check("Exercise 36.2.4: boundary of the sum of all faces = boundary edges with induced orientation (0 if closed)", ok)
ns5, same5, printed5 = ch36.run_printed(PAGE, "code-ex-36-2-5", "out-ex-36-2-5", dict(ns))
check("Exercise 36.2.5: code-ex-36-2-5 (after code-36-2-1) prints exactly out-ex-36-2-5", same5, printed5)
check("Exercise 36.2.5: rank boundary1 = 35, rank boundary2 = 48 (84 - 35 - 48 = 1)",
      np.linalg.matrix_rank(ns5["B1"]) == 35 and np.linalg.matrix_rank(ns5["B2"]) == 48)

summary()
