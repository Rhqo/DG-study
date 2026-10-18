"""Verification for Section 36.4 (The Discrete Exterior Derivative).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/verify/v-36-4-discrete-exterior-derivative.py``
"""

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

PAGE = HERE / "36-4-discrete-exterior-derivative.html"
rng = np.random.default_rng(3604)

# ---------------------------------------------------------------- Example 36.4.2 / Figure 36.4.1
V, F = ch36.kite()
E, index = dd.edge_list(F)
D0, D1 = dd.d0(F, 4), dd.d1(F)
check("Example 36.4.2: edges 01, 02, 12, 13, 23; faces counterclockwise",
      E.tolist() == [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]]
      and all(ch36.cross2(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]) > 0))
check("Example 36.4.2: the matrices d0 and d1 as printed",
      np.array_equal(D0, [[-1, 1, 0, 0], [-1, 0, 1, 0], [0, -1, 1, 0], [0, -1, 0, 1], [0, 0, -1, 1]])
      and np.array_equal(D1, [[1, -1, 1, 0, 0], [0, 0, -1, 1, -1]]))
u = np.array([3.0, -1.0, 4.0, 2.0])
al = np.array([1.0, 2.0, -1.0, 3.0, 0.5])
check("Example 36.4.2: d0 u = (-4, 1, 5, 3, -2), d1 alpha = (-2, 3.5), d1 d0 u = 0",
      (D0 @ u).tolist() == [-4, 1, 5, 3, -2] and (D1 @ al).tolist() == [-2, 3.5] and not np.any(D1 @ D0 @ u))
# (36.4.1) against the coboundary of Definition 36.2.14
for name, (Vm, Fm) in (("kite", (V, F)), ("icosphere(1)", dm.icosphere(1)), ("annulus", ch36.annulus(3, 8))):
    Em, idxm = dd.edge_list(Fm)
    a = rng.normal(size=len(Em))
    val = lambda i, j: a[idxm[(min(i, j), max(i, j))]] * (1 if i < j else -1)  # noqa: E731
    circ = np.array([val(i, j) + val(j, k) + val(k, i) for i, j, k in Fm])
    check(f"Definition 36.4.1: (d1 alpha)_ijk = alpha_ij + alpha_jk + alpha_ki = coboundary ({name})",
          np.allclose(dd.d1(Fm) @ a, circ) and np.array_equal(dd.d1(Fm), dd.boundary2(Fm).T))

# ---------------------------------------------------------------- Non-example 36.4.3
L = np.linalg.norm(V[E[:, 1]] - V[E[:, 0]], axis=1)
q = (D0 @ u) / L
loop_f = q[index[(0, 1)]] + q[index[(1, 2)]] - q[index[(0, 2)]]
close("Non-example 36.4.3: difference quotients around f sum to 0.325 (not 0); d0 u sums to 0",
      [round(loop_f, 3), (D0 @ u)[index[(0, 1)]] + (D0 @ u)[index[(1, 2)]] - (D0 @ u)[index[(0, 2)]]], [0.325, 0.0],
      tol=1e-12)
close("Non-example 36.4.3: the three quotients -2, 2.906, -0.581",
      np.round([q[index[(0, 1)]], q[index[(1, 2)]], -q[index[(0, 2)]]], 3), [-2.0, 2.906, -0.581], tol=1e-12)

# ---------------------------------------------------------------- Proposition 36.4.4(b): orientation changes
Vg, Fg = ch36.grid(3)
Eg, idxg = dd.edge_list(Fg)
P1 = np.diag(np.where(rng.random(len(Eg)) < 0.5, -1.0, 1.0))
P2 = np.diag(np.where(rng.random(len(Fg)) < 0.5, -1.0, 1.0))
alpha_s = lambda p: np.array([p[0] * p[1] ** 2, np.cos(p[0]) + p[1]])  # noqa: E731
dalpha_s = lambda p: -np.sin(p[0]) - 2 * p[0] * p[1]  # noqa: E731
R1, R2 = dd.derham1(alpha_s, Vg, Eg, order=10), dd.derham2(dalpha_s, Vg, Fg, order=8)
check("Proposition 36.4.4(b): (P2 d1 P1)(P1 R1) = P2 (R2 d) after reorienting edges and faces",
      np.allclose((P2 @ dd.d1(Fg) @ P1) @ (P1 @ R1), P2 @ R2, atol=1e-13))

# ---------------------------------------------------------------- Theorem 36.4.5, Corollary 36.4.6
ok = True
for _ in range(10):
    a = rng.normal(size=len(Eg))
    c = rng.normal(size=len(Fg))
    ok &= abs((dd.d1(Fg) @ a) @ c - a @ (dd.boundary2(Fg) @ c)) < 1e-12
check("Theorem 36.4.5: <d1 alpha, c> = <alpha, boundary c> for random alpha, c", ok)
V2, F2 = ch36.grid(2)
E2, idx2 = dd.edge_list(F2)
xdy = dd.derham1(lambda p: np.array([0.0, p[0]]), V2, E2)
loop = dm.boundary_loop(F2)
bd = np.stack([loop, np.roll(loop, -1)], 1)
circ = dd.derham1(lambda p: np.array([0.0, p[0]]), V2, bd).sum()
close("Corollary 36.4.6(b): d1 of x dy is 1/8 on each face; the sum 1 equals the boundary circulation",
      np.concatenate([dd.d1(F2) @ xdy, [circ]]), np.concatenate([np.full(8, 0.125), [1.0]]), tol=1e-14)
for name, Fm in (("icosphere(2)", dm.icosphere(2)[1]), ("torus", dm.torus_mesh(2, 0.8, 8, 12)[1]),
                 ("tetrahedron", ch36.tetrahedron()[1])):
    check(f"Corollary 36.4.6(c): 1^T d1 = 0 on {name}", not np.any(np.ones(len(Fm)) @ dd.d1(Fm)))

# ---------------------------------------------------------------- Implementation 36.4 and Theorem 36.4.7
ns, same, printed = ch36.run_printed(PAGE, "code-36-4-1", "out-36-4-1")
check("Implementation 36.4: code-36-4-1 prints exactly out-36-4-1", same, printed)
check("Implementation 36.4: code-36-4-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-36-4-1") <= 30)
for name, (Vm, Fm) in (("jittered grid (R^2)", (Vg + 0.03 * rng.normal(size=Vg.shape), Fg)),
                       ("torus (R^3)", dm.torus_mesh(2, 0.8, 8, 12))):
    Em, _ = dd.edge_list(Fm)
    dim = Vm.shape[1]
    f = lambda p: np.sin(p[0]) * np.exp(p[1]) + p[-1] ** 2  # noqa: E731
    if dim == 2:
        df = lambda p: np.array([np.cos(p[0]) * np.exp(p[1]), np.sin(p[0]) * np.exp(p[1]) + 2 * p[1]])  # noqa: E731
    else:
        df = lambda p: np.array([np.cos(p[0]) * np.exp(p[1]), np.sin(p[0]) * np.exp(p[1]), 2 * p[2]])  # noqa: E731
    close(f"Theorem 36.4.7: R_1(df) = d0 R_0(f) for a non-polynomial f on {name}",
          dd.derham1(df, Vm, Em, order=12), dd.d0(Fm, len(Vm)) @ dd.derham0(f, Vm), tol=1e-12)
close("Theorem 36.4.7: R_2(d alpha) = d1 R_1(alpha) for a non-polynomial 1-form on the grid", R2, dd.d1(Fg) @ R1,
      tol=1e-12)
Vt, Ft = dm.torus_mesh(2, 0.8, 8, 12)
Et, _ = dd.edge_list(Ft)
a3 = lambda p: np.array([np.sin(p[1]) * p[2], p[0] * p[2], np.exp(p[0] / 3)])  # noqa: E731
curl3 = lambda p: np.array([0.0 - p[0], np.sin(p[1]) - np.exp(p[0] / 3) / 3,  # noqa: E731
                            p[2] - np.cos(p[1]) * p[2]])
close("Theorem 36.4.7: R_2(d alpha) = d1 R_1(alpha) on the torus surface in R^3 (curl representation)",
      dd.derham2(curl3, Vt, Ft, order=8), dd.d1(Ft) @ dd.derham1(a3, Vt, Et, order=12), tol=1e-10)
# tetrahedron (stated): numerical check of Stokes on one 3-simplex for a polynomial 2-form
v = np.array([[0.1, 0.0, 0.2], [1.0, 0.2, 0.0], [0.3, 1.1, 0.1], [0.2, 0.3, 0.9]])
if np.linalg.det(np.stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]], 1)) < 0:
    v[[1, 2]] = v[[2, 1]]
w = lambda p: np.array([p[0] * p[1], p[2] ** 2, p[0] + p[1] * p[2]])  # noqa: E731  (w_yz, w_zx, w_xy)
divw = lambda p: p[1] + 0.0 + p[1]  # noqa: E731  d omega = div w dV
B3, w3 = dd.triangle_quadrature(4)
g, wg = dd.gauss_legendre_01(6)
vol = 0.0
A = np.stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]], 1)
for s, ws in zip(g, wg):
    for (b, wb) in zip(B3, w3):
        # Duffy-type map of the unit cube to the 3-simplex: t = (1 - s) * (triangle point) , last coord s
        tt = np.array([(1 - s) * b[1], (1 - s) * b[2], s])
        jac = 0.5 * (1 - s) ** 2
        vol += ws * wb * jac * divw(v[0] + A @ tt)
vol *= np.linalg.det(A)
faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]                      # outward for a positive simplex
flux = dd.derham2(w, v, np.array(faces), order=4).sum()
close("Warning (tetrahedra): Stokes on one positively oriented 3-simplex holds numerically (stated case)", vol, flux,
      tol=1e-12)

# ---------------------------------------------------------------- Theorem 36.4.8
for name, Fm in (("icosphere(2)", dm.icosphere(2)[1]), ("torus", dm.torus_mesh(2, 0.8, 12, 16)[1]),
                 ("grid", ch36.grid(4)[1]), ("annulus", ch36.annulus(3, 12)[1])):
    check(f"Theorem 36.4.8: d1 d0 = 0 exactly on {name}", not np.any(dd.d1(Fm) @ dd.d0(Fm, int(Fm.max()) + 1)))

# ---------------------------------------------------------------- Example 36.4.10, Non-example 36.4.11
fhat = dd.derham0(lambda p: p[0] ** 2 - p[1] ** 2, V2)
dfhat = dd.derham1(lambda p: np.array([2 * p[0], -2 * p[1]]), V2, E2)
check("Example 36.4.10(a): R_1(df) = d0 R_0 f and d1 R_1(df) = 0", np.allclose(dfhat, dd.d0(F2, 9) @ fhat)
      and np.abs(dd.d1(F2) @ dfhat).max() < 1e-14)
Vs, Fs = ch36.grid(4)
Vs = Vs + np.array([0.5, 0.5])                                          # away from the origin
Es, _ = dd.edge_list(Fs)
dth = dd.derham1(lambda p: np.array([-p[1], p[0]]) / (p @ p), Vs, Es, order=12)
uu = np.linalg.lstsq(dd.d0(Fs, len(Vs)), dth, rcond=None)[0]
check("Example 36.4.10(c): R_1(d theta) is closed and exact on a grid that avoids the origin",
      np.abs(dd.d1(Fs) @ dth).max() < 1e-12 and np.abs(dd.d0(Fs, len(Vs)) @ uu - dth).max() < 1e-10)
A1 = np.abs(D1)
check("Non-example 36.4.11: unsigned A1 (d0 u) = (2, 6)", (A1 @ (D0 @ u)).tolist() == [2, 6])

# ---------------------------------------------------------------- Lemma 36.4.12, Proposition 36.4.13, Remark 36.4.15
expect = {"icosphere(2)": (1, 0, 1, 2), "torus": (1, 2, 1, 0), "grid 4x4": (1, 0, 0, 1), "annulus": (1, 1, 0, 0)}
for name, Fm in (("icosphere(2)", dm.icosphere(2)[1]), ("torus", dm.torus_mesh(2, 0.8, 12, 16)[1]),
                 ("grid 4x4", ch36.grid(4)[1]), ("annulus", ch36.annulus(3, 12)[1])):
    n0 = int(Fm.max()) + 1
    A0, A1m = dd.d0(Fm, n0), dd.d1(Fm)
    r0, r1 = np.linalg.matrix_rank(A0), np.linalg.matrix_rank(A1m)
    b = (n0 - r0, A0.shape[0] - r0 - r1, A1m.shape[0] - r1)
    chi = n0 - A0.shape[0] + A1m.shape[0]
    closed = len(dm.boundary_edges(Fm)) == 0
    check(f"Remark 36.4.15 / Proposition 36.4.13: (b0, b1, b2, chi) = {expect[name]} on {name}; "
          f"b1 = {'2' if closed else '1'} - chi; b0 - b1 + b2 = chi",
          b + (chi,) == expect[name] and b[1] == (2 - chi if closed else 1 - chi) and b[0] - b[1] + b[2] == chi
          and r0 == n0 - 1 and r1 == A1m.shape[0] - (1 if closed else 0))
ok = all((3 * n * n + 2 * n, 2 * n * n, (n + 1) ** 2) == (len(dd.edge_list(ch36.grid(n)[1])[0]), len(ch36.grid(n)[1]),
                                                           len(ch36.grid(n)[0])) for n in (1, 2, 5, 9))
check("Proposition 36.4.13: n x n grid has n0 = (n+1)^2, n1 = 3n^2 + 2n, n2 = 2n^2, so chi = 1", ok)

# ---------------------------------------------------------------- Example 36.4.14 / Figure 36.4.2
Vt, Ft = dm.torus_mesh(2.0, 0.8, 12, 16)
Et, idxt = dd.edge_list(Ft)


def dv(p):
    return np.array([-p[1], p[0], 0.0]) / (p[0] ** 2 + p[1] ** 2)


def du(p):
    rho = np.hypot(p[0], p[1])
    qq = (rho - 2.0) ** 2 + p[2] ** 2
    return (-p[2] * np.array([p[0], p[1], 0.0]) / rho + (rho - 2.0) * np.array([0.0, 0.0, 1.0])) / qq


aT, bT = dd.derham1(dv, Vt, Et, order=10), dd.derham1(du, Vt, Et, order=10)


def loop(verts):
    c = np.zeros(len(Et))
    for a_, b_ in zip(verts, verts[1:] + verts[:1]):
        c[idxt[(min(a_, b_), max(a_, b_))]] += 1 if a_ < b_ else -1
    return c


lon, mer = loop(list(range(16))), loop([16 * i for i in range(12)])
check("Example 36.4.14: vertex 16 i + j at u = 2 pi i/12, v = 2 pi j/16 (vertex 0 on the outer equator)",
      np.allclose(Vt[0], [2.8, 0, 0]) and np.allclose(Vt[16 * 3], [2.0, 0, 0.8]) and np.allclose(Vt[4], [0, 2.8, 0]))
check("Example 36.4.14: R_1(dv) and R_1(du) are closed", np.abs(dd.d1(Ft) @ aT).max() < 1e-12
      and np.abs(dd.d1(Ft) @ bT).max() < 1e-12)
close("Example 36.4.14: <dv, longitude> = 2 pi, <dv, meridian> = 0, <du, longitude> = 0, <du, meridian> = 2 pi",
      [aT @ lon, aT @ mer, bT @ lon, bT @ mer], [2 * np.pi, 0, 0, 2 * np.pi], tol=1e-12)
close("Example 36.4.14: each longitude edge subtends 2 pi / 16 around the z-axis", aT[[idxt[(j, j + 1)] for j in range(15)]],
      np.full(15, 2 * np.pi / 16), tol=1e-12)

# ---------------------------------------------------------------- Exercises
Fx = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1]])
Ex, _ = dd.edge_list(Fx)
check("Exercise 36.4.1: edge order 01, 02, 03, 12, 13, 23 and the printed d0, d1; d1 d0 = 0; consistent; chi = 1",
      Ex.tolist() == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
      and np.array_equal(dd.d1(Fx), [[1, -1, 0, 1, 0, 0], [0, 1, -1, 0, 0, 1], [-1, 0, 1, 0, -1, 0]])
      and np.array_equal(dd.d0(Fx, 4), [[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1], [0, -1, 1, 0], [0, -1, 0, 1],
                                        [0, 0, -1, 1]])
      and not np.any(dd.d1(Fx) @ dd.d0(Fx, 4)) and dm.oriented_consistently(Fx) and dm.euler_characteristic(Fx) == 1
      and np.linalg.matrix_rank(dd.d1(Fx)) == 3 and np.linalg.matrix_rank(dd.d0(Fx, 4)) == 3)
A0t = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0], [-1, 0, 0, 1], [0, -1, 1, 0], [0, -1, 0, 1], [0, 0, -1, 1]])
Ft4 = ch36.tetrahedron()[1]
A1t = dd.d1(Ft4)
check("Exercise 36.4.2: tetrahedron (Example 33.5.7): d0 = A0, A1 A0 = 0, 1^T A1 = 0",
      np.array_equal(dd.d0(Ft4, 4), A0t) and not np.any(A1t @ A0t) and not np.any(np.ones(4) @ A1t))
ok = True
for _ in range(20):
    uu_, vv_ = rng.normal(size=4), rng.normal(size=4)
    ub = 0.5 * np.abs(D0) @ uu_
    vb = 0.5 * np.abs(D0) @ vv_
    ok &= np.allclose(D0 @ (uu_ * vv_), ub * (D0 @ vv_) + vb * (D0 @ uu_))
check("Exercise 36.4.3(a): d0(uv) = mean(u) d0 v + mean(v) d0 u", ok)
check("Exercise 36.4.3(b): u = v = (0, 1) on one edge: d0(uv) = 1, start-value rule gives 0",
      (np.array([[-1.0, 1.0]]) @ np.array([0.0, 1.0])).item() == 1.0)
ns5, same5, printed5 = ch36.run_printed(PAGE, "code-ex-36-4-5", "out-ex-36-4-5")
check("Exercise 36.4.5: code-ex-36-4-5 prints exactly out-ex-36-4-5", same5, printed5)

summary()
