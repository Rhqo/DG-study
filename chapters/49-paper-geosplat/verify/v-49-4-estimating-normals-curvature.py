"""Verification for Section 49.4 (Estimating Normals and Curvature).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-4-estimating-normals-curvature.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import geo49  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "49-4-estimating-normals-curvature.html"

# ---------------------------------------------------------------- Example 49.4.2: the manifold estimator on the sphere
X = geo49.sphere_points(50000, seed=0)
taus, eigs, angs = [], [], []
for i in list(range(0, 50000, 50000 // 12))[:12]:
    r = geo49.manifold_estimate(X, X[i], 0.01, n_ref=X[i])
    taus.append(r["tau"])
    eigs.append(r["eigK"])
    angs.append(math.degrees(math.acos(min(1.0, abs(r["n"] @ X[i])))))
taus, eigs = np.mean(taus, 0), np.mean(eigs, 0)
close("Example 49.4.2: mean tau_1 = -1.01", taus[0], -1.01, tol=0.01)
close("Example 49.4.2: mean tau_2 = -0.92", taus[1], -0.92, tol=0.01)
check("Example 49.4.2: mean eigenvalues of K about (0.01, 0.96, 1.02)", np.allclose(eigs, [0.010, 0.960, 1.017], atol=0.01))
check("Example 49.4.2: normal error below 0.6 degrees", max(angs) < 0.6)
# cross product of the two tangent eigenvectors is the normal (up to sign)
r = geo49.manifold_estimate(X, X[0], 0.01, n_ref=X[0])
cp = np.cross(r["U"][:, 0], r["U"][:, 1])
close("Theorem 49.4.1: u_1 x u_2 = +-normal (unit length)", abs(cp @ r["n"]), 1.0, tol=1e-12)
check("Exercise 49.4.1: (1,0,0) x (0,0.6,0.8) = (0,-0.8,0.6)", np.allclose(np.cross([1, 0, 0], [0, 0.6, 0.8]), [0, -0.8, 0.6]))

# ---------------------------------------------------------------- Proposition 49.4.3: Koszul formula (paper (23))
x, y = sp.symbols("x y", real=True)
co = [x, y]
g = sp.Matrix([[1 + x ** 2, x * y], [x * y, 2 + y ** 2]])      # a Riemannian metric near the origin
Gam = dgsym.christoffel(g, co)                                # Gam[k][i][j] = Gamma^k_ij


def nabla(U, V):
    return sp.Matrix([sum(U[i] * sp.diff(V[k], co[i]) for i in range(2))
                      + sum(Gam[k][i][j] * U[i] * V[j] for i in range(2) for j in range(2)) for k in range(2)])


def ip(U, V):
    return (U.T * g * V)[0]


def dirder(U, f):
    return sum(U[i] * sp.diff(f, co[i]) for i in range(2))


def br(U, V):
    return sp.Matrix([dirder(U, V[k]) - dirder(V, U[k]) for k in range(2)])


U1, U2, U3 = sp.Matrix([1 + y, x]), sp.Matrix([x * y, 1]), sp.Matrix([y ** 2, x - 1])
lhs = ip(nabla(U1, U2), U3)
rhs = sp.Rational(1, 2) * (dirder(U1, ip(U2, U3)) + dirder(U2, ip(U3, U1)) - dirder(U3, ip(U1, U2))
                           + ip(br(U1, U2), U3) - ip(br(U1, U3), U2) - ip(br(U2, U3), U1))
check("Prop 49.4.3 (49.4.3) = paper (23): Koszul formula for polynomial fields and metric", sp.simplify(lhs - rhs) == 0)
check("(49.4.6) (21): Levi-Civita is torsion-free", sp.simplify(nabla(U1, U2) - nabla(U2, U1) - br(U1, U2)) == sp.zeros(2, 1))
check("(49.4.4) (19): metric compatibility",
      sp.simplify(dirder(U1, ip(U2, U3)) - ip(nabla(U1, U2), U3) - ip(U2, nabla(U1, U3))) == 0)
# Exercise 49.4.2: polar coordinates, g = diag(1, r^2): g(nabla_{d theta} d theta, d r) = -r
rr, th = sp.symbols("r theta", positive=True)
Gp = dgsym.christoffel(sp.diag(1, rr ** 2), [rr, th])
check("Exercise 49.4.2: nabla_{d_theta} d_theta = -r d_r", sp.simplify(Gp[0][1][1] + rr) == 0 and Gp[1][1][1] == 0)

# ---------------------------------------------------------------- Proposition 49.4.4: closed form = [BLM22] (31)-(32)
Xs = geo49.sphere_points(20000, seed=5)
m = np.ones(len(Xs))
for i0, eps in ((0, 0.15), (7, 0.3)):
    T = np.eye(3)[None] - Xs[:, :, None] * Xs[:, None, :]
    Bt = geo49.blm22_tensor(Xs, T, m, i0, eps)
    Vb = geo49.tangent_basis(Xs[i0])
    ref = Vb.T @ np.einsum("ijk,k->ij", Bt, Xs[i0]) @ Vb       # (32): T^T {<B_ij:, n>}_{i,j <= 3} T
    minus = geo49.geosplat_B(Xs, Xs, m, i0, eps, sign=-1.0, Vi=Vb)
    plus = geo49.geosplat_B(Xs, Xs, m, i0, eps, sign=+1.0, Vi=Vb)
    close(f"Prop 49.4.4: minus-sign closed form = [BLM22] (7.2)+(32) (eps = {eps})", np.abs(minus - ref).max(), 0.0, tol=1e-13)
    check(f"Prop 49.4.4: the printed plus sign differs (eps = {eps})", np.abs(plus - ref).max() > 1e-4)
# vectorized version (Implementation 49.4) agrees with the loop version
close("geosplat_B_fast = geosplat_B", np.abs(geo49.geosplat_B_fast(Xs, Xs, m, 3, 0.25) - geo49.geosplat_B(Xs, Xs, m, 3, 0.25)).max(),
      0.0, tol=1e-13)
# Exercise 49.4.5: the printed V~^j = I - u_3^i (u_3^i)^T makes B identically zero
i0, eps = 0, 0.2
u = Xs[i0]
Vb = geo49.tangent_basis(u)
d_ = Xs[i0] - Xs
dist = np.linalg.norm(d_, axis=1)
idx = np.where((dist > 0) & (dist < eps))[0]
Pi = np.eye(3) - np.outer(u, u)
tot = np.zeros((2, 2))
for j in idx:
    A = Pi @ Vb
    Sym = 0.5 * (np.outer(u, d_[j]) + np.outer(d_[j], u))
    tot += np.abs(2 * A.T @ Sym @ A + (d_[j] @ Pi @ u) * (Vb.T @ Pi @ Vb - np.eye(2)))
close("Exercise 49.4.5: with the printed V~^j every B^{i,j} vanishes", tot.max(), 0.0, tol=1e-14)
# Exercise 49.4.4: orders of the two terms in the angle theta between neighbouring normals (circle of radius 1 in 3D)
for th_ in (0.1, 0.05):
    ui, uj = np.array([0.0, 0, 1]), np.array([math.sin(th_), 0, math.cos(th_)])
    mui, muj = ui.copy(), uj.copy()
    dd = mui - muj
    Pj = np.eye(3) - np.outer(uj, uj)
    V = geo49.tangent_basis(ui)
    f1 = np.abs(2 * (Pj @ V).T @ (0.5 * (np.outer(ui, dd) + np.outer(dd, ui))) @ (Pj @ V)).max()
    f2 = np.abs((dd @ Pj @ ui) * (V.T @ Pj @ V - np.eye(2))).max()
    close(f"Exercise 49.4.4: first term ~ 2 theta^2 / |delta| ~ 2 theta (theta = {th_})", f1 / (2 * th_ ** 2), 1.0, tol=0.02)
    close(f"Exercise 49.4.4: second term ~ theta^4 (theta = {th_})", f2 / th_ ** 4, 1.0, tol=0.02)

# ---------------------------------------------------------------- Example 49.4.5 and Figure 49.4.2 numbers
X4 = geo49.sphere_points(40000, seed=1)
m4 = np.ones(len(X4))
Q = range(0, 40000, 4000)


def mean_ev(eps, sign, inv):
    return np.mean([np.linalg.eigvalsh(geo49.geosplat_B_fast(X4, X4, m4, i, eps, sign=sign, with_inv_eps=inv)).mean() for i in Q])


close("Example 49.4.5: sphere, eps = 0.2, derived (minus, 1/eps): -0.984", mean_ev(0.2, -1, True), -0.984, tol=2e-3)
close("Example 49.4.5: sphere, eps = 0.2, printed plus: -0.964", mean_ev(0.2, 1, True), -0.964, tol=2e-3)
close("Example 49.4.5: sphere, eps = 0.4, derived: -0.939", mean_ev(0.4, -1, True), -0.939, tol=2e-3)
close("Example 49.4.5: sphere, eps = 0.4, printed plus: -0.860", mean_ev(0.4, 1, True), -0.860, tol=2e-3)
close("Example 49.4.5: sphere, eps = 0.2, no 1/eps: -0.197 (= eps x curvature)", mean_ev(0.2, -1, False), -0.197, tol=2e-3)
close("Exercise 49.4.3: scene scaled by 10 (eps 2): without 1/eps the value is unchanged",
      np.mean([np.linalg.eigvalsh(geo49.geosplat_B_fast(10 * X4, X4, m4, i, 2.0, with_inv_eps=False)).mean() for i in Q]),
      mean_ev(0.2, -1, False), tol=1e-12)
close("Exercise 49.4.3: scene scaled by 10: with 1/eps the curvature is divided by 10",
      np.mean([np.linalg.eigvalsh(geo49.geosplat_B_fast(10 * X4, X4, m4, i, 2.0)).mean() for i in Q]),
      mean_ev(0.2, -1, True) / 10, tol=1e-12)
R_, r_ = 2.0, 0.8
T = geo49.torus_points(150000, seed=2)
rho = np.hypot(T[:, 0], T[:, 1])
NT = (T - np.stack([R_ * T[:, 0] / rho, R_ * T[:, 1] / rho, 0 * rho], axis=1)) / r_
q0 = np.array([[2.8, 0, 0]])
T2, N2 = np.vstack([q0, T]), np.vstack([[[1.0, 0, 0]], NT])
ev = sorted(np.linalg.eigvalsh(geo49.geosplat_B_fast(T2, N2, np.ones(len(T2)), 0, 0.2)), key=abs, reverse=True)
close("Example 49.4.5: torus outer equator tau_1 = -1.195 (true -1.25)", ev[0], -1.195, tol=2e-3)
close("Example 49.4.5: torus outer equator tau_2 = -0.363 (true -0.357)", ev[1], -0.363, tol=2e-3)

# ---------------------------------------------------------------- Implementation 49.4 (code-49-4-1)
ns = dgcode.run_block(PAGE, "code-49-4-1")
check("Implementation 49.4: at most 30 lines", dgcode.code_lines(PAGE, "code-49-4-1") <= 30)
check("Implementation 49.4: eigenvalues (-1.006, -0.963)", np.allclose(np.sort(ns["tau"]), [-1.006, -0.963], atol=1e-3))
Xi = ns["X"]
close("Implementation 49.4: agrees with lib geosplat_B", np.abs(np.sort(ns["tau"]) - np.sort(np.linalg.eigvalsh(
    geo49.geosplat_B(Xi, Xi, np.ones(len(Xi)), 0, 0.2)))).max(), 0.0, tol=1e-12)

summary()
