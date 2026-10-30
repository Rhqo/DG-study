"""Verification for Section 49.5 (The Manifold Estimator in Detail).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-5-manifold-estimator.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import geo49  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "49-5-manifold-estimator.html"
rng = np.random.default_rng(49_5)

# ---------------------------------------------------------------- (41), (63): v_d = P e_d maximizes d phi_d / ds among |v| = |P e_d|
n = np.array([0.6, 0.0, 0.8])
P = np.eye(3) - np.outer(n, n)
ok = True
for d in range(3):
    best = P[:, d]
    for _ in range(2000):
        v = P @ rng.normal(size=3)
        v *= np.linalg.norm(best) / np.linalg.norm(v)
        ok &= v[d] <= best[d] + 1e-12
check("(41)/(63): the coordinate-derived vector P e_d increases phi_d fastest", ok)

# ---------------------------------------------------------------- Proposition 49.5.2 and the normalization warning
ev, E = np.linalg.eigh(P)
check("Prop 49.5.2: K = P has eigenvalues (0, 1, 1)", np.allclose(ev, [0, 1, 1]))
check("Prop 49.5.2: the 0-eigenvector is the normal", abs(abs(E[:, 0] @ n) - 1) < 1e-12)
Dg = np.diag(np.sqrt(np.diag(P)))
Kn = np.linalg.inv(Dg) @ P @ np.linalg.inv(Dg)          # inner products of the normalized vectors v_d / |v_d|
evn, En = np.linalg.eigh(Kn)
dn = Dg @ n / np.linalg.norm(Dg @ n)
check("Warning (E.4.3): with normalized v_d the null vector is D n, not n", abs(abs(En[:, 0] @ dn) - 1) < 1e-12)
close("Exercise 49.5.1: D = diag(0.8, 1, 0.6)", np.abs(np.diag(Dg) - [0.8, 1, 0.6]).max(), 0.0, tol=1e-12)
close("Exercise 49.5.1: angle between D n and n is 8.13 degrees", math.degrees(math.acos(abs(dn @ n))), 8.13, tol=0.01)
check("Exercise 49.5.1: normalized eigenvalues are not (0, 1, 1)", not np.allclose(evn, [0, 1, 1], atol=0.05))
close("Exercise 49.5.1: their largest eigenvalue", evn[-1], float(np.linalg.eigvalsh(Kn)[-1]), tol=1e-15)

# ---------------------------------------------------------------- Example 49.5.4: (1/2) Lbnz_Delta[x_a, x_b] = P_ab on S^2
th, ph = sp.symbols("theta phi", positive=True)
xs = [sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)]


def lap(f):
    return sp.simplify(sp.diff(sp.sin(th) * sp.diff(f, th), th) / sp.sin(th) + sp.diff(f, ph, 2) / sp.sin(th) ** 2)


check("Example 49.5.4: Delta x = -2 x on the unit sphere", all(sp.simplify(lap(x) + 2 * x) == 0 for x in xs))
okP = True
for a in range(3):
    for b in range(3):
        lb = lap(xs[a] * xs[b]) - xs[a] * lap(xs[b]) - xs[b] * lap(xs[a])
        okP &= sp.simplify(lb / 2 - ((1 if a == b else 0) - xs[a] * xs[b])) == 0
check("Example 49.5.4: (1/2) Lbnz[x_a, x_b] = delta_ab - x_a x_b", okP)
check("Example 49.5.4: Delta(x_a x_b) = 2 delta_ab - 6 x_a x_b",
      all(sp.simplify(lap(xs[a] * xs[b]) - (2 * (a == b) - 6 * xs[a] * xs[b])) == 0 for a in range(3) for b in range(3)))

# ---------------------------------------------------------------- Proposition 49.5.6: kernel Lbnz = weighted second moment
X = rng.normal(size=(500, 3))
q = rng.normal(size=3)
a = geo49.generator_weights(q, X, 0.7)
f, h = X[:, 0] ** 2 + X[:, 1], np.sin(X[:, 2])
fq, hq = q[0] ** 2 + q[1], np.sin(q[2])
close("Prop 49.5.6: Lbnz_L[f, h](q) = sum a_j (f_j - f_q)(h_j - h_q)",
      geo49.lbnz(a, fq, f, hq, h), geo49.lbnz_literal(a, fq, f, hq, h), tol=1e-10)
D = X - q
K = np.array([[geo49.lbnz(a, q[i], X[:, i], q[j], X[:, j]) / 2 for j in range(3)] for i in range(3)])
close("Prop 49.5.6: K = (1/2) sum a_j (x_j - q)(x_j - q)^T", np.abs(K - 0.5 * np.einsum("j,ja,jb->ab", a, D, D)).max(), 0.0, tol=1e-12)
close("Exercise 49.5.2: Lbnz of the identity generator part: f = h = constant gives 0",
      geo49.lbnz(a, 3.0, 3.0 + 0 * f, 2.0, 2.0 + 0 * h), 0.0, tol=1e-15)

# ---------------------------------------------------------------- Proposition 49.5.9 on the unit sphere (continuum Gamma)
# Gamma(f, h)(y) = <P(y) grad f, grad h> for functions extended to R^3; y on the unit sphere, outward normal n = q.
yv = sp.Matrix(sp.symbols("y1:4", real=True))
qn = np.array([0.3, -0.4, math.sqrt(1 - 0.25)])
nv = sp.Matrix(qn.tolist())
Py = sp.eye(3) - yv * yv.T / (yv.T * yv)[0]


def Gam(fe, he):
    gf = sp.Matrix([sp.diff(fe, v) for v in yv])
    gh = sp.Matrix([sp.diff(he, v) for v in yv])
    return (gf.T * Py * gh)[0]


eta = (nv.T * (yv - nv))[0]
subs = {yv[i]: qn[i] for i in range(3)}
Pq = np.eye(3) - np.outer(qn, qn)
A_eta = np.array([[float(4 * Gam(eta, Gam(yv[a_], yv[b_])).subs(subs)) for b_ in range(3)] for a_ in range(3)])
A_mid = np.array([[float(4 * Gam(yv[a_], Gam(eta, yv[b_])).subs(subs)) for b_ in range(3)] for a_ in range(3)])
close("Prop 49.5.9: A[eta, phi_a, phi_b](q) = 0", np.abs(A_eta).max(), 0.0, tol=1e-12)
close("Prop 49.5.9: A[phi_a, eta, phi_b](q)/4 = s(n, P e_a, P e_b) = -P_ab (unit sphere, outward)",
      np.abs(A_mid / 4 + Pq).max(), 0.0, tol=1e-12)
S8 = (A_mid + A_mid.T - A_eta) / 8
close("Prop 49.5.8 (50): the shape operator matrix on coordinate vectors is -P", np.abs(S8 + Pq).max(), 0.0, tol=1e-12)
U = np.linalg.eigh(Pq)[1][:, 1:]
check("Thm 49.5.10 (51): S = U^T S3 U has eigenvalues (-1, -1)", np.allclose(np.linalg.eigvalsh(U.T @ S8 @ U), [-1, -1]))
check("Warning (126): U U^T = P, not I", np.allclose(U @ U.T, Pq) and not np.allclose(U @ U.T, np.eye(3)))

# ---------------------------------------------------------------- (47): Lambda = t Sigma^i Sigma^j is not symmetric
Si = np.diag([1.0, 4.0, 0.01])
R = np.array([[math.cos(0.5), -math.sin(0.5), 0], [math.sin(0.5), math.cos(0.5), 0], [0, 0, 1]])
Sj = R @ np.diag([2.0, 0.5, 0.01]) @ R.T
L47 = Si @ Sj
check("Exercise 49.5.5: Sigma^i Sigma^j is not symmetric when they do not commute", np.abs(L47 - L47.T).max() > 0.5)
check("Exercise 49.5.5: t (Sigma^i + Sigma^j)/2 is symmetric positive definite",
      np.allclose(Si + Sj, (Si + Sj).T) and np.all(np.linalg.eigvalsh((Si + Sj) / 2) > 0))

# ---------------------------------------------------------------- Example 49.4.2 numbers reproduced by the implementation
ns = dgcode.run_block(PAGE, "code-49-5-1")
check("Implementation 49.5: at most 30 lines", dgcode.code_lines(PAGE, "code-49-5-1") <= 30)
Xi = ns["X"]
ref = geo49.manifold_estimate(Xi, Xi[0], 0.01, n_ref=Xi[0])
close("Implementation 49.5: same principal curvatures as lib/geo49.py",
      np.abs(np.sort(ns["tau_out"]) - np.sort(ref["tau"])).max(), 0.0, tol=1e-10)
check("Implementation 49.5: outward curvatures near (-1, -1)", np.all(np.abs(ns["tau_out"] + 1) < 0.1))
close("Implementation 49.5: same normal up to sign", abs(ns["n"] @ ref["n"]), 1.0, tol=1e-10)

# ---------------------------------------------------------------- Figure 49.5.1 numbers (noise)
N, t = 20000, 0.003
X0 = geo49.sphere_points(N, seed=3, pole=False)
rr = np.random.default_rng(0)
Q = rr.choice(N, 200, replace=False)
res = {}
for sig in (0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03):
    Xn = X0 + sig * rr.normal(size=X0.shape)
    if sig not in (0.02, 0.03):
        continue
    e = [[], [], []]
    for i in Q:
        qq, n0 = Xn[i], X0[i]
        e[0].append(math.degrees(math.acos(min(1, abs(geo49.local_pca_normal(Xn, qq, 30) @ n0)))))
        aa = geo49.generator_weights(qq, Xn, t)
        Dd = Xn - qq
        e[1].append(math.degrees(math.acos(min(1, abs(np.linalg.eigh(np.einsum("j,ja,jb->ab", aa, Dd, Dd))[1][:, 0] @ n0)))))
        w = aa / aa.sum()
        Dc = Xn - w @ Xn
        e[2].append(math.degrees(math.acos(min(1, abs(np.linalg.eigh(np.einsum("j,ja,jb->ab", w, Dc, Dc))[1][:, 0] @ n0)))))
    res[sig] = [(np.median(x), np.percentile(x, 90)) for x in e]
close("Figure 49.5.1: sigma 0.02 median, local PCA 7.7", res[0.02][0][0], 7.74, tol=0.01)
close("Figure 49.5.1: sigma 0.02 median, GeoSplat K 6.5", res[0.02][1][0], 6.50, tol=0.01)
close("Figure 49.5.1: sigma 0.02 median, kernel PCA 5.0", res[0.02][2][0], 4.97, tol=0.01)
close("Figure 49.5.1: sigma 0.03 90th percentile, GeoSplat K 63", res[0.03][1][1], 63.0, tol=0.5)
close("Figure 49.5.1: sigma 0.03 90th percentile, local PCA 24", res[0.03][0][1], 24.4, tol=0.1)

# ---------------------------------------------------------------- Proposition 49.5.11: density (two points of Figure 49.5.3)
Xd = geo49.sphere_points_density(60000, a=0.9, seed=0)
for z in (-0.85, 0.95):
    qd = np.array([math.sqrt(1 - z * z), 0.0, z])
    X2 = np.vstack([qd[None], Xd])
    rp = geo49.manifold_estimate(X2, qd, 0.005, mode="paper", density=len(X2) / (4 * math.pi), n_ref=qd)
    s = np.trace(rp["K"]) / 2
    close(f"Prop 49.5.11: tr K / 2 = p/pbar = {1 + 0.9 * z:.3f} at z = {z}", s, 1 + 0.9 * z, tol=0.15 * (1 + 0.9 * z) + 0.02)
    close(f"Prop 49.5.11: raw mean curvature = -(p/pbar)^2 at z = {z}", rp["tau"].mean(), -(1 + 0.9 * z) ** 2,
          tol=0.25 * (1 + 0.9 * z) ** 2 + 0.03)
    close(f"Prop 49.5.11: corrected by (tr K/2)^2 gives -1 at z = {z}", rp["tau"].mean() / s ** 2, -1.0, tol=0.2)
close("Exercise 49.5.4: four times denser: K x 4, curvature x 16", 4.0 ** 2, 16.0, tol=0)

summary()
