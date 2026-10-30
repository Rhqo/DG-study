"""Verification for Section 48.6 (Implementation and Numerical Illustrations).

Run from the project root:
    PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-6-numerical-illustrations.py
(about one minute and 2 GB of memory: the torus experiment of Figure 48.6.1 is recomputed.)
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import var48 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "48-6-numerical-illustrations.html"
rng = np.random.default_rng(48)
n, d = 3, 2


def close_each(name, got, exp, tols):
    """entrywise |got - exp| <= tols (dgcheck.close takes one tolerance)."""
    got, exp = np.asarray(got, float), np.asarray(exp, float)
    ok = got.shape == exp.shape and bool(np.all(np.abs(got - exp) <= np.broadcast_to(np.asarray(tols, float), exp.shape)))
    return check(name, ok, f"got {np.round(got, 4).tolist()}")


# ---------------------------------------------------------------- Exercise 48.6.1: c_xi = c_rho for the natural pair
c_rho = n * v.omega(n) * v.radial_moment(v.rho, n - 1)
c_xi = n * v.omega(n) * v.radial_moment(lambda t: v.xi(t, n), n - 1)
close("Exercise 48.6.1: int xi = int rho on R^3 (natural pair)", c_xi, c_rho, 1e-10)
C_rho = d * v.omega(d) * v.radial_moment(v.rho, d - 1) / c_rho          # constants of the normalised profiles
C_xi = d * v.omega(d) * v.radial_moment(lambda t: v.xi(t, n), d - 1) / c_xi
close("C_xi / C_rho = d/n for the normalised natural pair", C_xi / C_rho, d / n, 1e-10)

# ---------------------------------------------------------------- Proposition 48.6.1 from the definitions
K = 60
x0 = np.zeros(3)
Xl = rng.uniform(-0.25, 0.25, (K, 3))
Nl = rng.normal(size=(K, 3))
Pl = v.proj_from_normal(Nl / np.linalg.norm(Nl, axis=1)[:, None])
ml = rng.uniform(0.5, 1.5, K)
Xl[0], P0 = x0, Pl[0]                                                # the atom x_{l0} itself
eps = 0.3


def beta_definition(eps):
    """-(C_xi/C_rho) (delta_ijk V * rho_eps)(x0) / (||V|| * xi_eps)(x0) with normalised kernels ((48.5.2), (48.5.3))."""
    Z = Xl - x0
    r = np.linalg.norm(Z, axis=1)
    u = np.zeros_like(Z)
    u[r > 0] = Z[r > 0] / r[r > 0, None]
    grad = v.drho(r / eps)[:, None] * u / (c_rho * eps ** (n + 1))    # grad rho_eps(x_l - x0)
    num = np.einsum("l,ljk,li->ijk", ml, Pl, np.einsum("lab,lb->la", Pl, grad))
    den = np.sum(ml * v.xi(r / eps, n)) / (c_xi * eps ** n)
    return -(C_xi / C_rho) * num / den


def beta_formula(eps, with_eps=True):
    """(48.6.1) written out (the printed formula of Section 7.1 when with_eps is False)."""
    Z = Xl - x0
    r = np.linalg.norm(Z, axis=1)
    s = r / eps
    w = np.zeros_like(Z)
    w[r > 0] = np.einsum("lab,lb->la", Pl[r > 0], Z[r > 0] / r[r > 0, None])
    Zs = np.sum(ml * v.xi(s, n))
    out = -(d / n) * np.einsum("l,ljk,li->ijk", ml * v.drho(s), Pl, w) / Zs
    return out / eps if with_eps else out


ok = True
for e in (0.15, 0.3, 0.6):
    ok &= np.allclose(beta_definition(e), beta_formula(e), atol=1e-12 * max(1, np.abs(beta_formula(e)).max()))
check("Proposition 48.6.1: (48.6.1) = definition (48.5.2)-(48.5.3) with normalised kernels (eps = 0.15, 0.3, 0.6)", ok)
check("Proposition 48.6.1: the printed formula (no 1/eps) is eps times the definition",
      np.allclose(beta_formula(eps, False), eps * beta_definition(eps)))
check("Proposition 48.6.1: (48.6.1) agrees with lib/var48.beta_tensor",
      np.allclose(beta_formula(eps), v.beta_tensor(x0, Xl, Pl, ml, eps, n, d)))
Aperp = beta_definition(eps) - np.einsum("jk,i->ijk", P0, np.einsum("qiq->i", beta_definition(eps)))
Dl = Pl - P0
r = np.linalg.norm(Xl - x0, axis=1)
w = np.zeros((K, 3))
w[1:] = np.einsum("lab,lb->la", Pl[1:], (Xl[1:] - x0) / r[1:, None])
a = -(d / n) * ml * v.drho(r / eps) / eps / np.sum(ml * v.xi(r / eps, n))
A_formula = np.einsum("l,ljk,li->ijk", a, Dl, w)
check("Proposition 48.6.1: (48.6.2) = beta - (P_l0)_jk H_i, the closed form (48.5.10) with c^V(x_l0) = P_l0",
      np.allclose(A_formula, Aperp) and np.allclose(A_formula, v.wsff_perp(x0, P0, Xl, Pl, ml, eps, n, d)))
T = (np.einsum("ljk,li->lijk", Dl, w) + np.einsum("lik,lj->lijk", Dl, w) - np.einsum("lij,lk->lijk", Dl, w))
B_formula = 0.5 * np.einsum("l,lijk->ijk", a, T)
check("Proposition 48.6.1: (48.6.3) = B of (48.6.2) by B^k_ij = (A_ijk + A_jik - A_kij)/2",
      np.allclose(B_formula, v.B_from_A(A_formula)))
Nrm = np.linalg.eigh(P0)[1][:, 0]
Dn = Dl @ Nrm
M = np.einsum("l,la,lb->ab", a, w, Dn) + np.einsum("l,la,lb->ab", a, Dn, w) - np.einsum("l,l,lab->ab", a, w @ Nrm, Dl)
check("Remark 48.6.3: (48.6.4) = contraction of (48.6.3) with N, symmetric, and D_l N = P_l N",
      np.allclose(0.5 * M, np.einsum("ijk,k->ij", B_formula, Nrm)) and np.allclose(M, M.T)
      and np.allclose(Dn, Pl @ Nrm) and np.allclose(0.5 * M, v.shape_perp_fast(x0, P0, Nrm, Xl, Pl, ml, eps)))

# ---------------------------------------------------------------- Exercise 48.6.2: the trace gives (47.7.1)
Hperp = -(d / n) * np.einsum("l,ab,lb->a", ml * v.drho(r / eps), np.eye(3) - P0, w) / eps / np.sum(ml * v.xi(r / eps, n))
check("Exercise 48.6.2: sum_q A_qiq = (47.7.1) (orthogonal approximate mean curvature)",
      np.allclose(np.einsum("qiq->i", A_formula), Hperp))

# ---------------------------------------------------------------- Remark 48.6.2: sign of N and choice of Q
Q1 = v.tangent_basis(Nrm)
th = 0.7
Q2 = Q1 @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
W1 = Q1.T @ (0.5 * M) @ Q1
W2 = Q2.T @ (0.5 * M) @ Q2
Wm = Q1.T @ np.einsum("ijk,k->ij", B_formula, -Nrm) @ Q1
k1, k2, km = np.linalg.eigvalsh(W1), np.linalg.eigvalsh(W2), np.linalg.eigvalsh(Wm)
check("Remark 48.6.2 / Exercise 48.1.1: eigenvalues do not depend on Q; -N flips them; K, |k1|+|k2|, |k1+k2| unchanged",
      np.allclose(k1, k2) and np.allclose(km, -k1[::-1]) and np.isclose(km.prod(), k1.prod())
      and np.isclose(np.abs(km).sum(), np.abs(k1).sum()) and np.isclose(abs(km.sum()), abs(k1.sum())))

# ---------------------------------------------------------------- Warning: the missing 1/eps on the unit sphere
Xs = v.fib_sphere(20000)
Ps = v.proj_from_normal(Xs)
ms = np.full(20000, 4 * np.pi / 20000)
kk = []
for i0 in (0, 5000, 10000, 15000, 19999):
    nb = v.ball(Xs, Xs[i0], 0.2)
    kk.append(v.principal_perp(Xs[i0], Ps[i0], Xs[i0], Xs[nb], Ps[nb], ms[nb], 0.2))
kk = np.array(kk)
print("unit sphere, eps = 0.2, five points:", np.round(kk, 4).tolist())
check("Warning: with 1/eps the eigenvalues lie in [-0.990, -0.979] (exact -1)",
      kk.min() > -0.9905 and kk.max() < -0.9785, str(kk))
check("Warning: without 1/eps they lie in [-0.198, -0.196] (eps times the curvature)",
      (0.2 * kk).min() > -0.1985 and (0.2 * kk).max() < -0.1955)

# ---------------------------------------------------------------- Exercise 48.6.3: scaling
lam = 3.0
Xb = lam * Xl
mb = lam ** d * ml
Ab = v.wsff_perp(x0, P0, Xb, Pl, mb, lam * eps, n, d)
check("Exercise 48.6.3: scaling x and eps by lambda divides (48.6.2) by lambda; the printed formula is invariant",
      np.allclose(Ab, A_formula / lam) and np.allclose(lam * eps * Ab, eps * A_formula))

# ---------------------------------------------------------------- Exercise 48.6.4: masses
check("Exercise 48.6.4 (a): multiplying all masses by c leaves (48.6.2) unchanged",
      np.allclose(v.wsff_perp(x0, P0, Xl, Pl, 7.3 * ml, eps, n, d), A_formula))
idx20, dist20 = v.knn(Xs, 20)
m_rule = math.pi * dist20[:, -1] ** 2 / 20
rat = m_rule.mean() / (4 * np.pi / 20000)
print("Exercise 48.6.4: mean of pi r_i^2 / N_mass over 4 pi / N =", round(rat, 4))
close("Exercise 48.6.4 (b): mass rule on the Fibonacci sphere: 0.95 of the area per point", rat, 0.95, 0.01)

# ---------------------------------------------------------------- Implementation box
check("Implementation 48.6: at most 30 lines", dgcode.code_lines(PAGE, "code-48-6-1") <= 30)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-48-6-1")
printed = dgcode.code_block(PAGE, "out-48-6-1").strip()
check("Implementation 48.6: printed output matches the page", buf.getvalue().strip() == printed,
      f"got:\n{buf.getvalue()}")
X = ns["X"]
Rm = np.linalg.norm(X[None] - X[:, None], axis=2)
nn = np.argsort(Rm, axis=1)[:, :40]
Y = X[nn] - X[nn].mean(axis=1, keepdims=True)
Nr = np.linalg.eigh(np.einsum("ika,ikb->iab", Y, Y))[1][:, :, 0]
Pi = v.proj_from_normal(Nr)
kl = np.zeros((len(X), 2))
for i in range(len(X)):
    Mx = v.shape_perp_fast(X[i], Pi[i], Nr[i], X[nn[i]], Pi[nn[i]], np.ones(40), Rm[i, nn[i, -1]])
    Q = v.tangent_basis(Nr[i])
    kl[i] = np.linalg.eigvalsh(Q.T @ Mx @ Q)
check("Implementation 48.6: same values as lib/var48 (shape_perp_fast) with the same planes",
      np.allclose(ns["kap"], kl, atol=1e-9))
eps_mean = Rm[np.arange(len(X)), nn[:, -1]].mean()
print("Implementation 48.6: mean eps =", round(eps_mean, 4))
check("Implementation 48.6: mean eps about 0.14 (28% of the radius 0.5); |kappa| 3% and K 6% low",
      abs(eps_mean - 0.14) < 0.005 and abs(np.abs(ns["kap"]).mean() / 2 - 0.97) < 0.005
      and abs(ns["kap"].prod(1).mean() / 4 - 0.94) < 0.005)

# ---------------------------------------------------------------- Remark 48.6.5, Example 48.6.9, Table 48.6.2 (cube)
X0 = v.cube_grid(61)
NC = len(X0)
check("Remark 48.6.5: the cube grid of spacing 1/60 has 6*61^2 - 12*61 + 8 = 21602 points",
      NC == 6 * 61 ** 2 - 12 * 61 + 8 == 21602)
onb = np.isclose(X0, 0) | np.isclose(X0, 1)
fax = np.argmax(onb, axis=1)
other = np.array([[c for c in range(3) if c != a_] for a_ in range(3)])
fc = np.take_along_axis(X0, other[fax], axis=1)
dd = np.minimum(fc, 1 - fc)
tmin, tmax = dd.min(1), dd.max(1)
face, edge, corner = tmin > 0.1, np.isclose(tmin, 0) & (tmax > 0.1), tmax < 0.05
h = 1 / 60
table = {(0.0, 40): [0.0589, 0, 14.69, 17.51], (0.01, 40): [0.0601, 2.70, 10.01, 10.86],
         (0.05, 40): [0.0805, 3.85, 4.02, 4.00], (0.05, 150): [0.1307, 1.78, 2.60, 3.03]}
res = {}
for (sig, k), row in table.items():
    Xc = X0 + sig * np.random.default_rng(0).normal(size=X0.shape) if sig > 0 else X0.copy()
    idx, dist = v.knn(Xc, k)
    P, nrm = v.pca_planes(Xc, k, idx=idx, dist=dist)
    kap, _ = v.curvatures_knn(Xc, P, nrm, np.ones(NC), k, idx=idx, dist=dist)
    S = np.abs(kap).sum(1)
    res[(sig, k)] = (kap, P, nrm, dist)
    got = [dist[:, -1].mean(), S[face].mean(), S[edge].mean(), S[corner].mean()]
    close_each(f"Table 48.6.2: (sigma, k) = ({sig}, {k}): mean eps, S face, S edge, S corner", got, row,
               [5e-5, 5e-3, 5e-3, 5e-3])
kap, P, nrm, dist = res[(0.0, 40)]
S, Kc = np.abs(kap).sum(1), kap.prod(1)
check("Example 48.6.9: no noise: S = 0 exactly on the faces", np.abs(S[face]).max() < 1e-12)
prof = np.array([S[np.isclose(tmin, t * h) & (tmax > 0.2)].mean() for t in range(10)])
close("Example 48.6.9: profile 14.69, 13.43, 9.05, 3.71, 1.76, 0.47, 0.045 for t = 0..6", prof[:7],
      [14.69, 13.43, 9.05, 3.71, 1.76, 0.47, 0.045], 6e-3)
check("Example 48.6.9 / Exercise 48.6.5: profile exactly 0 from t = 7", np.all(prof[7:] < 1e-12) and prof[6] > 0.04)
close("Example 48.6.9: mean K near the vertices (both in-face coordinates within 0.05) = 73.5", Kc[corner].mean(), 73.5,
      0.05)
vtx = tmax < 1e-9
close_each("Example 48.6.9: at a vertex kappa = (9.91, 10.70), K = 105", [*np.sort(np.abs(kap[vtx][0])),
           Kc[vtx].mean()], [9.91, 10.70, 105.3], [5e-3, 5e-3, 0.05])
tilt = np.degrees(np.arccos(np.clip(np.abs(np.einsum("ij,ij->i", nrm, np.eye(3)[fax])), 0, 1)))
tl = [tilt[np.isclose(tmin, t * h) & (tmax > 0.2)].mean() for t in range(6)]
close_each("Exercise 48.6.5: PCA planes tilted 45, 15.8, 5.8, 0.67 degrees at t = 0..3", tl[:4],
           [45.0, 15.8, 5.8, 0.67], [0.05, 0.05, 0.05, 0.005])
inner = face & (tmin > 4.5 * h)
check("Exercise 48.6.5: PCA planes exact from t = 4; eps = sqrt(13) h on the faces",
      max(tl[4:]) < 1e-6 and np.allclose(dist[inner, -1], math.sqrt(13) * h))
reg_c = tmax < 0.1
sumK, partC = Kc.sum() * h * h, Kc[reg_c].sum() * h * h
partE = Kc[(~reg_c) & (tmin < 0.1)].sum() * h * h
close_each("Remark 48.6.10: sum K h^2 = 4.55 (not 4 pi); vertex regions 4.58 (0.57 each), edge bands -0.03",
           [sumK, partC, partC / 8, partE], [4.55, 4.58, 0.572, -0.033], [5e-3, 5e-3, 5e-4, 5e-3])
check("Remark 48.6.10: faces contribute 0", abs(Kc[face].sum()) < 1e-9)

# ---------------------------------------------------------------- Example 48.6.7, Figure 48.6.1 (torus)
R_, r_ = 2.0, 0.8
KS = [20, 40, 80, 160, 320]
expect = {20000: {20: (0.228, 0.094), 40: (0.153, 0.056), 80: (0.113, 0.037), 160: (0.108, 0.030), 320: (0.174, 0.035)},
          80000: {20: (0.238, 0.099), 40: (0.160, 0.056), 80: (0.109, 0.035), 160: (0.075, 0.024), 320: (0.064, 0.018)}}
expK = {20000: [0.119, 0.066, 0.047, 0.057, 0.091], 80000: [0.122, 0.061, 0.036, 0.026, 0.030]}
expE = {20000: [0.137, 0.197, 0.281, 0.398, 0.561], 80000: [0.0687, 0.0987, 0.141, 0.200, 0.282]}
for NT in (20000, 80000):
    u, w_ = v.torus_sample(NT, R_, r_, seed=3)
    XT = v.torus_point(u, w_, R_, r_)
    Nex = v.torus_outward(u, w_)
    ex = np.sort(np.stack(v.torus_kappa_outward(u, R_, r_), 1), 1)
    sub = np.arange(0, NT, NT // 4000)[:4000]
    idx_all, dist_all = v.knn(XT, max(KS))
    errs, Kerr, epsm, angs = [], [], [], []
    for k in KS:
        idx, dist = idx_all[:, :k], dist_all[:, :k]
        P, nrm = v.pca_planes(XT, k, idx=idx, dist=dist)
        nrm = nrm * np.sign(np.einsum("ij,ij->i", nrm, Nex))[:, None]
        kp = np.zeros((len(sub), 2))
        for a_, i in enumerate(sub):
            Mx = v.shape_perp_fast(XT[i], P[i], nrm[i], XT[idx[i]], P[idx[i]], np.ones(k), dist[i, -1] * (1 + 1e-9))
            Q = v.tangent_basis(nrm[i])
            W = Q.T @ Mx @ Q
            kp[a_] = np.linalg.eigvalsh(0.5 * (W + W.T))
        errs.append(np.abs(kp - ex[sub]).mean(0))
        Kerr.append(np.abs(kp.prod(1) - ex[sub].prod(1)).mean())
        epsm.append(dist[sub, -1].mean())
        angs.append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", nrm[sub], Nex[sub]), -1, 1))).mean())
    errs = np.array(errs)
    print(f"torus N = {NT}: eps", np.round(epsm, 4).tolist(), "errors", np.round(errs, 3).tolist(),
          "K errors", np.round(Kerr, 3).tolist(), "normal angle (deg)", np.round(angs, 2).tolist())
    close(f"Example 48.6.7: N = {NT}: mean eps for k = 20..320", epsm, expE[NT], 2e-3)
    close(f"Example 48.6.7: N = {NT}: mean errors of (kappa_min, kappa_max) for k = 20..320",
          errs.ravel(), np.array([expect[NT][k] for k in KS]).ravel(), 2e-3)
    close(f"Example 48.6.7: N = {NT}: mean errors of K for k = 20..320", Kerr, expK[NT], 2e-3)
    close(f"Example 48.6.7: N = {NT}, k = 40: mean normal angle (1.0 / 0.5 degrees)", angs[1],
          1.0 if NT == 20000 else 0.5, 0.06)
    check(f"Example 48.6.7: N = {NT}: best k for kappa_min is {160 if NT == 20000 else 320}",
          KS[int(np.argmin(errs[:, 0]))] == (160 if NT == 20000 else 320))
summary()
