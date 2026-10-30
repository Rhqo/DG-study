"""Verification for Section 48.2 (The Second Fundamental Form in Projection Form).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-2-projection-form.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var48 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

R, r = 2.0, 0.8


def chart_A(xf, nf, u, w, hu, hw):
    """A_ijk = <grad^M P_jk, e_i> from central differences of x(u, w) and P = I - N N^T on a chart."""
    def P(a, b):
        return v.proj_from_normal(nf(a, b))
    xu = (xf(u + hu, w) - xf(u - hu, w)) / (2 * hu)
    xw = (xf(u, w + hw) - xf(u, w - hw)) / (2 * hw)
    Pu = (P(u + hu, w) - P(u - hu, w)) / (2 * hu)
    Pw = (P(u, w + hw) - P(u, w - hw)) / (2 * hw)
    g = np.array([[xu @ xu, xu @ xw], [xw @ xu, xw @ xw]])
    gi = np.linalg.inv(g)
    grads = [Pu, Pw]
    tang = [xu, xw]
    A = np.zeros((3, 3, 3))
    for a in range(2):
        for b in range(2):
            A += gi[a, b] * np.einsum("jk,i->ijk", grads[a], tang[b])
    return A, xu, xw, gi


tor_x = lambda a, b: v.torus_point(a, b, R, r)  # noqa: E731
tor_n = lambda a, b: v.torus_outward(a, b)       # noqa: E731


def torus_BN(u, w, h=1e-4):
    A, xu, xw, gi = chart_A(tor_x, tor_n, u, w, h, h)
    B = v.B_from_A(A)
    N = tor_n(u, w)
    return A, B, np.einsum("ijk,k->ij", B, N), N, xu, xw, gi


# ---------------------------------------------------------------- Proposition 48.2.1 on random torus points
rng = np.random.default_rng(3)
ok_eig = ok_null = ok_tr = ok_flip = ok_a = True
for _ in range(12):
    u, w = rng.uniform(-np.pi, np.pi, 2)
    A, B, BN, N, xu, xw, gi = torus_BN(u, w)
    Q = v.tangent_basis(N)
    ev = np.linalg.eigvalsh(Q.T @ BN @ Q)
    ex = np.sort(v.torus_kappa_outward(u, R, r))
    ok_eig &= np.allclose(ev, ex, atol=1e-6)
    ok_null &= np.allclose(BN @ N, 0, atol=1e-7)
    Hvec = np.einsum("qiq->i", A)
    ok_tr &= abs(np.trace(BN) - ex.sum()) < 1e-6 and np.allclose(Hvec, ex.sum() * N, atol=1e-6)
    ok_flip &= np.allclose(np.einsum("ijk,k->ij", B, -N), -BN)
    # (a): B^k_ij = <W P e_i, P e_j> N_k with W U = -D_U N computed independently from the chart
    Nu = (tor_n(u + 1e-4, w) - tor_n(u - 1e-4, w)) / 2e-4
    Nw = (tor_n(u, w + 1e-4) - tor_n(u, w - 1e-4)) / 2e-4
    Pm = v.proj_from_normal(N)
    WPe = np.zeros((3, 3))          # column i = W P e_i = -D_{P e_i} N
    for i in range(3):
        c = gi @ np.array([xu @ Pm[:, i], xw @ Pm[:, i]])   # coordinates of P e_i in (x_u, x_w)
        WPe[:, i] = -(c[0] * Nu + c[1] * Nw)
    Bpred = np.einsum("ai,aj,k->ijk", WPe, Pm, N)
    ok_a &= np.allclose(B, Bpred, atol=1e-6)
check("Proposition 48.2.1(a): B^k_ij = <W P e_i, P e_j> N_k on 12 random points of T_{2,0.8}", ok_a)
check("Proposition 48.2.1(b): eigenvalues of Q^T B_N Q = (-1/r, -cos u/(R + r cos u)) (outward normal)", ok_eig)
check("Proposition 48.2.1(b): B_N N = 0", ok_null)
check("Proposition 48.2.1(c): tr B_N = kappa_1 + kappa_2 and sum_q A_qiq = (kappa_1 + kappa_2) N", ok_tr)
check("Proposition 48.2.1(d): -N gives -B_N", ok_flip)

# sphere of radius 0.6 with the outward normal: B_N = -P/r, both principal curvatures -1/r
rs = 0.6
sph_x = lambda a, b: rs * np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])  # noqa: E731
sph_n = lambda a, b: np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])       # noqa: E731
A, *_ = chart_A(sph_x, sph_n, 0.4, 1.1, 1e-4, 1e-4)
Ns = sph_n(0.4, 1.1)
BNs = np.einsum("ijk,k->ij", v.B_from_A(A), Ns)
close("Proposition 48.2.1: sphere (r = 0.6, outward N): B_N = -P/r", BNs, -v.proj_from_normal(Ns) / rs, 1e-6)

# ---------------------------------------------------------------- Example 48.2.2
_, _, BN0, N0, *_ = torus_BN(0.0, 0.0)
close("Example 48.2.2: outer equator (2.8, 0, 0): B_N = diag(0, -0.3571, -1.25)", BN0,
      np.diag([0.0, -1 / 2.8, -1.25]), 1e-6)
close("Example 48.2.2: outer equator N = e_1", N0, [1.0, 0.0, 0.0], 1e-12)
_, _, BNpi, Npi, *_ = torus_BN(np.pi, 0.0)
close("Example 48.2.2: inner equator (1.2, 0, 0): N = -e_1, B_N = diag(0, 0.8333, -1.25)",
      np.concatenate([Npi, BNpi.ravel()]), np.concatenate([[-1.0, 0.0, 0.0], np.diag([0.0, 1 / 1.2, -1.25]).ravel()]),
      1e-6)
close("Example 48.2.2: K = 0.4464 (outer), -1.0417 (inner)", [1.25 / 2.8, -1.25 / 1.2], [0.4464, -1.0417], 5e-5)
_, _, BNtop, *_ = torus_BN(np.pi / 2, 0.3)
close("Example 48.2.2: top circle u = pi/2: eigenvalues of B_N are (-1.25, 0, 0)", np.linalg.eigvalsh(BNtop),
      [-1.25, 0.0, 0.0], 1e-6)

# ---------------------------------------------------------------- Figure 48.2.1 (48 x 96 grid)
NU, NV = 48, 96
hu, hv = 2 * np.pi / NU, 2 * np.pi / NV
us = np.arange(NU) * hu - np.pi
kap = np.zeros((NU, 2))
Kerr = 0.0
Kvals = []
for a, u in enumerate(us):
    A, xu, xw, gi = chart_A(tor_x, tor_n, u, 0.0, hu, hv)
    N = tor_n(u, 0.0)
    Q = np.stack([np.array([-np.sin(u), 0.0, np.cos(u)]), np.array([0.0, 1.0, 0.0])], 1)
    W = Q.T @ np.einsum("ijk,k->ij", v.B_from_A(A), N) @ Q
    kap[a] = np.linalg.eigvalsh(0.5 * (W + W.T))
    Kvals.append(np.prod(kap[a]))
ex = np.sort(np.stack(v.torus_kappa_outward(us, R, r), 1), 1)
err = np.abs(kap - ex).max()
Kex = np.cos(us) / (r * (R + r * np.cos(us)))
check("Figure 48.2.1: grid includes u = 0 and u = -pi", np.isclose(us, 0).any() and np.isclose(us[0], -np.pi))
close("Figure 48.2.1: max |kappa_grid - kappa| = 0.011", err, 0.0107, 2e-4)
close("Figure 48.2.1: max |K_grid - K| = 0.011", np.abs(np.array(Kvals) - Kex).max(), 0.0111, 2e-4)
close("Figure 48.2.1: K ranges over [-1.042, 0.446]", [Kex.min(), Kex.max()], [-1.0417, 0.4464], 5e-5)
ratio = (np.sin(2 * hu) / (2 * hu)) / (np.sin(hu) / hu)
close("Figure 48.2.1: difference ratio sin(2h)/2h / (sin h/h) = 0.991, -1.25 becomes -1.239",
      [ratio, kap[:, 0].mean()], [0.9915, -1.2393], 1e-4)

# ---------------------------------------------------------------- Remark 48.2.3: A contracted with N is not symmetric
uu, vv = v.torus_sample(20000, seed=1)
X = v.torus_point(uu, vv)
Nn = v.torus_outward(uu, vv)
P = v.proj_from_normal(Nn)
m = np.full(len(X), 4 * np.pi ** 2 * R * r / len(X))
nb = v.ball(X, X[0], 0.25)
A = v.wsff_perp(X[0], P[0], X[nb], P[nb], m[nb], 0.25, 3, 2)
AN = np.einsum("ijk,k->ij", A, Nn[0])
BN = np.einsum("ijk,k->ij", v.B_from_A(A), Nn[0])
check("Remark 48.2.3: the point is near the inner equator (cos u < -0.99)", np.cos(uu[0]) < -0.99)
close("Remark 48.2.3: max |M_ij - M_ji| of sum_k A_ijk N_k = 0.14", np.abs(AN - AN.T).max(), 0.139, 1e-3)
check("Remark 48.2.3: B_N is exactly symmetric", np.abs(BN - BN.T).max() < 1e-14)
check("Remark 48.2.3: smooth case: sum_k A_ijk N_k = B_N",
      np.allclose(np.einsum("ijk,k->ij", torus_BN(0.7, 0.2)[0], tor_n(0.7, 0.2)), torus_BN(0.7, 0.2)[2], atol=1e-6))

# ---------------------------------------------------------------- Exercise 48.2.1
At = rng.normal(size=(3, 3, 3))
Bt = v.B_from_A(At)
close("Exercise 48.2.1: B^k_ji - B^k_ij = (A_kij - A_kji)/2",
      np.transpose(Bt, (1, 0, 2)) - Bt, 0.5 * (np.einsum("kij->ijk", At) - np.einsum("kji->ijk", At)), 1e-12)
As = 0.5 * (At + np.transpose(At, (0, 2, 1)))
Bs = v.B_from_A(As)
check("Exercise 48.2.1: A_ijk = A_ikj gives B^k_ij = B^k_ji", np.allclose(Bs, np.transpose(Bs, (1, 0, 2))))
check("Exercise 48.2.1: a general A gives a non-symmetric B", not np.allclose(Bt, np.transpose(Bt, (1, 0, 2))))

# ---------------------------------------------------------------- Exercise 48.2.2: cylinder
rc = 0.5
cyl_x = lambda a, b: np.array([rc * np.cos(a), rc * np.sin(a), b])  # noqa: E731
cyl_n = lambda a, b: np.array([np.cos(a), np.sin(a), 0.0])          # noqa: E731
A, *_ = chart_A(cyl_x, cyl_n, 0.0, 0.0, 1e-4, 1e-4)
BNc = np.einsum("ijk,k->ij", v.B_from_A(A), cyl_n(0.0, 0.0))
close("Exercise 48.2.2: cylinder r = 0.5 at (r, 0, 0): B_N = diag(0, -1/r, 0)", BNc, np.diag([0.0, -2.0, 0.0]), 1e-6)
Qc = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
close("Exercise 48.2.2: Q^T B_N Q = diag(-1/r, 0)", Qc.T @ BNc @ Qc, np.diag([-2.0, 0.0]), 1e-6)

# ---------------------------------------------------------------- Exercise 48.2.3
A0, B0, *_ = torus_BN(0.0, 0.0)
close("Exercise 48.2.3: outer equator: sum B^2 = sum kappa^2 = 1.6901", np.sum(B0 ** 2), 1.6901, 1e-4)
close("Exercise 48.2.3: outer equator: sum A^2 = 2 sum kappa^2 = 3.3801", np.sum(A0 ** 2), 3.3801, 1e-4)
u1, w1 = 2.1, -0.4
A1, B1, *_ = torus_BN(u1, w1)
k1 = np.array(v.torus_kappa_outward(u1, R, r))
check("Exercise 48.2.3: the identities at a generic point", np.isclose(np.sum(B1 ** 2), np.sum(k1 ** 2), atol=1e-6)
      and np.isclose(np.sum(A1 ** 2), 2 * np.sum(k1 ** 2), atol=1e-6))

# ---------------------------------------------------------------- Exercise 48.2.4: a circle in R^3
rcir, th, h = 0.7, 0.9, 1e-5
cx = lambda t: rcir * np.array([np.cos(t), np.sin(t), 0.0])  # noqa: E731
ct = lambda t: np.array([-np.sin(t), np.cos(t), 0.0])        # noqa: E731
dPds = (np.outer(ct(th + h), ct(th + h)) - np.outer(ct(th - h), ct(th - h))) / (2 * h * rcir)
tau, x = ct(th), cx(th)
Acir = np.einsum("i,jk->ijk", tau, dPds)
Bcir = v.B_from_A(Acir)
close("Exercise 48.2.4: B^k_ij = -tau_i tau_j x_k / r^2", Bcir, -np.einsum("i,j,k->ijk", tau, tau, x) / rcir ** 2, 1e-8)
close("Exercise 48.2.4: H = sum_q A_qiq = -x/r^2", np.einsum("qiq->i", Acir), -x / rcir ** 2, 1e-8)
vals = []
for al in (0.0, 0.6, np.pi / 2):
    nu = np.cos(al) * x / rcir + np.sin(al) * np.array([0.0, 0.0, 1.0])
    vals.append(tau @ np.einsum("ijk,k->ij", Bcir, nu) @ tau + np.cos(al) / rcir)
close("Exercise 48.2.4: tau^T B_nu tau = -cos(alpha)/r", vals, [0.0, 0.0, 0.0], 1e-8)

summary()
