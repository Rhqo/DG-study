"""Verification for Section 48.1 (Problem and Key Idea).

Run from the project root: PYTHONPATH=tools python3 chapters/48-paper-varifold-curvatures/verify/v-48-1-problem-key-idea.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import itertools
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var48 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

n, d = 2, 1


def ellipse_cloud(N):
    t = 2 * np.pi * np.arange(N) / N
    X = np.stack([2 * np.cos(t), np.sin(t)], axis=1)
    G = np.stack([-2 * np.sin(t), np.cos(t)], axis=1)
    sp = np.linalg.norm(G, axis=1)
    T = G / sp[:, None]
    return t, X, T, v.proj_from_tangent(T), sp * 2 * np.pi / N


def angle_from(tau, U):
    return np.arctan2(tau[0] * U[:, 1] - tau[1] * U[:, 0], U @ tau)


def contracted(i0, X, T, P, m, eps):
    """(-tau0^T sum_k B^k_ij n0_k tau0, xi-weighted mean of theta_l/s_l, quotients) with n0 outward."""
    x0, tau0 = X[i0], T[i0]
    n0 = np.array([tau0[1], -tau0[0]])
    nb = v.ball(X, x0, eps)
    B = v.B_from_A(v.wsff_perp(x0, P[i0], X[nb], P[nb], m[nb], eps, n, d))
    kh = -float(tau0 @ np.einsum("ijk,k->ij", B, n0) @ tau0)
    oth = nb[nb != i0]
    s = (X[oth] - x0) @ tau0
    q = angle_from(tau0, T[oth]) / s
    w = m[oth] * v.xi(np.linalg.norm(X[oth] - x0, axis=1) / eps, n)
    return kh, float(np.sum(w * q) / np.sum(w)), q, len(nb)


# ---------------------------------------------------------------- Figure 48.1.1
t, X, T, P, m = ellipse_cloud(200)
kh, mq, q, cnt = contracted(25, X, T, P, m, 0.3)
check("Figure 48.1.1: 12 points (including x_0) in B_0.3(x_0)", cnt == 12)
close("Figure 48.1.1: kappa(pi/4) = 2/(2.5)^1.5 = 0.5060", 2 / 2.5 ** 1.5, 0.50596, 1e-4)
close("Figure 48.1.1: epsilon-WSFF-perp contracted with the outward normal gives -0.5074", kh, 0.5074, 1e-4)
close("Figure 48.1.1: xi-weighted mean of the difference quotients = 0.5128", mq, 0.5128, 1e-4)
close("Figure 48.1.1: difference quotients range from 0.445 to 0.611", [q.min(), q.max()], [0.445, 0.611], 6e-4)

# ---------------------------------------------------------------- (48.1.2) = (6.22) = beta - c_jk H_i
x0 = X[25]
nb = v.ball(X, x0, 0.3)
beta = v.beta_tensor(x0, X[nb], P[nb], m[nb], 0.3, n, d)
H = v.H_from_beta(beta)
A622 = beta - np.einsum("jk,i->ijk", P[25], H)
A_centered = v.wsff_perp(x0, P[25], X[nb], P[nb], m[nb], 0.3, n, d)
close("(48.1.2): centered form equals beta_ijk - c_jk sum_q beta_qiq of (6.22)", A_centered, A622, 1e-12)
Aperp_sys = v.solve_system(beta - np.einsum("jk,i->ijk", P[25], P[25] @ H), P[25])
close("(48.1.2): equals the solution of (3.4) with b = beta-perp of (6.21) and c = P_0", A_centered, Aperp_sys, 1e-12)

# ---------------------------------------------------------------- Exercise 48.1.2: dimension count
n3 = 3
symA = [(i, j, k) for i in range(n3) for j in range(n3) for k in range(n3) if j <= k]
symB = [(i, j, k) for i in range(n3) for j in range(n3) for k in range(n3) if i <= j]
check("Exercise 48.1.2: 18 independent entries of A (A_ijk = A_ikj) and of B (B^k_ij = B^k_ji)",
      len(symA) == 18 and len(symB) == 18)
rng = np.random.default_rng(0)
Bt = rng.normal(size=(3, 3, 3))
Bt = 0.5 * (Bt + np.transpose(Bt, (1, 0, 2)))
At = v.A_from_B(Bt)
check("Exercise 48.1.2: A = B + B^T is symmetric in (j, k) and (3.8) recovers B", np.allclose(At, np.transpose(
    At, (0, 2, 1))) and np.allclose(v.B_from_A(At), Bt))
M = np.zeros((27, 27))
for c_, (i, j, k) in enumerate(itertools.product(range(3), repeat=3)):
    E = np.zeros((3, 3, 3))
    E[i, j, k] = 1.0
    E = 0.5 * (E + np.transpose(E, (1, 0, 2)))
    M[:, c_] = v.A_from_B(E).ravel()
check("Exercise 48.1.2: B -> A has rank 18", np.linalg.matrix_rank(M) == 18)

# ---------------------------------------------------------------- Exercise 48.1.3: the vertex
kh0, mq0, q0, _ = contracted(0, X, T, P, m, 0.2)
close("Exercise 48.1.3: vertex quotients (eps = 0.2) range 1.945..1.998", [q0.min(), q0.max()], [1.945, 1.998],
      6e-4)
check("Exercise 48.1.3: vertex quotients are all below kappa = 2 and even in s", np.all(q0 < 2) and np.allclose(
    np.sort(q0[: len(q0) // 2]), np.sort(q0[len(q0) // 2:])))

# ---------------------------------------------------------------- Exercise 48.1.4: the circle
Nc = 2000
tc = 2 * np.pi * np.arange(Nc) / Nc
Xc = np.stack([np.cos(tc), np.sin(tc)], axis=1)
Tc = np.stack([-np.sin(tc), np.cos(tc)], axis=1)
Pc = v.proj_from_tangent(Tc)
mc = np.full(Nc, 2 * np.pi / Nc)
khc, mqc, qc, _ = contracted(0, Xc, Tc, Pc, mc, 0.2)
check("Exercise 48.1.4: on the unit circle every quotient theta/s = phi/sin(phi) >= 1", np.all(qc >= 1.0))
close("Exercise 48.1.4: unit circle, eps = 0.2: weighted mean 1.0032", mqc, 1.0032, 1e-4)
close("Exercise 48.1.4: unit circle, eps = 0.2: exact contraction 0.9859", khc, 0.9859, 1e-4)
close("Section 48.1: natural pair C_xi/C_rho = d/n = 1/2 (n = 2, d = 1)", v.ratio(1, 2), 0.5, 1e-9)

summary()
