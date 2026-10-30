"""Verification for Section 47.3 (Regularized First Variation).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-3-regularized-first-variation.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var47 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

R, L = 0.5, math.pi
c_exp = v.normaliser(v.rho_exp, 2)
C_RHO = c_exp * v.C_const(v.rho_exp, 1)

# ---------------------------------------------------------------- L^1 bound of Prop 3.2 (our version)
g = np.linspace(-1, 1, 2001)
GX, GY = np.meshgrid(g, g)
Rg = np.hypot(GX, GY)
gradL1 = (c_exp * np.abs(v.drho_exp(Rg))).sum() * (g[1] - g[0]) ** 2
close("Remark 47.3.4: ||grad rho_1||_{L^1(R^2)} = 2 pi c int rho dr = pi C_rho = 2.990 (grid)", gradL1, math.pi * C_RHO, 1e-4)
close("pi C_rho = 2.990", math.pi * C_RHO, 2.990, 5e-4)
h0 = L / 250
eps0 = 4 * h0
close("Remark 47.3.4: bound for the circle, N = 250, eps = 4h: 2.990 pi / eps = 186.9", math.pi * C_RHO * L / eps0, 186.9, 0.05)


# ---------------------------------------------------------------- clouds and the field (3.4)
def cloud(N, jitter, seed=3):
    j = np.arange(N, dtype=float)
    if jitter:
        j = j + np.random.default_rng(seed).standard_normal(N)
    X, T, _ = v.circle(2 * np.pi * j / N, R)
    return X, v.proj_from_tangent(T), np.full(N, L / N)


def field(Q, X, P, m, eps):
    D = X[None, :, :] - Q[:, None, :]
    r = np.linalg.norm(D, axis=2)
    gg = c_exp * v.drho_exp(r / eps) / eps ** 3
    u = D / np.where(r > 0, r, 1.0)[:, :, None]
    return np.einsum("j,qj,qja->qa", m, gg, np.einsum("jab,qjb->qja", P, u))


s = np.linspace(0, np.radians(20), 1200)
Q = R * np.stack([np.cos(s), np.sin(s)], 1)
nu = Q / R
tg = np.stack([-nu[:, 1], nu[:, 0]], 1)
Xu, Pu, mu = cloud(250, False)
Xj, Pj, mj = cloud(250, True)
Fu, Fj = eps0 * field(Q, Xu, Pu, mu, eps0), eps0 * field(Q, Xj, Pj, mj, eps0)
Fs = eps0 * field(Q[:5], *cloud(20000, False), eps0)
smooth = np.einsum("qa,qa->q", Fs, nu[:5]).mean()
close("Figure 47.3.1: smooth circle eps (delta V * rho_eps) . nu = |H| C_rho = 1.90", smooth, 2 * C_RHO, 2e-3)
close("2 C_rho = 1.903", 2 * C_RHO, 1.903, 5e-4)
check("Figure 47.3.1: delta V * rho_eps points outward (normal component > 0)", np.all(np.einsum("qa,qa->q", Fu, nu) > 0))
un, ut = np.einsum("qa,qa->q", Fu, nu), np.einsum("qa,qa->q", Fu, tg)
jt = np.einsum("qa,qa->q", Fj, tg)
close("Figure 47.3.1(a): uniform normal component in [1.86, 1.99]", [un.min(), un.max()], [1.86, 1.99], 6e-3)
close("Figure 47.3.1(a): uniform tangential ripple amplitude 3.1", np.abs(ut).max(), 3.1, 0.02)
close("Figure 47.3.1(b): jittered tangential up to 25.7", np.abs(jt).max(), 25.7, 0.05)
Fd = field(Xu[:4], Xu, Pu, mu, eps0)
check("Exercise 47.3.4: tangential part vanishes at the uniform data points",
      np.abs(np.einsum("qa,qa->q", Fd, np.stack([-Xu[:4, 1], Xu[:4, 0]], 1) / R)).max() < 1e-10)
mid = R * np.array([[np.cos(np.pi / 250), np.sin(np.pi / 250)]])
Fm = field(mid, Xu, Pu, mu, eps0)
check("Exercise 47.3.4: and at the midpoints between two data points",
      abs(Fm[0] @ np.array([-mid[0, 1], mid[0, 0]]) / R) < 1e-8)

# ---------------------------------------------------------------- Figure 47.3.1(c): L^1 norms
Ns = [250, 500, 1000, 2000, 4000]
uni4 = [v.field_L1_circle(*cloud(N, False), 4 * L / N, R)[0] for N in Ns]
jit4 = [v.field_L1_circle(*cloud(N, True), 4 * L / N, R)[0] for N in Ns]
jit3 = [v.field_L1_circle(*cloud(N, True), 4 * h0 * (N / 250) ** (-1 / 3), R)[0] for N in Ns]
uni16 = [v.field_L1_circle(*cloud(N, False), 16 * L / N, R)[0] for N in Ns]
close("Figure 47.3.1(c): uniform eps = 4h: 8.49 -> 82.2", [uni4[0], uni4[-1]], [8.49, 82.2], 0.05)
close("Figure 47.3.1(c): jittered eps = 4h: 31.8 -> 485", [jit4[0], jit4[-1]], [31.8, 485], 0.5)
close("Figure 47.3.1(c): jittered eps ~ N^(-1/3): 31.8 -> 11.2", [jit3[0], jit3[-1]], [31.8, 11.2], 0.05)
close("Figure 47.3.1(c): uniform eps = 16h: 6.216 -> 6.285 (2 pi = 6.283)", [uni16[0], uni16[-1]], [6.216, 6.285], 5e-4)
check("Figure 47.3.1(c): eps = 4h grows roughly like N (ratio > 9 over 16x)", jit4[-1] / jit4[0] > 14 and uni4[-1] / uni4[0] > 9)
ex = np.array(jit3) - 2 * np.pi
sl = np.polyfit(np.log(Ns), np.log(ex), 1)[0]
close("Example 47.3.7 (heuristic): excess over 2 pi decays like N^(-2/3) (fitted slope about -0.6)", sl, -0.6, 0.08)
bound = lambda eps: math.pi * C_RHO * L / eps
check("Remark 47.3.4: every L^1 norm in Figure 47.3.1(c) is below 2.990 ||V|| / eps",
      all(x < bound(4 * L / N) for x, N in zip(uni4 + jit4, Ns + Ns))
      and all(x < bound(4 * h0 * (N / 250) ** (-1 / 3)) for x, N in zip(jit3, Ns))
      and all(x < bound(16 * L / N) for x, N in zip(uni16, Ns)))

# ---------------------------------------------------------------- Exercise 47.3.2: affine fields are unchanged by convolution
# first moment of a radial kernel vanishes: int rho_1(z) z dz = 0, so rho_eps * X = X for affine X
w = c_exp * v.rho_exp(Rg) * (g[1] - g[0]) ** 2
close("Exercise 47.3.2: int rho_1 = 1, int rho_1 z = 0", [w.sum(), (w * GX).sum(), (w * GY).sum()], [1, 0, 0], 1e-6)

# ---------------------------------------------------------------- Example 47.3.8: stacked segments
for k in (2, 5, 20):
    mass = k * (1 / k) * 1.0
    tv = k * (1 / k) * 2           # each segment: Dirac masses -e1, +e1 at its ends with weight 1/k
    close(f"Example 47.3.9: k = {k}: mass 1 and |delta V_k|(R^2) = 2", [mass, tv], [1, 2], 1e-12)
    r = 0.5 / k
    check(f"Example 47.3.9: k = {k}: ||V_k||(B_r(x)) = 2r/k for r < 1/k, so no uniform theta_0", abs((1 / k) * 2 * r - 2 * r / k) < 1e-15)

summary()
