"""Verification for Section 47.1 (Problem and Key Idea).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-1-problem-key-idea.py
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

K = v.pairs(2)

# ---------------------------------------------------------------- Definition 47.1.1, Example 47.1.2: normalisation
c_exp = v.normaliser(v.rho_exp, 2)
close("Example 47.1.2: exp profile normalised in R^2, c = 2.1436", c_exp, 2.1436, 5e-5)
close("Example 47.1.2 / Exercise 47.1.1: tent profile normalised in R^2, c' = 3/pi = 0.9549", v.normaliser(v.rho_tent, 2),
      3 / math.pi, 1e-12)
close("Exercise 47.1.1: tent profile normalised in R^3, also c' = 3/pi", v.normaliser(v.rho_tent, 3), 3 / math.pi, 1e-12)
close("c' = 0.9549", 3 / math.pi, 0.9549, 5e-5)
for n in (2, 3):
    for name, prof in (("exp", v.rho_exp), ("tent", v.rho_tent)):
        p = v.pairs(n)["natural " + name]
        close(f"Example 47.1.2: natural {name} partner -r rho'/n has the same normaliser as rho (n = {n})",
              v.normaliser(p.xi, n), v.normaliser(prof, n), 1e-9)

# check the normalisation by a 2D grid sum for the exp kernel (eps = 1)
g = np.linspace(-1, 1, 1601)
GX, GY = np.meshgrid(g, g)
R = np.hypot(GX, GY)
close("Definition 47.1.1: int_{R^2} rho_1 = 1 (grid sum)", (c_exp * v.rho_exp(R)).sum() * (g[1] - g[0]) ** 2, 1.0, 1e-6)
close("Definition 47.1.1: int_{R^2} xi_1 = 1 for the natural exp partner (grid sum)",
      (c_exp * K["natural exp"].xi(R)).sum() * (g[1] - g[0]) ** 2, 1.0, 1e-6)

# ---------------------------------------------------------------- constants C_rho, C_xi for d = 1
C_rho_exp = c_exp * v.C_const(v.rho_exp, 1)
C_xi_exp = c_exp * v.C_const(K["natural exp"].xi, 1)
close("Example 47.1.2: C_rho = 0.9517 (normalised exp, d = 1)", C_rho_exp, 0.9517, 5e-5)
close("Example 47.1.2: C_xi = 0.4759 (natural exp, d = 1)", C_xi_exp, 0.4759, 5e-5)
ct = 3 / math.pi
close("Example 47.1.2 / Exercise 47.1.2: C_rho = c' for tent (d = 1)", ct * v.C_const(v.rho_tent, 1), ct, 1e-12)
close("Example 47.1.2 / Exercise 47.1.2: C_xi = c'/2 = 0.4775 for natural tent (d = 1)",
      ct * v.C_const(K["natural tent"].xi, 1), 0.4775, 5e-5)
for name in ("natural exp", "natural tent"):
    close(f"Exercise 47.1.2: C_xi / C_rho = 1/2 for the {name} pair (d = 1, n = 2)", K[name].ratio(1), 0.5, 1e-9)
for name in ("exp", "tent"):
    close(f"Exercise 47.1.2: C_xi / C_rho = 1 for the {name} pair", K[name].ratio(1), 1.0, 1e-12)

# ---------------------------------------------------------------- Figure 47.1.2 statements
r = np.linspace(0, 1, 200001)[:-1]
xi = K["natural exp"].xi(r)
rmax = r[np.argmax(xi)]
check(f"Figure 47.1.2(a): natural exp partner peaks near r = 0.8 (argmax {rmax:.4f})", abs(rmax - 0.8) < 0.03)
check("Figure 47.1.2(a): xi_exp(0) = 0", K["natural exp"].xi(np.array([0.0]))[0] == 0.0)
close("Figure 47.1.2(b): jump of the normalised natural tent partner at r = 1 is c'/2 = 0.477",
      ct * K["natural tent"].xi(np.array([1 - 1e-12]))[0], 0.477, 5e-4)
check("Non-example 47.1.3: rho_tent has slope -1 at 0+ (so rho(|x|) has a cone point)",
      abs((v.rho_tent(1e-6) - v.rho_tent(0.0)) / 1e-6 + 1) < 1e-9)

# ---------------------------------------------------------------- Figure 47.1.1: the key idea on a circle
N, RAD, EPS = 60, 0.5, 0.2
pair = K["natural exp"]
t = 2 * np.pi * np.arange(N) / N
X, T, Hx = v.circle(t, RAD)
P = v.proj_from_tangent(T)
m = np.full(N, np.pi / N)
x = X[0]
D = X - x
rr = np.linalg.norm(D, axis=1)
nb = (rr > 0) & (rr < EPS)
check("Figure 47.1.1: six neighbours of x in B_0.2(x)", nb.sum() == 6)
u = D[nb] / rr[nb, None]
cj = (m[nb] * (-pair.drho(rr[nb] / EPS)))[:, None] * np.einsum("kab,kb->ka", P[nb], u)
S = cj.sum(0)
close("Figure 47.1.1: S = (-0.0383, 0) (unnormalised exp profile)", S, [-0.0383, 0.0], 5e-5)
# the same sum with the normalised kernel is - eps^(n+1) (delta V * rho_eps)(x)
dV = sum(m[j] * P[j] @ (c_exp * v.drho_exp(np.array([rr[j] / EPS]))[0] * EPS ** -3 * D[j] / rr[j])
         for j in np.where(nb)[0])
close("Figure 47.1.1: c_exp * S = -eps^3 (delta V * rho_eps)(x)", c_exp * S, -EPS ** 3 * dV, 1e-12)
H, Hp = v.amc_single(x, P[0], X, P, m, EPS, pair, 1)
close("Figure 47.1.1: approximate mean curvature (-1.961, 0)", H, [-1.961, 0.0], 5e-4)
check("Figure 47.1.1: within 2% of the exact (-2, 0)", abs(np.linalg.norm(H) - 2) / 2 < 0.02)
# H is invariant under rescaling rho or xi
pr = v.Pair("scaled", lambda s: 3 * v.rho_exp(s), lambda s: 3 * v.drho_exp(s), lambda s: 0.7 * pair.xi(s), 2)
close("(47.1.1) does not depend on the normalisation of rho and xi", v.amc_single(x, P[0], X, P, m, EPS, pr, 1)[0], H,
      1e-12)

# ---------------------------------------------------------------- Exercise 47.1.3: lone point
Hexp, _ = v.amc_single(x, P[0], X[:1], P[:1], m[:1], EPS, K["exp"], 1)
check("Exercise 47.1.3: exp pair with only x in the ball gives H = 0", np.all(Hexp == 0))
den = float(np.sum(m[:1] * pair.xi(np.array([0.0]))))
check("Exercise 47.1.3: natural exp denominator vanishes with only x in the ball", den == 0.0)

summary()
