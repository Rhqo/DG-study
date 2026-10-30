"""Verification for Section 47.5 (Natural Kernel Pairs).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-5-natural-kernel-pairs.py
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")  # keep BLAS from using every core

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import var47 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

# ---------------------------------------------------------------- Proposition 47.5.2
for n in (2, 3):
    for d in range(1, n):
        for name, prof in (("exp", v.rho_exp), ("tent", v.rho_tent)):
            p = v.pairs(n)["natural " + name]
            close(f"Prop 47.5.2: natural {name} partner has the same normaliser (n = {n})", v.normaliser(p.xi, n),
                  v.normaliser(prof, n), 1e-9)
            close(f"Prop 47.5.2: C_xi / C_rho = d/n = {d}/{n} (natural {name})", p.ratio(d), d / n, 1e-9)

# ---------------------------------------------------------------- (5.11): integral zero for all pairs, integrand zero for NKP
s = (np.arange(200000) + 0.5) / 200000
for d in (1, 2):
    for name, prof, dprof in (("exp", v.rho_exp, v.drho_exp), ("tent", v.rho_tent, v.drho_tent)):
        Crr = 1.0                                     # (rho, rho)
        f = (s * dprof(s) + d * Crr * prof(s)) * s ** (d - 1)
        close(f"(5.11): int (s rho' + d C xi) s^(d-1) = 0 for ({name}, {name}), d = {d}", f.mean(), 0.0, 1e-6)
        p = v.pairs(3)["natural " + name]
        Cnat = 3 / d
        g = (s * dprof(s) + d * Cnat * p.xi(s)) * s ** (d - 1)
        check(f"(5.12): integrand vanishes identically for the natural {name} pair (n = 3, d = {d})", np.abs(g).max() < 1e-12)
        close(f"(5.12): C_(rho,xi) = C_rho / C_xi = n/d for the natural {name} pair (n = 3, d = {d})",
              1 / p.ratio(d), 3 / d, 1e-9)
fexp = v.normaliser(v.rho_exp, 2) * (s * v.drho_exp(s) + v.rho_exp(s))
close("Figure 47.5.1(a): (rho, rho) exp integrand starts at rho(0) = 0.789 and dips to about -1.18 near s = 0.8",
      [fexp[0], fexp.min(), s[np.argmin(fexp)]], [0.789, -1.18, 0.8], 0.02)

# ---------------------------------------------------------------- (5.3): the normal part of x^M is <D^2u y, y> (no 1/2)
for k in (0.7, 2.0):
    y = 1e-3
    x = np.array([y, 0.5 * k * y ** 2])                       # graph of u(y) = k y^2 / 2
    t = np.array([1.0, k * y]) / math.hypot(1.0, k * y)       # unit tangent at x
    xM = (x @ t) * t
    close(f"(5.3): normal part of x^M is k y^2 = <D^2u(0) y, y> (k = {k}), not u(y) = k y^2/2", xM[1] / (k * y ** 2),
          1.0, 1e-5)

# ---------------------------------------------------------------- Proposition 47.5.4 (ours): circle identity
rng = np.random.default_rng(55)
R, EPS = 0.5, 0.05
worst = 0.0
all_ok = True
for trial in range(200):
    K = int(rng.integers(1, 40))
    phimax = 2 * np.arcsin(EPS / (2 * R))
    phi = np.concatenate([[0.0], rng.uniform(-phimax, phimax, K)])
    m = rng.uniform(0.1, 2.0, K + 1)
    X = R * np.stack([np.cos(phi), np.sin(phi)], 1)
    P = v.proj_from_tangent(np.stack([-np.sin(phi), np.cos(phi)], 1))
    for nm in ("natural exp", "natural tent"):
        pr = v.pairs(2)[nm]
        H, Hp = v.amc_single(X[0], P[0], X, P, m, EPS, pr, 1)
        t = 2 * R * np.abs(np.sin(phi / 2)) / EPS
        w = m * pr.xi(t) * (t < 1)
        pred = (1 / R) * np.sum(w * np.cos(phi / 2) ** 2) / np.sum(w)
        worst = max(worst, abs(-Hp[0] - pred))
        all_ok &= bool((-Hp[0] <= 1 / R + 1e-12) and (1 / R - (-Hp[0]) <= (1 / R) * (EPS / (2 * R)) ** 2 + 1e-12))
check("Prop 47.5.4: H^perp . (-nu) = kappa sum m xi cos^2(phi/2) / sum m xi for 200 random neighbourhoods (masses, K)",
      worst < 1e-12, f"worst {worst:.2e}")
check("Prop 47.5.4: 0 <= kappa - estimate <= kappa (eps kappa / 2)^2 in every trial", all_ok)
close("Prop 47.5.4: (eps kappa / 2)^2 = 0.0025 for eps = 0.05, kappa = 2", (EPS * 2 / 2) ** 2, 0.0025, 1e-15)

# ---------------------------------------------------------------- Exercise 47.5.3: (rho, rho) with two neighbours
tt = np.linspace(0.001, 0.999, 9999)
ratio2 = -2 * tt * v.drho_exp(tt) / (v.rho_exp(np.zeros(1))[0] + 2 * v.rho_exp(tt))
close("Exercise 47.5.3: -2 t rho'(t) / (rho(0) + 2 rho(t)) ranges from 0 to 2.56 (at t = 0.82)",
      [ratio2[0], ratio2.max(), tt[np.argmax(ratio2)]], [0.0, 2.56, 0.82], 0.01)

# ---------------------------------------------------------------- Remark 47.5.5: cylinder with one-directional neighbours
pair3 = v.pairs(3)["natural exp"]
a = np.linspace(-0.045, 0.045, 19)
ph = a / R
Xc = np.stack([R * np.cos(ph), R * np.sin(ph), 0 * ph], 1)
Pc = v.proj_from_normal(np.stack([np.cos(ph), np.sin(ph), 0 * ph], 1))
Hc, _ = v.amc_single(Xc[9], Pc[9], Xc, Pc, np.ones(19), EPS, pair3, 2)
close("Exercise 47.5.4: cylinder, neighbours only around the circle: estimate about 2 * 2 = 4 (true 2)", -Hc[0], 4.0, 0.01)
Xa = np.stack([R + 0 * a, 0 * a, a], 1)
Pa = v.proj_from_normal(np.stack([1 + 0 * a, 0 * a, 0 * a], 1))
Ha, _ = v.amc_single(Xa[9], Pa[9], Xa, Pa, np.ones(19), EPS, pair3, 2)
close("Exercise 47.5.4: cylinder, neighbours only along the axis: estimate 0", np.linalg.norm(Ha), 0.0, 1e-12)

# ---------------------------------------------------------------- Figure 47.5.1(b): Monte Carlo values
spec = importlib.util.spec_from_file_location("fig451", HERE.parents[1] / "figures" / "fig-47-5-1-natural-kernel-pairs.py")
fig451 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fig451)
err = fig451.rms_errors()
close("Figure 47.5.1(b): circle and sphere, natural exp: relative RMS error 0.0012 and 0.0013 for every K",
      [err[("circle", "natural exp")].max(), err[("sphere", "natural exp")].max()], [0.0012, 0.0013], 1e-4)
close("Figure 47.5.1(b): circle, exp (rho, rho): 0.59 (K = 4) -> 0.083 (K = 256)",
      [err[("circle", "exp")][0], err[("circle", "exp")][-1]], [0.59, 0.083], 0.01)
close("Figure 47.5.1(b): cylinder, natural exp: 0.42 (K = 4) -> 0.054 (K = 256)",
      [err[("cylinder", "natural exp")][0], err[("cylinder", "natural exp")][-1]], [0.42, 0.054], 0.01)
sl = np.polyfit(np.log(fig451.KS), np.log(err[("cylinder", "natural exp")]), 1)[0]
close("Figure 47.5.1(b): cylinder natural exp decays like K^(-1/2)", sl, -0.5, 0.05)

summary()
