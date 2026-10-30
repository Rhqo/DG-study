"""Verification for Section 47.4 (Approximate Mean Curvature and Its Convergence).

Run from the project root: PYTHONPATH=tools python3 chapters/47-paper-varifold-approximation/verify/v-47-4-approximate-mean-curvature.py
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
import var47 as v  # noqa: E402
import dgcode  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "47-4-approximate-mean-curvature.html"
K = v.pairs(2)
c_exp = v.normaliser(v.rho_exp, 2)

# ---------------------------------------------------------------- Example 47.4.2: printed (4.3) versus Def 4.1
N, R, EPS = 20000, 0.5, 0.01
t = 2 * np.pi * np.arange(N) / N
X, T, Hx = v.circle(t, R)
P = v.proj_from_tangent(T)
m = np.full(N, math.pi / N)
pk = K["natural exp"]
H, Hp = v.amc_single(X[0], P[0], X, P, m, EPS, pk, 1)
close("Example 47.4.2: Def 4.1 on the circle r = 0.5 (N = 20000, eps = 0.01, natural exp) gives (-1.9999, 0)", H,
      [-1.9999, 0.0], 1e-4)
printed = -H * (1 / pk.ratio(1))           # printed (4.3) = -delta V*rho / ||V||*xi = -(C_rho/C_xi) H
D = X - X[0]
r = np.linalg.norm(D, axis=1)
sel = (r > 0) & (r < EPS)
num = np.einsum("k,kab,kb->a", m[sel] * pk.drho(r[sel] / EPS), P[sel], D[sel] / r[sel, None])
den = np.sum(m[r < EPS] * pk.xi(r[r < EPS] / EPS))
printed_direct = num / (EPS * den)
close("Example 47.4.2: printed (4.3) evaluated literally gives (+4.00, 0)", printed_direct, [4.0, 0.0], 1e-3)
close("Example 47.4.2: printed (4.3) = -(C_rho / C_xi) H = -2 H for the natural pair (d = 1, n = 2)", printed_direct,
      printed, 1e-10)
close("Example 47.4.2: C_xi / C_rho = d/n = 1/2 for the natural exp pair", pk.ratio(1), 0.5, 1e-9)

# ---------------------------------------------------------------- Example 47.4.4 (Theorem 4.3 on a smooth ellipse)
Ne = 400000
te = 2 * np.pi * np.arange(Ne) / Ne
Xe, Te, He = v.ellipse(te, 2.0, 1.0)
Pe = v.proj_from_tangent(Te)
me = np.sqrt(4 * np.sin(te) ** 2 + np.cos(te) ** 2) * 2 * np.pi / Ne
vals = {}
for eps in (0.05, 0.1, 0.3):
    for nm in ("exp", "natural exp"):
        vals[(eps, nm)] = np.linalg.norm(v.amc_single(Xe[0], Pe[0], Xe, Pe, me, eps, K[nm], 1)[0])
close("Example 47.4.4: ellipse vertex, exp pair: 1.995, 1.980, 1.840 (eps = 0.05, 0.1, 0.3)",
      [vals[(e, "exp")] for e in (0.05, 0.1, 0.3)], [1.995, 1.980, 1.840], 6e-4)
close("Example 47.4.4: ellipse vertex, natural exp pair: 1.994, 1.977, 1.819",
      [vals[(e, "natural exp")] for e in (0.05, 0.1, 0.3)], [1.994, 1.977, 1.819], 6e-4)
check("Example 47.4.4: the bias decreases roughly like eps^2 (ratio of errors 0.3 -> 0.1 about 8)",
      6 < (2 - vals[(0.3, "exp")]) / (2 - vals[(0.1, "exp")]) < 10)

# ---------------------------------------------------------------- Example 47.4.5: the corner of a square
C_RHO = c_exp * v.C_const(v.rho_exp, 1)
const = math.sqrt(2) * c_exp * math.exp(-1) / C_RHO
close("Example 47.4.5: |H^V| eps at the corner = sqrt2 rho(0) / C_rho = 1.172", const, 1.172, 5e-4)
M = 200000
s = (np.arange(M) + 0.5) / M
Xs = np.vstack([np.stack([s, 0 * s], 1), np.stack([0 * s, s], 1)])
Ps = np.concatenate([np.repeat(np.array([[[1.0, 0], [0, 0]]]), M, 0), np.repeat(np.array([[[0.0, 0], [0, 1]]]), M, 0)])
ms = np.full(2 * M, 1.0 / M)
for eps in (0.1, 0.05, 0.02):
    for nm in ("exp", "natural exp", "natural tent"):
        Hc = v.amc_single(np.zeros(2), np.eye(2) * 0.5, Xs, Ps, ms, eps, K[nm], 1)[0]
        cst = math.sqrt(2) * v.rho_exp(np.array([0.0]))[0] / v.C_const(v.rho_exp, 1) if "exp" in nm else \
            math.sqrt(2) * 1.0 / v.C_const(v.rho_tent, 1)
        close(f"Example 47.4.5: corner, {nm}, eps = {eps}: H = (sqrt2 rho(0)/(C_rho eps)) (1, 1)/sqrt2", Hc,
              [cst / eps / math.sqrt(2)] * 2, 1e-4 * cst / eps)
close("Example 47.4.5: exp value 8.286 at eps = 0.1 (each component)", const / 0.1 / math.sqrt(2), 8.286, 5e-4)

# ---------------------------------------------------------------- Figure 47.4.1: tangential artefact
Nn, en = 2000, 0.1
u = (np.arange(Nn) + 0.5) / Nn
tn = 2 * np.pi * u + 0.5 * np.sin(2 * np.pi * u)
Xn, Tn, _ = v.circle(tn, 1.0)
Pn = v.proj_from_tangent(Tn)
Hn, Hpn, _ = v.amc(Xn, Pn, np.full(Nn, 2 * np.pi / Nn), en, K["natural exp"], 1)
tan = np.einsum("ka,ka->k", Hn, Tn)
nor = -np.einsum("ka,ka->k", Hn, Xn)
pred = 0.5 * np.sin(2 * np.pi * u) / (1 + 0.5 * np.cos(2 * np.pi * u)) ** 2
close("Figure 47.4.1: max |theta'/theta| = 0.847 at t = 2.73", [np.abs(pred).max(), tn[np.argmax(np.abs(pred))]],
      [0.847, 2.73], 3e-3)
check("Figure 47.4.1: tangential part of H^V matches theta'/theta within 0.013", np.abs(tan - pred).max() < 0.013)
close("Figure 47.4.1: normal part of H^V is 0.9988 (exact 1)", [nor.min(), nor.max()], [0.9988, 0.9988], 1e-4)
check("Figure 47.4.1: H^{V,perp} has no tangential part", np.abs(np.einsum("ka,ka->k", Hpn, Tn)).max() < 1e-12)


# ---------------------------------------------------------------- Figure 47.4.2: rates
def shape(tt):
    return v.ellipse(tt, 2.0, 1.0)


L = v.curve_length(shape)
close("ellipse (2 cos t, sin t) has length 9.6884", L, 9.6884, 5e-5)
res = {}
for Nr in (1000, 4000, 16000):
    rng = np.random.default_rng(12)
    tr = 2 * np.pi * (np.arange(Nr) + rng.standard_normal(Nr)) / Nr
    Xr, Tr, Hr = shape(tr)
    Pr = v.proj_from_tangent(Tr)
    mr = np.sqrt(4 * np.sin(tr) ** 2 + np.cos(tr) ** 2) * 2 * np.pi / Nr
    for f in (4, 16, 64):
        eps = f * L / Nr
        if eps > 0.4:
            continue
        a, b, _ = v.amc(Xr, Pr, mr, eps, K["exp"], 1)
        _, bn, _ = v.amc(Xr, Pr, mr, eps, K["natural exp"], 1)
        e = lambda A: float(np.linalg.norm(A - Hr, axis=1).mean())
        res[(Nr, f)] = (e(a), e(b), e(bn))
close("Figure 47.4.2(a): H^V error at eps/h = 16 is 0.568, 2.00, 8.07 for N = 1000, 4000, 16000",
      [res[(n, 16)][0] for n in (1000, 4000, 16000)], [0.568, 2.00, 8.07], 0.01)
close("Figure 47.4.2(b): H^perp error (exp) at eps/h = 16 is 0.046, 0.043, 0.043",
      [res[(n, 16)][1] for n in (1000, 4000, 16000)], [0.046, 0.043, 0.043], 1e-3)
close("Figure 47.4.2(b): H^perp error (natural exp) at eps/h = 16 is 8.5e-3, 9.5e-4, 1.8e-4",
      [res[(n, 16)][2] for n in (1000, 4000, 16000)], [8.5e-3, 9.5e-4, 1.8e-4], 1e-4)
sl_v = math.log(res[(16000, 64)][0] / res[(16000, 4)][0]) / math.log(16)
sl_p = math.log(res[(16000, 64)][1] / res[(16000, 4)][1]) / math.log(16)
close("Figure 47.4.2: slopes about -2.2 (H^V) and -1.3 (H^perp, exp) for N = 16000, eps/h 4..64", [sl_v, sl_p],
      [-2.2, -1.3], 0.1)

# ---------------------------------------------------------------- Implementation 47.4: the printed code
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-47-4-1")
check("Implementation 47.4: code-47-4-1 has at most 30 lines", dgcode.code_lines(PAGE, "code-47-4-1") <= 30)
check("Implementation 47.4: code-47-4-1 prints exactly out-47-4-1",
      buf.getvalue().strip() == dgcode.code_block(PAGE, "out-47-4-1").strip(), buf.getvalue())
Hc, _ = ns["amc"](X[::50], P[::50], m[::50] * 50, 0.05, True)
Hl, _, _ = v.amc(X[::50], P[::50], m[::50] * 50, 0.05, pk, 1)
close("Implementation 47.4: the printed amc agrees with lib/var47.amc", Hc, Hl, 1e-6)

# ---------------------------------------------------------------- Exercise 47.4.2: thresholds
h = 1e-4
close("Exercise 47.4.2: eps >> sqrt(h) = 1e-2 versus eps >> h = 1e-4", [math.sqrt(h), h], [1e-2, 1e-4], 1e-15)

summary()
