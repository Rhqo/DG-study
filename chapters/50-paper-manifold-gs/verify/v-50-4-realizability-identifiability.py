"""Verification for Section 50.4 (Realizability Is Not Identifiability).

Run from the project root: PYTHONPATH=tools python3 chapters/50-paper-manifold-gs/verify/v-50-4-realizability-identifiability.py
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import mgs50 as g  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "50-4-realizability-identifiability.html"
rng = np.random.default_rng(504)

# ---------------------------------------------------------------- Proposition 50.4.3 (symbolic, generic functions)
u, v = sp.symbols("u v", real=True)
X = sp.Matrix([sp.Function(f"x{k}")(u, v) for k in range(3)])
n = sp.Matrix([sp.Function(f"n{k}")(u, v) for k in range(3)])
Xu, Xv = X.diff(u), X.diff(v)
lhs = sp.diff(n.dot(Xv), u) - sp.diff(n.dot(Xu), v)
h_uv, h_vu = -n.diff(u).dot(Xv), -n.diff(v).dot(Xu)
check("Proposition 50.4.3: d<n, dx> = h^c_vu - h^c_uv for generic x and n", sp.simplify(lhs - (h_vu - h_uv)) == 0)
# Exercise 50.4.2: n = N gives h^c = h^s (sphere graph chart, symbolic)
z = sp.sqrt(1 - u ** 2 - v ** 2)
Xs = sp.Matrix([u, v, z])
Ns = Xs.diff(u).cross(Xs.diff(v))
Ns = Ns / sp.sqrt(Ns.dot(Ns))
hs = sp.Matrix(2, 2, lambda i, j: Xs.diff((u, v)[i], (u, v)[j]).dot(Ns))
hc = sp.Matrix(2, 2, lambda i, j: -Ns.diff((u, v)[i]).dot(Xs.diff((u, v)[j])))
check("Exercise 50.4.2: n = N gives h^c = h^s on the sphere chart", sp.simplify(hs - hc) == sp.zeros(2, 2))
fN = sp.lambdify((u, v), Ns - Xs, "numpy")
check("Example 50.4.6: on the sphere graph chart N = x",
      all(np.allclose(np.array(fN(*rng.uniform(-0.5, 0.5, 2)), float), 0, atol=1e-12) for _ in range(20)))

# ---------------------------------------------------------------- Example 50.4.6 / Table 50.4.1 (residuals)
uu, vv, cases = g.residual_cases()
R = {k: g.residuals(Xc, nc, uu, vv) for k, (Xc, nc) in cases.items()}
for k in ("A", "D"):
    check(f"Example 50.4.6 ({k}): all five residuals vanish", max(R[k].values()) < 1e-10)
for k in "ABCD":
    close(f"Example 50.4.6 ({k}): normal curl = antisymmetry", R[k]["curl"], R[k]["antisymmetry"], 1e-10)
expect = {"B": (0.386, 0.538, 0.060, 0.032), "C": (0.803, 0.0, 1.439, 0.334)}
for k, vals in expect.items():
    got = [round(R[k][key], 3) for key in ("disagreement", "antisymmetry", "gauss", "codazzi")]
    close(f"Table 50.4.1 ({k}): disagreement, antisymmetry, Gauss, Codazzi", got, vals, 1e-9)
# review: the RMS values are converged (24 and 40 Gauss-Legendre nodes agree to 4 digits)
R40 = {k: g.residuals(*cases[k], uu, vv, m=40) for k in ("B", "C")}
close("Table 50.4.1: residual quadrature converged (24 vs 40 nodes)",
      [R[k][key] for k in ("B", "C") for key in ("disagreement", "antisymmetry", "gauss", "codazzi")],
      [R40[k][key] for k in ("B", "C") for key in ("disagreement", "antisymmetry", "gauss", "codazzi")], 1e-4)
# C is the normal of the ellipsoid z = 2 sqrt(1 - u^2 - v^2) at the same (u, v)
f = 2 * z
nE = sp.Matrix([-f.diff(u), -f.diff(v), 1])
nC = sp.Matrix([2 * u, 2 * v, z])
check("Example 50.4.6 (C): N + (u, v, 0) is parallel to the ellipsoid normal", sp.simplify(nE.cross(nC)) == sp.zeros(3, 1))
Xc, nc = cases["C"]
Xc, nc = sp.Matrix(Xc), sp.Matrix(nc)
gm = sp.Matrix([[Xc.diff(uu).dot(Xc.diff(uu)), Xc.diff(uu).dot(Xc.diff(vv))],
                [Xc.diff(uu).dot(Xc.diff(vv)), Xc.diff(vv).dot(Xc.diff(vv))]])
hcC = sp.Matrix(2, 2, lambda i, j: -nc.diff((uu, vv)[i]).dot(Xc.diff((uu, vv)[j])))
ratio = sp.lambdify((uu, vv), hcC.det() / gm.det(), "numpy")
Kg = sp.lambdify((uu, vv), dgsym.sectional(gm, [uu, vv], [1, 0], [0, 1]), "numpy")
close("Example 50.4.6 (C): K(g) = 1, det h^c / det g = 4 at the top and 0.96 at the corner",
      [float(Kg(0.0, 0.0)), float(Kg(0.3, 0.2)), float(ratio(0.0, 0.0)), float(ratio(0.5, 0.5))], [1, 1, 4, 0.96], 1e-9)

# ---------------------------------------------------------------- Example 50.4.8 (the two terms)
at = g.decomposition_atoms()
XS, NS, WS = at["S"]
check("Example 50.4.8: 31 x 31 atoms, sphere weights 1 / z", len(WS) == 961 and np.allclose(WS, 1 / XS[:, 2]))
first = {k: g.mmd(*at[k], XS, NS, WS, 0.1, 0.5) for k in ("B", "C")}
second = g.mmd(*at["D"], XS, NS, WS, 0.1, 0.5)
close("Table 50.4.1: d_V(V, Pi_Z V) for B and C", [round(first["B"], 3), round(first["C"], 3)], [0.065, 0.140], 1e-9)
close("Table 50.4.1: d_V(Pi_Z V, V_*) for D", round(second, 3), 0.223, 1e-9)
check("Example 50.4.8: D is the farthest although compatible", second > first["C"] > first["B"])

# ---------------------------------------------------------------- Proposition 50.4.10 (random couplings)
worst = 0.0
for _ in range(300):
    m1, m2 = rng.integers(2, 7), rng.integers(2, 7)
    XA, NA = rng.normal(size=(m1, 3)), g.unit(rng.normal(size=(m1, 3)))
    XB, NB = rng.normal(size=(m2, 3)), g.unit(rng.normal(size=(m2, 3)))
    a, b = rng.dirichlet(np.ones(m1)), rng.dirichlet(np.ones(m2))
    pi = np.outer(a, b)                                            # the independent coupling
    for _ in range(20):                                            # random mass-preserving moves keep the marginals
        i, j, k, l = rng.integers(m1), rng.integers(m1), rng.integers(m2), rng.integers(m2)
        if i == j or k == l:
            continue
        t = rng.uniform(0, min(pi[i, l], pi[j, k]))
        pi[i, k] += t
        pi[j, l] += t
        pi[i, l] -= t
        pi[j, k] -= t
    check("Proposition 50.4.10: pi is a coupling", np.allclose(pi.sum(1), a) and np.allclose(pi.sum(0), b) and pi.min() >= -1e-15)
    sig, tau = rng.uniform(0.1, 2), rng.uniform(0.2, 2)
    worst = max(worst, g.mmd(XA, NA, a, XB, NB, b, sig, tau) / g.transport_bound(XA, NA, XB, NB, pi, sig, tau))
check(f"Proposition 50.4.10: MMD <= transport for 300 random couplings (max ratio {worst:.3f})", worst <= 1 + 1e-12)
check("Remark 50.4.11: MMD <= sqrt 2 (the kernel is at most 1)",
      g.mmd(np.zeros((1, 3)), np.array([[0, 0, 1.0]]), [1], 100 * np.ones((1, 3)), np.array([[1.0, 0, 0]]), [1], 0.1, 0.5)
      <= math.sqrt(2) + 1e-12)
for d, m_e, b_e in ((0.05, 0.4848, 0.5), (0.3, 1.4063, 3.0), (1.0, 1.4142, 10.0)):
    close(f"Exercise 50.4.3 (delta = {d}): MMD and bound", [math.sqrt(2 * (1 - math.exp(-d * d / (2 * 0.01)))), d / 0.1],
          [m_e, b_e], 5e-5)

# ---------------------------------------------------------------- Example 50.4.12 / Figure 50.4.2
for sig, sh_e, rs_e, h1_e, h2_e in ((0.05, 0.0122, 0.237, 0.0044, 0.0101), (0.2, 0.0115, 0.129, 0.0071, 0.0247)):
    sh = g.mmd_shift(0.01, sig)
    close(f"Example 50.4.12 (sigma = {sig}): shift by 0.01", round(sh, 4), sh_e, 1e-9)
    close(f"Example 50.4.12 (sigma = {sig}): holes r = 0.1, 0.2",
          [round(g.mmd_hole(0.1, sig), 4), round(g.mmd_hole(0.2, sig), 4)], [h1_e, h2_e], 1e-9)
    close(f"Example 50.4.12 (sigma = {sig}): crossing radius r*", round(g.crossing_radius(sig, sh), 3), rs_e, 1e-9)
close("Example 50.4.12: 0.01 sqrt(2 pi) / 2 = 0.0125", 0.01 * math.sqrt(2 * math.pi) / 2, 0.0125, 5e-5)
close("Example 50.4.12: area shares 0.8 % and 3.1 %", [100 * math.pi * 0.01 / 4, 100 * math.pi * 0.04 / 4], [0.8, 3.1], 0.05)
close("Figure 50.4.2: a small hole costs its area share pi r^2 / 4", g.mmd_hole(0.01, 0.2) / (math.pi * 1e-4 / 4), 1.0, 0.05)
# the shift distance against a direct quadrature of (50.4.8) (positions on a grid, planes equal)
t = np.linspace(-1, 1, 81)
P = np.stack(np.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
w = np.ones(len(P)) / len(P)
Xa = np.c_[P, np.zeros(len(P))]
Na = np.tile([0, 0, 1.0], (len(P), 1))
close("Example 50.4.12: the closed form of the shift agrees with a grid quadrature",
      g.mmd(Xa, Na, w, Xa + [0, 0, 0.01], Na, w, 0.2, 1.0), g.mmd_shift(0.01, 0.2), 2e-3)

# ---------------------------------------------------------------- Example 50.4.15 and Exercise 50.4.4
cases = {"textureless": ([0.5, 0, 0, 0], (-0.2, 0.2), 0.0, 0.0), "one view": ([0.5, 0.3, 0, -0.1], (0.0,), 0.444, 0.0),
         "baseline 0.4": ([0.5, 0.3, 0, -0.1], (-0.2, 0.2), 0.641, 0.333),
         "baseline 0.1": ([0.5, 0.3, 0, -0.1], (-0.05, 0.05), 0.629, 0.0856)}
sig = {}
for name, (coef, offs, j_e, s_e) in cases.items():
    jd, s = g.restricted_sigma(coef, offs)
    sig[name] = s
    close(f"Example 50.4.15 ({name}): |J_d| and sigma_min", [round(jd, 3), round(s, 4 if s < 0.1 else 3)], [j_e, s_e], 1e-9)
check("Example 50.4.15: one view gives sigma_min = 0 to rounding", sig["one view"] < 1e-12)
close("Example 50.4.15: with appearance fixed one view has |J_d|^2 = 0.197", round(g.restricted_sigma([0.5, 0.3, 0, -0.1], (0.0,))[0] ** 2, 3), 0.197, 1e-9)
close("Example 50.4.15: Fisher sigma^2 = 0.111 and 0.0073", [sig["baseline 0.4"] ** 2, sig["baseline 0.1"] ** 2],
      [0.111, 0.0073], 5e-4)
close("Example 50.4.15: Fisher ratio about 15", sig["baseline 0.4"] ** 2 / sig["baseline 0.1"] ** 2, 15.0, 0.2)
for coef, offs, positive in (([0.5, 0.3], (-0.2, 0.2), True), ([0.5, 0.0], (-0.2, 0.2), False),
                             ([0.5, 0.3], (0.1, 0.1), False), ([0.5, 0.3], (0.0,), False)):
    s = g.restricted_sigma(coef, offs)[1]
    check(f"Exercise 50.4.4 (c = {coef}, o = {offs}): sigma_min > 0 iff c1 != 0 and o1 != o2", (s > 1e-6) == positive)
# the least-squares meaning: min over dc |J_d - J_c dc| = sigma_min (one geometric parameter)
x = np.linspace(-0.4, 0.4, 101)
Jd, Jc = [], []
for o in (-0.2, 0.2):
    s_ = o + 2.0 * x
    Jd.append((0.3 - 0.3 * s_ ** 2) * x)
    Jc.append(np.stack([s_ ** k for k in range(4)], 1))
Jd, Jc = np.concatenate(Jd), np.vstack(Jc)
res = Jd - Jc @ np.linalg.lstsq(Jc, Jd, rcond=None)[0]
close("Definition 50.4.14: min_dc |J_d - J_c dc| = sigma_min", np.linalg.norm(res), sig["baseline 0.4"], 1e-10)

# ---------------------------------------------------------------- Exercise 50.4.1: constant tilted normal on a plane
beta = sp.Rational(1, 5)
Rp = g.residuals([uu, vv, 0], sp.Matrix([sp.sin(beta), 0, sp.cos(beta)]), uu, vv)
check("Exercise 50.4.1: all residuals vanish for a constant tilted normal", max(Rp.values()) < 1e-12)

# ---------------------------------------------------------------- Implementation 50.4.1
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ns = dgcode.run_block(PAGE, "code-50-4-1")
check("Implementation 50.4.1: at most 30 lines", dgcode.code_lines(PAGE, "code-50-4-1") <= 30)
check("Implementation 50.4.1: printed output equals the page",
      buf.getvalue().strip() == dgcode.code_block(PAGE, "out-50-4-1").strip())
check("Implementation 50.4.1: all 200 ratios below 1", max(ns["ratios"]) < 1)

summary()
