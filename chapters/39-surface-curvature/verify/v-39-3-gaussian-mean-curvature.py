"""Verification for Section 39.3 (Gaussian and Mean Curvature).

Run from the project root: PYTHONPATH=tools python3 chapters/39-surface-curvature/verify/v-39-3-gaussian-mean-curvature.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import ddg39  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "39-3-gaussian-mean-curvature.html"
ex = dgsym.EXAMPLES
u, v = sp.symbols("u v", real=True)

# ---------------------------------------------------------------- Implementation 39.3 and Exercise 39.3.6
check("Implementation 39.3: at most 30 lines", dgcode.code_lines(PAGE, "code-39-3-1") <= 30)
ns, out = ddg39.run_printed(PAGE, "code-39-3-1")
check("Implementation 39.3: printed output matches the page", out.strip() == ddg39.printed_block(PAGE, "out-39-3-1").strip(),
      out)
ci = ns["curvature_integrals"]
A, IK, IH = ci(ns["sphere"], 0, np.pi, 0, 2 * np.pi)
close("Implementation 39.3: unit sphere (4 pi, 4 pi, -4 pi)", [A, IK, IH], [4 * np.pi, 4 * np.pi, -4 * np.pi], 1e-4)
A, IK, IH = ci(ns["torus"], 0, 2 * np.pi, 0, 2 * np.pi)
close("Implementation 39.3: torus area 4 pi^2 R r, int K = 0, int H = 2 pi^2 R (inward)", [A, IK, IH],
      [4 * np.pi ** 2 * 1.6, 0.0, 2 * np.pi ** 2 * 2], 1e-4)
A, IK, IH = ci(ns["ellipsoid"], 0, np.pi, 0, 2 * np.pi, n=800)
close("Implementation 39.3: ellipsoid int K = 4 pi (n = 800)", IK, 4 * np.pi, 1e-4)
check("Theorem 39.3.7: Minkowski on the ellipsoid (2, 1.5, 1)", -IH > np.sqrt(4 * np.pi * A) + 0.5, f"{-IH} {np.sqrt(4*np.pi*A)}")
check("Exercise 39.3.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-39-3-6") <= 30)
ns2, out2 = ddg39.run_printed(PAGE, "code-ex-39-3-6", dict(ns))
check("Exercise 39.3.6: printed output matches the page",
      out2.strip() == ddg39.printed_block(PAGE, "out-ex-39-3-6").strip(), out2)
w = sp.symbols("w", real=True)
close("Exercise 39.3.6: closed forms -2 sqrt2 pi, -4 pi tanh 1",
      [float(-2 * sp.pi * sp.integrate((1 + w ** 2) ** sp.Rational(-3, 2), (w, -1, 1))),
       float(-2 * sp.pi * sp.integrate(sp.sech(w) ** 2, (w, -1, 1)))],
      [-2 * np.sqrt(2) * np.pi, -4 * np.pi * np.tanh(1.0)], 1e-12)

# ---------------------------------------------------------------- (39.3.1), Example 39.3.2 (symbolic)
for key in ("sphere", "cylinder", "torus"):
    e = ex[key]
    K, H = dgsym.K_H(e["expr"], *e["coords"], e["positive"])
    Wm = dgsym.shape_operator(e["expr"], *e["coords"], e["positive"])
    sym_equal(f"(39.3.1): K = det W, H = tr W / 2 ({key})", [K, H], [Wm.det(), Wm.trace() / 2])
    sym_equal(f"Example 39.3.2: K of the {key}", K, e["expected"]["K"])
sym_equal("Example 39.3.2(a): sphere H = -1/r", dgsym.K_H(ex["sphere"]["expr"], *ex["sphere"]["coords"], ex["sphere"]["positive"])[1],
          ex["sphere"]["expected"]["H"])
sym_equal("Example 39.3.2(b): cylinder H = -1/(2r)", dgsym.K_H(ex["cylinder"]["expr"], *ex["cylinder"]["coords"])[1],
          ex["cylinder"]["expected"]["H"])
to = ex["torus"]
Rr, rr = to["params"]
ut = to["coords"][0]
Kt, Ht = dgsym.K_H(to["expr"], *to["coords"], to["positive"])
sym_equal("Example 39.3.2(c): torus H = (R + 2 r cos u)/(2 r (R + r cos u)) (inward)", Ht,
          (Rr + 2 * rr * sp.cos(ut)) / (2 * rr * (Rr + rr * sp.cos(ut))))
Hn = sp.lambdify(ut, Ht.subs({Rr: 2, rr: sp.Rational(4, 5)}), modules="numpy")
Kn = sp.lambdify(ut, Kt.subs({Rr: 2, rr: sp.Rational(4, 5)}), modules="numpy")
close("Example 39.3.2(c): H(0) = 0.804, H(pi) = 0.208; K range [-1.042, 0.446]",
      [Hn(0.0), Hn(np.pi), Kn(np.pi), Kn(0.0)], [0.8035714, 0.2083333, -1.0416667, 0.4464286], 1e-6)
check("Example 39.3.2(c): H > 0 on the torus (2, 0.8)", min(Hn(x) for x in np.linspace(0, 2 * np.pi, 721)) > 0)
sad = ex["saddle"]
Ks, Hs = dgsym.K_H(sad["expr"], *sad["coords"])
sym_equal("Example 39.3.2(d): saddle origin K = -4, H = 0", [Ks.subs(sad["point"]), Hs.subs(sad["point"])], [-4, 0])
q = {u: sp.Rational(1, 2), v: -sp.Rational(2, 5)}
close("Example 39.3.2(d): saddle q K = -0.5739, H = -0.0839", [float(Ks.subs(q)), float(Hs.subs(q))], [-0.573921, -0.083926], 1e-6)
# kappa_{1,2} = H +- sqrt(H^2 - K) and H^2 - K = ((k1 - k2)/2)^2
k1s, k2s = sp.symbols("k1 k2", real=True)
sym_equal("(39.3.1): H^2 - K = ((k1 - k2)/2)^2", ((k1s + k2s) / 2) ** 2 - k1s * k2s, ((k1s - k2s) / 2) ** 2)

# ---------------------------------------------------------------- Proposition 39.3.1 (normal flip) numerically
for uu in (0.3, 2.0, 3.0):
    Kin, Hin = ddg39.KH(ddg39.torus, uu, 0.5)
    rev = lambda a, b: ddg39.torus(a, -b)  # noqa: E731
    Kout, Hout = ddg39.KH(rev, uu, -0.5)
    Nin, Nout = ddg39.unit_normal(ddg39.torus, uu, 0.5), ddg39.unit_normal(rev, uu, -0.5)
    check(f"Prop 39.3.1: torus u = {uu}: K same, H flips, H N same", abs(Kin - Kout) < 1e-6 and abs(Hin + Hout) < 1e-6
          and np.allclose(Hin * Nin, Hout * Nout, atol=1e-6) and np.allclose(Nin, -Nout, atol=1e-8))
# Exercise 39.3.1 (cylinder both normals)
k1, k2, *_ = ddg39.principal(ddg39.cylinder, 0.4, 0.2)
k1b, k2b, *_ = ddg39.principal(lambda a, b: ddg39.cylinder(-a, b), -0.4, 0.2)
close("Exercise 39.3.1: cylinder (r = 1) outward (0, -1), inward (1, 0)", [k1, k2, k1b, k2b], [0, -1, 1, 0], 1e-6)

# ---------------------------------------------------------------- Example 39.3.3, Non-example 39.3.4
cat = sp.Matrix([sp.cosh(v) * sp.cos(u), sp.cosh(v) * sp.sin(u), v])
hel = sp.Matrix([v * sp.cos(u), v * sp.sin(u), u])
Kc, Hc = dgsym.K_H(cat, u, v)
Kh, Hh = dgsym.K_H(hel, u, v)
sym_equal("Example 39.3.3: catenoid H = 0, K = -1/cosh^4 v", [Hc, Kc], [0, -1 / sp.cosh(v) ** 4])
sym_equal("Example 39.3.3: helicoid H = 0, K = -1/(1 + v^2)^2", [Hh, Kh], [0, -1 / (1 + v ** 2) ** 2])
fu, fv = 2 * u, -2 * v
num = (1 + fv ** 2) * 2 - 2 * fu * fv * 0 + (1 + fu ** 2) * (-2)
sym_equal("Non-example 39.3.4: graph H formula, numerator 8(v^2 - u^2)", [Hs, num], [num / (2 * (1 + fu ** 2 + fv ** 2) ** sp.Rational(3, 2)), 8 * (v ** 2 - u ** 2)])
# Exercise 39.3.2(b),(c): H = 0 implies K <= 0
check("Exercise 39.3.2(b): H = 0 implies K = -k1^2", sp.simplify((k1s * k2s).subs(k2s, -k1s) + k1s ** 2) == 0)

# ---------------------------------------------------------------- Theorem 39.3.5, Example 39.3.6, Exercise 39.3.3
e_, r_, k_ = sp.symbols("epsilon r k", positive=True)
A_sph = 2 * sp.pi * r_ ** 2 * (1 - sp.cos(e_ / r_))
ser = sp.series(A_sph, e_, 0, 6).removeO()
sym_equal("Example 39.3.6(a): sphere A = pi e^2 - (pi/12)(1/r^2) e^4 + O(e^6)", ser, sp.pi * e_ ** 2 - sp.pi / 12 / r_ ** 2 * e_ ** 4)
rho = sp.symbols("rho", positive=True)
for Kc_, f in [(k_ ** 2, sp.sin(k_ * rho) / k_), (0, rho), (-k_ ** 2, sp.sinh(k_ * rho) / k_)]:
    check(f"Exercise 39.3.3: f'' + K f = 0, f(0) = 0, f'(0) = 1 (K = {Kc_})",
          sp.simplify(f.diff(rho, 2) + Kc_ * f) == 0 and f.subs(rho, 0) == 0 and sp.limit(f.diff(rho), rho, 0) == 1)
    Aser = sp.series(2 * sp.pi * sp.integrate(f, (rho, 0, e_)), e_, 0, 5).removeO()
    Lser = sp.series(2 * sp.pi * f.subs(rho, e_), e_, 0, 4).removeO()
    sym_equal(f"Theorem 39.3.5 / Exercise 39.3.3: A and L expansions (K = {Kc_})", [Aser, Lser],
              [sp.pi * e_ ** 2 - sp.pi / 12 * Kc_ * e_ ** 4, 2 * sp.pi * e_ - sp.pi / 3 * Kc_ * e_ ** 3])
    sym_equal(f"(39.3.3): sqrt G = rho - K rho^3/6 + O(rho^4) (K = {Kc_})", sp.series(f, rho, 0, 4).removeO(), rho - Kc_ * rho ** 3 / 6)
sym_equal("Example 39.3.6(b): hyperbolic A = 2 pi (cosh e - 1)", 2 * sp.pi * sp.integrate(sp.sinh(rho), (rho, 0, e_)), 2 * sp.pi * (sp.cosh(e_) - 1))
sym_equal("Example 39.3.6(a): sqrt G = r sin(rho/r) on the sphere", sp.sqrt(ex["sphere"]["expected"]["G"]).subs(ex["sphere"]["coords"][0], rho / r_),
          r_ * sp.sin(rho / r_), {rho: (0.1, 2.5), r_: (1.0, 2.0)})
close("Figure 39.3.2: eps = 1 cap area 2.888, ratio 0.919 vs 11/12; eps = 2 ratio 0.708 vs 2/3",
      [2 * np.pi * (1 - np.cos(1)), 2 * (1 - np.cos(1)), 11 / 12, 2 * (1 - np.cos(2)) / 4, 1 - 4 / 12],
      [2.888366, 0.919395, 0.916667, 0.708073, 0.666667], 1e-6)
close("Figure 39.3.2: sin 1 = 0.841", np.sin(1.0), 0.841471, 1e-6)

# ---------------------------------------------------------------- Exercise 39.3.4 (torus N_u x N_v = K x_u x x_v)
Xt = to["expr"]
vt = to["coords"][1]
Nt = dgsym.unit_normal(Xt, ut, vt, to["positive"])
sym_equal("Exercise 39.3.4: N_u x N_v = K x_u x x_v (torus)", Nt.diff(ut).cross(Nt.diff(vt)), Kt * Xt.diff(ut).cross(Xt.diff(vt)))

# ---------------------------------------------------------------- Exercise 39.3.5 (torus total mean curvature)
dA = rr * (Rr + rr * sp.cos(ut))
sym_equal("Exercise 39.3.5: int H dA = 2 pi^2 R", sp.integrate(sp.integrate(sp.simplify(Ht * dA), (ut, 0, 2 * sp.pi)), (vt, 0, 2 * sp.pi)),
          2 * sp.pi ** 2 * Rr)
close("Exercise 39.3.5: same-area sphere 4 pi sqrt(pi R r) = 28.17 vs 39.48",
      [4 * np.pi * np.sqrt(np.pi * 1.6), 2 * np.pi ** 2 * 2], [28.1738, 39.4784], 1e-4)
# Theorem 39.3.7 on spheres: equality
close("Theorem 39.3.7: equality on a sphere of radius r = 1.7", 4 * np.pi * 1.7, np.sqrt(4 * np.pi * 4 * np.pi * 1.7 ** 2), 1e-12)

summary()
