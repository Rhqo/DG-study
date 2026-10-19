"""Verification for Section 39.4 (The Gauss-Bonnet Theorem).

Run from the project root: PYTHONPATH=tools python3 chapters/39-surface-curvature/verify/v-39-4-gauss-bonnet.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgmesh as dm  # noqa: E402
import dgsym  # noqa: E402
import ddg39  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "39-4-gauss-bonnet.html"
ex = dgsym.EXAMPLES

# ---------------------------------------------------------------- Implementation 39.4 and Exercise 39.4.5
check("Implementation 39.4: at most 30 lines", dgcode.code_lines(PAGE, "code-39-4-1") <= 30)
ns, out = ddg39.run_printed(PAGE, "code-39-4-1")
check("Implementation 39.4: printed output matches the page", out.strip() == ddg39.printed_block(PAGE, "out-39-4-1").strip(),
      out)
ct = ns["cap_terms"]
for t0 in np.linspace(0.2, np.pi - 0.2, 9):
    A, B = ct(ns["bumpy"], t0)
    close(f"Prop 39.4.6: bumpy cap t0 = {t0:.3f}: area + boundary = 2 pi", A + B, 2 * np.pi, 2e-5)
A, B = ct(ns["bumpy"], np.pi)
close("Example 39.4.7: whole bumpy sphere int K dA = 4 pi, boundary term -> -2 pi", [A, B], [4 * np.pi, -2 * np.pi], 2e-5)
# independent check of the area term with the 2D numeric curvature
T, V = np.meshgrid((np.arange(600) + 0.5) * np.pi / 600, (np.arange(200) + 0.5) * 2 * np.pi / 200, indexing="ij")
Kg, Hg, dA = ddg39.KH_grid(ddg39.bumpy_sphere, T, V)
close("Example 39.4.7: 2D quadrature of K dA on the bumpy sphere = 4 pi", float((Kg * dA).sum() * (np.pi / 600) * (2 * np.pi / 200)),
      4 * np.pi, 2e-3)
Kb = ddg39.revolution_K(ddg39.bumpy_profile, np.linspace(0.0, np.pi, 40001)[1:-1])
tb = sp.symbols("tb", real=True)
rb = 1 + sp.Rational(1, 5) * sp.cos(4 * tb)
Kb_expr = ((rb * sp.sin(tb)).diff(tb) * (rb * sp.cos(tb)).diff(tb, 2) - (rb * sp.sin(tb)).diff(tb, 2) * (rb * sp.cos(tb)).diff(tb)) \
    * (rb * sp.cos(tb)).diff(tb) / (rb * sp.sin(tb) * ((rb * sp.sin(tb)).diff(tb) ** 2 + (rb * sp.cos(tb)).diff(tb) ** 2) ** 2)
check("Example 39.4.7: K ranges over [-5.01, 9.34] (max 3025/324 at the poles) and changes sign",
      abs(Kb.min() + 5.0095) < 1e-3 and abs(Kb.max() - 3025 / 324) < 1e-3 and sp.limit(Kb_expr, tb, 0) == sp.Rational(3025, 324))
check("Exercise 39.4.5: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-39-4-5") <= 30)
ns2, out2 = ddg39.run_printed(PAGE, "code-ex-39-4-5", dict(ns))
check("Exercise 39.4.5: printed output matches the page",
      out2.strip() == ddg39.printed_block(PAGE, "out-ex-39-4-5").strip(), out2)

# ---------------------------------------------------------------- Example 39.4.1 (plane domains)
t = sp.symbols("t", real=True)
rr = 1 + sp.Rational(3, 10) * sp.cos(5 * t)
fl = sp.Matrix([rr * sp.cos(t), rr * sp.sin(t)])
d1, d2 = fl.diff(t), fl.diff(t, 2)
ks_ds = sp.lambdify(t, (d1[0] * d2[1] - d1[1] * d2[0]) / (d1[0] ** 2 + d1[1] ** 2), modules="numpy")   # kappa_s |gamma'| dt
tt = np.linspace(0, 2 * np.pi, 200001)
vals = ks_ds(tt)
total = float(np.sum(0.5 * (vals[1:] + vals[:-1]) * np.diff(tt)))
close("Example 39.4.1: flower curve total curvature = 2 pi", total, 2 * np.pi, 1e-8)
sgn = np.sign(vals)
check("Example 39.4.1: flower kappa_s changes sign ten times", int(np.sum(sgn[1:] != sgn[:-1])) == 10)

# ---------------------------------------------------------------- Example 39.4.3 (closed surfaces)
to = ex["torus"]
Rr, rr_ = to["params"]
ut, vt = to["coords"]
Kt, Ht = dgsym.K_H(to["expr"], *to["coords"], to["positive"])
E, F, G = dgsym.first_ff(to["expr"], ut, vt, to["positive"])
KdA = sp.simplify(Kt * sp.sqrt(E * G - F ** 2))
sym_equal("Example 39.4.3(c): torus K dA = cos u du dv", KdA, sp.cos(ut))
check("Example 39.4.3(c): torus int K dA = 0, outer half 4 pi, inner half -4 pi",
      sp.integrate(sp.cos(ut), (ut, 0, 2 * sp.pi)) * 2 * sp.pi == 0
      and sp.integrate(sp.cos(ut), (ut, -sp.pi / 2, sp.pi / 2)) * 2 * sp.pi == 4 * sp.pi
      and sp.integrate(sp.cos(ut), (ut, sp.pi / 2, 3 * sp.pi / 2)) * 2 * sp.pi == -4 * sp.pi)
sp_ = ex["sphere"]
check("Example 39.4.3(a): sphere K * area = 4 pi", sp.simplify(sp_["expected"]["K"] * 4 * sp.pi * sp_["params"][0] ** 2 - 4 * sp.pi) == 0)
Vi, Fi = dm.icosahedron()
check("Example 39.4.3(a): icosahedron chi = 12 - 30 + 20 = 2", dm.counts(Fi) == (12, 30, 20) and dm.euler_characteristic(Fi) == 2)

# ---------------------------------------------------------------- Example 39.4.5 (polar caps, hemisphere, octant)
th0 = sp.symbols("theta0", positive=True)
capA = sp.integrate(sp.integrate(sp.sin(sp.Symbol("x")), (sp.Symbol("x"), 0, th0)), (sp.Symbol("p"), 0, 2 * sp.pi))
sym_equal("(39.4.3): polar cap int K dA = 2 pi (1 - cos theta0)", capA, 2 * sp.pi * (1 - sp.cos(th0)))
sym_equal("(39.4.3): boundary int kappa_g ds = cot(theta0) * 2 pi sin(theta0)", sp.cot(th0) * 2 * sp.pi * sp.sin(th0), 2 * sp.pi * sp.cos(th0))
sym_equal("(39.4.3): sum = 2 pi", capA + 2 * sp.pi * sp.cos(th0), 2 * sp.pi)
for th in (np.pi / 3, 2 * np.pi / 3, 0.4):
    kap, kn, kg, *_ = ddg39.darboux(ddg39.sphere, lambda s: (th, s), 0.2)
    A, B = ddg39.cap_terms(lambda x: (np.sin(x), np.cos(x)), th)
    close(f"Figure 39.4.1: theta0 = {th:.3f}: kappa_g = cot, profile terms match", [kg, A, B],
          [1 / np.tan(th), 2 * np.pi * (1 - np.cos(th)), 2 * np.pi * np.cos(th)], 1e-5)
close("Figure 39.4.1: theta0 = 2 pi/3: kappa_g = -0.577, boundary -pi, area 3 pi",
      [1 / np.tan(2 * np.pi / 3), 2 * np.pi * np.cos(2 * np.pi / 3), 2 * np.pi * (1 - np.cos(2 * np.pi / 3))],
      [-0.57735, -np.pi, 3 * np.pi], 1e-5)
close("Example 39.4.5: octant triangle int K dA = 2 pi - 3 pi/2 = pi/2 = 4 pi/8", 2 * np.pi - 3 * np.pi / 2, 4 * np.pi / 8, 1e-15)

# ---------------------------------------------------------------- Proposition 39.4.6 (symbolic, arc-length profile)
s, v = sp.symbols("s v", real=True)
rho = sp.Function("rho")(s)
z = sp.Function("z")(s)
X = sp.Matrix([rho * sp.cos(v), rho * sp.sin(v), z])
Xs, Xv = X.diff(s), X.diff(v)
n_ = Xs.cross(Xv)
# with rho'^2 + z'^2 = 1: |x_s x x_v| = rho, N = (-z' cos v, -z' sin v, rho')
Nexp = sp.Matrix([-z.diff(s) * sp.cos(v), -z.diff(s) * sp.sin(v), rho.diff(s)])
sym_equal("Prop 39.4.6: x_s x x_v = rho (-z' cos v, -z' sin v, rho')", n_, rho * Nexp)
tvec = sp.Matrix([-sp.sin(v), sp.cos(v), 0])
sym_equal("Prop 39.4.6: N x t = -x_s", Nexp.cross(tvec), -Xs)
# K = -rho''/rho under rho'^2 + z'^2 = 1: a unit-speed profile with tangent angle s + 0.3 sin 2s (numeric)
ss = np.linspace(0, 2.0, 20001)
ang = ss + 0.3 * np.sin(2 * ss)
r1 = np.cos(ang)
z1 = -np.sin(ang)
rr_num = np.concatenate([[0], np.cumsum(0.5 * (r1[1:] + r1[:-1]) * np.diff(ss))])
r2 = np.gradient(r1, ss)
z2 = np.gradient(z1, ss)
Kformula = (r1 * z2 - r2 * z1) * z1 / rr_num.clip(1e-12)
sel = slice(2000, 18000)
check("Prop 39.4.6: K = (r'z'' - r''z') z'/r equals -r''/r for a unit-speed profile",
      np.max(np.abs(Kformula[sel] + r2[sel] / rr_num[sel])) < 1e-6)
A_ = 2 * np.pi * np.sum((-r2[:-1] - r2[1:]) / 2 * np.diff(ss))
close("Prop 39.4.6: int K dA = 2 pi (1 - rho'(s0)) for that profile (s0 = 2)", A_, 2 * np.pi * (1 - r1[-1]), 1e-5)

# ---------------------------------------------------------------- Non-example 39.4.8 (catenoid)
w = sp.symbols("w", real=True)
check("Non-example 39.4.8: catenoid total curvature -2 pi [tanh]_{-oo}^{oo} = -4 pi (tanh' = sech^2)",
      sp.simplify(sp.tanh(w).diff(w) - sp.sech(w) ** 2) == 0
      and -2 * sp.pi * (sp.limit(sp.tanh(w), w, sp.oo) - sp.limit(sp.tanh(w), w, -sp.oo)) == -4 * sp.pi)

# ---------------------------------------------------------------- Exercises 39.4.1-39.4.3
check("Exercise 39.4.1: chi = 2 - 2g - b gives -4 pi (g = 2) and -2 pi (sphere minus three disks)",
      2 * np.pi * (2 - 4) == -4 * np.pi and 2 * np.pi * (2 - 0 - 3) == -2 * np.pi)
al = sp.symbols("alpha", positive=True)
sym_equal("Exercise 39.4.2: lune triangle area = alpha", sp.integrate(sp.integrate(sp.sin(sp.Symbol("x")), (sp.Symbol("x"), 0, sp.pi / 2)), (sp.Symbol("p"), 0, al)), al)
sym_equal("Exercise 39.4.2: GB gives 2 pi - (pi/2 + pi/2 + pi - alpha) = alpha", 2 * sp.pi - (sp.pi / 2 + sp.pi / 2 + sp.pi - al), al)
# frustum: kappa_g on the two circles from the Darboux frame
h = 0.7
c = 1 / np.sqrt(1 + h * h)
fr = lambda rho_, vv: np.array([rho_ * np.cos(vv), rho_ * np.sin(vv), h * (2.0 - rho_) + 0 * vv])  # noqa: E731
_, _, kg2, *_ = ddg39.darboux(fr, lambda q: (2.0, q), 0.3)
_, _, kg1, *_ = ddg39.darboux(fr, lambda q: (1.0, -q), 0.3)
close("Exercise 39.4.3: frustum boundary terms 2 pi c and -2 pi c (sum 0)", [kg2 * 2 * np.pi * 2.0, kg1 * 2 * np.pi * 1.0],
      [2 * np.pi * c, -2 * np.pi * c], 1e-6)
check("Exercise 39.4.3: K = 0 on the frustum", abs(ddg39.KH(fr, 1.5, 0.3)[0]) < 1e-6)

# ---------------------------------------------------------------- proof sketch counting identity
for n0, n1, n2 in [(12, 30, 20), (128, 384, 256), (6, 12, 8)]:
    check(f"Proof sketch: 2 pi n0 - pi n2 = 2 pi chi when 3 n2 = 2 n1 ({n0}, {n1}, {n2})",
          3 * n2 == 2 * n1 and abs((2 * np.pi * n0 - np.pi * n2) - 2 * np.pi * (n0 - n1 + n2)) < 1e-12)
V, F = dm.grid_mesh(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
n0, n1, n2 = dm.counts(F)
nb = len(dm.boundary_edges(F))
check("Proof sketch: with boundary 2 pi n0 - pi n2 - pi n0_bdry = 2 pi chi (grid, n1_bdry = n0_bdry)",
      abs((2 * np.pi * n0 - np.pi * n2 - np.pi * nb) - 2 * np.pi * (n0 - n1 + n2)) < 1e-12 and 3 * n2 == 2 * n1 - nb)

summary()
