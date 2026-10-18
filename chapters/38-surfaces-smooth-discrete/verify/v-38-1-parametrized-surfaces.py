"""Verification for Section 38.1 (Parametrized Surfaces and the Differential).

Run from the project root: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/verify/v-38-1-parametrized-surfaces.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import ddg38  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "38-1-parametrized-surfaces.html"
u, v, s, t, rho, lam = sp.symbols("u v s t rho lambda", real=True)

# ---------------------------------------------------------------- Implementation 38.1 and Exercise 38.1.5
check("Implementation 38.1: at most 30 lines", dgcode.code_lines(PAGE, "code-38-1-1") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-38-1-1")
check("Implementation 38.1: printed output matches the page", out.strip() == ddg38.printed_block(PAGE, "out-38-1-1").strip(),
      out)
ad = ns["area_density"]
grid = np.linspace(0, 2 * np.pi, 361)
close("Implementation 38.1: torus minimum = r (R - r) = 0.96", ad(ns["torus"], grid, grid).min(), 0.8 * 1.2, 1e-8)
check("Exercise 38.1.5: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-38-1-5") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-ex-38-1-5", ns)
check("Exercise 38.1.5: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-ex-38-1-5").strip(), out)
for la in np.linspace(0, 1, 11):
    a = abs(1 - 2 * la)
    exact = a if a * a <= 3 / 7 else np.sqrt((9 * a * a + 3) / 16)
    close(f"Exercise 38.1.5: m({la:.1f}) closed form", ns["m"][int(round(la * 10))], exact, 1e-8)

# ---------------------------------------------------------------- Example 38.1.2, Non-example 38.1.3
ex = dgsym.EXAMPLES
xs = ex["saddle"]["expr"]
y = xs.subs({ex["saddle"]["coords"][0]: rho * sp.cos(t), ex["saddle"]["coords"][1]: rho * sp.sin(t)})
sym_equal("Example 38.1.2(d): saddle in polar coordinates z = rho^2 cos 2t", y[2], rho ** 2 * sp.cos(2 * t))
h = sp.Matrix([rho * sp.cos(t), rho * sp.sin(t)])
sym_equal("Example 38.1.2(d): det Dh = rho", sp.simplify(h.jacobian([rho, t]).det()), rho)
gam = sp.Matrix([sp.cos(u + v), sp.sin(u + v), sp.Rational(3, 10) * (u + v)])
check("Non-example 38.1.3: x_u = x_v for x(u, v) = gamma(u + v)", sp.simplify(gam.diff(u) - gam.diff(v)) == sp.zeros(3, 1))
check("Non-example 38.1.3: Jacobian has rank 1", gam.jacobian([u, v]).rank(simplify=True) == 1)

# ---------------------------------------------------------------- Proposition 38.1.4 (numeric) and Example 38.1.5
R, r = ddg38.R_TORUS, ddg38.r_TORUS
rng = np.random.default_rng(381)
for _ in range(4):
    q = rng.uniform(-3, 3, 2)
    X = rng.normal(size=2)
    xu, xv = ddg38.partials(ddg38.torus, np.array(q[0]), np.array(q[1]))
    J = np.stack([xu, xv], 1)
    hh = 1e-6
    dd = (ddg38.torus(q[0] + hh * X[0], q[1] + hh * X[1]) - ddg38.torus(q[0] - hh * X[0], q[1] - hh * X[1])) / (2 * hh)
    close("Prop 38.1.4(a)(b): Jacobian times X = directional derivative (torus)", J @ X, dd, 1e-6)
    # (c): the curve c(t) = q + t X + t^2 Y has c'(0) = X
    Y = rng.normal(size=2)
    c = lambda tt: ddg38.torus(q[0] + tt * X[0] + tt * tt * Y[0], q[1] + tt * X[1] + tt * tt * Y[1])  # noqa: E731
    close("Prop 38.1.4(c): velocity of x(c(t)) at 0 is dx(X)", (c(hh) - c(-hh)) / (2 * hh), J @ X, 1e-6)
xt = ex["torus"]["expr"]
Rs, rs = ex["torus"]["params"]
ut, vt = ex["torus"]["coords"]
Xu, Xv = xt.diff(ut), xt.diff(vt)
sym_equal("(38.1.2): x_u of the torus", Xu, rs * sp.Matrix([-sp.sin(ut) * sp.cos(vt), -sp.sin(ut) * sp.sin(vt), sp.cos(ut)]))
sym_equal("(38.1.2): x_v of the torus", Xv, (Rs + rs * sp.cos(ut)) * sp.Matrix([-sp.sin(vt), sp.cos(vt), 0]))
sym_equal("Example 38.1.5(a): <x_u, x_v> = 0", Xu.dot(Xv), 0)
al = sp.symbols("alpha", real=True)
dX = sp.cos(al) * Xu + sp.sin(al) * Xv
sym_equal("Example 38.1.5(a): |dx(X)|^2 = r^2 cos^2 a + (R + r cos u)^2 sin^2 a", dX.dot(dX),
          rs ** 2 * sp.cos(al) ** 2 + (Rs + rs * sp.cos(ut)) ** 2 * sp.sin(al) ** 2)
val = (Xu + Xv).subs({Rs: 2, rs: sp.Rational(4, 5), ut: sp.pi / 2, vt: 0})
check("Example 38.1.5(a): dx(d_u + d_v) at (pi/2, 0) = (-0.8, 2, 0)", sp.simplify(val - sp.Matrix([-sp.Rational(4, 5), 2, 0]))
      == sp.zeros(3, 1))
for qq in (np.array([0.0, -1.0]), np.array([2 * np.pi / 3, -0.25])):
    xu, xv = ddg38.partials(ddg38.torus, np.array(qq[0]), np.array(qq[1]))
    close("Figure 38.1.1: semi-axes 0.45 r and 0.45 (R + r cos u)", np.sort(0.45 * np.linalg.svd(np.stack([xu, xv], 1),
          compute_uv=False)), np.sort([0.45 * r, 0.45 * (R + r * np.cos(qq[0]))]), 1e-7)
close("Figure 38.1.1: 0.45 * 2.8 = 1.26 and 0.45 * 1.6 = 0.72", [0.45 * (R + r), 0.45 * (R + r * np.cos(2 * np.pi / 3))],
      [1.26, 0.72], 1e-12)
xsa = sp.Matrix([u, v, u ** 2 - v ** 2])
X1, X2 = sp.symbols("X1 X2", real=True)
dxs = (xsa.diff(u) * X1 + xsa.diff(v) * X2).subs({u: sp.Rational(1, 2), v: -sp.Rational(2, 5)})
check("Example 38.1.5(c): saddle dx_q(X) = (X1, X2, X1 + 0.8 X2) at q = (0.5, -0.4)",
      sp.simplify(dxs - sp.Matrix([X1, X2, X1 + sp.Rational(4, 5) * X2])) == sp.zeros(3, 1))
xsp = ex["sphere"]["expr"]
th, ph = ex["sphere"]["coords"]
rsp = ex["sphere"]["params"][0]
sym_equal("Example 38.1.5(b): |x_theta| = r, |x_phi|^2 = r^2 sin^2 theta",
          [xsp.diff(th).dot(xsp.diff(th)), xsp.diff(ph).dot(xsp.diff(ph))], [rsp ** 2, rsp ** 2 * sp.sin(th) ** 2])

# ---------------------------------------------------------------- Proposition 38.1.7 (iv), Example 38.1.8
cr = Xu.cross(Xv)
sym_equal("Example 38.1.8(a): |x_u x x_v|^2 = r^2 (R + r cos u)^2", cr.dot(cr), rs ** 2 * (Rs + rs * sp.cos(ut)) ** 2)
close("Example 38.1.8(a): minimum r (R - r) = 0.96 for (2, 0.8)", 0.8 * (2 - 0.8), 0.96, 1e-12)
crs = xsp.diff(th).cross(xsp.diff(ph))
sym_equal("Example 38.1.8(b): x_theta x x_phi = r sin(theta) x", crs, rsp * sp.sin(th) * xsp)
check("Example 38.1.8(b): x_phi = 0 at theta = 0", xsp.diff(ph).subs(th, 0) == sp.zeros(3, 1))
cyl = sp.Matrix([rs * sp.cos(u), rs * sp.sin(u), v])
sym_equal("Example 38.1.8(c): cylinder x_u x x_v", cyl.diff(u).cross(cyl.diff(v)), sp.Matrix([rs * sp.cos(u), rs * sp.sin(u), 0]))
sym_equal("Example 38.1.8(c): saddle x_u x x_v = (-2u, 2v, 1)", xsa.diff(u).cross(xsa.diff(v)), sp.Matrix([-2 * u, 2 * v, 1]))
# (iv): dx ^ dx (d_u, d_v) = 2 x_u x x_v by (35.5.5)
wedge = Xu.cross(Xv) - Xv.cross(Xu)
sym_equal("Prop 38.1.7(iv): (dx ^ dx)(d_u, d_v) = 2 x_u x x_v (torus)", wedge, 2 * cr)

# ---------------------------------------------------------------- Non-examples 38.1.9, 38.1.11
cone = sp.Matrix([rho * sp.cos(t), rho * sp.sin(t), rho])
sym_equal("Non-example 38.1.9(a): cone x_rho x x_t", cone.diff(rho).cross(cone.diff(t)),
          sp.Matrix([-rho * sp.cos(t), -rho * sp.sin(t), rho]))
fold = sp.Matrix([u, v ** 2, 0])
check("Non-example 38.1.9(b): fold x(u, v) = x(u, -v) and x_v = 0 at v = 0",
      fold.subs(v, -v) == fold and fold.diff(v).subs(v, 0) == sp.zeros(3, 1))
cub = sp.Matrix([u ** 3, v, 0])
check("Non-example 38.1.11: (u^3, v, 0) has x_u = 0 at u = 0", cub.diff(u).subs(u, 0) == sp.zeros(3, 1))
us = np.linspace(-2, 2, 4001)
check("Non-example 38.1.11: u -> u^3 is strictly increasing (injective)", np.all(np.diff(us ** 3) > 0))

# ---------------------------------------------------------------- Non-example 38.1.14 (figure-eight cylinder)
f8 = sp.Matrix([sp.sin(u), sp.sin(u) * sp.cos(u), v])
cr8 = f8.diff(u).cross(f8.diff(v))
c = sp.symbols("c", real=True)
sym_equal("Non-example 38.1.14: |x_u x x_v|^2 = 4c^4 - 3c^2 + 1 (c = cos u)",
          sp.expand(cr8.dot(cr8)).subs(sp.sin(2 * u), 2 * sp.sin(u) * sp.cos(u)).rewrite(sp.cos),
          (4 * c ** 4 - 3 * c ** 2 + 1).subs(c, sp.cos(u)))
cc = np.linspace(-1, 1, 200001)
close("Non-example 38.1.14: min of 4c^4 - 3c^2 + 1 on [-1, 1] is 7/16", (4 * cc ** 4 - 3 * cc ** 2 + 1).min(), 7 / 16, 1e-9)
check("Non-example 38.1.14: x(0, v) = x(pi, v)", sp.simplify(f8.subs(u, 0) - f8.subs(u, sp.pi)) == sp.zeros(3, 1))
n0 = cr8.subs(u, 0)
n1 = cr8.subs(u, sp.pi)
check("Non-example 38.1.14: normals (1, -1, 0)/sqrt 2 and (1, 1, 0)/sqrt 2",
      sp.simplify(n0 / sp.sqrt(n0.dot(n0)) - sp.Matrix([1, -1, 0]) / sp.sqrt(2)) == sp.zeros(3, 1)
      and sp.simplify(n1 / sp.sqrt(n1.dot(n1)) - sp.Matrix([1, 1, 0]) / sp.sqrt(2)) == sp.zeros(3, 1))

# ---------------------------------------------------------------- Non-example 38.1.15, (38.1.4)
ev = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), (1 - 2 * lam) * sp.cos(th)])
crev = ev.diff(th).cross(ev.diff(ph))
sym_equal("(38.1.4): x_theta x x_phi of the family", crev,
          sp.Matrix([(1 - 2 * lam) * sp.sin(th) ** 2 * sp.cos(ph), (1 - 2 * lam) * sp.sin(th) ** 2 * sp.sin(ph),
                     sp.sin(th) * sp.cos(th)]))
eq = crev.subs(th, sp.pi / 2)
sym_equal("Non-example 38.1.15: |x_theta x x_phi|^2 = (1 - 2 lambda)^2 on the equator", eq.dot(eq), (1 - 2 * lam) ** 2)
for la_, sgn in ((0.2, 1), (0.8, -1)):
    val = float(eq.dot(ev.subs(th, sp.pi / 2)).subs({lam: la_, ph: 0.7}))
    check(f"Non-example 38.1.15: normal on the equator {'outward' if sgn > 0 else 'inward'} at lambda = {la_}",
          np.sign(val) == sgn)

# ---------------------------------------------------------------- Exercises 38.1.1-38.1.4
sym_equal("Exercise 38.1.1: cylinder dx(d_u + d_v) at (pi/2, 0) = (-r, 0, 1)",
          (cyl.diff(u) + cyl.diff(v)).subs({u: sp.pi / 2, v: 0}), sp.Matrix([-rs, 0, 1]))
yv = -xsp
cry = yv.diff(th).cross(yv.diff(ph))
sym_equal("Exercise 38.1.2: y_theta x y_phi = -r^2 sin(theta) (y / r)", cry, -rsp ** 2 * sp.sin(th) * yv / rsp)
ypol = xsa.subs({u: rho * sp.cos(t), v: rho * sp.sin(t)})
cpol = ypol.diff(rho).cross(ypol.diff(t))
sym_equal("Exercise 38.1.3(c): y_rho x y_t = (-2 rho^2 cos t, 2 rho^2 sin t, rho)", cpol,
          sp.Matrix([-2 * rho ** 2 * sp.cos(t), 2 * rho ** 2 * sp.sin(t), rho]))
sym_equal("Exercise 38.1.3(a)(c): y_rho x y_t = det(Dh) (x_u x x_v) o h", cpol,
          rho * xsa.diff(u).cross(xsa.diff(v)).subs({u: rho * sp.cos(t), v: rho * sp.sin(t)}))
wu = sp.Matrix([u * v, v, u ** 2])
sym_equal("Exercise 38.1.4(a): Whitney umbrella x_u x x_v = (-2u, 2u^2, v)", wu.diff(u).cross(wu.diff(v)),
          sp.Matrix([-2 * u, 2 * u ** 2, v]))
check("Exercise 38.1.4(b): x(u, 0) = x(-u, 0)", sp.simplify(wu.subs(v, 0) - wu.subs({u: -u, v: 0})) == sp.zeros(3, 1))
a1, b1 = sp.symbols("a1 b1", real=True)
# second coordinate forces b2 = b1, third forces a2 = +-a1; with a2 = -a1 the difference is (2 a1 b1, 0, 0)
check("Exercise 38.1.4(b): x(a, b) - x(-a, b) = (2ab, 0, 0), so a second preimage needs b = 0",
      sp.simplify(wu.subs({u: a1, v: b1}) - wu.subs({u: -a1, v: b1}) - sp.Matrix([2 * a1 * b1, 0, 0])) == sp.zeros(3, 1))

# ---------------------------------------------------------------- Conventions box: Crane's sphere
fC = sp.Matrix([sp.cos(u) * sp.sin(v), sp.sin(u) * sp.sin(v), sp.cos(v)])
sym_equal("Conventions: Crane's f_u x f_v = -sin(v) f (inward)", fC.diff(u).cross(fC.diff(v)), -sp.sin(v) * fC)
check("Conventions: Crane's (u, v) = our (phi, theta)", sp.simplify(fC - xsp.subs({rsp: 1, th: v, ph: u})) == sp.zeros(3, 1))

summary()
