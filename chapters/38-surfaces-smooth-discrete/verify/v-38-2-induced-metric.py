"""Verification for Section 38.2 (The Induced Metric and Conformal Coordinates).

Run from the project root: PYTHONPATH=tools python3 chapters/38-surfaces-smooth-discrete/verify/v-38-2-induced-metric.py
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

PAGE = HERE.parents[1] / "38-2-induced-metric.html"
u, v, t, ph, rho, w = sp.symbols("u v t phi rho w", real=True)
r = sp.symbols("r", positive=True)
ex = dgsym.EXAMPLES

# ---------------------------------------------------------------- Implementation 38.2 and Exercise 38.2.6
check("Implementation 38.2: at most 30 lines", dgcode.code_lines(PAGE, "code-38-2-1") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-38-2-1")
check("Implementation 38.2: printed output matches the page", out.strip() == ddg38.printed_block(PAGE, "out-38-2-1").strip(),
      out)
close("Implementation 38.2: metric at t = 0.5 is sech^2(0.5) I", ns["metric"](ns["mercator"], np.array([0.3, 0.5])),
      np.eye(2) / np.cosh(0.5) ** 2, 1e-8)
close("(38.2.9): loxodrome length beta = 60 deg, T = 3", ns["length_via_metric"](ns["mercator"], ns["line"], 0, 3),
      np.arctan(np.sinh(3.0)) / np.cos(np.pi / 3), 1e-6)
check("Exercise 38.2.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-38-2-6") <= 30)
ns, out = ddg38.run_printed(PAGE, "code-ex-38-2-6", ns)
check("Exercise 38.2.6: printed output matches the page",
      out.strip() == ddg38.printed_block(PAGE, "out-ex-38-2-6").strip(), out)
close("Exercise 38.2.6: pi/2 - arctan(sinh 20) ~ 4e-9", np.pi / 2 - np.arctan(np.sinh(20.0)), 2 * np.exp(-20), 1e-10)

# ---------------------------------------------------------------- Proposition 38.2.2
rng = np.random.default_rng(382)
for _ in range(4):
    q = rng.uniform(-2, 2, 2)
    xu, xv = ddg38.partials(ddg38.torus, np.array(q[0]), np.array(q[1]))
    J = np.stack([xu, xv], 1)
    X, Y = rng.normal(size=2), rng.normal(size=2)
    close("Prop 38.2.2(b): g(X, Y) = <dx X, dx Y> = X^T (Dx^T Dx) Y", (J @ X) @ (J @ Y), X @ (J.T @ J) @ Y, 1e-9)
    check("Prop 38.2.2(a): g positive definite at an immersion point", np.all(np.linalg.eigvalsh(J.T @ J) > 0))
xs = sp.Matrix([u, v, u ** 2 - v ** 2])
E, F, G = dgsym.first_ff(xs, u, v)
sym_equal("(38.2.2): saddle E, F, G", [E, F, G], [1 + 4 * u ** 2, -4 * u * v, 1 + 4 * v ** 2])
sym_equal("(38.2.2): EG - F^2 = 1 + 4u^2 + 4v^2", E * G - F ** 2, 1 + 4 * u ** 2 + 4 * v ** 2)
gm = sp.Matrix([[E, F], [F, G]])
h = sp.Matrix([rho * sp.cos(t), rho * sp.sin(t)])
Dh = h.jacobian([rho, t])
gt = sp.simplify(Dh.T * gm.subs({u: h[0], v: h[1]}) * Dh)
y = xs.subs({u: h[0], v: h[1]})
Et, Ft, Gt = dgsym.first_ff(y, rho, t)
sym_equal("Prop 38.2.2(c): Dh^T g Dh = metric of y (polar saddle)", [gt[0, 0], gt[0, 1], gt[1, 1]], [Et, Ft, Gt])
sym_equal("(38.2.3): polar saddle E~, F~, G~", [Et, Ft, Gt],
          [1 + 4 * rho ** 2 * sp.cos(2 * t) ** 2, -2 * rho ** 3 * sp.sin(4 * t), rho ** 2 + 4 * rho ** 4 * sp.sin(2 * t) ** 2])
# Warning box: plane in polar coordinates
pl = sp.Matrix([rho * sp.cos(t), rho * sp.sin(t), 0])
sym_equal("Warning: plane in polar coordinates g = diag(1, rho^2)", list(dgsym.first_ff(pl, rho, t)), [1, 0, rho ** 2])

# ---------------------------------------------------------------- Example 38.2.3, Non-example 38.2.4
sp_ = ex["sphere"]
Es, Fs, Gs = dgsym.first_ff(sp_["expr"], *sp_["coords"], positive=sp_["positive"])
rr = sp_["params"][0]
th = sp_["coords"][0]
sym_equal("Example 38.2.3(a): sphere E, F, G", [Es, Fs, Gs], [rr ** 2, 0, rr ** 2 * sp.sin(th) ** 2])
to = ex["torus"]
Et_, Ft_, Gt_ = dgsym.first_ff(to["expr"], *to["coords"], positive=to["positive"])
Rs, rs = to["params"]
sym_equal("Example 38.2.3(a): torus E, F, G", [Et_, Ft_, Gt_], [rs ** 2, 0, (Rs + rs * sp.cos(to["coords"][0])) ** 2])
th0 = 0.4
close("Non-example 38.2.4: latitude circle length 2 pi r sin(theta0) (r = 1)",
      ns["length_via_metric"](lambda q: ddg38.sphere(q[0], q[1]), lambda s: np.array([th0, s]), 0, 2 * np.pi, n=400),
      2 * np.pi * np.sin(th0), 1e-6)

# ---------------------------------------------------------------- Proposition 38.2.5, Example 38.2.6
a = sp.symbols("a", positive=True)
A = sp.integrate(sp.integrate(sp.sqrt(1 + 4 * rho ** 2) * rho, (rho, 0, a)), (t, 0, 2 * sp.pi))
sym_equal("(38.2.4): saddle patch area", A, sp.pi / 6 * ((1 + 4 * a ** 2) ** sp.Rational(3, 2) - 1))
close("Example 38.2.6: a = 1/2 gives 0.9574", float(A.subs(a, sp.Rational(1, 2))), 0.9574, 5e-5)
close("Example 38.2.6: a = 1/2 exact (pi/6)(2 sqrt 2 - 1)", float(A.subs(a, sp.Rational(1, 2))), np.pi / 6 * (2 * np.sqrt(2) - 1),
      1e-12)
ser = sp.series(A, a, 0, 6).removeO()
sym_equal("Example 38.2.6: A = pi a^2 + pi a^4 + O(a^6)", ser, sp.pi * a ** 2 + sp.pi * a ** 4)
# Proposition 38.2.5(b): angle from g equals angle in R^3
q = np.array([0.5, -0.4])
xu, xv = ddg38.partials(ddg38.saddle, np.array(q[0]), np.array(q[1]))
J = np.stack([xu, xv], 1)
X, Y = np.array([1.0, 0.3]), np.array([-0.2, 1.0])
g = J.T @ J
close("Prop 38.2.5(b): cos angle via g", (X @ g @ Y) / np.sqrt((X @ g @ X) * (Y @ g @ Y)),
      (J @ X) @ (J @ Y) / (np.linalg.norm(J @ X) * np.linalg.norm(J @ Y)), 1e-12)

# ---------------------------------------------------------------- Proposition 38.2.8, Examples 38.2.9, 38.2.10
ws = 1 + u ** 2 + v ** 2
st = r * sp.Matrix([2 * u / ws, 2 * v / ws, (u ** 2 + v ** 2 - 1) / ws])
E2, F2, G2 = dgsym.first_ff(st, u, v)
sym_equal("Example 38.2.9: stereographic E = G = 4 r^2/(1+u^2+v^2)^2, F = 0", [E2, F2, G2],
          [4 * r ** 2 / ws ** 2, 0, 4 * r ** 2 / ws ** 2])
mu_st = sp.log(2 * r) - sp.log(ws)
sym_equal("(38.2.6): e^{2 mu} matches", sp.exp(2 * mu_st), 4 * r ** 2 / ws ** 2)
sym_equal("Example 38.2.9: the image lies on S^2(r)", st.dot(st), r ** 2)
cr0 = st.diff(u).cross(st.diff(v)).subs({u: 0, v: 0})
check("Example 38.2.9: at (0, 0) x_u x x_v = (0, 0, 4r^2) and x = (0, 0, -r) (inward)",
      sp.simplify(cr0 - sp.Matrix([0, 0, 4 * r ** 2])) == sp.zeros(3, 1) and st.subs({u: 0, v: 0}) == sp.Matrix([0, 0, -r]))
mer = r * sp.Matrix([sp.cos(ph) / sp.cosh(t), sp.sin(ph) / sp.cosh(t), sp.tanh(t)])
E3, F3, G3 = dgsym.first_ff(mer, ph, t)
sym_equal("Example 38.2.10: Mercator E = G = r^2 sech^2 t, F = 0", [E3, F3, G3],
          [r ** 2 / sp.cosh(t) ** 2, 0, r ** 2 / sp.cosh(t) ** 2])
sym_equal("(38.2.8): e^{2 mu} matches", sp.exp(2 * (sp.log(r) - sp.log(sp.cosh(t)))), r ** 2 / sp.cosh(t) ** 2)
sym_equal("Example 38.2.10: x_phi x x_t = r^2 sech^2 t (x / r) (outward)", mer.diff(ph).cross(mer.diff(t)),
          r * mer / sp.cosh(t) ** 2)
thv = sp.symbols("theta", positive=True)
tt = sp.log(sp.cot(thv / 2))
for val in (0.3, 1.1, 2.5):
    close(f"Example 38.2.10: sin(theta) = sech t, cos(theta) = tanh t at theta = {val}",
          [float(1 / sp.cosh(tt.subs(thv, val))), float(sp.tanh(tt.subs(thv, val)))], [np.sin(val), np.cos(val)], 1e-12)
be = sp.symbols("beta", positive=True)
cp = sp.Matrix([sp.tan(be), 1])
gg = (r ** 2 / sp.cosh(t) ** 2) * (cp.T * cp)[0]
sym_equal("(38.2.9): g(c', c') = r^2 sech^2 t / cos^2 beta", gg, r ** 2 / (sp.cosh(t) ** 2 * sp.cos(be) ** 2))
Tt = sp.symbols("T", positive=True)
sym_equal("(38.2.9): integral of sech from 0 to T = arctan(sinh T)", sp.diff(sp.atan(sp.sinh(Tt)), Tt), 1 / sp.cosh(Tt))
close("(38.2.9): beta = 60 deg, r = 1: pole length = pi", (np.pi / 2) / np.cos(np.pi / 3), np.pi, 1e-12)
# angle with the meridian in R^3 is beta (Proposition 38.2.8)
for tv in (0.0, 1.3):
    vel = mer.diff(ph) * sp.tan(be) + mer.diff(t)
    md = mer.diff(t)
    cosang = (vel.dot(md) / sp.sqrt(vel.dot(vel) * md.dot(md))).subs({be: sp.pi / 3, t: tv, ph: 0.4, r: 1})
    close(f"Example 38.2.10: loxodrome meets the meridian at 60 deg (t = {tv})", float(cosang), 0.5, 1e-12)
# Proposition 38.2.8 (iii): |dx(X)| constant on the unit circle iff E = G, F = 0
al = sp.symbols("alpha", real=True)
for (Ev, Fv, Gv, conf) in ((2.0, 0.0, 2.0, True), (2.0, 0.0, 3.0, False), (2.0, 0.5, 2.0, False)):
    vals = [Ev * np.cos(x_) ** 2 + 2 * Fv * np.cos(x_) * np.sin(x_) + Gv * np.sin(x_) ** 2 for x_ in (0, np.pi / 2, np.pi / 4)]
    check(f"Prop 38.2.8(iii): (E, F, G) = ({Ev}, {Fv}, {Gv}) circle-to-circle = {conf}", (np.ptp(vals) < 1e-12) == conf)
# Figure 38.2.1: at theta = pi/8 the ellipse ratio is 1/sin(pi/8) = 2.61
close("Figure 38.2.1: 1 / sin(pi/8) = 2.61", 1 / np.sin(np.pi / 8), 2.6131, 1e-4)

# ---------------------------------------------------------------- Non-example 38.2.11
sym_equal("Non-example 38.2.11(b): E - G = 4(u^2 - v^2)", E - G, 4 * (u ** 2 - v ** 2))

# ---------------------------------------------------------------- Exercises
cyl = sp.Matrix([r * sp.cos(u), r * sp.sin(u), v])
sym_equal("Exercise 38.2.1: cylinder E, F, G", list(dgsym.first_ff(cyl, u, v)), [r ** 2, 0, 1])
s_ = sp.symbols("s", real=True)
cy2 = sp.Matrix([r * sp.cos(s_ / r), r * sp.sin(s_ / r), v])
sym_equal("Exercise 38.2.1: y(s, v) has g = I", list(dgsym.first_ff(cy2, s_, v)), [1, 0, 1])
sym_equal("Exercise 38.2.2: polar saddle F~ = -2 rho^3 sin 4t", Ft, -2 * rho ** 3 * sp.sin(4 * t))
R_, r_ = 2.0, 0.8
uu = np.linspace(0, 2 * np.pi, 400001)
wprime = r_ / (R_ + r_ * np.cos(uu))
w2pi = np.sum((wprime[1:] + wprime[:-1]) / 2 * np.diff(uu))
close("Exercise 38.2.3(c): w(2 pi) = 2 pi r / sqrt(R^2 - r^2) = 2.742", w2pi, 2 * np.pi * r_ / np.sqrt(R_ ** 2 - r_ ** 2), 1e-9)
close("Exercise 38.2.3(c): value 2.742", 2 * np.pi * r_ / np.sqrt(R_ ** 2 - r_ ** 2), 2.742, 5e-4)
uu0 = 1.1
du_dw = (R_ + r_ * np.cos(uu0)) / r_
gt3 = np.diag([du_dw, 1]) @ np.diag([r_ ** 2, (R_ + r_ * np.cos(uu0)) ** 2]) @ np.diag([du_dw, 1])
close("Exercise 38.2.3(b): torus in (w, v) has g = (R + r cos u)^2 I", gt3, (R_ + r_ * np.cos(uu0)) ** 2 * np.eye(2), 1e-12)
lam_, alp = 1.7, 0.6
Rm = np.array([[np.cos(alp), -np.sin(alp)], [np.sin(alp), np.cos(alp)]])
close("Exercise 38.2.4: (lam R)^T (e^{2mu} I) (lam R) = lam^2 e^{2 mu} I", (lam_ * Rm).T @ (2.3 * np.eye(2)) @ (lam_ * Rm),
      lam_ ** 2 * 2.3 * np.eye(2), 1e-12)
yv, eps = sp.symbols("y epsilon", positive=True)
sym_equal("Exercise 38.2.5(a): hyperbolic length log(1/eps)", sp.integrate(1 / yv, (yv, eps, 1)), -sp.log(eps))

summary()
