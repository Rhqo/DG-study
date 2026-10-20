"""Verification for Section 42.2 (Dirichlet Energy, Harmonic Functions, and the Poisson Equation).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/verify/v-42-2-dirichlet-poisson.py``
"""

import pathlib
import sys

import dgcode
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch42  # noqa: E402

from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE / "42-2-dirichlet-poisson.html"
x, y = sp.symbols("x y", real=True)
r, t = sp.symbols("r t", positive=True)
th, ph = sp.symbols("theta phi", positive=True)
pol = {x: r * sp.cos(t), y: r * sp.sin(t)}


def disk_energy(u):
    g = (sp.diff(u, x) ** 2 + sp.diff(u, y) ** 2).subs(pol)
    return sp.simplify(sp.Rational(1, 2) * sp.integrate(sp.integrate(sp.expand(g) * r, (r, 0, 1)), (t, 0, 2 * sp.pi)))


def lb_sph(f):
    return sp.simplify(sp.diff(sp.sin(th) * sp.diff(f, th), th) / sp.sin(th) + sp.diff(f, ph, 2) / sp.sin(th) ** 2)


def sphere_int(e):
    return sp.simplify(sp.integrate(sp.integrate(sp.expand(e * sp.sin(th)), (th, 0, sp.pi)), (ph, 0, 2 * sp.pi)))


def sph_grad2(f):
    return sp.diff(f, th) ** 2 + sp.diff(f, ph) ** 2 / sp.sin(th) ** 2


Z, X = sp.cos(th), sp.sin(th) * sp.cos(ph)

# ---------------------------------------------------------------- Example 42.2.2
check("Example 42.2.2(a): E_D(x) = pi/2 on the unit disk", sp.simplify(disk_energy(x) - sp.pi / 2) == 0)
check("Example 42.2.2(b): E_D(x^2 - y^2) = pi on the unit disk", sp.simplify(disk_energy(x ** 2 - y ** 2) - sp.pi) == 0)
check("Example 42.2.2(c): |grad z|^2 = 1 - z^2 and E_D(z) = 4 pi / 3 on the unit sphere",
      sp.simplify(sph_grad2(Z) - (1 - Z ** 2)) == 0 and sp.simplify(sphere_int(sph_grad2(Z)) / 2 - 4 * sp.pi / 3) == 0)
check("Example 42.2.2(c): -1/2 int z Delta z = int z^2 = 4 pi / 3", sp.simplify(-sphere_int(Z * lb_sph(Z)) / 2 - 4 * sp.pi / 3) == 0)

# ---------------------------------------------------------------- Non-example 42.2.3
for delta in (0.25, 0.1, 0.02):
    n = 4000
    xs = (np.arange(n + 1)) / n
    ud = np.minimum(1.0, np.maximum(0.0, 0.5 + (xs - 0.5) / (2 * delta)))
    check(f"Non-example 42.2.3: the ramp of half width {delta} is 0 on the left edge and 1 on the right edge",
          ud[0] == 0.0 and ud[-1] == 1.0)
    E = 0.5 * np.sum((np.diff(ud) * n) ** 2) / n          # u_delta is piecewise linear in x; the length in y is 1
    close(f"Non-example 42.2.3: E_D of the ramp of half width {delta} is 1/(4 delta)", E, 1 / (4 * delta),
          tol=2e-3 / (4 * delta))
close("Non-example 42.2.3: E_D(x) = 1/2 on [0, 1]^2", 0.5 * float(sp.integrate(sp.Integer(1), (x, 0, 1), (y, 0, 1))), 0.5, tol=0)

# ---------------------------------------------------------------- Theorem 42.2.4, Figure 42.2.1, Exercise 42.2.6
tt = sp.symbols("tau", real=True)
uf, vf = sp.Function("u")(x, y), sp.Function("v")(x, y)
g2 = lambda f: sp.diff(f, x) ** 2 + sp.diff(f, y) ** 2  # noqa: E731
sym_equal("Theorem 42.2.4 proof (42.2.3): |grad(u + t v)|^2 expands with 2 t <grad u, grad v> + t^2 |grad v|^2",
          sp.expand(g2(uf + tt * vf)), sp.expand(g2(uf) + 2 * tt * (sp.diff(uf, x) * sp.diff(vf, x) + sp.diff(uf, y) * sp.diff(vf, y)) + tt ** 2 * g2(vf)))
uh = sp.sin(sp.pi * x) * sp.sinh(sp.pi * y) / sp.sinh(sp.pi)
wb = y * sp.sin(sp.pi * x)
sq = lambda e: sp.integrate(sp.integrate(sp.expand(e), (x, 0, 1)), (y, 0, 1))  # noqa: E731
EDu = sp.simplify(sq(g2(uh)) / 2)
EDw = sp.simplify(sq(g2(wb)) / 2)
EDd = sp.simplify(sq(g2(wb - uh)) / 2)
check("Figure 42.2.1: u is harmonic and equals g on the boundary",
      sp.simplify(sp.diff(uh, x, 2) + sp.diff(uh, y, 2)) == 0 and uh.subs(y, 1) == sp.sin(sp.pi * x) and uh.subs(y, 0) == 0
      and uh.subs(x, 0) == 0 and sp.simplify(uh.subs(x, 1)) == 0)
close("Figure 42.2.1, Exercise 42.2.6: E_D(u) = (pi/4) coth(pi) = 0.7883", [float(EDu), float(sp.pi / 4 / sp.tanh(sp.pi))],
      [0.7883, 0.7883], tol=5e-5)
check("Exercise 42.2.6: E_D(w) = pi^2/12 + 1/4 = 1.0725", sp.simplify(EDw - (sp.pi ** 2 / 12 + sp.Rational(1, 4))) == 0
      and abs(float(EDw) - 1.0725) < 5e-5)
close("Theorem 42.2.4(b) (42.2.2): E_D(w) = E_D(u) + E_D(w - u), difference 0.2841", [float(EDw - EDu), float(EDd)],
      [0.2841, float(EDw - EDu)], tol=5e-5)
bd_int = sp.integrate(sp.sin(sp.pi * x) * sp.diff(uh, y).subs(y, 1), (x, 0, 1)) / 2
check("Exercise 42.2.6: E_D(u) = (1/2) boundary integral of u <grad u, n>", sp.simplify(bd_int - EDu) == 0)
n = 32
xs = np.linspace(0, 1, n + 1)
V, F = dm.grid_mesh(xs, xs)
L = dm.cotan_laplacian(V, F)
B = np.array(dm.boundary_vertices(F))
I = np.setdiff1d(np.arange(len(V)), B)
ud = np.zeros(len(V))
ud[B] = np.where(np.isclose(V[B, 1], 1.0), np.sin(np.pi * V[B, 0]), 0.0)
ud[I] = np.linalg.solve(L[np.ix_(I, I)], -L[np.ix_(I, B)] @ ud[B])
uex = np.sin(np.pi * V[:, 0]) * np.sinh(np.pi * V[:, 1]) / np.sinh(np.pi)
close("Figure 42.2.1: discrete harmonic interpolant on grid_mesh(32): error 2.8e-4, energy -u^T L u / 2 = 0.7890",
      [np.abs(ud - uex).max(), -0.5 * ud @ L @ ud], [2.8e-4, 0.7890], tol=5e-5)

# ---------------------------------------------------------------- Figure 42.2.2 (History box)
eps = sp.symbols("epsilon", positive=True)
phie = sp.atan(x / eps) / sp.atan(1 / eps)
Jint = sp.integrate(x ** 2 * sp.diff(phie, x) ** 2, (x, -1, 1))
Jcl = (eps * sp.atan(1 / eps) - eps ** 2 / (1 + eps ** 2)) / sp.atan(1 / eps) ** 2
close("Figure 42.2.2: J(phi_eps) has the closed form (eps = 0.3, 0.07)", [float(Jint.subs(eps, e)) for e in (0.3, 0.07)],
      [float(Jcl.subs(eps, e)) for e in (0.3, 0.07)], tol=1e-12)
close("Figure 42.2.2: J = 0.288, 0.063, 0.013 for eps = 0.5, 0.1, 0.02", [float(Jcl.subs(eps, e)) for e in (0.5, 0.1, 0.02)],
      [0.288, 0.063, 0.013], tol=6e-4)
check("Figure 42.2.2: J(phi_eps) ~ 2 eps / pi as eps -> 0", sp.limit(Jcl / (2 * eps / sp.pi), eps, 0, "+") == 1)

# ---------------------------------------------------------------- Example 42.2.6, Proposition 42.2.8, Corollary 42.2.9
for hf in (x ** 2 - y ** 2, x * y, x ** 3 - 3 * x * y ** 2, sp.log(x ** 2 + y ** 2) / 2):
    check(f"Example 42.2.6: {hf} is harmonic", sp.simplify(sp.diff(hf, x, 2) + sp.diff(hf, y, 2)) == 0)
check("Example 42.2.6: x^2 - y^2 = r^2 cos 2 theta", sp.simplify((x ** 2 - y ** 2).subs(pol) - r ** 2 * sp.cos(2 * t)) == 0)
hnum = lambda X_, Y_: X_ ** 3 - 3 * X_ * Y_ ** 2 + np.log(np.hypot(X_ - 2.0, Y_))  # noqa: E731
pp = np.array([0.3, -0.4])
check("Proposition 42.2.8: circle means equal the center value (radii 0.2, 0.7, 1.2)",
      all(abs(ch42.circle_mean(hnum, pp, e) - hnum(*pp)) < 1e-12 for e in (0.2, 0.7, 1.2)))
G = np.linspace(-1, 1, 201)
GX, GY = np.meshgrid(G, G)
inside = GX ** 2 + GY ** 2 < 1
vals = hnum(GX, GY)
tc = np.linspace(0, 2 * np.pi, 4000)
bmax = hnum(np.cos(tc), np.sin(tc)).max()
check("Corollary 42.2.9: on the unit disk the maximum of a harmonic function is on the boundary",
      vals[inside].max() <= bmax + 1e-12)

# ---------------------------------------------------------------- Example 42.2.12, Exercises 42.2.3, 42.2.4
check("Example 42.2.12(a): Delta(-z/2) = z, int z = 0", sp.simplify(lb_sph(-Z / 2) - Z) == 0 and sphere_int(Z) == 0)
check("Example 42.2.12(b): int 1 dA = 4 pi", sphere_int(sp.Integer(1)) == 4 * sp.pi)
check("Exercise 42.2.3(a),(b): Delta(-x/2) = x, int (x + 1) = 4 pi", sp.simplify(lb_sph(-X / 2) - X) == 0
      and sphere_int(X + 1) == 4 * sp.pi)
check("Exercise 42.2.3(c),(d): int z^2 = 4 pi / 3, Delta(-(z^2 - 1/3)/6) = z^2 - 1/3",
      sphere_int(Z ** 2) == 4 * sp.pi / 3 and sp.simplify(lb_sph(-(Z ** 2 - sp.Rational(1, 3)) / 6) - (Z ** 2 - sp.Rational(1, 3))) == 0)
Jfun = lambda u, f: sp.simplify(sphere_int(sph_grad2(u)) / 2 + sphere_int(f * u))  # noqa: E731
us = -Z / 2
check("Exercise 42.2.4: J(-z/2) = -E_D(-z/2) = (1/2) int f u = -pi/3",
      sp.simplify(Jfun(us, Z) + sp.pi / 3) == 0 and sp.simplify(sphere_int(sph_grad2(us)) / 2 - sp.pi / 3) == 0
      and sp.simplify(sphere_int(Z * us) / 2 + sp.pi / 3) == 0)
e_ = sp.symbols("e", real=True)
check("Exercise 42.2.4, Proposition 42.2.13(b): J(u + e x) = J(u) + e^2 E_D(x) = -pi/3 + 4 pi e^2 / 3",
      sp.simplify(Jfun(us + e_ * X, Z) - (-sp.pi / 3 + 4 * sp.pi * e_ ** 2 / 3)) == 0)
check("Proposition 42.2.13(a): J(c) = c int f is unbounded when int f != 0 (f = 1 + z)",
      sp.simplify(Jfun(sp.Symbol("c"), 1 + Z) - 4 * sp.pi * sp.Symbol("c")) == 0)

# ---------------------------------------------------------------- Remark 42.2.14, Exercise 42.2.2
Gf = lambda P: np.log(np.hypot(*P)) / (2 * np.pi)  # noqa: E731


def flux(center, rad, m=4000):
    s = 2 * np.pi * np.arange(m) / m
    P = np.array([center[0] + rad * np.cos(s), center[1] + rad * np.sin(s)])
    grad = P / (2 * np.pi * (P[0] ** 2 + P[1] ** 2))
    return np.sum(grad[0] * np.cos(s) + grad[1] * np.sin(s)) * rad * 2 * np.pi / m


close("Exercise 42.2.2: flux of grad (1/2pi) log r through circles about 0 is 1", [flux((0, 0), 0.3), flux((0, 0), 2.0)], [1, 1],
      tol=1e-12)
close("Exercise 42.2.2: flux through circles that do not enclose 0 is 0 (and enclosing, off-center: 1)",
      [flux((2.0, 1.0), 0.8), flux((0.3, -0.2), 1.0)], [0.0, 1.0], tol=1e-10)

# ---------------------------------------------------------------- Exercise 42.2.1
check("Exercise 42.2.1: E_D(x^2 - y^2) = pi", sp.simplify(disk_energy(x ** 2 - y ** 2) - sp.pi) == 0)
wr = r * sp.cos(2 * t)
EDwr = sp.integrate(sp.integrate((sp.diff(wr, r) ** 2 + sp.diff(wr, t) ** 2 / r ** 2) * r, (r, 0, 1)), (t, 0, 2 * sp.pi)) / 2
check("Exercise 42.2.1: E_D(r cos 2 theta) = 5 pi / 4", sp.simplify(EDwr - 5 * sp.pi / 4) == 0)

# ---------------------------------------------------------------- Exercise 42.2.5
ns, same, printed = ch42.run_printed(PAGE, "code-ex-42-2-5", "out-ex-42-2-5")
check("Exercise 42.2.5: code-ex-42-2-5 prints exactly out-ex-42-2-5", same, printed)
check("Exercise 42.2.5: code block is at most 30 lines", dgcode.code_lines(PAGE, "code-ex-42-2-5") <= 30)
errs2d = []
for N in (8, 16, 32):
    xs = np.linspace(0.0, 1.0, N + 1)
    V2, F2 = dm.grid_mesh(xs, xs)
    L2 = dm.cotan_laplacian(V2, F2)
    m2 = ch42.mass(V2, F2, "circumcentric")
    ex = np.sin(np.pi * V2[:, 0]) * np.sin(np.pi * V2[:, 1])
    inn = (V2[:, 0] > 1e-9) & (V2[:, 0] < 1 - 1e-9) & (V2[:, 1] > 1e-9) & (V2[:, 1] < 1 - 1e-9)
    u2 = np.zeros(len(V2))
    u2[inn] = np.linalg.solve(L2[np.ix_(inn, inn)], m2[inn] * (-2 * np.pi ** 2 * ex[inn]))
    errs2d.append(np.abs(u2 - ex).max())
errs1d = [np.abs(ns["solve_1d"](N)[1] - np.sin(np.pi * ns["solve_1d"](N)[0])).max() for N in (8, 16, 32)]
close("Exercise 42.2.5: the 1D errors equal the 2D square-grid Poisson errors of Section 36.6 (separation of variables)",
      errs1d, errs2d, tol=1e-12)

summary()
