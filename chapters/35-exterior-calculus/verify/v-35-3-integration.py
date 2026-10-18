"""Section 35.3 (Integration of Differential Forms): verification of every computation (GUIDELINES 13).

Integrals are computed exactly with sympy (through lib/ch35.integrate_patch, i.e. Definition 35.3.8) and numerically
with the printed Implementation code; simplex integrals are compared with tools/dgdec.py (derham1, derham2).
Run from the project root: PYTHONPATH=tools python3 chapters/35-exterior-calculus/verify/v-35-3-integration.py
"""
import contextlib
import io
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from dgcheck import check, close, sym_equal, summary  # noqa: E402
import dgdec as dd  # noqa: E402
import ch35  # noqa: E402

PAGE = "35-3-integration.html"
x, y, z = sp.symbols("x y z", real=True)
r = sp.symbols("r", positive=True)
th, ph, t, s, u, v = sp.symbols("theta phi t s u v", real=True)
a_, b_ = sp.symbols("a b", positive=True)
X2, X3 = [x, y], [x, y, z]
rng = np.random.default_rng(353)

# ---------------------------------------------------------------------------
# Proposition 35.3.2 and Example 35.3.3
P = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
c = ch35.pullback_top({(0, 1): x ** 2 + y ** 2}, X2, P, [r, th])
sym_equal("Proposition 35.3.2: polar pullback of (x^2+y^2) dx^dy = r^3", c, r ** 3)
sym_equal("Example 35.3.3(b): int_B (x^2 + y^2) dx^dy = pi/2", sp.integrate(c, (r, 0, 1), (th, 0, 2 * sp.pi)), sp.pi / 2)
sym_equal("Example 35.3.3(b): int_B dx^dy = pi",
          sp.integrate(ch35.pullback_top({(0, 1): 1}, X2, P, [r, th]), (r, 0, 1), (th, 0, 2 * sp.pi)), sp.pi)
check("Example 35.3.3(a): int_Q 6xy^2 dx^dy = 1", sp.integrate(6 * x * y ** 2, (x, 0, 1), (y, 0, 1)) == 1)
check("Example 35.3.3(a): 6xy^2 dy^dx = -6xy^2 dx^dy", ch35.wedge({(1,): 6 * x * y ** 2}, {(0,): 1}) == {(0, 1): -6 * x * y ** 2})
# a map with negative Jacobian determinant gives the opposite sign: (u, v) -> (v, u)
Psw = sp.Matrix([v, u])
c_sw = ch35.pullback_top({(0, 1): 6 * x * y ** 2}, X2, Psw, [u, v])
check("Proposition 35.3.2: det DP < 0 gives the integrand -f(P) |det DP|",
      sp.integrate(c_sw, (u, 0, 1), (v, 0, 1)) == -1)

# ---------------------------------------------------------------------------
# Proposition 35.3.5: Riemann sums for x dy on the unit circle (Figure 35.3.1)
for n in (8, 16, 64, 128):
    tt = 2 * np.pi * np.arange(n + 1) / n
    Pn = np.stack([np.cos(tt), np.sin(tt)], 1)
    Sn = float(np.sum(Pn[:-1, 0] * (Pn[1:, 1] - Pn[:-1, 1])))
    close(f"Proposition 35.3.5 / Figure 35.3.1: S_{n} = (n/2) sin(2 pi/n)", Sn, n / 2 * np.sin(2 * np.pi / n), 1e-12)
close("Figure 35.3.1: S_8 = 2.828", 4 * np.sin(np.pi / 4), 2.828, 5e-4)
close("Figure 35.3.1: pi - S_128 = 0.0013", np.pi - 64 * np.sin(2 * np.pi / 128), 0.0013, 5e-5)
v8 = [np.cos(2 * np.pi * i / 8) * (np.sin(2 * np.pi * (i + 1) / 8) - np.sin(2 * np.pi * i / 8)) for i in range(8)]
check("Figure 35.3.1: chord values 0.207, 0.500, 0.707, ... (max 0.707, two zeros)",
      np.isclose(max(v8), 0.7071, atol=1e-4) and sum(abs(q) < 1e-12 for q in v8) == 2)
check("Figure 35.3.1 (text): the zeros start at (0, 1) and (0, -1) (x_i = 0), the maxima at (1, 0) and (-1, 0)",
      [i for i in range(8) if abs(v8[i]) < 1e-12] == [2, 6]
      and [i for i in range(8) if np.isclose(v8[i], max(v8))] == [0, 4])
for n in (8, 64):
    tt = 2 * np.pi * np.arange(n) / n
    px, py = np.cos(tt), np.sin(tt)
    shoelace = 0.5 * np.sum(px * np.roll(py, -1) - np.roll(px, -1) * py)
    close(f"Text: S_{n} equals the area of the inscribed regular {n}-gon (shoelace)", shoelace, n / 2 * np.sin(2 * np.pi / n),
          1e-12)

# Example 35.3.6
gam = sp.Matrix([sp.cos(t), sp.sin(t)])
sym_equal("Example 35.3.6(a): int x dy over the unit circle = pi",
          sp.integrate(ch35.pullback_top({(1,): x}, X2, gam, [t]), (t, 0, 2 * sp.pi)), sp.pi)
gbar = sp.Matrix([sp.cos(t), -sp.sin(t)])
sym_equal("Example 35.3.6(a): clockwise = -pi",
          sp.integrate(ch35.pullback_top({(1,): x}, X2, gbar, [t]), (t, 0, 2 * sp.pi)), -sp.pi)
rr = sp.sqrt(x ** 2 + y ** 2)
dth = ch35.one_form([-y / rr ** 2, x / rr ** 2])
sym_equal("Example 35.3.6(b): int dtheta over the unit circle = 2 pi",
          sp.integrate(ch35.pullback_top(dth, X2, gam, [t]), (t, 0, 2 * sp.pi)), 2 * sp.pi)
hel = sp.Matrix([a_ * sp.cos(t), a_ * sp.sin(t), b_ * t])
alh = {(0,): -y, (1,): x, (2,): z}
ch_ = ch35.pullback_top(alh, X3, hel, [t])
sym_equal("Example 35.3.6(c): integrand a^2 + b^2 t", ch_, a_ ** 2 + b_ ** 2 * t)
val = sp.integrate(ch_, (t, 0, 2 * sp.pi))
sym_equal("Example 35.3.6(c): int = 2 pi a^2 + 2 pi^2 b^2", val, 2 * sp.pi * a_ ** 2 + 2 * sp.pi ** 2 * b_ ** 2)
close("Example 35.3.6(c): (a, b) = (1, 0.3): 2 pi + 0.18 pi^2 = 8.0597", float(val.subs({a_: 1, b_: sp.Rational(3, 10)})),
      8.0597, 5e-5)
# Non-example 35.3.7: the length does not change sign under reversal
L1 = sp.integrate(sp.sqrt(sp.simplify(sum(c_ ** 2 for c_ in gam.diff(t)))), (t, 0, 2 * sp.pi))
L2 = sp.integrate(sp.sqrt(sp.simplify(sum(c_ ** 2 for c_ in gbar.diff(t)))), (t, 0, 2 * sp.pi))
check("Non-example 35.3.7: length of the circle is 2 pi in both directions", L1 == L2 == 2 * sp.pi)

# ---------------------------------------------------------------------------
# Proposition 35.3.9: independence of the parametrization (exact, sympy)
phi = sp.Matrix([u, v, u * v + u ** 2])
om = {(1, 2): x, (0, 2): -y ** 2, (0, 1): 1 + z}           # x dy^dz + y^2 dz^dx + (1 + z) dx^dy
I0 = ch35.integrate_patch(om, X3, phi, [u, v], [(u, 0, 1), (v, 0, 1)])
phi_h = phi.subs({u: s, v: t ** 2}, simultaneous=True)    # h(s, t) = (s, t^2), det Dh = 2t > 0 inside
I1 = ch35.integrate_patch(om, X3, phi_h, [s, t], [(s, 0, 1), (t, 0, 1)])
phi_sw = phi.subs({u: t, v: s}, simultaneous=True)        # h(s, t) = (t, s), det Dh = -1
I2 = ch35.integrate_patch(om, X3, phi_sw, [s, t], [(s, 0, 1), (t, 0, 1)])
check("Proposition 35.3.9: det Dh > 0 keeps the integral", sp.simplify(I0 - I1) == 0)
check("Proposition 35.3.9: det Dh < 0 flips the sign", sp.simplify(I0 + I2) == 0 and I0 != 0)
# the integrand transforms by det Dh (proof)
cc = ch35.pullback_top(om, X3, phi, [u, v])
cc_h = ch35.pullback_top(om, X3, phi_h, [s, t])
sym_equal("Proposition 35.3.9 (proof): integrand of phi o h = det Dh * (integrand o h)", cc_h,
          2 * t * cc.subs({u: s, v: t ** 2}, simultaneous=True))

# ---------------------------------------------------------------------------
# Example 35.3.10 (sphere)
sph = sp.Matrix([r * sp.sin(th) * sp.cos(ph), r * sp.sin(th) * sp.sin(ph), r * sp.cos(th)])
xt, xp = sph.diff(th), sph.diff(ph)
sym_equal("Example 35.3.10: x_theta x x_phi = r^2 sin(theta) N", xt.cross(xp), r * sp.sin(th) * sph,
          {th: (0.05, 3.1)})
rho = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
sigma = ch35.two_form_r3([x / rho, y / rho, z / rho])
cs = sp.simplify(ch35.pullback_top(sigma, X3, sph, [th, ph]).subs(sp.sqrt(r ** 2 * sp.sin(th) ** 2), r * sp.sin(th)))
sym_equal("Example 35.3.10(a): sigma(x_theta, x_phi) = r^2 sin(theta)", cs, r ** 2 * sp.sin(th), {th: (0.05, 3.1)})
sym_equal("Example 35.3.10(a): int sigma = 4 pi r^2",
          sp.integrate(r ** 2 * sp.sin(th), (th, 0, sp.pi), (ph, 0, 2 * sp.pi)), 4 * sp.pi * r ** 2)
csh = ch35.pullback_top({(0, 1): 1}, X3, sph, [th, ph])
sym_equal("Example 35.3.10(b): (dx^dy)(x_theta, x_phi) = r^2 sin cos", csh, r ** 2 * sp.sin(th) * sp.cos(th))
sym_equal("Example 35.3.10(b): int over the upper hemisphere = pi r^2",
          sp.integrate(csh, (th, 0, sp.pi / 2), (ph, 0, 2 * sp.pi)), sp.pi * r ** 2)
sym_equal("Example 35.3.10(b): int over the whole sphere = 0", sp.integrate(csh, (th, 0, sp.pi), (ph, 0, 2 * sp.pi)), 0)
h12 = np.pi / 12
cellv = [(np.pi / 6) * 0.5 * (np.sin((i + 1) * h12) ** 2 - np.sin(i * h12) ** 2) for i in (1, 3, 5)]
cella = [(np.pi / 6) * (np.cos(i * h12) - np.cos((i + 1) * h12)) for i in (1, 3, 5)]
close("Figure 35.3.2 / text: shadow areas 0.0479, 0.0654, 0.0175", cellv, [0.0479, 0.0654, 0.0175], 5e-5)
close("Figure 35.3.2 / text: sphere areas of the cells 0.052, 0.108, 0.136", cella, [0.052, 0.108, 0.136], 6e-4)

# ---------------------------------------------------------------------------
# Lemma 35.3.11: vol(Delta^k) = 1/k!
ts = sp.symbols("t1:6", nonnegative=True)
for k in range(1, 5):
    expr = sp.Integer(1)
    # integrate t_k first over [0, 1 - t_1 - ... - t_{k-1}], then the rest
    lims = []
    for i in range(k - 1, -1, -1):
        lims.append((ts[i], 0, 1 - sum(ts[:i])))
    vol = sp.integrate(expr, *lims)
    check(f"Lemma 35.3.11: vol(Delta^{k}) = 1/{k}!", vol == sp.Rational(1, math.factorial(k)))

# Proposition 35.3.12(a): constant forms on simplices in R^4 (exact integral of the constant integrand)
for k in (1, 2, 3):
    Xs = rng.normal(size=(k + 1, 4))
    A = rng.normal(size=(k, 4))            # omega = alpha^1 ^ ... ^ alpha^k (a generic decomposable constant form)
    omega_val = np.linalg.det(A @ (Xs[1:] - Xs[0]).T)
    # Definition 35.3.8 with phi_sigma: integrand constant = omega(x_1 - x_0, ...); integral = integrand * vol
    close(f"Proposition 35.3.12(a): k = {k}: int = omega(...)/k! (constant integrand times vol Delta^k)",
          omega_val * float(sp.Rational(1, math.factorial(k))), omega_val / math.factorial(k), 1e-12)

# (b) alternation under all permutations of the vertices
for k in (2, 3):
    Xs = rng.normal(size=(k + 1, 4))
    A = rng.normal(size=(k, 4))
    F = lambda pts: np.linalg.det(A @ (pts[1:] - pts[0]).T)  # noqa: E731
    ok = all(np.isclose(F(Xs[list(p)]), ch35.perm_sign(p) * F(Xs)) for p in itertools.permutations(range(k + 1)))
    check(f"Proposition 35.3.12(b): F(x_pi) = sgn(pi) F(x) for all {math.factorial(k + 1)} permutations (k = {k})", ok)

# (c) permuted parametrizations of a triangle: det Dh = sgn(pi), and integrals of a non-constant form
V3 = np.array([[1.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
wfun = lambda p: np.array([p[1] * p[2], p[0] ** 2, p[0] * p[1]])  # noqa: E731
ref = dd.derham2(wfun, V3, np.array([[0, 1, 2]]))[0]
for p in itertools.permutations(range(3)):
    val_p = dd.derham2(wfun, V3[list(p)], np.array([[0, 1, 2]]))[0]
    check(f"Proposition 35.3.12(c): triangle vertices in order {p}: integral = sgn * reference",
          np.isclose(val_p, ch35.perm_sign(p) * ref))
    # the affine map h with phi_{sigma pi} = phi_sigma o h has det = sgn(pi)
    E0 = (V3[1:] - V3[0]).T
    Ep = (V3[list(p)][1:] - V3[list(p)][0]).T
    Dh = np.linalg.lstsq(E0, Ep, rcond=None)[0]
    check(f"Proposition 35.3.12(c): det Dh = sgn(pi) for {p}", np.isclose(np.linalg.det(Dh), ch35.perm_sign(p)))

# Example 35.3.13
T2 = np.array([[0.0, 0.0], [2.0, 0.5], [0.6, 1.6]])
close("Example 35.3.13(a): int_sigma dx^dy = 1.45", dd.derham2(lambda p: 1.0, T2, np.array([[0, 1, 2]]))[0], 1.45, 1e-12)
close("Example 35.3.13(a): int (dx + dy) on the first edge = 2.5", dd.derham1(lambda p: np.array([1.0, 1.0]), T2,
                                                                           np.array([[0, 1]]))[0], 2.5, 1e-12)
crs = np.cross(V3[1] - V3[0], V3[2] - V3[0])
check("Example 35.3.13(b): (x1 - x0) x (x2 - x0) = (2, 1, 2)", np.allclose(crs, [2, 1, 2]))
close("Example 35.3.13(b): flux of w = (1,1,1) = 2.5", dd.derham2(lambda p: np.ones(3), V3, np.array([[0, 1, 2]]))[0],
      2.5, 1e-12)
close("Example 35.3.13(b): int dx^dy = shadow area 1", dd.derham2(lambda p: np.array([0, 0, 1.0]), V3,
                                                                  np.array([[0, 1, 2]]))[0], 1.0, 1e-12)
close("Example 35.3.13(c): int x dy over [(0,0),(1,1)] = 1/2",
      dd.derham1(lambda p: np.array([0.0, p[0]]), np.array([[0.0, 0], [1, 1]]), np.array([[0, 1]]))[0], 0.5, 1e-12)

# ---------------------------------------------------------------------------
# Implementation code-35-3-1
ns = ch35.run_block(PAGE, "code-35-3-1")
close("Implementation code-35-3-1: I_helix = 2 pi + 0.18 pi^2", ns["I_helix"], 2 * np.pi + 0.18 * np.pi ** 2, 1e-11)
close("Implementation code-35-3-1: area = 4 pi", ns["area"], 4 * np.pi, 1e-11)
close("Implementation code-35-3-1: shadow = pi", ns["shadow"], np.pi, 1e-11)
close("Text: I_helix = 8.0597", ns["I_helix"], 8.0597, 5e-5)
# against the independent numeric line integral of lib/ch35
ref_h = ch35.line_integral_num(lambda p: np.array([-p[1], p[0], p[2]]),
                               lambda q: np.array([np.cos(q), np.sin(q), 0.3 * q]),
                               lambda q: np.array([-np.sin(q), np.cos(q), 0.3]), 0, 2 * np.pi)
close("Implementation code-35-3-1: I_helix = lib line integral", ns["I_helix"], ref_h, 1e-11)
check("Implementation box code-35-3-1: at most 30 lines", ch35.code_lines(PAGE, "code-35-3-1") <= 30)

# ---------------------------------------------------------------------------
# Exercises
ell = sp.Matrix([2 * sp.cos(t), sp.sin(t)])
al = {(1,): x / 2, (0,): -y / 2}
sym_equal("Exercise 35.3.1: integrand = 1", ch35.pullback_top(al, X2, ell, [t]), 1)
sym_equal("Exercise 35.3.1: shifted start, same value 2 pi",
          sp.integrate(ch35.pullback_top(al, X2, ell.subs(t, t + sp.pi), [t]), (t, 0, 2 * sp.pi)), 2 * sp.pi)
sym_equal("Exercise 35.3.1: reversed = -2 pi",
          sp.integrate(ch35.pullback_top(al, X2, ell.subs(t, -t), [t]), (t, 0, 2 * sp.pi)), -2 * sp.pi)
check("Exercise 35.3.2(d): cyclic permutation (1, 2, 0) is even", ch35.perm_sign((1, 2, 0)) == 1)
cyl = sp.Matrix([r * sp.cos(u), r * sp.sin(u), v])
sym_equal("Exercise 35.3.3(a): flux of (x, y, 0) through the cylinder = 2 pi r^2",
          ch35.integrate_patch(ch35.two_form_r3([x, y, 0]), X3, cyl, [u, v], [(u, 0, 2 * sp.pi), (v, 0, 1)]),
          2 * sp.pi * r ** 2)
check("Exercise 35.3.3(b): (dx^dy)(x_u, x_v) = 0 on the cylinder", ch35.pullback_top({(0, 1): 1}, X3, cyl, [u, v]) == 0)
# Exercise 35.3.4: C_id maps to Delta^k by a unipotent lower-triangular map
for k in (2, 3, 4):
    M = sp.eye(k) - sp.Matrix(k, k, lambda i, j: 1 if i == j + 1 else 0)
    check(f"Exercise 35.3.4: k = {k}: the map t -> s has det 1", M.det() == 1)
pts = rng.uniform(size=(20000, 3))
frac = np.mean(np.all(np.diff(pts, axis=1) >= 0, axis=1))
close("Exercise 35.3.4: C_id has volume 1/3! (Monte Carlo, 20000 points)", frac, 1 / 6, 0.01)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch35.run_block(PAGE, "code-ex-35-3-5", dict(ns))
check("Exercise 35.3.5: output equals out-ex-35-3-5", buf.getvalue().strip() == ch35.output_block(PAGE, "out-ex-35-3-5"),
      buf.getvalue())
check("Exercise 35.3.5: printed values", buf.getvalue().strip() == "2.5 [-1.  2. -1.] True True", buf.getvalue())

summary()
