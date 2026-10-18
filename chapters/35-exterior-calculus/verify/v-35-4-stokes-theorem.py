"""Section 35.4 (Stokes's Theorem): verification of every computation (GUIDELINES 13).

Stokes's theorem is checked exactly (sympy) on boxes, on the standard simplex, on triangles and parametrized patches,
and numerically on meshes (tools/dgdec.py); the orientation rules are checked against tools/dgdec.induced_boundary.
Run from the project root: PYTHONPATH=tools python3 chapters/35-exterior-calculus/verify/v-35-4-stokes-theorem.py
"""
import contextlib
import io
import itertools
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from dgcheck import check, close, sym_equal, summary  # noqa: E402
import dgdec as dd  # noqa: E402
import ch35  # noqa: E402

PAGE = "35-4-stokes-theorem.html"
x, y, z, w = sp.symbols("x y z w", real=True)
s, t, u, v, r, th, ph = sp.symbols("s t u v r theta phi", real=True)
X2, X3 = [x, y], [x, y, z]
rng = np.random.default_rng(354)

# ---------------------------------------------------------------------------
# Example 35.4.2(b): det(n, J n) = 1
J = sp.Matrix([[0, -1], [1, 0]])
a_ = sp.symbols("a", real=True)
n_ = sp.Matrix([sp.cos(a_), sp.sin(a_)])
check("Example 35.4.2(b): det(n, J n) = 1 and J(J n) = -n",
      sp.simplify(sp.Matrix.hstack(n_, J * n_).det()) == 1 and sp.simplify(J * J * n_ + n_) == sp.zeros(2, 1))

# Proposition 35.4.3: det(e_i, e_1, ..., e_i-hat, ..., e_n) = (-1)^(i-1)
for n in range(2, 6):
    ok = True
    for i in range(n):
        cols = [i] + [j for j in range(n) if j != i]
        ok &= np.isclose(np.linalg.det(np.eye(n)[:, cols]), (-1) ** i)
    check(f"Proposition 35.4.3: face signs (-1)^(i-1) for n = {n}", ok)

# Proposition 35.4.4: outward normal first = (-1)^m on random positively oriented simplices
for k in range(1, 5):
    ok = True
    for _ in range(20):
        Vt = rng.normal(size=(k + 1, k))
        if np.linalg.det((Vt[1:] - Vt[0]).T) < 0:
            Vt = Vt[[1, 0] + list(range(2, k + 1))]          # swapping v_0 and v_1 flips the orientation
        assert np.linalg.det((Vt[1:] - Vt[0]).T) > 0
        for m in range(k + 1):
            face = [i for i in range(k + 1) if i != m]
            cen = Vt[face].mean(0)
            # outward vector: normal of the face hyperplane pointing away from v_m
            if k == 1:
                nvec = Vt[face[0]] - Vt[m]                       # a point gets + if the outward direction is positive
                ok &= np.sign(nvec[0]) == (-1) ** m
                continue
            Ed = (Vt[face[1:]] - Vt[face[0]]).T                         # k x (k-1) edge vectors of the face
            Qm, _ = np.linalg.qr(np.column_stack([Ed, rng.normal(size=k)]))
            nvec = Qm[:, -1]
            if np.dot(nvec, Vt[m] - cen) > 0:
                nvec = -nvec
            ok &= np.sign(np.linalg.det(np.column_stack([nvec, Ed]))) == (-1) ** m
    check(f"Proposition 35.4.4: Stokes orientation of the face opposite v_m is (-1)^m (k = {k}, 20 random simplices)", ok)
    check(f"Proposition 35.4.4: agrees with dgdec.induced_boundary (k = {k})",
          [sg for sg, _ in dd.induced_boundary(tuple(range(k + 1)))] == [(-1) ** m for m in range(k + 1)])

# ---------------------------------------------------------------------------
# Theorem 35.4.8 (box), exact: R^2 and R^3


def box_boundary_integral(form, xs, bounds):
    """(35.4.1): sum_i (-1)^(i-1) (int_{psi_i^+} - int_{psi_i^-})."""
    n = len(xs)
    tot = 0
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for c, sgn in [(bounds[i][1], 1), (bounds[i][0], -1)]:
            phi = sp.Matrix([xs[j] if j != i else c for j in range(n)])
            params = [xs[j] for j in others]
            val = ch35.integrate_patch(form, xs, phi, params, [(xs[j], bounds[j][0], bounds[j][1]) for j in others])
            tot += (-1) ** i * sgn * val
    return sp.simplify(tot)


om2 = {(0,): x * y ** 2 + sp.sin(y), (1,): x ** 3 * y}
b2 = [(0, 1), (-1, 2)]
lhs = sp.integrate(ch35.d(om2, X2).get((0, 1), 0), (x, 0, 1), (y, -1, 2))
sym_equal("Theorem 35.4.8: box in R^2", lhs, box_boundary_integral(om2, X2, b2))
om3 = {(1, 2): x ** 2 * y, (0, 2): y * z + x, (0, 1): sp.exp(z) * x}
b3 = [(0, 1), (0, 2), (-1, 1)]
d3 = ch35.d(om3, X3).get((0, 1, 2), 0)
lhs3 = sp.integrate(d3, (x, 0, 1), (y, 0, 2), (z, -1, 1))
sym_equal("Theorem 35.4.8: box in R^3", lhs3, box_boundary_integral(om3, X3, b3))

# Theorem 35.4.9 (standard simplex), exact in R^2 and R^3
def simplex_param(verts, ts):
    verts = [sp.Matrix(vv) for vv in verts]
    return verts[0] + sum((ti * (verts[i + 1] - verts[0]) for i, ti in enumerate(ts)), sp.zeros(len(verts[0]), 1))


def int_over_standard_simplex(expr, ts):
    lims = []
    for i in range(len(ts) - 1, -1, -1):
        lims.append((ts[i], 0, 1 - sum(ts[:i])))
    return sp.integrate(expr, *lims)


for n, xs, form in [(2, X2, {(0,): x ** 2 * y + y, (1,): x * y ** 3}),
                    (3, X3, {(1, 2): x * y * z, (0, 2): x ** 2 + z, (0, 1): y ** 2 * z})]:
    E = [[0] * n] + [list(row) for row in np.eye(n, dtype=int)]
    tt = sp.symbols(f"q1:{n + 1}", real=True)
    lhs = int_over_standard_simplex(ch35.d(form, xs).get(tuple(range(n)), 0).subs(dict(zip(xs, tt))), tt)
    rhs = 0
    for m in range(n + 1):
        face = [E[i] for i in range(n + 1) if i != m]
        tf = sp.symbols(f"f1:{n}", real=True)
        phi = simplex_param(face, tf)
        c = ch35.pullback_top(form, xs, phi, list(tf))
        rhs += (-1) ** m * int_over_standard_simplex(c, tf)
    sym_equal(f"Theorem 35.4.9: standard simplex in R^{n}", lhs, sp.simplify(rhs))

# Lemma 35.4.10 (generic coefficients, polynomial patch in R^3)
A = [sp.Function(f"A{i}")(*X3) for i in range(3)]
alpha = ch35.one_form(A)
phi = sp.Matrix([s + t ** 2, s * t, s ** 2 - t])
sub = dict(zip(X3, list(phi)))
a_s = ch35.evaluate(ch35.subs(alpha, sub), [phi.diff(s)])
a_t = ch35.evaluate(ch35.subs(alpha, sub), [phi.diff(t)])
lhs = sp.diff(a_t, s) - sp.diff(a_s, t)
rhs = ch35.evaluate(ch35.subs(ch35.d(alpha, X3), sub), [phi.diff(s), phi.diff(t)])
sym_equal("Lemma 35.4.10: d_s[alpha(phi_t)] - d_t[alpha(phi_s)] = (d alpha)(phi_s, phi_t)", sp.expand(lhs - rhs), 0)

# Corollary 35.4.11 on a triangle in R^3 (exact)
xv = [sp.Matrix([1, 0, 0]), sp.Matrix([0, 2, 0]), sp.Matrix([0, 0, 1])]
al3 = {(0,): y * z, (1,): x ** 2, (2,): x * y + z}
tri = simplex_param(xv, [s, t])
lhs = int_over_standard_simplex(ch35.pullback_top(ch35.d(al3, X3), X3, tri, [s, t]), [s, t])
edges = 0
for i, j in [(0, 1), (1, 2), (2, 0)]:
    seg = xv[i] + t * (xv[j] - xv[i])
    edges += sp.integrate(ch35.pullback_top(al3, X3, seg, [t]), (t, 0, 1))
sym_equal("Corollary 35.4.11 (35.4.4): int_sigma d alpha = sum of the three edge integrals (triangle in R^3)", lhs, edges)
# a non-injective C^2 patch over a rectangle (polar map on [0, 1] x [0, 2 pi]) with alpha = x dy
pol = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
lhs = ch35.integrate_patch({(0, 1): 1}, X2, pol, [r, th], [(r, 0, 1), (th, 0, 2 * sp.pi)])
rhs = sum(sp.integrate(ch35.pullback_top({(1,): x}, X2, pol.subs(sb), [pv]), lim) for sb, pv, lim in [
    ({th: 0}, r, (r, 0, 1)), ({r: 1}, th, (th, 0, 2 * sp.pi)), ({th: 2 * sp.pi}, r, (r, 1, 0)), ({r: 0}, th, (th, 2 * sp.pi, 0))])
check("Example 35.4.13(a): disk via the polar rectangle: int dx^dy = pi = sum over the rectangle's edges",
      sp.simplify(lhs - sp.pi) == 0 and sp.simplify(rhs - sp.pi) == 0)

# ---------------------------------------------------------------------------
# Proposition 35.4.12
# (b) plane divergence theorem on the unit disk for X = (x^3, y)
Xd = [x ** 3, y]
lhs = ch35.integrate_patch({(0, 1): ch35.div(Xd, X2)}, X2, pol, [r, th], [(r, 0, 1), (th, 0, 2 * sp.pi)])
circ = sp.Matrix([sp.cos(th), sp.sin(th)])
flux = sp.integrate((sp.Matrix(Xd).subs({x: sp.cos(th), y: sp.sin(th)}).dot(circ)), (th, 0, 2 * sp.pi))
sym_equal("Proposition 35.4.12(b): int_disk div X = boundary flux (X = (x^3, y))", lhs, flux)
sym_equal("Proposition 35.4.12(b): star X^flat on the circle = <X, n>|gamma'|",
          ch35.pullback_top(ch35.hodge(ch35.flat(Xd), 2), X2, circ, [th]),
          sp.Matrix(Xd).subs({x: sp.cos(th), y: sp.sin(th)}).dot(circ))
# (d) classical Stokes on the upper hemisphere, X = (-y, x, 0): both sides 2 pi
sph = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
Xr = [-y, x, 0]
lhs = ch35.integrate_patch(ch35.two_form_r3(ch35.curl(Xr, X3)), X3, sph, [th, ph], [(th, 0, sp.pi / 2), (ph, 0, 2 * sp.pi)])
eq_ = sph.subs(th, sp.pi / 2)
rhs = sp.integrate(ch35.pullback_top(ch35.flat(Xr), X3, eq_, [ph]), (ph, 0, 2 * sp.pi))
check("Proposition 35.4.12(d): hemisphere, X = (-y, x, 0): int <curl X, N> dA = circulation on the equator = 2 pi",
      sp.simplify(lhs - 2 * sp.pi) == 0 and sp.simplify(rhs - 2 * sp.pi) == 0)

# Example 35.4.13
# (b) inscribed polygon: sum of exact edge integrals of x dy = (n/2) sin(2 pi / n); S_n difference = 0
for n in (5, 8, 13):
    tt = 2 * np.pi * np.arange(n + 1) / n
    P = np.stack([np.cos(tt), np.sin(tt)], 1)
    exact = float(np.sum(0.5 * (P[:-1, 0] + P[1:, 0]) * (P[1:, 1] - P[:-1, 1])))
    Sn = float(np.sum(P[:-1, 0] * (P[1:, 1] - P[:-1, 1])))
    close(f"Example 35.4.13(b): n = {n}: edge integrals of x dy = (n/2) sin(2 pi/n)", exact,
          n / 2 * np.sin(2 * np.pi / n), 1e-12)
    close(f"Example 35.4.13(b): n = {n}: (1/2) sum dx dy = 0, so S_n = polygon area", Sn, exact, 1e-12)
    close(f"Example 35.4.13(b): n = {n}: polygon area by derham2 over the fan", float(dd.derham2(
        lambda p: 1.0, np.vstack([P[:-1], [[0, 0]]]), np.array([[n, i, (i + 1) % n] for i in range(n)])).sum()),
        exact, 1e-12)
# (c) annulus
dth = ch35.one_form([-y / (x ** 2 + y ** 2), x / (x ** 2 + y ** 2)])
outer = sp.integrate(ch35.pullback_top(dth, X2, 2 * circ, [th]), (th, 0, 2 * sp.pi))
inner = -sp.integrate(ch35.pullback_top(dth, X2, circ, [th]), (th, 0, 2 * sp.pi))
check("Example 35.4.13(c): annulus, d theta: 2 pi - 2 pi = 0", sp.simplify(outer - 2 * sp.pi) == 0 and sp.simplify(inner + 2 * sp.pi) == 0)
rdth = ch35.one_form([-y / sp.sqrt(x ** 2 + y ** 2), x / sp.sqrt(x ** 2 + y ** 2)])
lhs = ch35.integrate_patch(ch35.d(rdth, X2), X2, pol, [r, th], [(r, 1, 2), (th, 0, 2 * sp.pi)])
bdy = sp.integrate(ch35.pullback_top(rdth, X2, 2 * circ, [th]), (th, 0, 2 * sp.pi)) - \
    sp.integrate(ch35.pullback_top(rdth, X2, circ, [th]), (th, 0, 2 * sp.pi))
check("Example 35.4.13(c): annulus, r dtheta: int dr^dtheta = 2 pi = 4 pi - 2 pi",
      sp.simplify(lhs - 2 * sp.pi) == 0 and sp.simplify(bdy - 2 * sp.pi) == 0)
# (d) unit cube
Xc = [x ** 2, y * z, x * z]
check("Example 35.4.13(d): div X = 3x + z, integral 2", sp.simplify(ch35.div(Xc, X3) - 3 * x - z) == 0
      and sp.integrate(ch35.div(Xc, X3), (x, 0, 1), (y, 0, 1), (z, 0, 1)) == 2)
check("Example 35.4.13(d): boundary flux = 2 by (35.4.1)",
      box_boundary_integral(ch35.hodge(ch35.flat(Xc), 3), X3, [(0, 1)] * 3) == 2)
check("Example 35.4.13(d): face fluxes 1, 1/2, 1/2 (x = 1, y = 1, z = 1)",
      sp.integrate(Xc[0].subs(x, 1), (y, 0, 1), (z, 0, 1)) == 1
      and sp.integrate(Xc[1].subs(y, 1), (x, 0, 1), (z, 0, 1)) == sp.Rational(1, 2)
      and sp.integrate(Xc[2].subs(z, 1), (x, 0, 1), (y, 0, 1)) == sp.Rational(1, 2))
# (e) Coulomb 2-form: closed, int over S^2 = 4 pi; int over S^2 of an exact form d eta = 0 (two hemispheres)
rho = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
om = ch35.two_form_r3([x / rho ** 3, y / rho ** 3, z / rho ** 3])
check("Example 35.4.13(e): omega is closed", ch35.d(om, X3) == {})
cS = sp.simplify(ch35.pullback_top(om, X3, sph, [th, ph]))
sym_equal("Example 35.4.13(e): integrand on the unit sphere = sin(theta)", cS, sp.sin(th), {th: (0.05, 3.1)})
check("Example 35.4.13(e): int_{S^2} omega = 4 pi", sp.integrate(sp.sin(th), (th, 0, sp.pi), (ph, 0, 2 * sp.pi)) == 4 * sp.pi)
eta = {(0,): x * y * z + y, (1,): sp.sin(z) * x, (2,): x ** 2 - y}
deta = ch35.d(eta, X3)
halves = [ch35.integrate_patch(deta, X3, sph, [th, ph], [(th, a, b), (ph, 0, 2 * sp.pi)])
          for a, b in [(0, sp.pi / 2), (sp.pi / 2, sp.pi)]]
equator = sp.integrate(ch35.pullback_top(eta, X3, eq_, [ph]), (ph, 0, 2 * sp.pi))
hv = [float(sp.N(h_)) for h_ in halves]                    # sympy leaves an elementary-free integral unevaluated
check("Example 35.4.13(e): upper half of int d eta = equator integral, lower half = minus it, total 0",
      abs(hv[0] - float(equator)) < 1e-10 and abs(hv[1] + float(equator)) < 1e-10 and abs(float(equator)) > 1)

# ---------------------------------------------------------------------------
# Proposition 35.4.14 and Figure 35.4.2: circulation density of r dtheta at p = (1, 0.5)
p0 = np.array([1.0, 0.5])
al_num = lambda q: np.array([-q[1], q[0]]) / np.hypot(*q)  # noqa: E731
ratios = []
for e in [0.8, 0.4, 0.2, 0.1, 0.05, 0.025]:
    c = [p0, p0 + [e, 0], p0 + [e, e], p0 + [0, e], p0]
    L = sum(ch35.line_integral_num(al_num, lambda q, a=a, b=b: a + q * (b - a), lambda q, a=a, b=b: b - a, 0, 1, m=30)
            for a, b in zip(c[:-1], c[1:]))
    tq, wq = ch35.gauss01(30)
    Aint = sum(wi * wj * e ** 2 / np.hypot(p0[0] + e * ti, p0[1] + e * tj) for ti, wi in zip(tq, wq) for tj, wj in zip(tq, wq))
    check(f"Figure 35.4.2: eps = {e}: boundary integral = integral of d alpha (Corollary 35.4.11)", abs(L - Aint) < 1e-12)
    ratios.append(L / e ** 2)
close("Figure 35.4.2: ratios 0.607, 0.722, 0.799, 0.844, 0.868, 0.881", ratios,
      [0.607, 0.722, 0.799, 0.844, 0.868, 0.881], 6e-4)
close("Figure 35.4.2: limit 1/|p| = 0.894", 1 / np.linalg.norm(p0), 0.894, 5e-4)
errs = np.abs(np.array(ratios) - 1 / np.linalg.norm(p0))
check("Figure 35.4.2: error roughly halves with eps (O(eps))", np.all(np.abs(errs[1:] / errs[:-1] - 0.5) < 0.15))
# symbolic: (d alpha)_p(u, v) for alpha = X^flat in R^3 is <curl X, u x v>
Xg = [sp.Function(f"X{i}")(*X3) for i in range(3)]
uu, vv = sp.Matrix(sp.symbols("u1:4", real=True)), sp.Matrix(sp.symbols("v1:4", real=True))
sym_equal("Text: d(X^flat)(u, v) = <curl X, u x v>", ch35.evaluate(ch35.d(ch35.flat(Xg), X3), [uu, vv]),
          sp.Matrix(ch35.curl(Xg, X3)).dot(uu.cross(vv)))


# Derivation: boundary of the boundary of a box cancels (n = 2, 3, 4), using the face signs of Proposition 35.4.3
def box_faces_signed(coords):
    """faces of the box on 'coords' (list of free coordinate indices): (sign, fixed (index, side), remaining coords)."""
    out = []
    for pos, i in enumerate(coords):
        rest = [j for j in coords if j != i]
        out.append(((-1) ** pos, (i, 1), rest))
        out.append(((-1) ** (pos + 1), (i, 0), rest))
    return out


for n in (2, 3, 4):
    total = {}
    for s1, f1, rest1 in box_faces_signed(list(range(n))):
        for s2, f2, rest2 in box_faces_signed(rest1):
            key = (frozenset([f1, f2]), tuple(rest2))
            total[key] = total.get(key, 0) + s1 * s2
    check(f"Derivation: every (n-2)-face of the {n}-box appears twice with opposite signs",
          all(c_ == 0 for c_ in total.values()) and len(total) == 4 * n * (n - 1) // 2)
# the explicit sign bookkeeping written in the Derivation (reviewer): via F_i^s the common face gets
# s(-1)^(i-1) s'(-1)^(j-2), via F_j^s' it gets s'(-1)^(j-1) s(-1)^(i-1); 1-based i < j
check("Derivation (text): the two induced signs of {x^i = c_i, x^j = c_j} have ratio -1 (n <= 6)",
      all(s * (-1) ** (i - 1) * t * (-1) ** (j - 2) == -(t * (-1) ** (j - 1) * s * (-1) ** (i - 1))
          for n in range(2, 7) for i in range(1, n + 1) for j in range(i + 1, n + 1) for s in (1, -1) for t in (1, -1)))

# ---------------------------------------------------------------------------
# Implementation code-35-4-1
ns = ch35.run_block(PAGE, "code-35-4-1")
check("Implementation code-35-4-1: Stokes on every triangle (< 1e-12)", ns["per_triangle"] < 1e-12, str(ns["per_triangle"]))
check("Implementation code-35-4-1: Green on the square (< 1e-12)", ns["whole"] < 1e-12, str(ns["whole"]))
check("Implementation code-35-4-1: 50 triangles, boundary loop of 20 vertices", len(ns["F"]) == 50 and len(ns["loop"]) == 20)
check("Implementation box code-35-4-1: at most 30 lines", ch35.code_lines(PAGE, "code-35-4-1") <= 30)

# ---------------------------------------------------------------------------
# Exercises
ex1 = {(0,): x * y ** 2, (1,): x ** 3}
check("Exercise 35.4.1: int_Q d alpha = 1/2 = boundary integral",
      sp.integrate(ch35.d(ex1, X2)[(0, 1)], (x, 0, 1), (y, 0, 1)) == sp.Rational(1, 2)
      and box_boundary_integral(ex1, X2, [(0, 1), (0, 1)]) == sp.Rational(1, 2))
e1, e2, e3 = np.eye(3)
check("Exercise 35.4.2: (e2 - e1) x (e3 - e1) = (1, 1, 1)", np.allclose(np.cross(e2 - e1, e3 - e1), [1, 1, 1]))
check("Exercise 35.4.2: e2 x e1 = -e3", np.allclose(np.cross(e2, e1), -e3))
check("Exercise 35.4.2: boundary signs of the tetrahedron", [sg for sg, _ in dd.induced_boundary((0, 1, 2, 3))] == [1, -1, 1, -1])
R_ = sp.symbols("R", positive=True)
sphR = R_ * sph
check("Exercise 35.4.3: flux of (x, y, z) through S^2(r) = 4 pi r^3",
      sp.simplify(ch35.integrate_patch(ch35.two_form_r3(X3), X3, sphR, [th, ph], [(th, 0, sp.pi), (ph, 0, 2 * sp.pi)])
                  - 4 * sp.pi * R_ ** 3) == 0)
# Exercise 35.4.4: shoelace on a random convex polygon vs fan triangulation
ang = np.sort(rng.uniform(0, 2 * np.pi, 9))
Pp = np.stack([np.cos(ang) * 1.3, np.sin(ang)], 1)
shoe = 0.5 * np.sum(Pp[:, 0] * np.roll(Pp[:, 1], -1) - np.roll(Pp[:, 0], -1) * Pp[:, 1])
fan = sum(0.5 * np.linalg.det(np.column_stack([Pp[i] - Pp[0], Pp[i + 1] - Pp[0]])) for i in range(1, 8))
close("Exercise 35.4.4(b): shoelace = area of the fan triangulation", shoe, fan, 1e-12)
ell = sp.Matrix([2 * sp.cos(th), sp.sin(th)])
check("Exercise 35.4.4(c): ellipse area 2 pi", sp.integrate(ch35.pullback_top({(1,): x / 2, (0,): -y / 2}, X2, ell, [th]),
                                                            (th, 0, 2 * sp.pi)) == 2 * sp.pi)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch35.run_block(PAGE, "code-ex-35-4-6", dict(ns))
check("Exercise 35.4.6: output equals out-ex-35-4-6", buf.getvalue().strip() == ch35.output_block(PAGE, "out-ex-35-4-6")
      == "True 6.2831853072 -6.2831853072", buf.getvalue())

summary()
