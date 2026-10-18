"""Section 35.5 (Inner Products and Vector-Valued Forms): verification of every computation (GUIDELINES 13).

The codifferential sign (-1)^(n(k+1)+1) is checked by exact L2 adjointness on unit cubes (forms vanishing on the
boundary), Crane's unsigned delta = star d star is compared, and vector-valued wedges are checked numerically and
symbolically.
Run from the project root:
PYTHONPATH=tools python3 chapters/35-exterior-calculus/verify/v-35-5-inner-products-vector-valued-forms.py
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
import ch35  # noqa: E402

PAGE = "35-5-inner-products-vector-valued-forms.html"
x, y, z, w = sp.symbols("x y z w", real=True)
r, th, ph, u, v, R_ = sp.symbols("r theta phi u v R", real=True)
rp = sp.symbols("rp", positive=True)
X2, X3, X4 = [x, y], [x, y, z], [x, y, z, w]
rng = np.random.default_rng(355)


def L2(a, b, xs):
    return sp.integrate(sum(a.get(I, 0) * b.get(I, 0) for I in set(a) | set(b)), *[(q, 0, 1) for q in xs])


def gen(n, k, xs, tag):
    return {I: sp.Function(tag + "".join(map(str, I)))(*xs) for I in ch35.increasing(n, k)}


def rand_poly(xs, deg=2):
    mons = [sp.Mul(*[q ** e for q, e in zip(xs, es)]) for es in itertools.product(range(deg + 1), repeat=len(xs))
            if sum(es) <= deg]
    return sum(int(c) * m for c, m in zip(rng.integers(-3, 4, len(mons)), mons))


# ---------------------------------------------------------------------------
# Proposition 35.5.1: alpha ^ star beta = <alpha, beta> dV (dict model and tensor model)
for n, xs in [(2, X2), (3, X3), (4, X4)]:
    for k in range(0, n + 1):
        a, b = gen(n, k, xs, "a"), gen(n, k, xs, "b")
        lhs = ch35.wedge(a, ch35.hodge(b, n))
        ip = sum(a[I] * b[I] for I in a)
        ok = ch35.equal(lhs, {tuple(range(n)): ip}) and ch35.equal(ch35.wedge(b, ch35.hodge(a, n)), lhs)
        check(f"Proposition 35.5.1: alpha ^ star beta = beta ^ star alpha = <alpha, beta> dV (k = {k}, R^{n})", ok)
# tensor-model cross-check of one case
a, b = gen(3, 1, X3, "s"), gen(3, 2, X3, "t")
Tw = ch35.wedge_tensor(ch35.to_tensor(a, 3, 1), ch35.to_tensor(b, 3, 2), 3, 1, 2)
check("Proposition 35.5.1: wedge (dict) = wedge (tensor model) for a 1-form and a 2-form",
      ch35.equal(ch35.from_tensor(Tw, 3, 3), ch35.wedge(a, b)))

# Example 35.5.4
pol = sp.Matrix([rp * sp.cos(th), rp * sp.sin(th)])
J = rp


def disk_int(expr, r0=0, r1=1):
    return sp.integrate(sp.simplify(expr.subs({x: rp * sp.cos(th), y: rp * sp.sin(th)}) * J), (rp, r0, r1),
                        (th, 0, 2 * sp.pi))


sym_equal("Example 35.5.4(a): ||x dy||^2 on the unit disk = pi/4", disk_int(x ** 2), sp.pi / 4)
check("Example 35.5.4(a): <<dx, dy>> = 0 (pointwise 0)", sum(p * q for p, q in [(1, 0), (0, 1)]) == 0)
df = ch35.d({(): x ** 2 - y ** 2}, X2)
sym_equal("Example 35.5.4(b): ||df||^2 = 2 pi", disk_int(sum(c ** 2 for c in df.values())), 2 * sp.pi)
rr = sp.sqrt(x ** 2 + y ** 2)
dth = ch35.one_form([-y / rr ** 2, x / rr ** 2])
drf = ch35.one_form([x / rr, y / rr])
sym_equal("Example 35.5.4(c): ||dtheta||^2 on the annulus = 2 pi log 2",
          disk_int(sum(c ** 2 for c in dth.values()), 1, 2), 2 * sp.pi * sp.log(2))
sym_equal("Example 35.5.4(c): <dr, dtheta> = 0", sum(dth[I] * drf[I] for I in dth), 0)
sym_equal("Example 35.5.4(c): |star dtheta|^2 = |dtheta|^2 pointwise",
          sum(c ** 2 for c in ch35.hodge(dth, 2).values()), sum(c ** 2 for c in dth.values()))
# Non-example 35.5.5
al1 = {(0,): x + y, (1,): x * y}
check("Non-example 35.5.5(a): alpha ^ alpha = 0 and alpha ^ beta = -beta ^ alpha",
      ch35.wedge(al1, al1) == {} and ch35.equal(ch35.wedge(al1, dth), ch35.scale(-1, ch35.wedge(dth, al1))))

# ---------------------------------------------------------------------------
# Theorem 35.5.8: exact adjointness on unit cubes with forms vanishing on the boundary, all k, n = 2, 3
for n, xs in [(2, X2), (3, X3)]:
    bump = sp.Mul(*[(q * (1 - q)) ** 2 for q in xs])
    for k in range(1, n + 1):
        al = {I: bump * rand_poly(xs, 1) for I in ch35.increasing(n, k - 1)}
        be = {I: rand_poly(xs, 2) for I in ch35.increasing(n, k)}
        lhs = L2(ch35.d(al, xs), be, xs)
        rhs = L2(al, ch35.codifferential(be, n, xs, k), xs)
        delta = ch35.hodge(ch35.d(ch35.hodge(be, n), xs), n)
        rhs_delta = L2(al, delta, xs)
        sgn = (-1) ** (n * (k + 1) + 1)
        check(f"Theorem 35.5.8: <<d alpha, beta>> = <<alpha, d* beta>> exactly (n = {n}, k = {k}), value != 0",
              sp.simplify(lhs - rhs) == 0 and lhs != 0)
        check(f"Conventions box: delta = star d star equals {sgn:+d} d* (n = {n}, k = {k})",
              sp.simplify(rhs_delta - sgn * rhs) == 0)
# (a), (b) on R^2, R^3, R^4
for n, xs in [(2, X2), (3, X3), (4, X4)]:
    Xn = [sp.Function(f"V{i}")(*xs) for i in range(n)]
    cd = ch35.codifferential(ch35.flat(Xn), n, xs, 1)
    check(f"Theorem 35.5.8(a): d* X^flat = -div X on R^{n}", ch35.equal(cd, {(): -ch35.div(Xn, xs)}))
    fn = sp.Function("g")(*xs)
    cdd = ch35.codifferential(ch35.d({(): fn}, xs), n, xs, 1)
    check(f"Theorem 35.5.8(b): d* d f = -Delta f on R^{n}", ch35.equal(cdd, {(): -ch35.laplacian(fn, xs)}))
    for k in range(2, n + 1):
        be = gen(n, k, xs, "c")
        dd2 = ch35.codifferential(ch35.codifferential(be, n, xs, k), n, xs, k - 1)
        check(f"Theorem 35.5.8(d): d* d* = 0 on {k}-forms of R^{n}", dd2 == {})
check("Theorem 35.5.8(c): signs for n = 2 all -1, n = 3 equal (-1)^k",
      all((-1) ** (2 * (k + 1) + 1) == -1 for k in range(1, 3)) and all((-1) ** (3 * (k + 1) + 1) == (-1) ** k for k in range(1, 4)))
# parity identity used in the proof
check("Theorem 35.5.8 (proof): k + (n-k+1)(k-1) = n(k+1) + 1 mod 2 for n <= 6",
      all((k + (n - k + 1) * (k - 1) - (n * (k + 1) + 1)) % 2 == 0 for n in range(1, 7) for k in range(1, n + 1)))

# Example 35.5.9
f2 = sp.Function("f")(*X2)
check("Example 35.5.9(a): d*(f dx^dy) = f_y dx - f_x dy",
      ch35.equal(ch35.codifferential({(0, 1): f2}, 2, X2, 2), {(0,): sp.diff(f2, y), (1,): -sp.diff(f2, x)}))
check("Example 35.5.9(b): d* dtheta = 0 and d dtheta = 0",
      ch35.codifferential(dth, 2, X2, 1) == {} and ch35.d(dth, X2) == {})
Xg = [sp.Function(f"X{i}")(*X3) for i in range(3)]
check("Example 35.5.9(c): d*(star X^flat) = (curl X)^flat in R^3",
      ch35.equal(ch35.codifferential(ch35.hodge(ch35.flat(Xg), 3), 3, X3, 2), ch35.flat(ch35.curl(Xg, X3))))
# Figure 35.5.1
gfun = sp.exp(-(x ** 2 + y ** 2))
val0 = ch35.codifferential(ch35.flat([x * gfun, y * gfun]), 2, X2, 1)[()].subs({x: 0, y: 0})
check("Figure 35.5.1: d* X^flat = -2 at the origin, 0 on |p| = 1",
      val0 == -2 and sp.simplify(ch35.codifferential(ch35.flat([x * gfun, y * gfun]), 2, X2, 1)[()].subs({x: 1, y: 0})) == 0)

# Conventions box / Exercise 35.5.6: Hodge Laplacian vs Crane's delta d + d delta on 1-forms of R^3
Xf = ch35.flat(Xg)
dds = ch35.d(ch35.codifferential(Xf, 3, X3, 1), X3)
dsd = ch35.codifferential(ch35.d(Xf, X3), 3, X3, 2)
hodge_lap = ch35.add(dds, dsd)
check("Exercise 35.5.6(a): (dd* + d*d) X^flat = (curl curl X - grad div X)^flat",
      ch35.equal(hodge_lap, ch35.add(ch35.flat(ch35.curl(ch35.curl(Xg, X3), X3)),
                                     ch35.scale(-1, ch35.flat(ch35.grad(ch35.div(Xg, X3), X3))))))
check("Exercise 35.5.6(a): = -(Delta X)^flat componentwise",
      ch35.equal(hodge_lap, ch35.flat([-ch35.laplacian(c, X3) for c in Xg])))
delta = lambda f_, n_: ch35.hodge(ch35.d(ch35.hodge(f_, n_), X3[:n_]), n_)  # noqa: E731
crane = ch35.add(delta(ch35.d(Xf, X3), 3), ch35.d(delta(Xf, 3), X3))
check("Exercise 35.5.6(b): delta d + d delta = (curl curl X + grad div X)^flat",
      ch35.equal(crane, ch35.add(ch35.flat(ch35.curl(ch35.curl(Xg, X3), X3)), ch35.flat(ch35.grad(ch35.div(Xg, X3), X3)))))
for Xt, expected in [([x ** 2, 0, 0], [2, 0, 0]), ([y ** 2, 0, 0], [-2, 0, 0])]:
    cr = ch35.add(delta(ch35.d(ch35.flat(Xt), X3), 3), ch35.d(delta(ch35.flat(Xt), 3), X3))
    check(f"Exercise 35.5.6(b): example X = {Xt}: delta d + d delta gives {expected}", ch35.equal(cr, ch35.flat(expected)))
for k in range(0, 3):
    al = gen(2, k, X2, "h")
    lhs = ch35.add(delta(ch35.d(al, X2), 2) if k < 2 else {}, ch35.d(delta(al, 2), X2) if k > 0 else {})
    hl = ch35.add(ch35.d(ch35.codifferential(al, 2, X2, k), X2) if k > 0 else {},
                  ch35.codifferential(ch35.d(al, X2), 2, X2, k + 1) if k < 2 else {})
    check(f"Exercise 35.5.6(c): n = 2, k = {k}: delta d + d delta = -(dd* + d*d)", ch35.equal(lhs, ch35.scale(-1, hl)))
# Conventions box (reviewer): the slides' integrand star(alpha) ^ beta = (-1)^(k(n-k)) alpha ^ star(beta)
for n, xs in [(2, X2), (3, X3), (4, X4)]:
    for k in range(0, n + 1):
        a, b = gen(n, k, xs, "p"), gen(n, k, xs, "q")
        check(f"Conventions box: star(alpha) ^ beta = (-1)^(k(n-k)) alpha ^ star(beta) (k = {k}, R^{n})",
              ch35.equal(ch35.wedge(ch35.hodge(a, n), b), ch35.scale((-1) ** (k * (n - k)), ch35.wedge(a, ch35.hodge(b, n)))))
slide_ip = ch35.wedge(ch35.hodge({(0,): sp.Integer(1)}, 2), {(0,): sp.Integer(1)})     # star dx ^ dx on R^2
check("Conventions box: with the integrand star(dx) ^ dx the unit square gives -1 (not positive)",
      sp.integrate(slide_ip.get((0, 1), 0), (x, 0, 1), (y, 0, 1)) == -1)

# ---------------------------------------------------------------------------
# Implementation code-35-5-1
ns = ch35.run_block(PAGE, "code-35-5-1")
check("Implementation code-35-5-1: lhs = rhs != 0", sp.simplify(ns["lhs"] - ns["rhs"]) == 0 and ns["lhs"] != 0, str(ns["lhs"]))
be = ns["beta"]
check("Implementation code-35-5-1: printed codiff = lib codifferential",
      ch35.equal(ch35.clean(ns["codiff"](be, 3, X3, 2)), ch35.codifferential(be, 3, X3, 2)))
check("Implementation code-35-5-1: printed star = lib hodge on all basis forms of R^4",
      all(ch35.equal(ch35.clean(ns["star"]({I: 1}, 4)), ch35.hodge({I: 1}, 4)) for k in range(5) for I in ch35.increasing(4, k)))
check("Implementation box code-35-5-1: at most 30 lines", ch35.code_lines(PAGE, "code-35-5-1") <= 30,
      str(ch35.code_lines(PAGE, "code-35-5-1")))

# ---------------------------------------------------------------------------
# Vector-valued forms: Proposition 35.5.13, 35.5.14 (numeric, constant forms on R^3)


def vv_eval(comps, vecs):
    """R^m-valued k-form given by m component dicts, evaluated on vectors (numeric)."""
    return np.array([float(ch35.evaluate(c, [sp.Matrix(q) for q in vecs])) if c else 0.0 for c in comps])


def vv_wedge(A, Bf, prod):
    """alpha ^_B beta for R^3-valued forms (component dicts) with B = 'cross' or 'dot'."""
    out = [dict() for _ in range(3)] if prod == "cross" else [dict()]
    E = np.eye(3)
    for a_, b_ in itertools.product(range(3), range(3)):
        wab = ch35.wedge(A[a_], Bf[b_])
        val = np.cross(E[a_], E[b_]) if prod == "cross" else np.array([E[a_] @ E[b_]])
        for c_, coef in enumerate(val):
            if coef:
                out[c_] = ch35.add(out[c_], ch35.scale(int(coef), wab))
    return out


def rand_const_vv(k):
    return [{I: int(q) for I, q in zip(ch35.increasing(3, k), rng.integers(-3, 4, 3)) if q} for _ in range(3)]


for k, l in [(1, 1), (1, 2), (2, 1)]:
    A, Bf = rand_const_vv(k), rand_const_vv(l)
    for prod, eps in [("cross", -1), ("dot", 1)]:
        lhs = vv_wedge(A, Bf, prod)
        rhs = vv_wedge(Bf, A, prod)
        ok = all(ch35.equal(p_, ch35.scale(eps * (-1) ** (k * l), q_)) for p_, q_ in zip(lhs, rhs))
        check(f"Proposition 35.5.13: alpha ^_B beta = eps (-1)^(kl) beta ^_B alpha ({prod}, k = {k}, l = {l})", ok)
A = rand_const_vv(1)
uu, vv_ = rng.normal(size=3), rng.normal(size=3)
aa = vv_wedge(A, A, "cross")
check("Proposition 35.5.14: (alpha ^ alpha)(u, v) = 2 alpha(u) x alpha(v)",
      np.allclose(vv_eval(aa, [uu, vv_]), 2 * np.cross(vv_eval(A, [uu]), vv_eval(A, [vv_]))))
check("Proposition 35.5.14: <alpha ^ alpha> = 0", all(not c for c in vv_wedge(A, A, "dot")))
B1 = rand_const_vv(1)
lhs = vv_eval(vv_wedge(A, B1, "cross"), [uu, vv_])
rhs = np.cross(vv_eval(A, [uu]), vv_eval(B1, [vv_])) - np.cross(vv_eval(A, [vv_]), vv_eval(B1, [uu]))
check("(35.5.5): cross product wedge of 1-forms evaluated", np.allclose(lhs, rhs))

# (35.5.6) and Example 35.5.15: the sphere
sph = sp.Matrix([R_ * sp.sin(th) * sp.cos(ph), R_ * sp.sin(th) * sp.sin(ph), R_ * sp.cos(th)])
xt, xp = sph.diff(th), sph.diff(ph)
dx_comp = [{(0,): xt[a_], (1,): xp[a_]} for a_ in range(3)]       # dx = x_theta dtheta + x_phi dphi on (theta, phi)
ww = vv_wedge(dx_comp, dx_comp, "cross")
Nout = sph / R_
sym_equal("Example 35.5.15: (dx ^ dx)(d_theta, d_phi) = 2 R^2 sin(theta) N",
          [c.get((0, 1), 0) for c in ww], list(2 * R_ ** 2 * sp.sin(th) * Nout))
check("Example 35.5.15: <dx ^ dx> = 0", all(not c for c in vv_wedge(dx_comp, dx_comp, "dot")))
close("Figure 35.5.2: |x_phi| = sin 1 = 0.841, |dx^dx| = 1.683", [np.sin(1.0), 2 * np.sin(1.0)], [0.841, 1.683], 5e-4)

# Proposition 35.5.16: d(x ^ dx) = dx ^ dx for a general polynomial surface patch, product rule (cross)
xpatch = sp.Matrix([u + v ** 2, u * v, u ** 2 - v ** 3])
x0 = [{(): xpatch[a_]} for a_ in range(3)]
dxp = [ch35.d(c, [u, v]) for c in x0]
lhs = [ch35.d(c, [u, v]) for c in vv_wedge(x0, dxp, "cross")]
rhs = vv_wedge(dxp, dxp, "cross")
check("Proposition 35.5.16: d(x ^ dx) = dx ^ dx (polynomial patch)", all(ch35.equal(p_, q_) for p_, q_ in zip(lhs, rhs)))
check("Proposition 35.5.16(c): d(dx) = 0", all(ch35.d(c, [u, v]) == {} for c in dxp))
check("(35.5.6): dx ^ dx = 2 x_u x x_v du^dv (polynomial patch)",
      all(sp.simplify(c.get((0, 1), 0) - 2 * q_) == 0 for c, q_ in zip(rhs, xpatch.diff(u).cross(xpatch.diff(v)))))
# product rule (D2) for R^3-valued forms with the cross product: random polynomial 1-forms on R^3
A1 = [{(i,): rand_poly(X3, 1) for i in range(3)} for _ in range(3)]
B1 = [{(i,): rand_poly(X3, 1) for i in range(3)} for _ in range(3)]
lhs = [ch35.d(c, X3) for c in vv_wedge(A1, B1, "cross")]
dA, dB = [ch35.d(c, X3) for c in A1], [ch35.d(c, X3) for c in B1]
rhs = [ch35.add(p_, ch35.scale(-1, q_)) for p_, q_ in zip(vv_wedge(dA, B1, "cross"), vv_wedge(A1, dB, "cross"))]
check("Proposition 35.5.16(b): d(alpha ^ beta) = d alpha ^ beta - alpha ^ d beta (R^3-valued 1-forms, cross)",
      all(ch35.equal(p_, q_) for p_, q_ in zip(lhs, rhs)))

# Smooth vs. Discrete: a flat triangle: int dx ^ dx = (x_j - x_i) x (x_k - x_i) = 2 A N
P0, P1, P2 = rng.normal(size=(3, 3))
c_ = np.cross(P1 - P0, P2 - P0)
area = 0.5 * np.linalg.norm(c_)
# dx is constant on the affine triangle: (dx ^ dx)(e1, e2) = 2 e1 x e2, integral = (1/2!) * that
close("Smooth vs. Discrete: int over a flat triangle of dx ^ dx = 2 A N", 0.5 * 2 * c_, 2 * area * c_ / np.linalg.norm(c_), 1e-12)

# ---------------------------------------------------------------------------
# Exercises
a1, b1 = {(0,): x, (1,): 1}, {(0,): 1, (1,): -y}
check("Exercise 35.5.1: <<alpha, beta>> = 0, ||alpha||^2 = 4/3, int alpha ^ beta = -5/4",
      L2(a1, b1, X2) == 0 and L2(a1, a1, X2) == sp.Rational(4, 3)
      and sp.integrate(ch35.wedge(a1, b1)[(0, 1)], (x, 0, 1), (y, 0, 1)) == -sp.Rational(5, 4))
check("Exercise 35.5.2(a): d*(xyz dx^dy) = xz dx - yz dy",
      ch35.equal(ch35.codifferential({(0, 1): x * y * z}, 3, X3, 2), {(0,): x * z, (1,): -y * z}))
check("Exercise 35.5.2(a): = (curl(0, 0, xyz))^flat", ch35.curl([0, 0, x * y * z], X3) == [x * z, -y * z, 0])
check("Exercise 35.5.2(b): d*((x^2+y^2+z^2) dV) = -(2x dy^dz + 2y dz^dx + 2z dx^dy)",
      ch35.equal(ch35.codifferential({(0, 1, 2): x ** 2 + y ** 2 + z ** 2}, 3, X3, 3),
                 ch35.scale(-1, ch35.two_form_r3([2 * x, 2 * y, 2 * z]))))
fb = (x * (1 - x) * y * (1 - y)) ** 2 * (1 + x)
dfb = ch35.d({(): fb}, X2)
check("Exercise 35.5.3: ||df||^2 = -<<f, Delta f>> > 0 (a bump on the unit square)",
      sp.simplify(L2(dfb, dfb, X2) + L2({(): fb}, {(): ch35.laplacian(fb, X2)}, X2)) == 0 and L2(dfb, dfb, X2) > 0)
cyl = sp.Matrix([rp * sp.cos(u), rp * sp.sin(u), v])
sym_equal("Exercise 35.5.4(a): cylinder dx^dx = 2r(cos u, sin u, 0) du^dv", list(2 * cyl.diff(u).cross(cyl.diff(v))),
          [2 * rp * sp.cos(u), 2 * rp * sp.sin(u), 0])
Rp = sp.symbols("Rp", positive=True)
tor = sp.Matrix([(Rp + rp * sp.cos(u)) * sp.cos(v), (Rp + rp * sp.cos(u)) * sp.sin(v), rp * sp.sin(u)])
Nt_out = sp.Matrix([sp.cos(u) * sp.cos(v), sp.cos(u) * sp.sin(v), sp.sin(u)])
sym_equal("Exercise 35.5.4(b): torus dx^dx = -2r(R + r cos u) N_out du^dv", list(2 * tor.diff(u).cross(tor.diff(v))),
          list(-2 * rp * (Rp + rp * sp.cos(u)) * Nt_out))
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch35.run_block(PAGE, "code-ex-35-5-5", dict(ns))
check("Exercise 35.5.5: output equals out-ex-35-5-5", buf.getvalue().strip() == ch35.output_block(PAGE, "out-ex-35-5-5")
      == "[True, True, False, False]", buf.getvalue())

summary()
