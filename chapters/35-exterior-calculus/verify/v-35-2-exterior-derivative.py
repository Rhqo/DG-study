"""Section 35.2 (The Exterior Derivative): verification of every computation (GUIDELINES 13).

The coordinate formula (35.2.3) (dict model in lib/ch35.py) is cross-checked against the invariant formula
(d omega)(e_i0..e_ik) = sum_m (-1)^m d_{i_m} omega(..e_im-hat..) (tensor model), and the printed Implementation code
is run as printed and compared with both.
Run from the project root: PYTHONPATH=tools python3 chapters/35-exterior-calculus/verify/v-35-2-exterior-derivative.py
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

PAGE = "35-2-exterior-derivative.html"
x, y, z, w = sp.symbols("x y z w", real=True)
r = sp.symbols("r", positive=True)
th = sp.symbols("theta", real=True)
X2, X3, X4 = [x, y], [x, y, z], [x, y, z, w]
rr = sp.sqrt(x ** 2 + y ** 2)
polar = {x: r * sp.cos(th), y: r * sp.sin(th)}
dom = {th: (-3, 3), r: (0.3, 2)}


def gen_form(n, k, xs, tag):
    return {I: sp.Function(tag + "".join(map(str, I)))(*xs) for I in ch35.increasing(n, k)}


# ---------------------------------------------------------------------------
# Proposition 35.2.2
f3 = sp.Function("f")(x, y, z)
check("Proposition 35.2.2(a): df = sum_i d_i f dx^i", ch35.equal(ch35.d({(): f3}, X3),
                                                               ch35.one_form(ch35.grad(f3, X3))))
for i, xi in enumerate(X3):
    check(f"Proposition 35.2.2(a): d(x^{i + 1}) = dx^{i + 1}", ch35.d({(): xi}, X3) == {(i,): 1})
# (b) for polar coordinates: d(r) = (x dx + y dy)/r, d(atan2(y, x)) = dtheta, df = f_r dr + f_theta dtheta
check("Proposition 35.2.2(b): d(r) = (x dx + y dy)/r", ch35.equal(ch35.d({(): rr}, X2), ch35.one_form([x / rr, y / rr])))
check("Proposition 35.2.2(b): d(atan2(y, x)) = (-y dx + x dy)/(x^2 + y^2)",
      ch35.equal(ch35.d({(): sp.atan2(y, x)}, X2), ch35.one_form([-y / rr ** 2, x / rr ** 2])))
fp = x ** 3 * y + sp.sin(x * y)
P = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
DPinv = P.jacobian([r, th]).inv()
fP = fp.subs(polar)
lhs = [sp.diff(fP, r) * DPinv[0, c] + sp.diff(fP, th) * DPinv[1, c] for c in range(2)]
sym_equal("Proposition 35.2.2(b): sum_j d(f o P)/du^j du^j has Cartesian coefficients grad f",
          lhs, [sp.diff(fp, x).subs(polar), sp.diff(fp, y).subs(polar)], dom)

# Example 35.2.3: saddle in polar coordinates
fs = r ** 2 * sp.cos(2 * th)
sym_equal("Example 35.2.3: saddle f = r^2 cos 2theta", fs, (x ** 2 - y ** 2).subs(polar), dom)
fr, fth = sp.diff(fs, r), sp.diff(fs, th)
sym_equal("Example 35.2.3: df = 2r cos2th dr - 2r^2 sin2th dth", [fr, fth],
          [2 * r * sp.cos(2 * th), -2 * r ** 2 * sp.sin(2 * th)], dom)
gradf = fr * P.diff(r) + fth / r ** 2 * P.diff(th)
sym_equal("Example 35.2.3: grad f = f_r d_r + (f_th/r^2) d_th = (2x, -2y)", list(gradf),
          [2 * r * sp.cos(th), -2 * r * sp.sin(th)], dom)
sym_equal("Example 35.2.3: grad coefficients (2r cos2th, -2 sin2th)", [fr, fth / r ** 2],
          [2 * r * sp.cos(2 * th), -2 * sp.sin(2 * th)], dom)

# ---------------------------------------------------------------------------
# Theorem 35.2.4: formula vs tensor model, (D2), (D3)
for n, k, xs in [(2, 1, X2), (3, 1, X3), (3, 2, X3), (4, 1, X4), (4, 2, X4)]:
    form = gen_form(n, k, xs, "a")
    dd = ch35.d(form, xs)
    T = ch35.d_tensor(ch35.to_tensor(form, n, k), n, k, xs)
    check(f"Theorem 35.2.4: (35.2.3) = invariant formula for k = {k} on R^{n}", ch35.equal(dd, ch35.from_tensor(T, n, k + 1)))
    check(f"Theorem 35.2.4 (D3): d d = 0 for k = {k} on R^{n}", ch35.d(dd, xs) == {})
check("Theorem 35.2.4 (D3): d d f = 0 on R^4", ch35.d(ch35.d({(): sp.Function("f")(*X4)}, X4), X4) == {})
for k, l in [(0, 1), (1, 1), (1, 2), (2, 1), (0, 2)]:
    a = gen_form(4, k, X4, "p")
    b = gen_form(4, l, X4, "q")
    lhs = ch35.d(ch35.wedge(a, b), X4)
    rhs = ch35.add(ch35.wedge(ch35.d(a, X4), b), ch35.scale((-1) ** k, ch35.wedge(a, ch35.d(b, X4))))
    check(f"Theorem 35.2.4 (D2): graded product rule, k = {k}, l = {l} on R^4", ch35.equal(lhs, rhs))
check("Theorem 35.2.4: d = 0 on n-forms of R^3", ch35.d(gen_form(3, 3, X3, "c"), X3) == {})

# Non-example 35.2.5
al, be = {(1,): x}, {(0,): z}
lhs = ch35.d(ch35.wedge(al, be), X3)
check("Non-example 35.2.5: d((x dy)^(z dx)) = -x dxdydz", ch35.equal(lhs, {(0, 1, 2): -x}))
check("Non-example 35.2.5: d alpha ^ beta = 0", ch35.wedge(ch35.d(al, X3), be) == {})
check("Non-example 35.2.5: alpha ^ d beta = x dxdydz", ch35.equal(ch35.wedge(al, ch35.d(be, X3)), {(0, 1, 2): x}))
check("Non-example 35.2.5: without the sign the rule fails", not ch35.equal(lhs, ch35.wedge(al, ch35.d(be, X3))))

# Proposition 35.2.6: d computed in polar coordinates equals d computed in Cartesian coordinates
a_rt = r ** 2 * sp.sin(th) + r ** 3
b_rt = r * sp.cos(2 * th)
polar_form = {(0,): a_rt, (1,): b_rt}                        # a dr + b dtheta in the (r, theta) coframe
dpol = ch35.d(polar_form, [r, th])                           # (b_r - a_theta) dr ^ dtheta
back = {r: rr, th: sp.atan2(y, x)}
cart = ch35.add(ch35.scale(a_rt.subs(back), ch35.one_form([x / rr, y / rr])),
                ch35.scale(b_rt.subs(back), ch35.one_form([-y / rr ** 2, x / rr ** 2])))
dcart = ch35.d(cart, X2)
sym_equal("Proposition 35.2.6: d(a dr + b dth) in polar = Cartesian d (dr^dth = dx^dy/r)",
          dcart.get((0, 1), 0).subs(polar), dpol.get((0, 1), 0) / r, dom)

# Example 35.2.7
check("Example 35.2.7(a): d(x dy) = dx^dy", ch35.d({(1,): x}, X2) == {(0, 1): 1})
dth_f = ch35.one_form([-y / rr ** 2, x / rr ** 2])
check("Example 35.2.7(b): d(dtheta) = 0", ch35.d(dth_f, X2) == {})
sym_equal("Example 35.2.7(b): d_y(-y/r^2) = (-(x^2+y^2) + 2y^2)/r^4", sp.diff(-y / rr ** 2, y),
          (-(x ** 2 + y ** 2) + 2 * y ** 2) / rr ** 4)
sym_equal("Example 35.2.7(b): d_x(x/r^2) = ((x^2+y^2) - 2x^2)/r^4", sp.diff(x / rr ** 2, x),
          ((x ** 2 + y ** 2) - 2 * x ** 2) / rr ** 4)
check("Example 35.2.7(c): d(r dtheta) = (1/r) dx^dy", ch35.equal(ch35.d(ch35.one_form([-y / rr, x / rr]), X2),
                                                                {(0, 1): 1 / rr}))
check("Example 35.2.7(c): polar: d(r dtheta) = dr^dtheta", ch35.d({(1,): r}, [r, th]) == {(0, 1): 1})
check("Example 35.2.7(d): d(xy dz) = y dx^dz + x dy^dz", ch35.equal(ch35.d({(2,): x * y}, X3), {(0, 2): y, (1, 2): x}))
check("Example 35.2.7(d): d(x dy^dz) = dx^dy^dz", ch35.d({(1, 2): x}, X3) == {(0, 1, 2): 1})
check("Example 35.2.7(e): saddle df = 2x dx - 2y dy, d(df) = 0",
      ch35.equal(ch35.d({(): x ** 2 - y ** 2}, X2), {(0,): 2 * x, (1,): -2 * y})
      and ch35.d(ch35.d({(): x ** 2 - y ** 2}, X2), X2) == {})

# ---------------------------------------------------------------------------
# Proposition 35.2.8 and Corollary 35.2.9 (generic coefficients)
f = sp.Function("f")(*X3)
Xv = [sp.Function(n)(*X3) for n in ("X1", "X2", "X3")]
check("Proposition 35.2.8(a): df = (grad f)^flat", ch35.equal(ch35.d({(): f}, X3), ch35.flat(ch35.grad(f, X3))))
dXf = ch35.d(ch35.flat(Xv), X3)
check("Proposition 35.2.8(b): d X^flat = star (curl X)^flat", ch35.equal(dXf, ch35.hodge(ch35.flat(ch35.curl(Xv, X3)), 3)))
check("Proposition 35.2.8(b): curl X = (star d X^flat)^sharp",
      all(sp.simplify(a - b) == 0 for a, b in zip(ch35.sharp(ch35.hodge(dXf, 3), 3), ch35.curl(Xv, X3))))
dsX = ch35.d(ch35.hodge(ch35.flat(Xv), 3), X3)
check("Proposition 35.2.8(c): d star X^flat = (div X) dxdydz", ch35.equal(dsX, {(0, 1, 2): ch35.div(Xv, X3)}))
check("Corollary 35.2.9: curl grad f = 0", all(sp.simplify(c) == 0 for c in ch35.curl(ch35.grad(f, X3), X3)))
sym_equal("Corollary 35.2.9: div curl X = 0", ch35.div(ch35.curl(Xv, X3), X3), 0)
check("Text: X = (yz, xz, xy) = grad(xyz) has curl 0", ch35.curl([y * z, x * z, x * y], X3) == [0, 0, 0]
      and [y * z, x * z, x * y] == ch35.grad(x * y * z, X3))
check("Text: X = (x, y, z): div 3, curl 0", ch35.div([x, y, z], X3) == 3 and ch35.curl([x, y, z], X3) == [0, 0, 0])
check("Text: X = (-y, x, 0): curl (0, 0, 2), div 0", ch35.curl([-y, x, 0], X3) == [0, 0, 2] and ch35.div([-y, x, 0], X3) == 0)
Pf, Qf = sp.Function("P")(x, y), sp.Function("Q")(x, y)
check("Text: R^2: d(P dx + Q dy) = (Q_x - P_y) dx^dy",
      ch35.equal(ch35.d({(0,): Pf, (1,): Qf}, X2), {(0, 1): sp.diff(Qf, x) - sp.diff(Pf, y)}))
check("Text: R^2: star d star (P dx + Q dy) = P_x + Q_y",
      ch35.equal(ch35.hodge(ch35.d(ch35.hodge({(0,): Pf, (1,): Qf}, 2), X2), 2), {(): sp.diff(Pf, x) + sp.diff(Qf, y)}))
check("Text: R^2: star(P dx + Q dy) = P dy - Q dx", ch35.equal(ch35.hodge({(0,): Pf, (1,): Qf}, 2), {(1,): Pf, (0,): -Qf}))

# ---------------------------------------------------------------------------
# Proposition 35.2.10 / 35.2.11: dtheta closed, not exact (the jump of the angle across the slit)
close("Proposition 35.2.11: theta -> pi from above (-1, 0)", float(sp.atan2(1e-9, -1)), np.pi, 1e-8)
close("Proposition 35.2.11: theta -> -pi from below (-1, 0)", float(sp.atan2(-1e-9, -1)), -np.pi, 1e-8)
q = np.array([1.0, 0.0])
ok = True
rng = np.random.default_rng(3521)
for p in rng.uniform(-3, 3, size=(2000, 2)):
    if p[1] == 0 and p[0] <= 0:
        continue
    for t in np.linspace(0, 1, 41)[1:]:
        s = q + t * (p - q)
        if abs(s[1]) < 1e-15 and s[0] <= 0:
            ok = False
check("Proposition 35.2.11: the slit plane is star-shaped with respect to (1, 0) (2000 random points)", ok)
check("Text: star dtheta = -dr/r = -d(log r) (exact)",
      ch35.equal(ch35.hodge(dth_f, 2), ch35.scale(-1, ch35.d({(): sp.log(rr)}, X2))))

# Proposition 35.2.12 (a), (b) in R^2, R^3, R^4
for n, xs in [(2, X2), (3, X3), (4, X4)]:
    Xn = [sp.Function(f"V{i}")(*xs) for i in range(n)]
    lhs = ch35.d(ch35.hodge(ch35.flat(Xn), n), xs)
    check(f"Proposition 35.2.12(a): d star X^flat = (div X) dV on R^{n}", ch35.equal(lhs, {tuple(range(n)): ch35.div(Xn, xs)}))
    fn = sp.Function("g")(*xs)
    lap = ch35.hodge(ch35.d(ch35.hodge(ch35.d({(): fn}, xs), n), xs), n)
    sym_equal(f"Proposition 35.2.12(b): star d star d f = sum d_i^2 f on R^{n}", lap.get((), 0), ch35.laplacian(fn, xs))
    for i in range(n):
        Ic = tuple(j for j in range(n) if j != i)
        check(f"Proposition 35.2.12: star dx^{i + 1} = (-1)^{i} dx^(..hat..) on R^{n}",
              ch35.hodge({(i,): 1}, n) == {Ic: (-1) ** i})

# Example 35.2.13: polar Laplacian (35.2.4)
for name, fx in [("x^3 y + exp(x) cos(y)", x ** 3 * y + sp.exp(x) * sp.cos(y)), ("saddle x^2 - y^2", x ** 2 - y ** 2),
                 ("log r", sp.log(rr)), ("x^2 y^2", x ** 2 * y ** 2)]:
    fP = sp.simplify(fx.subs(polar))
    lap_pol = sp.diff(r * sp.diff(fP, r), r) / r + sp.diff(fP, th, 2) / r ** 2
    sym_equal(f"Example 35.2.13 (35.2.4): polar Laplacian of {name}", lap_pol, ch35.laplacian(fx, X2).subs(polar), dom)
# the derivation in the (r, theta) coframe: star df = r f_r dtheta - (f_theta/r) dr, d star df
fg = sp.Function("F")(r, th)
star_df = {(1,): r * sp.diff(fg, r), (0,): -sp.diff(fg, th) / r}
dsd = ch35.d(star_df, [r, th])
sym_equal("Example 35.2.13: d star df = (d_r(r f_r) + f_thth / r) dr^dth", dsd.get((0, 1), 0),
          sp.diff(r * sp.diff(fg, r), r) + sp.diff(fg, th, 2) / r)
sym_equal("Example 35.2.13(a): saddle Delta = 4cos2th - 4cos2th = 0",
          sp.diff(r * sp.diff(fs, r), r) / r + sp.diff(fs, th, 2) / r ** 2, 0)
check("Example 35.2.13(b): star d log r = dtheta", ch35.equal(ch35.hodge(ch35.d({(): sp.log(rr)}, X2), 2), dth_f))
sym_equal("Example 35.2.13(c): Delta theta = 0", ch35.laplacian(sp.atan2(y, x), X2), 0)

# Figure 35.2.2: star df = d(2xy)
check("Figure 35.2.2: star d(x^2 - y^2) = d(2xy)",
      ch35.equal(ch35.hodge(ch35.d({(): x ** 2 - y ** 2}, X2), 2), ch35.d({(): 2 * x * y}, X2)))

# ---------------------------------------------------------------------------
# Implementation code-35-2-1
ns = ch35.run_block(PAGE, "code-35-2-1")
check("Implementation code-35-2-1: d(df) = {}", ns["ddf"] == {})
cX = ch35.curl(ns["X"], X3)
dXflat = ns["dXflat"]
check("Implementation code-35-2-1: dXflat = (curl X)^1 dy^dz - (curl X)^2 dx^dz + (curl X)^3 dx^dy",
      ch35.equal(dXflat, {(1, 2): cX[0], (0, 2): -cX[1], (0, 1): cX[2]}))
check("Implementation code-35-2-1: d(dtheta) = {}", ns["d_dtheta"] == {})
check("Implementation code-35-2-1: d(r dtheta) = {(0, 1): 1/r}", ch35.equal(ns["d_r_dtheta"], {(0, 1): 1 / rr}))
for n, k, xs in [(3, 1, X3), (3, 2, X3), (4, 2, X4)]:
    form = gen_form(n, k, xs, "m")
    check(f"Implementation code-35-2-1: printed d = lib d (k = {k}, R^{n})", ch35.equal(ns["d"](form, xs), ch35.d(form, xs)))
check("Implementation box code-35-2-1: at most 30 lines", ch35.code_lines(PAGE, "code-35-2-1") <= 30)

# ---------------------------------------------------------------------------
# Exercises
check("Exercise 35.2.1(a): d(e^{xy} dx) = -x e^{xy} dx^dy", ch35.equal(ch35.d({(0,): sp.exp(x * y)}, X2),
                                                                     {(0, 1): -x * sp.exp(x * y)}))
check("Exercise 35.2.1(b)", ch35.equal(ch35.d({(0,): x * y * z, (1,): x * y * z, (2,): x * y * z}, X3),
                                       {(0, 1): y * z - x * z, (0, 2): y * z - x * y, (1, 2): x * z - x * y}))
check("Exercise 35.2.1(c): = 3 dxdydz", ch35.d(ch35.two_form_r3([x, y, z]), X3) == {(0, 1, 2): 3})
check("Exercise 35.2.2(b): d(y dx) = dy^dx = -dx^dy", ch35.d({(0,): y}, X2) == {(0, 1): -1})
fg2, gg2 = sp.Function("f")(*X2), sp.Function("g")(*X2)
check("Exercise 35.2.2(c): d(f dg) = df ^ dg", ch35.equal(ch35.d(ch35.scale(fg2, ch35.d({(): gg2}, X2)), X2),
                                                         ch35.wedge(ch35.d({(): fg2}, X2), ch35.d({(): gg2}, X2))))
# Exercise 35.2.3: polar divergence
Xr, Xt = sp.Function("Xr")(r, th), sp.Function("Xt")(r, th)
Xflat_pol = {(0,): Xr, (1,): r ** 2 * Xt}
star_pol = {(1,): r * Xr, (0,): -r * Xt}                     # star dr = r dth, star dth = -dr/r
check("Exercise 35.2.3: star X^flat = r X^r dth - r X^th dr (from (35.1.6))",
      ch35.equal(star_pol, ch35.add(ch35.scale(Xr, {(1,): r}), ch35.scale(r ** 2 * Xt, {(0,): -1 / r}))))
dsp = ch35.d(star_pol, [r, th]).get((0, 1), 0)
sym_equal("Exercise 35.2.3: div X = (1/r) d_r(r X^r) + d_th X^th", dsp / r,
          sp.diff(r * Xr, r) / r + sp.diff(Xt, th))
# Cartesian check for a concrete field
Xr_c, Xt_c = r ** 2 * sp.cos(th), sp.sin(th) / r
Xcart = Xr_c * P.diff(r) + Xt_c * P.diff(th)
div_cart = (sp.diff(Xcart[0].subs({r: rr, th: sp.atan2(y, x)}), x) + sp.diff(Xcart[1].subs({r: rr, th: sp.atan2(y, x)}), y))
sym_equal("Exercise 35.2.3: polar divergence formula agrees with Cartesian for a concrete field",
          div_cart.subs(polar), sp.diff(r * Xr_c, r) / r + sp.diff(Xt_c, th), dom)
check("Exercise 35.2.3: d_theta = (-y, x), r d_r = (x, y)",
      sp.simplify(P.diff(th) - sp.Matrix([-y, x]).subs(polar)) == sp.zeros(2, 1)
      and sp.simplify(r * P.diff(r) - sp.Matrix([x, y]).subs(polar)) == sp.zeros(2, 1))


# Exercise 35.2.4: the operator D satisfies (D1), (D2) but not (D3)
def Dop(form):
    k = ch35.degree(form)
    if k is None or k == 2:
        return {}
    if k == 0:
        return ch35.d(form, X2)
    a, b = form.get((0,), 0), form.get((1,), 0)
    return ch35.add(ch35.d({(0,): a}, X2), ch35.d({(1,): b}, X2), {(0, 1): a})


rng = np.random.default_rng(3524)


def rand_poly():
    return sum(int(c) * x ** i * y ** j for (i, j), c in zip(itertools.product(range(3), range(3)),
                                                               rng.integers(-2, 3, 9)))


okD2 = True
for k, l in [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0)]:
    for _ in range(3):
        a = {I: rand_poly() for I in ch35.increasing(2, k)}
        b = {I: rand_poly() for I in ch35.increasing(2, l)}
        lhs = Dop(ch35.wedge(a, b))
        rhs = ch35.add(ch35.wedge(Dop(a), b), ch35.scale((-1) ** k, ch35.wedge(a, Dop(b))))
        okD2 &= ch35.equal(lhs, rhs)
check("Exercise 35.2.4: D satisfies the graded product rule on random polynomial forms", okD2)
check("Exercise 35.2.4: D(Dx) = dx^dy != 0", Dop(Dop({(): x})) == {(0, 1): 1})
# Exercise 35.2.5
rho = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
sym_equal("Exercise 35.2.5: div(p/|p|^3) = 0", ch35.div([x / rho ** 3, y / rho ** 3, z / rho ** 3], X3), 0)
check("Exercise 35.2.5: d omega = 0", ch35.d(ch35.two_form_r3([x / rho ** 3, y / rho ** 3, z / rho ** 3]), X3) == {})
# Exercise 35.2.6 (coding)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch35.run_block(PAGE, "code-ex-35-2-6", dict(ns))
check("Exercise 35.2.6: output equals out-ex-35-2-6 (True True True)",
      buf.getvalue().strip() == ch35.output_block(PAGE, "out-ex-35-2-6") == "True True True", buf.getvalue())

summary()
