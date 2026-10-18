"""Section 35.1 (Differential Forms): verification of every computation (GUIDELINES 13).

Pointwise algebra is checked in the dict model of lib/ch35.py (sorted index tuples), cross-checked against the
tensor model (function tensors with sympy permutation signs) where a wedge or a star is involved.
Run from the project root: PYTHONPATH=tools python3 chapters/35-exterior-calculus/verify/v-35-1-differential-forms.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from dgcheck import check, close, sym_equal, summary  # noqa: E402
import ch35  # noqa: E402

PAGE = "35-1-differential-forms.html"
x, y, z = sp.symbols("x y z", real=True)
r = sp.symbols("r", positive=True)
th, ph = sp.symbols("theta phi", real=True)
u, v, s = sp.symbols("u v s", real=True)
X2, X3 = [x, y], [x, y, z]

# ---------------------------------------------------------------------------
# Proposition 35.1.4: beta^I = sum_J det(B_IJ) e^J for a coframe beta^i = sum_j B^i_j e^j (n = 3)
B = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"b{i}{j}"))
betas = [ch35.one_form(list(B.row(i))) for i in range(3)]
for I in [(0, 1), (0, 2), (1, 2)]:
    w = ch35.wedge(betas[I[0]], betas[I[1]])
    pred = ch35.clean({J: B.extract(list(I), list(J)).det() for J in ch35.increasing(3, 2)})
    check(f"Proposition 35.1.4: beta^{I} = sum_J det(B_IJ) e^J", ch35.equal(w, pred))
w3 = ch35.wedge(ch35.wedge(betas[0], betas[1]), betas[2])
sym_equal("Proposition 35.1.4: beta^123 = det B e^123", w3.get((0, 1, 2), 0), B.det())

# ---------------------------------------------------------------------------
# Example 35.1.5: polar coordinate fields and their dual basis (35.1.4)
P = sp.Matrix([r * sp.cos(th), r * sp.sin(th)])
DP = P.jacobian([r, th])
sym_equal("Example 35.1.5: d/dr = (cos th, sin th), d/dth = (-r sin th, r cos th)", DP,
          sp.Matrix([[sp.cos(th), -r * sp.sin(th)], [sp.sin(th), r * sp.cos(th)]]))
rows = DP.inv()
xr = {x: r * sp.cos(th), y: r * sp.sin(th)}
dr_c = [x / sp.sqrt(x ** 2 + y ** 2), y / sp.sqrt(x ** 2 + y ** 2)]
dth_c = [-y / (x ** 2 + y ** 2), x / (x ** 2 + y ** 2)]
sym_equal("Example 35.1.5 (35.1.4): dr = (x dx + y dy)/r is the first row of DP^-1",
          [sp.sympify(c).subs(xr) for c in dr_c], list(rows.row(0)), {th: (-3, 3)})
sym_equal("Example 35.1.5 (35.1.4): dtheta = (-y dx + x dy)/r^2 is the second row of DP^-1",
          [sp.sympify(c).subs(xr) for c in dth_c], list(rows.row(1)), {th: (-3, 3)})
pairing = sp.Matrix([[sum(a * b for a, b in zip([sp.sympify(c).subs(xr) for c in row], DP.col(j)))
                      for j in range(2)] for row in (dr_c, dth_c)])
sym_equal("Example 35.1.5: dual basis check (dr, dtheta)(d/dr, d/dtheta) = I", pairing, sp.eye(2), {th: (-3, 3)})
# Example 1.3.9 gives beta^1 = (cos th, sin th), beta^2 = (-sin th, cos th)/r
sym_equal("Example 35.1.5: agrees with Example 1.3.9", list(rows.row(0)) + list(rows.row(1)),
          [sp.cos(th), sp.sin(th), -sp.sin(th) / r, sp.cos(th) / r])

# Common misconception box: P(x, s) = (x, x + s)
Pm = sp.Matrix([x, x + s])
DPm = Pm.jacobian([x, s])
check("Misconception box: in coordinates (x, s), d/dx = (1, 1)", list(DPm.col(0)) == [1, 1])
check("Misconception box: in coordinates (x, s), dx = first row of DP^-1 = (1, 0)", list(DPm.inv().row(0)) == [1, 0])
check("Misconception box: DP^-1 = [[1, 0], [-1, 1]]", DPm.inv() == sp.Matrix([[1, 0], [-1, 1]]))

# ---------------------------------------------------------------------------
# Example 35.1.7 and Proposition 35.1.6(d)
alpha = {(1,): x}
Xrot = [-y, x]
sym_equal("Example 35.1.7: (x dy)(X) = x^2 for X = -y d/dx + x d/dy", ch35.evaluate(alpha, [Xrot]), x ** 2)
sym_equal("Example 35.1.7: X^flat(X) = x^2 + y^2", ch35.evaluate(ch35.flat(Xrot), [Xrot]), x ** 2 + y ** 2)

# Example 35.1.8
check("Example 35.1.8: star dx = dy, star dy = -dx", ch35.hodge({(0,): 1}, 2) == {(1,): 1}
      and ch35.hodge({(1,): 1}, 2) == {(0,): -1})
check("Example 35.1.8: star 1 = dx^dy, star(dx^dy) = 1", ch35.hodge({(): 1}, 2) == {(0, 1): 1}
      and ch35.hodge({(0, 1): 1}, 2) == {(): 1})
check("Example 35.1.8: star(x dy) = -x dx", ch35.equal(ch35.hodge({(1,): x}, 2), {(0,): -x}))
check("Example 35.1.8: star(-y dx + x dy) = -x dx - y dy",
      ch35.equal(ch35.hodge({(0,): -y, (1,): x}, 2), {(0,): -x, (1,): -y}))
al, be = {(0,): y, (1,): 1}, {(0,): 1, (1,): x}
ab = ch35.wedge(al, be)
check("Example 35.1.8: (y dx + dy) ^ (dx + x dy) = (xy - 1) dx^dy", ch35.equal(ab, {(0, 1): x * y - 1}))
Tab = ch35.wedge_tensor(ch35.to_tensor(al, 2), ch35.to_tensor(be, 2), 2, 1, 1)
check("Example 35.1.8: tensor-model wedge agrees", ch35.equal(ch35.from_tensor(Tab, 2, 2), ab))
t = sp.symbols("t", positive=True)
check("Example 35.1.8: on xy = 1 the coefficient vectors (y, 1), (1, x) are parallel",
      sp.simplify(sp.Matrix([[1 / t, 1], [1, t]]).det()) == 0)

# ---------------------------------------------------------------------------
# Example 35.1.9 (35.1.6): Hodge star in polar coordinates (Cartesian coefficients)
rr = sp.sqrt(x ** 2 + y ** 2)
dr_f, dth_f = ch35.one_form(dr_c), ch35.one_form(dth_c)
check("Example 35.1.9: star dr = r dtheta", ch35.equal(ch35.hodge(dr_f, 2), ch35.scale(rr, dth_f)))
check("Example 35.1.9: star dtheta = -dr/r", ch35.equal(ch35.hodge(dth_f, 2), ch35.scale(-1 / rr, dr_f)))
check("Example 35.1.9: dr ^ dtheta = dx^dy / r", ch35.equal(ch35.wedge(dr_f, dth_f), {(0, 1): 1 / rr}))
check("Example 35.1.9: star 1 = r dr ^ dtheta", ch35.equal(ch35.scale(rr, ch35.wedge(dr_f, dth_f)), {(0, 1): 1}))
sym_equal("Example 35.1.9: |dr| = 1", sum(c ** 2 for c in dr_c), 1)
sym_equal("Example 35.1.9: |dtheta| = 1/r", sp.sqrt(sum(c ** 2 for c in dth_c)), 1 / rr)
sym_equal("Example 35.1.9: <dr, dtheta> = 0", sum(a * b for a, b in zip(dr_c, dth_c)), 0)
# matrix of star in the coframe (dr, dtheta): columns are star dr = (0, r), star dtheta = (-1/r, 0)
M = sp.Matrix([[0, -1 / r], [r, 0]])
check("Example 35.1.9: star in the coframe (dr, dtheta) has matrix [[0, -1/r], [r, 0]]",
      sp.simplify(M * sp.Matrix([1, 0]) - sp.Matrix([0, r])) == sp.zeros(2, 1)
      and sp.simplify(M * sp.Matrix([0, 1]) - sp.Matrix([-1 / r, 0])) == sp.zeros(2, 1))
# Exercise 34.5.4 gives the same table
# Figure 35.1.2: (e^h - 1)/h = 1.14 for h = 2 pi / 24
h = 2 * np.pi / 24
close("Figure 35.1.2: (e^h - 1)/h = 1.14", (np.exp(h) - 1) / h, 1.14, 5e-3)

# ---------------------------------------------------------------------------
# Proposition 35.1.10: dx^1 ^ ... ^ dx^n = det DP du^1 ^ ... ^ du^n = sqrt(det G) du
check("Proposition 35.1.10: polar det DP = r", sp.simplify(DP.det()) == r)
for name, Pg, coords, dom in [
    ("polynomial map (u + v^2/3, v + u^3/3)", sp.Matrix([u + v ** 2 / 3, v + u ** 3 / 3]), [u, v],
     {u: (-0.5, 0.5), v: (-0.5, 0.5)}),
    ("spherical coordinates of R^3", sp.Matrix([r * sp.sin(th) * sp.cos(ph), r * sp.sin(th) * sp.sin(ph),
                                                r * sp.cos(th)]), [r, th, ph], {th: (0.1, 3.0), ph: (0, 6)}),
]:
    J = Pg.jacobian(coords)
    n = len(coords)
    # dx^i = sum_j J^i_j du^j: wedge them in the du-coframe (dict model over the u-indices)
    dxs = [ch35.one_form(list(J.row(i))) for i in range(n)]
    w = dxs[0]
    for f in dxs[1:]:
        w = ch35.wedge(w, f)
    sym_equal(f"Proposition 35.1.10: {name}: dx^1..dx^n = det DP du^1..du^n", w.get(tuple(range(n)), 0), J.det(), dom)
    G = J.T * J
    sym_equal(f"Proposition 35.1.10: {name}: det G = (det DP)^2", G.det(), J.det() ** 2, dom)
check("Proposition 35.1.10: spherical coordinates det DP = r^2 sin(theta)",
      sp.simplify(sp.Matrix([r * sp.sin(th) * sp.cos(ph), r * sp.sin(th) * sp.sin(ph), r * sp.cos(th)])
                  .jacobian([r, th, ph]).det() - r ** 2 * sp.sin(th)) == 0)

# Example 35.1.11: the dictionary of three-space
Xv = [sp.Function(f"X{i}")(x, y, z) for i in (1, 2, 3)]
uu = sp.Matrix(sp.symbols("u1:4", real=True))
vv = sp.Matrix(sp.symbols("v1:4", real=True))
sXf = ch35.hodge(ch35.flat(Xv), 3)
check("Example 35.1.11: star X^flat = X1 dy^dz + X2 dz^dx + X3 dx^dy", ch35.equal(sXf, ch35.two_form_r3(Xv)))
sym_equal("Example 35.1.11: (star X^flat)(u, v) = <X, u x v>", ch35.evaluate(sXf, [uu, vv]),
          sp.Matrix(Xv).dot(uu.cross(vv)))
fz = sp.Function("f")(x, y, z)
check("Example 35.1.11: star f = f dx^dy^dz", ch35.equal(ch35.hodge({(): fz}, 3), {(0, 1, 2): fz}))
check("Example 35.1.11: z dx^dy = star((0, 0, z)^flat)", ch35.equal(ch35.hodge(ch35.flat([0, 0, z]), 3), {(0, 1): z}))

# ---------------------------------------------------------------------------
# Implementation box code-35-1-1
ns = ch35.run_block(PAGE, "code-35-1-1")
Pg_ = ns["P"]
Xg, Yg = Pg_[..., 0], Pg_[..., 1]
check("Implementation code-35-1-1: alpha(X) = x^2 on the grid", np.allclose(ns["values"], Xg ** 2))
check("Implementation code-35-1-1: alpha ^ beta = xy - 1 on the grid", np.allclose(ns["ab"], Xg * Yg - 1))
check("Implementation code-35-1-1: star(x dy) = -x dx on the grid",
      np.allclose(ns["star_x_dy"][0], -Xg) and np.allclose(ns["star_x_dy"][1], 0))
check("Implementation code-35-1-1: 81 grid points", Pg_.shape == (9, 9, 2))
check("Implementation box code-35-1-1: at most 30 lines", ch35.code_lines(PAGE, "code-35-1-1") <= 30)

# ---------------------------------------------------------------------------
# Exercise 35.1.1
a1, b1 = {(0,): z, (1,): x}, {(1,): 1, (2,): -1}
w = ch35.wedge(a1, b1)
check("Exercise 35.1.1: alpha ^ beta = -x dy^dz + z dz^dx + z dx^dy", ch35.equal(w, ch35.two_form_r3([-x, z, z])))
check("Exercise 35.1.1: cyclic coefficients = a x b",
      [sp.simplify(c) for c in ch35.two_form_vector(w)] ==
      [sp.simplify(c) for c in sp.Matrix([z, x, 0]).cross(sp.Matrix([0, 1, -1]))])
check("Exercise 35.1.1: star(alpha ^ beta) = -x dx + z dy + z dz",
      ch35.equal(ch35.hodge(w, 3), {(0,): -x, (1,): z, (2,): z}))
check("Exercise 35.1.1: star alpha = z dy^dz + x dz^dx", ch35.equal(ch35.hodge(a1, 3), ch35.two_form_r3([z, x, 0])))
sym_equal("Exercise 35.1.1: alpha(X) = yz", ch35.evaluate(a1, [[y, 0, 1]]), y * z)
# Exercise 35.1.2(c)
om = {(0, 1): 1, (2, 3): 1}
check("Exercise 35.1.2(c): (dx1^dx2 + dx3^dx4)^2 = 2 dx1234", ch35.wedge(om, om) == {(0, 1, 2, 3): 2})

# Exercise 35.1.3
Pe = sp.Matrix([2 * u + v, v])
J = Pe.jacobian([u, v])
check("Exercise 35.1.3(a): d/du = (2, 0), d/dv = (1, 1)", list(J.col(0)) == [2, 0] and list(J.col(1)) == [1, 1])
Ji = J.inv()
check("Exercise 35.1.3(a): du = (dx - dy)/2, dv = dy",
      list(Ji.row(0)) == [sp.Rational(1, 2), -sp.Rational(1, 2)] and list(Ji.row(1)) == [0, 1])
G = J.T * J
check("Exercise 35.1.3(b): G = [[4, 2], [2, 2]], sqrt(det G) = 2 = det DP",
      G == sp.Matrix([[4, 2], [2, 2]]) and sp.sqrt(G.det()) == 2 == J.det())
du_c = ch35.one_form(list(Ji.row(0)))
dv_c = ch35.one_form(list(Ji.row(1)))
check("Exercise 35.1.3(b): du ^ dv = dx^dy / 2", ch35.wedge(du_c, dv_c) == {(0, 1): sp.Rational(1, 2)})
check("Exercise 35.1.3(c): star du = du + dv", ch35.equal(ch35.hodge(du_c, 2), ch35.add(du_c, dv_c)))
check("Exercise 35.1.3(c): du ^ star du = |du|^2 dx^dy = (1/2) dx^dy",
      ch35.wedge(du_c, ch35.hodge(du_c, 2)) == {(0, 1): sp.Rational(1, 2)})

# Exercise 35.1.4: dx = (x/r) dr - y dtheta, dy = (y/r) dr + x dtheta, and x dy in the polar coframe
dx_pol = ch35.add(ch35.scale(x / rr, dr_f), ch35.scale(-y, dth_f))
dy_pol = ch35.add(ch35.scale(y / rr, dr_f), ch35.scale(x, dth_f))
check("Exercise 35.1.4: dx = (x/r) dr - y dtheta", ch35.equal(dx_pol, {(0,): 1}))
check("Exercise 35.1.4: dy = (y/r) dr + x dtheta", ch35.equal(dy_pol, {(1,): 1}))
check("Exercise 35.1.4: x dy = (xy/r) dr + x^2 dtheta",
      ch35.equal(ch35.add(ch35.scale(x * y / rr, dr_f), ch35.scale(x ** 2, dth_f)), {(1,): x}))
sym_equal("Exercise 35.1.4: |x dy|^2 in the polar coframe = x^2",
          (x * y / rr) ** 2 * 1 + (x ** 2) ** 2 / rr ** 2, x ** 2)

# Exercise 35.1.5 (coding)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ch35.run_block(PAGE, "code-ex-35-1-5", dict(ns))
check("Exercise 35.1.5: output equals out-ex-35-1-5 (True True True True)",
      buf.getvalue().strip() == ch35.output_block(PAGE, "out-ex-35-1-5") == "True True True True", buf.getvalue())

summary()
