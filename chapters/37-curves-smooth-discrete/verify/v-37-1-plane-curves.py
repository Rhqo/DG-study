"""Verification for Section 37.1 (Plane Curves).

Run from the project root: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/verify/v-37-1-plane-curves.py
"""
import contextlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import ddg37  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "37-1-plane-curves.html"
check("Implementation 37.1: at most 30 lines", dgcode.code_lines(PAGE, "code-37-1-1") <= 30)
NS = dgcode.run_block(PAGE, "code-37-1-1")
cfc = NS["curve_from_curvature"]

t, u = sp.symbols("t u", real=True)
r = sp.symbols("r", positive=True)

# ---------------------------------------------------------------- Example 37.1.1, Figure 37.1.1
circ = dgsym.EXAMPLES["circle"]["expr"][:2, :]
sym_equal("Example 37.1.1(a): |gamma'| = r for the circle", sp.sqrt(sp.simplify(circ.diff(t).dot(circ.diff(t)))),
          dgsym.EXAMPLES["circle"]["params"][0])
ell = ddg37.ELLIPSE_EXPR
sym_equal("Example 37.1.1(b): |gamma'|^2 = 4 sin^2 t + cos^2 t = 1 + 3 sin^2 t (ellipse)",
          ell.diff(ddg37.T_SYM).dot(ell.diff(ddg37.T_SYM)), 1 + 3 * sp.sin(ddg37.T_SYM) ** 2)
tt = np.linspace(0, 2 * np.pi, 100001)
spd = ddg37.speed("ellipse", tt)
close("Example 37.1.1(b): speed ranges over [1, 2] (t = 0 and t = pi/2)", [spd.min(), spd.max(),
      ddg37.speed("ellipse", np.array([0.0]))[0], ddg37.speed("ellipse", np.array([np.pi / 2]))[0]],
      [1, 2, 1, 2], 1e-9)
close("Example 37.1.1(b): length of the ellipse 9.6884", round(ddg37.smooth_length("ellipse"), 4), 9.6884, 1e-12)

# ---------------------------------------------------------------- Proposition 37.1.2, Exercise 37.1.1
h = 2 * u
gt = ell.subs(ddg37.T_SYM, h)
sym_equal("Prop 37.1.2: d(gamma o h)_u(1) = h'(u) gamma'(h(u)) (h(u) = 2u, ellipse)",
          list(gt.diff(u)), list(2 * ell.diff(ddg37.T_SYM).subs(ddg37.T_SYM, h)))
vals = [float(sp.sqrt(gt.diff(u).dot(gt.diff(u))).subs(u, uv)) for uv in (0, sp.pi / 4)]
close("Exercise 37.1.1: |d tilde gamma_u(1)| = 2 at u = 0 and 4 at u = pi/4", vals, [2, 4], 1e-12)

# ---------------------------------------------------------------- Example 37.1.3, Non-example 37.1.4, Example 37.1.5
fl = ddg37.FLOWER_EXPR
rr = ddg37.FLOWER_R
sym_equal("Example 37.1.3: flower |gamma'|^2 = r^2 + r'^2", fl.diff(ddg37.T_SYM).dot(fl.diff(ddg37.T_SYM)),
          rr ** 2 + rr.diff(ddg37.T_SYM) ** 2)
check("Example 37.1.3: flower |gamma'|^2 >= 0.49 (regular)", (ddg37.speed("flower", tt) ** 2).min() >= 0.49 - 1e-12)
cusp = sp.Matrix([t ** 3, t ** 2])
cd = cusp.diff(t)
check("Non-example 37.1.4: gamma'(0) = 0 for (t^3, t^2)", list(cd.subs(t, 0)) == [0, 0])
tp = sp.symbols("tp", positive=True)
unit_pos = (cd / sp.sqrt(cd.dot(cd))).subs(t, tp)
sym_equal("Non-example 37.1.4 / Ex 37.1.2: t(t) = (3t, 2)/sqrt(9t^2+4) for t > 0", list(sp.simplify(unit_pos)),
          [3 * tp / sp.sqrt(9 * tp ** 2 + 4), 2 / sp.sqrt(9 * tp ** 2 + 4)])
tn = sp.symbols("tn", negative=True)
unit_neg = sp.simplify((cd / sp.sqrt(cd.dot(cd))).subs(t, tn))
sym_equal("Non-example 37.1.4 / Ex 37.1.2: t(t) = (3|t|, -2)/sqrt(9t^2+4) for t < 0", list(unit_neg),
          [-3 * tn / sp.sqrt(9 * tn ** 2 + 4), -2 / sp.sqrt(9 * tn ** 2 + 4)])
check("Non-example 37.1.4: limits (0, +1) and (0, -1)",
      [sp.limit(c, tp, 0, "+") for c in unit_pos] == [0, 1])
f8 = ddg37.FIG8_EXPR
sym_equal("Example 37.1.5: figure-eight gamma' = (cos t, cos 2t)", list(f8.diff(ddg37.T_SYM)),
          [sp.cos(ddg37.T_SYM), sp.cos(2 * ddg37.T_SYM)])
check("Example 37.1.5: figure-eight is regular (min speed > 0) and gamma(0) = gamma(pi)",
      ddg37.speed("figure8", tt).min() > 0.5 and np.allclose(ddg37.point("figure8", np.array([0.0, np.pi])), 0))

# ---------------------------------------------------------------- Derivation (37.1.4) and Proposition 37.1.7
s = np.linspace(0, 2 * np.pi, 2001)
_, G = cfc(np.ones_like(s), 2 * np.pi, 0.0, (0.0, -1.0))
close("Derivation: kappa = 1 gives the unit circle (center (0,0) when p0 = (0,-1), phi0 = 0)",
      np.abs(np.linalg.norm(G, axis=1) - 1).max(), 0.0, 1e-5)
for c, L_, closes in ((1.0, 2 * np.pi, True), (0.5, 8 * np.pi, True), (-2.0, 3 * np.pi, True), (0.0, 1.0, False),
                      (1.0, 3.0, False)):
    ss = np.linspace(0, L_, 4001)
    phi, Gc = cfc(c * np.ones_like(ss), L_, 0.3)
    dist = np.linalg.norm(Gc[-1] - Gc[0])
    check(f"Exercise 37.1.4 / Prop 37.1.7: kappa = {c}, L = {L_:.3f} closes = {closes} "
          f"(rotation index {(phi[-1] - phi[0]) / (2 * np.pi):.2f})", (dist < 1e-6) == closes)


def J1(x, terms=40):
    return sum((-1) ** m * (x / 2) ** (2 * m + 1) / (math.factorial(m) * math.factorial(m + 1)) for m in range(terms))


for a in (0.5, 1.5, 3.0):
    sb = np.linspace(0, 2 * np.pi, 8001)
    _, G3 = cfc(1 + a * np.cos(sb), 2 * np.pi, 0.0)
    close(f"Example 37.1.9: gap = (-2 pi J_1({a}), 0)", G3[-1] - G3[0], [-2 * np.pi * J1(a), 0.0], 1e-6)
sb = np.linspace(0, 2 * np.pi, 8001)
phi3, G3 = cfc(1 + 0.5 * np.cos(sb), 2 * np.pi, 0.0)
close("Example 37.1.9: total curvature 2 pi and gap -1.5222", [phi3[-1] - phi3[0], round(G3[-1, 0], 4)],
      [2 * np.pi, -1.5222], 1e-9)

# ---------------------------------------------------------------- Example 37.1.8 (ellipse recovered numerically)
L = ddg37.smooth_length("ellipse")
errs, gaps = [], []
for m in (100, 200, 400, 800):
    sj = np.linspace(0, L, m + 1)
    tj = ddg37.t_of_s("ellipse", sj, 200000)
    phi, Y = cfc(ddg37.kappa("ellipse", tj), L)
    errs.append(ddg37.rigid_align_error(Y, ddg37.point("ellipse", tj)))
    gaps.append(np.linalg.norm(Y[-1] - Y[0]))
    if m == 100:
        close("Example 37.1.8: int kappa_s ds = 2 pi", phi[-1] - phi[0], 2 * np.pi, 1e-6)
quoted = [2.6e-3, 6.4e-4, 1.6e-4, 4.0e-5]
check("Example 37.1.8: alignment errors 2.6e-3, 6.4e-4, 1.6e-4, 4.0e-5 (two digits)",
      all(abs(e - q) / q < 0.03 for e, q in zip(errs, quoted)), str(errs))
order = np.polyfit(np.log([100, 200, 400, 800]), np.log(errs), 1)[0]
check(f"Example 37.1.8: second order in m (observed {-order:.2f})", abs(order + 2) < 0.05)
check("Example 37.1.8: closing distance below 2e-9", max(gaps) < 2e-9, str(gaps))

# ---------------------------------------------------------------- Exercise 37.1.5 (printed code and output)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    dgcode.run_block(PAGE, "code-ex-37-1-5", NS)
printed = dgcode.code_block(PAGE, "out-ex-37-1-5").strip()
check("Exercise 37.1.5: printed output matches", buf.getvalue().strip() == printed, buf.getvalue())
close("Exercise 37.1.5: the closing a is the first zero of J_1", J1(3.831707), 0.0, 1e-6)
check("Exercise 37.1.5: J_1 > 0 on (0, 3.8) (first positive zero)",
      all(J1(x) > 0 for x in np.linspace(0.01, 3.8, 400)))

summary()
