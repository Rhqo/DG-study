"""Verification for Section 39.1 (The Weingarten Map).

Run from the project root: PYTHONPATH=tools python3 chapters/39-surface-curvature/verify/v-39-1-weingarten-map.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import ddg39  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "39-1-weingarten-map.html"
ex = dgsym.EXAMPLES
t, u, v = sp.symbols("t u v", real=True)

# ---------------------------------------------------------------- Implementation 39.1 and Exercise 39.1.5
check("Implementation 39.1: at most 30 lines", dgcode.code_lines(PAGE, "code-39-1-1") <= 30)
ns, out = ddg39.run_printed(PAGE, "code-39-1-1")
check("Implementation 39.1: printed output matches the page", out.strip() == ddg39.printed_block(PAGE, "out-39-1-1").strip(),
      out)
g, hh, W = ns["weingarten"](ns["saddle"], 0.5, -0.4)
close("Implementation 39.1: saddle W at q", W, [[0.764659, 0.373005], [-0.373005, -0.932511]], 1e-5)
check("Exercise 39.1.5: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-39-1-5") <= 30)
ns2, out2 = ddg39.run_printed(PAGE, "code-ex-39-1-5", dict(ns))
check("Exercise 39.1.5: printed output matches the page",
      out2.strip() == ddg39.printed_block(PAGE, "out-ex-39-1-5").strip(), out2)
for q in [(0.7, 0.3), (1.2, 2.0), (2.5, 4.0)]:
    Wq = ddg39.weingarten(ddg39.ellipsoid, *q)
    close(f"Exercise 39.1.5: det W = K_ellipsoid at {q}", np.linalg.det(Wq), ddg39.ellipsoid_K(ddg39.ellipsoid(*q)), 1e-6)
    check(f"Exercise 39.1.5: both eigenvalues negative at {q} (outward N)", np.all(np.linalg.eigvals(Wq).real < 0))
# outward normal of the ellipsoid parametrization
n = ddg39.unit_normal(ddg39.ellipsoid, 0.7, 0.3)
check("Exercise 39.1.5: x_theta x x_phi is outward", n @ ddg39.ellipsoid(0.7, 0.3) > 0)

# ---------------------------------------------------------------- Example 39.1.1, (39.1.1)
gam = sp.Matrix([2 * sp.cos(t), sp.sin(t)])
d1, d2 = gam.diff(t), gam.diff(t, 2)
sp_ = sp.sqrt(d1.dot(d1))
ks = sp.simplify((d1[0] * d2[1] - d1[1] * d2[0]) / sp_ ** 3)
sym_equal("Example 39.1.1: kappa_s = 2/(4 sin^2 t + cos^2 t)^(3/2)", ks, 2 / (4 * sp.sin(t) ** 2 + sp.cos(t) ** 2) ** sp.Rational(3, 2))
check("Example 39.1.1: kappa_s(0) = 2, kappa_s(pi/2) = 1/4",
      sp.simplify(ks.subs(t, 0) - 2) == 0 and sp.simplify(ks.subs(t, sp.pi / 2) - sp.Rational(1, 4)) == 0)
ns_ = sp.Matrix([-d1[1], d1[0]]) / sp_
sym_equal("(39.1.1): d n_s = -kappa_s d gamma (ellipse)", ns_.diff(t), -ks * d1)
sym_equal("Example 39.1.1: n_s = -(cos t, 2 sin t)/|gamma'|", ns_, -sp.Matrix([sp.cos(t), 2 * sp.sin(t)]) / sp_)
# Figure 39.1.1 numbers (arc length spacing)
tt = np.linspace(0, 2 * np.pi, 200001)
spd = np.sqrt(4 * np.sin(tt) ** 2 + np.cos(tt) ** 2)
L = float(np.sum(0.5 * (spd[1:] + spd[:-1]) * np.diff(tt)))
close("Figure 39.1.1: perimeter 9.688, ds = L/20 = 0.484", [L, L / 20], [9.688448, 0.484422], 1e-5)

# ---------------------------------------------------------------- Proposition 39.1.2 and Example 39.1.3
def W_of(entry):
    X = entry["expr"]
    a, b = entry["coords"]
    pos = entry["positive"]
    Nn = dgsym.unit_normal(X, a, b, pos)
    Xu, Xv = X.diff(a), X.diff(b)
    Nu, Nv = Nn.diff(a), Nn.diff(b)
    g = sp.Matrix([[Xu.dot(Xu), Xu.dot(Xv)], [Xv.dot(Xu), Xv.dot(Xv)]])
    h = -sp.Matrix([[Xu.dot(Nu), Xu.dot(Nv)], [Xv.dot(Nu), Xv.dot(Nv)]])      # h[k, j] = -<x_k, N_j>
    Wm = dgsym.simp(g.inv() * h, pos)
    return X, Xu, Xv, Nu, Nv, Wm, h


for key, expect in [("sphere", lambda P: -sp.eye(2) / P[0]),
                    ("cylinder", lambda P: sp.diag(-1 / P[0], 0)),
                    ("torus", None)]:
    e = ex[key]
    X, Xu, Xv, Nu, Nv, Wm, h = W_of(e)
    a, b = e["coords"]
    sym_equal(f"(39.1.2): -dN = dx W ({key}, d/du)", -Nu, Xu * Wm[0, 0] + Xv * Wm[1, 0])
    sym_equal(f"(39.1.2): -dN = dx W ({key}, d/dv)", -Nv, Xu * Wm[0, 1] + Xv * Wm[1, 1])
    sym_equal(f"Prop 39.1.2(b) = (8.5.5): W = shape_operator ({key})", Wm, dgsym.shape_operator(X, a, b, e["positive"]))
    sym_equal(f"Prop 39.1.5: h symmetric ({key})", h[0, 1], h[1, 0])
    if expect is not None:
        sym_equal(f"Example 39.1.3: W of the {key}", Wm, expect(e["params"]))
Rs, rs = ex["torus"]["params"]
ut = ex["torus"]["coords"][0]
_, _, _, _, _, Wt, _ = W_of(ex["torus"])
sym_equal("Example 39.1.3(c): torus W = diag(1/r, cos u/(R + r cos u))", Wt,
          sp.diag(1 / rs, sp.cos(ut) / (Rs + rs * sp.cos(ut))))
num = lambda uu: [float(Wt[0, 0].subs({Rs: 2, rs: sp.Rational(4, 5), ut: uu})),  # noqa: E731
                  float(Wt[1, 1].subs({Rs: 2, rs: sp.Rational(4, 5), ut: uu}))]
close("Example 39.1.3(c): u = 0, pi/2, pi values", num(0) + num(sp.pi / 2) + num(sp.pi),
      [1.25, 1 / 2.8, 1.25, 0.0, 1.25, -1 / 1.2], 1e-12)
check("Example 39.1.3(c): torus N is inward (x_u x x_v)",
      sp.simplify(dgsym.unit_normal(ex["torus"]["expr"], *ex["torus"]["coords"], ex["torus"]["positive"]) - ex["torus"]["expected"]["normal"]) == sp.zeros(3, 1))
# saddle at origin
sad = ex["saddle"]
Xs, Xsu, Xsv, Nsu, Nsv, Ws, hs = W_of(sad)
sym_equal("Example 39.1.3(d): saddle W(0) = diag(2, -2)", Ws.subs({u: 0, v: 0}), sp.diag(2, -2))
sym_equal("Example 39.1.3(d): N_u(0) = (-2, 0, 0), N_v(0) = (0, 2, 0)",
          [Nsu.subs({u: 0, v: 0}), Nsv.subs({u: 0, v: 0})], [sp.Matrix([-2, 0, 0]), sp.Matrix([0, 2, 0])])
# inward cylinder (Exercise 39.1.1)
cyl = ex["cylinder"]
rc = cyl["params"][0]
Xc = cyl["expr"]
Nin = -sp.Matrix([sp.cos(u), sp.sin(u), 0])
sym_equal("Exercise 39.1.1: -dN_in = (1/r) x_u du", -Nin.diff(u), Xc.diff(u) / rc)
sym_equal("Exercise 39.1.1: Crane S (outward) = diag(1/r, 0): dN = (1/r) x_u du", cyl["expected"]["normal"].diff(u), Xc.diff(u) / rc)

# ---------------------------------------------------------------- Example 39.1.4 (saddle at q)
q = {u: sp.Rational(1, 2), v: -sp.Rational(2, 5)}
gq = sp.Matrix([[Xsu.dot(Xsu), Xsu.dot(Xsv)], [Xsv.dot(Xsu), Xsv.dot(Xsv)]]).subs(q)
sym_equal("Example 39.1.4: g(q) = [[2, 0.8], [0.8, 1.64]]", gq, sp.Matrix([[2, sp.Rational(4, 5)], [sp.Rational(4, 5), sp.Rational(41, 25)]]))
hq = sp.simplify(hs.subs(q))
sym_equal("Example 39.1.4: h(q) = diag(2, -2)/sqrt(2.64)", hq, sp.diag(2, -2) / sp.sqrt(sp.Rational(66, 25)))
Wq = sp.simplify(Ws.subs(q))
close("Example 39.1.4: W(q) numbers", [float(a) for a in Wq], [0.7646593, 0.3730045, -0.3730045, -0.9325113], 1e-6)
check("Example 39.1.4: W(q) not symmetric, gW = h symmetric",
      sp.simplify(Wq[0, 1] - Wq[1, 0]) != 0 and sp.simplify(gq * Wq - hq) == sp.zeros(2, 2))
close("Example 39.1.4: det W = -4/2.64^2 = -0.5739, tr W / 2 = -0.0839",
      [float(Wq.det()), float(Wq.trace() / 2)], [-4 / 2.64 ** 2, -0.0839260], 1e-6)
Kg = (2 * (-2) - 0) / (1 + 1 + 0.64) ** 2
close("Example 39.1.4: graph formula K = (f_uu f_vv - f_uv^2)/(1 + f_u^2 + f_v^2)^2", float(Wq.det()), Kg, 1e-12)

# ---------------------------------------------------------------- Proposition 39.1.5 on a general graph and a random immersion
f = sp.Function("f")
Xg = sp.Matrix([u, v, f(u, v)])
Ng = dgsym.unit_normal(Xg, u, v)
lhs = Ng.diff(u).dot(Xg.diff(v)) - Ng.diff(v).dot(Xg.diff(u))
check("Prop 39.1.5: <N_u, x_v> = <N_v, x_u> for every graph", sp.simplify(lhs) == 0)
rng = np.random.default_rng(391)
cfs = rng.normal(size=(3, 6)) * 0.3
def xr(a, b):
    return np.array([a + cfs[0] @ [a * a, a * b, b * b, a ** 3, b ** 3, a * b * b],
                     b + cfs[1] @ [a * a, a * b, b * b, a ** 3, b ** 3, a * b * b],
                     cfs[2] @ [a * a, a * b, b * b, a ** 3, b ** 3, a * b * b]])
worst = 0.0
for _ in range(5):
    a, b = rng.uniform(-0.5, 0.5, 2)
    _, hh, _ = ns["weingarten"](xr, a, b)
    worst = max(worst, abs(hh[0, 1] - hh[1, 0]))
check("Prop 39.1.5: hh from dN is symmetric for a random polynomial immersion", worst < 1e-6, f"{worst}")

# ---------------------------------------------------------------- Non-example 39.1.6: J is skew-adjoint for g
gg = np.array([[2.0, 0.8], [0.8, 1.64]])
# J = g-rotation by +90 degrees: J = sqrt(det g)^{-1} [[-g12, -g22], [g11, g12]]
J = np.array([[-gg[0, 1], -gg[1, 1]], [gg[0, 0], gg[0, 1]]]) / np.sqrt(np.linalg.det(gg))
check("Non-example 39.1.6: J^2 = -1, g J skew (so J is not self-adjoint)",
      np.allclose(J @ J, -np.eye(2)) and np.allclose(gg @ J, -(gg @ J).T) and not np.allclose(gg @ J, (gg @ J).T))

# ---------------------------------------------------------------- Exercise 39.1.2 (reparametrization) on the saddle (polar)
rho, th = sp.symbols("rho theta", positive=True)
h_map = sp.Matrix([rho * sp.cos(th), rho * sp.sin(th)])
Y = Xs.subs({u: h_map[0], v: h_map[1]}, simultaneous=True)
NY = dgsym.unit_normal(Y, rho, th)
Yr, Yt = Y.diff(rho), Y.diff(th)
gY = sp.Matrix([[Yr.dot(Yr), Yr.dot(Yt)], [Yt.dot(Yr), Yt.dot(Yt)]])
hY = -sp.Matrix([[Yr.dot(NY.diff(rho)), Yr.dot(NY.diff(th))], [Yt.dot(NY.diff(rho)), Yt.dot(NY.diff(th))]])
WY = gY.inv() * hY
Dh = h_map.jacobian([rho, th])
rhs = Dh.inv() * Ws.subs({u: h_map[0], v: h_map[1]}, simultaneous=True) * Dh
sym_equal("Exercise 39.1.2: W_y = Dh^{-1} W_x Dh (saddle, polar, det Dh = rho > 0)", WY, rhs,
          {rho: (0.2, 1.2), th: (0.1, 3.0)})
# Exercise 39.1.4(a): sphere with phi -> phi + theta: F != 0 and W = -I/r
spx = ex["sphere"]
thh, phh = spx["coords"]
rr = spx["params"][0]
Z = spx["expr"].subs(phh, phh + thh)
Zt, Zp = Z.diff(thh), Z.diff(phh)
check("Exercise 39.1.4(a): F = r^2 sin^2 theta != 0 for x(theta, phi + theta)", sp.simplify(Zt.dot(Zp) - rr ** 2 * sp.sin(thh) ** 2) == 0)
sym_equal("Exercise 39.1.4(a): W = -I/r in that parametrization", dgsym.shape_operator(Z, thh, phh, spx["positive"]), -sp.eye(2) / rr,
          {thh: (0.3, 2.8), phh: (0.1, 6.0), rr: (0.5, 2.0)})

# ---------------------------------------------------------------- Exercise 39.1.3 (parallel surfaces) on the torus
Xt = ex["torus"]["expr"]
vt = ex["torus"]["coords"][1]
Nt = dgsym.unit_normal(Xt, ut, vt, ex["torus"]["positive"])
s_ = sp.symbols("s", real=True)
Xs_ = Xt + s_ * Nt
lhs = sp.Matrix.hstack(Xs_.diff(ut), Xs_.diff(vt))
rhs = sp.Matrix.hstack(Xt.diff(ut), Xt.diff(vt)) * (sp.eye(2) - s_ * Wt)
sym_equal("Exercise 39.1.3(a): d x_s = dx (I - s W) (torus)", lhs, rhs)
sym_equal("Exercise 39.1.3(c): torus x_s = torus with tube radius r - s",
          Xs_, Xt.subs(rs, rs - s_))
check("Exercise 39.1.3(c): cos u/(R + r cos u) <= 1/(R + r) < 1/r at (2, 0.8)",
      max(np.cos(x) / (2 + 0.8 * np.cos(x)) for x in np.linspace(0, 2 * np.pi, 1001)) <= 1 / 2.8 + 1e-12 and 1 / 2.8 < 1.25)

summary()
