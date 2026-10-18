"""Verification for Section 37.6 (Curvature Flows on Curves).

Run from the project root: PYTHONPATH=tools python3 chapters/37-curves-smooth-discrete/verify/v-37-6-curve-flows.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgdec as dd  # noqa: E402
import ddg37  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "37-6-curve-flows.html"
for cid in ("code-37-6-1", "code-37-6-2"):
    check(f"Implementation {cid}: at most 30 lines", dgcode.code_lines(PAGE, cid) <= 30)
NS = dgcode.run_block(PAGE, "code-37-6-1")
dgcode.run_block(PAGE, "code-37-6-2", NS)
rng = np.random.default_rng(376)

# ---------------------------------------------------------------- Proposition 37.6.6
for _ in range(4):
    X = rng.normal(size=(9, 2))
    K, ell = NS["length_laplacian"](X)
    close("Prop 37.6.6: K(gamma) gamma = grad L (dgdec.length_gradient)", K @ X, dd.length_gradient(X, True), 1e-12)
    ev = np.linalg.eigvalsh(K)
    check("Prop 37.6.6: K symmetric PSD with one-dimensional kernel (constants)",
          np.allclose(K, K.T) and ev.min() > -1e-12 and np.sum(ev < 1e-10) == 1 and np.allclose(K @ np.ones(9), 0))
# finite differences of the length
X = rng.normal(size=(7, 2))
g = np.zeros_like(X)
for i in range(7):
    for c in range(2):
        dX = np.zeros_like(X)
        dX[i, c] = 1e-6
        g[i, c] = (dd.curve_length(X + dX, True) - dd.curve_length(X - dX, True)) / 2e-6
close("Prop 37.6.6: (37.6.4) agrees with finite differences of L", NS["length_laplacian"](X)[0] @ X, g, 1e-8)

# ---------------------------------------------------------------- Proposition 37.6.2 (energy decrease, continuous time)
X0, _ = ddg37.sample("flower", 64)
lb = ddg37.dual_lengths(X0)
gL = dd.length_gradient(X0, True)
for name, Minv in (("M = I", np.ones(64)), ("M = diag(lbar)", 1 / lb)):
    v = -Minv[:, None] * gL
    eps = 1e-7
    dLdt = (dd.curve_length(X0 + eps * v, True) - dd.curve_length(X0 - eps * v, True)) / (2 * eps)
    close(f"Prop 37.6.2: dL/dt = -grad L^T M^-1 grad L ({name})", dLdt, -np.sum(Minv[:, None] * gL ** 2), 1e-6)
close("Example 37.6.3(a): -grad L = kappa^B n (the integrated B flow)", -gL,
      dd.curvature_length_variation(X0, True)[:, None] * dd.curve_vertex_normals(X0, True), 1e-12)

# ---------------------------------------------------------------- Proposition 37.6.4, 37.6.5
for h in (0.5, 1.5, 2.2, 3.0):
    u = (1 - h) ** np.arange(30)
    check(f"Prop 37.6.4(a): explicit with h lambda = {h}: decays iff < 2, oscillates iff > 1",
          (abs(u[-1]) < 1e-3) == (h < 2) and (np.all(u > 0) == (h < 1)))
    ui = (1 / (1 + h)) ** np.arange(30)
    check(f"Prop 37.6.4(b): implicit with h lambda = {h} decreases monotonically", np.all(np.diff(ui) < 0))
close("Figure 37.6.2: h lambda = 2.2 gives u_4 = 1.44^2 = 2.07", (1 - 2.2) ** 4, 2.0736, 1e-12)
close("Exercise 37.6.1: h lambda = 3: explicit (-2)^k, implicit 4^-k, exact e^-3 = 0.0498",
      [(1 - 3.0) ** 3, (1 / 4.0) ** 3, round(np.exp(-3), 4)], [-8, 1 / 64, 0.0498], 1e-12)
A = rng.normal(size=(6, 6))
Kq = A @ A.T
Mq = np.diag(rng.uniform(0.5, 2, 6))
lmax = np.linalg.eigvals(np.linalg.solve(Mq, Kq)).real.max()
for fac, bounded in ((0.99, True), (1.01, False)):
    h = fac * 2 / lmax
    x = rng.normal(size=6)
    for _ in range(4000):
        x = x - h * np.linalg.solve(Mq, Kq @ x)
    check(f"Prop 37.6.5(a): explicit step bounded iff h lambda_max <= 2 (h = {fac} * 2/lambda_max)",
          (np.linalg.norm(x) < 1e3) == bounded)
x = rng.normal(size=6)
Es = []
for _ in range(20):
    x = np.linalg.solve(Mq + 50.0 * Kq, Mq @ x)
    Es.append(0.5 * x @ Kq @ x)
check("Prop 37.6.5(b): implicit step with a huge h decreases E", np.all(np.diff(Es) <= 1e-14))

# ---------------------------------------------------------------- Proposition 37.6.7 and Example 37.6.8
ok = True
for _ in range(20):
    X = ddg37.sample("flower", 30)[0] + 0.05 * rng.normal(size=(30, 2))
    for h in (1e-3, 0.1, 10.0):
        Y = NS["semi_implicit_step"](X, h)
        ok &= dd.curve_length(Y, True) <= dd.curve_length(X, True) + 1e-12
check("Prop 37.6.7: the semi-implicit step never increases the length (random polygons, h up to 10)", ok)
X0, _ = ddg37.sample("flower", 64)
h0 = ddg37.explicit_critical_step(X0)
close("Example 37.6.8: h0 = 2 / lambda_max(M^-1 K) = 0.004137", round(h0, 6), 0.004137, 1e-12)
close("Implementation 37.6: page steps = lib steps",
      np.r_[NS["explicit_step"](X0, 1e-3).ravel(), NS["semi_implicit_step"](X0, 1e-2).ravel()],
      np.r_[ddg37.explicit_step(X0, 1e-3).ravel(), ddg37.semi_implicit_step(X0, 1e-2).ravel()], 1e-12)
Xe, Xs = X0.copy(), X0.copy()
mx, grew, grew_s = 0.0, 0, 0
for _ in range(54):
    Ye, Ys = NS["explicit_step"](Xe, 0.9 * h0), NS["semi_implicit_step"](Xs, 0.9 * h0)
    grew += dd.curve_length(Ye, True) > dd.curve_length(Xe, True)
    grew_s += dd.curve_length(Ys, True) > dd.curve_length(Xs, True) + 1e-14
    Xe, Xs = Ye, Ys
    mx = max(mx, np.abs(dd.turning_angles(Xe, True)).max())
close("Example 37.6.8: explicit 0.9 h0: max |theta| reaches 2.159, length grows in 5 steps; t = 0.201",
      [round(mx, 3), grew, round(54 * 0.9 * h0, 3)], [2.159, 5, 0.201], 1e-12)
check("Example 37.6.8: semi-implicit 0.9 h0 never grows the length", grew_s == 0)
close("Example 37.6.8: initial max |theta| of the flower polygon = 0.833",
      round(np.abs(dd.turning_angles(X0, True)).max(), 3), 0.833, 1e-12)

# ---------------------------------------------------------------- Figure 37.6.1 (inner product), Motivation
dt = 1e-4
hE = dt / ddg37.dual_lengths(X0).mean()
XE, XW = X0.copy(), X0.copy()
ell0 = dd.curve_edge_lengths(X0, True)
for _ in range(2000):
    XE = ddg37.explicit_step(XE, hE, weighted=False)
    XW = ddg37.explicit_step(XW, dt, weighted=True)
rE = (dd.curve_edge_lengths(XE, True) / ell0).min()
rW = (dd.curve_edge_lengths(XW, True) / ell0).min()
close("Motivation / Figure 37.6.1: min edge ratio at t = 0.2: 0.003 (M = I) and 0.18 (M = diag lbar)",
      [round(rE, 3), round(rW, 2)], [0.003, 0.18], 1e-12)
_lE, _lW = dd.curve_edge_lengths(XE, True), dd.curve_edge_lengths(XW, True)
_dE = np.linalg.norm(XE - XE.mean(0), axis=1)
_dW = np.linalg.norm(XW - XW.mean(0), axis=1)
close("Figure 37.6.1(a): distance to the vertex centroid 0.84..0.90 (M = I), 0.801..0.805 (M = diag lbar)",
      [round(_dE.min(), 2), round(_dE.max(), 2), round(_dW.min(), 3), round(_dW.max(), 3)],
      [0.84, 0.90, 0.801, 0.805], 1e-12)
close("Figure 37.6.1(a): longest / shortest edge about 400 (M = I)", _lE.max() / _lE.min(), 400, 25)
close("Figure 37.6.1(a): longest / shortest edge 6.3 (M = diag lbar)", round(_lW.max() / _lW.min(), 1), 6.3, 1e-12)
_short = np.argsort(_lE / ell0)[:5]                # the five shortest edges (relative to their start)
_tips = np.angle(np.exp(1j * (2 * np.pi * (_short + 0.5) / 64 - np.round((_short + 0.5) / 64 * 5) * 2 * np.pi / 5)))
check("Figure 37.6.1(a): with M = I the five shortest edges sit at the five former petal tips "
      "(within one edge of t = 2 pi j / 5) and keep 0.3%..3.5% of their length",
      len(set(np.round((_short + 0.5) / 64 * 5) % 5)) == 5 and np.all(np.abs(_tips) < 2 * np.pi / 64)
      and 0.003 <= (_lE / ell0)[_short].min() and (_lE / ell0)[_short].max() <= 0.035)

# ---------------------------------------------------------------- Example 37.6.10, Exercise 37.6.3
r = sp.symbols("r", positive=True)
sym_equal("Example 37.6.10: E(circle) = pi / r", sp.Rational(1, 2) * (1 / r) ** 2 * 2 * sp.pi * r, sp.pi / r)
for n, val in ((8, 3.2238), (16, 3.1619), (64, 3.1429)):
    P = ddg37.sample("circle", n)[0]
    th = dd.turning_angles(P, True)
    Ed = ddg37.elastic_energy(th, dd.curve_dual_lengths(P, True))
    close(f"Example 37.6.10: regular {n}-gon E^d = pi^2/(n sin(pi/n)) = {val}",
          [round(Ed, 4), Ed], [val, np.pi ** 2 / (n * np.sin(np.pi / n))], 1e-12)
    close(f"Exercise 37.6.3: E^d = 2 pi^2 / L for the regular {n}-gon", Ed, 2 * np.pi ** 2 / dd.curve_length(P, True),
          1e-12)

# ---------------------------------------------------------------- Derivation: L^2 gradient of the elastic energy
t, eps = sp.symbols("t epsilon", real=True)
g = ddg37.ELLIPSE_EXPR.subs(ddg37.T_SYM, t)
gp = g.diff(t)
spd = sp.sqrt(gp.dot(gp))
Tn = gp / spd
Nn = sp.Matrix([-Tn[1], Tn[0]])
f = sp.cos(2 * t)
ge = g + eps * f * Nn
d1, d2 = ge.diff(t), ge.diff(t, 2)
k_e = (d1[0] * d2[1] - d1[1] * d2[0]) / sp.sqrt(d1.dot(d1)) ** 3
lhs_f = sp.lambdify(t, sp.diff(sp.Rational(1, 2) * k_e ** 2 * sp.sqrt(d1.dot(d1)), eps).subs(eps, 0), "numpy")
k0 = (gp[0] * g.diff(t, 2)[1] - gp[1] * g.diff(t, 2)[0]) / spd ** 3
kss = sp.diff(sp.diff(k0, t) / spd, t) / spd
rhs_f = sp.lambdify(t, (kss + k0 ** 3 / 2) * f * spd, "numpy")
tq = 2 * np.pi * np.arange(4000) / 4000
close("Derivation: d/de E(gamma + e f n_s) = int (kappa'' + kappa^3/2) f ds (ellipse, f = cos 2t)",
      lhs_f(tq).mean() * 2 * np.pi, rhs_f(tq).mean() * 2 * np.pi, 1e-9)
rr = sp.symbols("rr", positive=True)
sym_equal("Derivation: a circle moves outward with r' = 1/(2 r^3)", (1 / rr) ** 3 / 2, 1 / (2 * rr ** 3))

# ---------------------------------------------------------------- isometric elastic flow (Prop 37.6.11, Example 37.6.12)
ell = dd.curve_edge_lengths(X0, True)
lbar = dd.curve_dual_lengths(X0, True)
th = dd.turning_angles(X0, True)
E = [ddg37.elastic_energy(th, lbar)]
th_lib = th.copy()
dev = 0.0
for _ in range(40):
    th = NS["isometric_elastic_step"](th, ell, 0.2)
    th_lib = ddg37.isometric_elastic_step(th_lib, ell, 0.2)
    dev = max(dev, np.abs(th - th_lib).max())
    E.append(ddg37.elastic_energy(th, lbar))
check("Implementation code-37-6-2 = lib isometric_elastic_step", dev < 1e-12)
Y = ddg37.rebuild_closed(ell, th)
close("Example 37.6.12: E^d 48.188 -> 2.2692, L = 8.9640, circle 2 pi^2 / L = 2.2020",
      [round(E[0], 3), round(E[-1], 4), round(ell.sum(), 4), round(2 * np.pi ** 2 / ell.sum(), 4)],
      [48.188, 2.2692, 8.9640, 2.2020], 1e-12)
check("Example 37.6.12: energy decreases at every step", np.all(np.diff(E) < 0))
rng_cs = np.random.default_rng(3761)
_cs_ok = all(E_k >= 2 * np.pi ** 2 / ell.sum() for E_k in E)
for _ in range(200):                       # Cauchy-Schwarz bound for random angles with sum 2 pi and random lbar
    _lb = rng_cs.uniform(0.1, 1.0, 12)
    _t = rng_cs.normal(size=12)
    _t += (2 * np.pi - _t.sum()) / 12
    _cs_ok &= 0.5 * np.sum(_t ** 2 / _lb) >= 2 * np.pi ** 2 / _lb.sum() - 1e-12
_t = 2 * np.pi * _lb / _lb.sum()
check("Fig 37.6.4 text: sum theta = 2 pi gives E^d >= 2 pi^2 / L, equality for theta proportional to lbar",
      _cs_ok and abs(0.5 * np.sum(_t ** 2 / _lb) - 2 * np.pi ** 2 / _lb.sum()) < 1e-12)
check("Prop 37.6.11 / Example 37.6.12: edge lengths kept within 1e-12, polygon closes with these angles",
      np.abs(dd.curve_edge_lengths(Y, True) - ell).max() < 1e-12
      and np.abs(dd.turning_angles(Y, True) - th).max() < 1e-12)
phi = ddg37.edge_angles(th)
check("Example 37.6.12: closing gap below 1e-12 and turning number 1",
      np.linalg.norm(ell @ np.stack([np.cos(phi), np.sin(phi)], 1)) < 1e-12 and abs(th.sum() / (2 * np.pi) - 1) < 1e-12)

# ---------------------------------------------------------------- Exercise 37.6.2 (regular polygon)
P = ddg37.sample("circle", 12)[0]
close("Exercise 37.6.2: explicit radius r - h/r, semi-implicit r^3/(r^2 + h) (r = 1, h = 0.05)",
      [np.linalg.norm(NS["explicit_step"](P, 0.05), axis=1).max(), np.linalg.norm(NS["semi_implicit_step"](P, 0.05), axis=1).max()],
      [0.95, 1 / 1.05], 1e-12)
# ---------------------------------------------------------------- Exercise 37.6.4 (uniform polygon bound)
for n in (16, 64):
    P = ddg37.sample("circle", n)[0]
    l0 = dd.curve_edge_lengths(P, True)[0]
    close(f"Exercise 37.6.4: lambda_max = 4 / l^2 for the regular {n}-gon", 2 / ddg37.explicit_critical_step(P),
          4 / l0 ** 2, 1e-9)

# ---------------------------------------------------------------- Exercises 37.6.5, 37.6.6 (printed code and output)
for ex in ("37-6-5", "37-6-6"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dgcode.run_block(PAGE, f"code-ex-{ex}", dict(NS))
    check(f"Exercise {ex.replace('-', '.')}: printed output matches",
          buf.getvalue().strip() == dgcode.code_block(PAGE, f"out-ex-{ex}").strip(), buf.getvalue())

summary()
