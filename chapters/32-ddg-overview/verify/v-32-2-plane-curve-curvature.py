"""Verification for Section 32.2 (Curvature of a Plane Curve).

Run from the project root: PYTHONPATH=tools python3 chapters/32-ddg-overview/verify/v-32-2-plane-curve-curvature.py
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgdec as dd  # noqa: E402
import dgsym  # noqa: E402
import ddg32  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "32-2-plane-curve-curvature.html"
t = ddg32.T_SYM

# ---------------------------------------------------------------- Example 32.2.1 (a) circle
circ = dgsym.EXAMPLES["circle"]
r = circ["params"][0]
g = circ["expr"]
sym_equal("Example 32.2.1(a): counterclockwise circle kappa_s = 1/r (dgsym EXAMPLES)",
          dgsym.signed_curvature(g, circ["coords"][0]), 1 / r, circ["domain"])
tt = circ["coords"][0]
g_cw = sp.Matrix([g[0].subs(tt, -tt), g[1].subs(tt, -tt), 0])
sym_equal("Example 32.2.1(a): clockwise circle kappa_s = -1/r", dgsym.signed_curvature(g_cw, tt), -1 / r,
          circ["domain"])

# ---------------------------------------------------------------- Example 32.2.1 (b) ellipse
ke = dgsym.signed_curvature(ddg32.ELLIPSE_EXPR, t)
sym_equal("Example 32.2.1(b): ellipse kappa_s = 2/(4 sin^2 t + cos^2 t)^(3/2)", ke,
          2 / (4 * sp.sin(t) ** 2 + sp.cos(t) ** 2) ** sp.Rational(3, 2), {t: (0, 2 * sp.pi)})
ts = np.linspace(0, 2 * np.pi, 200001)
kk = ddg32.kappa("ellipse", ts)
close("Example 32.2.1(b): ellipse max kappa_s = 2 (t = 0), min 1/4 (t = pi/2)", [kk.max(), kk.min()], [2, 0.25], 1e-9)

# ---------------------------------------------------------------- Example 32.2.1 (c) flower, (32.2.2)
rr = sp.Function("r")(t)
gpol = sp.Matrix([rr * sp.cos(t), rr * sp.sin(t)])
kpol = dgsym.signed_curvature(gpol, t)
polar = (rr ** 2 + 2 * rr.diff(t) ** 2 - rr * rr.diff(t, 2)) / (rr ** 2 + rr.diff(t) ** 2) ** sp.Rational(3, 2)
check("(32.2.2): polar formula = dgsym.signed_curvature for a general r(t)", sp.simplify(kpol - polar) == 0)
kf = dgsym.signed_curvature(ddg32.FLOWER_EXPR, t)
fpol = polar.subs(rr, ddg32.FLOWER_R).doit()
sym_equal("Example 32.2.1(c): flower kappa_s by (32.2.2) = dgsym.signed_curvature", sp.simplify(fpol), kf,
          {t: (0, 2 * sp.pi)})
close("Example 32.2.1(c): kappa_s(0) = 5.207, kappa_s(pi/5) = -13.878",
      [round(float(kf.subs(t, 0)), 3), round(float(kf.subs(t, sp.pi / 5)), 3)], [5.207, -13.878], 1e-12)
close("Example 32.2.1(c): (1.69 + 9.75)/1.3^3 and (0.49 - 5.25)/0.7^3",
      [(1.69 + 9.75) / 1.3 ** 3, (0.49 - 5.25) / 0.7 ** 3], [float(kf.subs(t, 0)), float(kf.subs(t, sp.pi / 5))], 1e-12)
kfl = ddg32.kappa("flower", ts)
check("Example 32.2.1(c): kappa_s changes sign 10 times in one turn",
      int(np.sum(np.sign(kfl[:-1]) != np.sign(kfl[1:]))) == 10)
m = 20000
tm = 2 * np.pi * np.arange(m) / m
spf = ddg32.speed("flower", tm)
close("Example 32.2.1(c): flower length = 9.0172", round(ddg32.smooth_length("flower"), 4), 9.0172, 1e-12)
close("Example 32.2.1(c): flower int kappa_s ds = 2 pi", (ddg32.kappa("flower", tm) * spf).sum() * 2 * np.pi / m,
      2 * np.pi, 1e-9)
check("Figure 32.2.1: r(t) > 0, so the flower curve is star-shaped about 0 and simple",
      np.all(1 + 0.3 * np.cos(5 * ts) > 0))

# ---------------------------------------------------------------- Proposition 32.2.2 (integrated form) on the ellipse
pt = ddg32.point("ellipse", tm)
dpt = np.stack([-2 * np.sin(tm), np.cos(tm)], 1)
phi = np.unwrap(np.arctan2(dpt[:, 1], dpt[:, 0]))
spe = ddg32.speed("ellipse", tm)
cum = np.r_[0, np.cumsum(0.5 * (ddg32.kappa("ellipse", tm)[1:] * spe[1:] + ddg32.kappa("ellipse", tm)[:-1] * spe[:-1])
                         * (2 * np.pi / m))]
close("Prop 32.2.2: int_0^s kappa_s ds = phi(s) - phi(0) on the ellipse (trapezoid)", cum - (phi - phi[0]),
      np.zeros(m), 1e-6)

# ---------------------------------------------------------------- Proposition 32.2.3 (first variation), numerical
M = 40000
tq = 2 * np.pi * np.arange(M) / M
P = ddg32.point("ellipse", tq)
d1 = np.stack([-2 * np.sin(tq), np.cos(tq)], 1)
spq = np.linalg.norm(d1, axis=1)
T = d1 / spq[:, None]
N = T @ ddg32.J.T
kq = ddg32.kappa("ellipse", tq)


def length_param(Q, dQ):
    return float(np.linalg.norm(dQ, axis=1).sum() * 2 * np.pi / M)


def variation(eta, deta, eps):
    return length_param(P + eps * eta, d1 + eps * deta)


eta = np.stack([np.sin(2 * tq), np.cos(3 * tq)], 1)
deta = np.stack([2 * np.cos(2 * tq), -3 * np.sin(3 * tq)], 1)
h = 1e-5
fd = (variation(eta, deta, h) - variation(eta, deta, -h)) / (2 * h)
rhs = -float((kq * (N * eta).sum(1) * spq).sum() * 2 * np.pi / M)
close("Prop 32.2.3: dL/deps = -int kappa_s <n_s, eta> ds (ellipse, eta = (sin 2t, cos 3t))", fd, rhs, 1e-7)
# eta = kappa_s n_s: derivative of kappa n with respect to t, by central differences of the exact field
dt = 1e-6
kn = lambda s: (ddg32.kappa("ellipse", s)[:, None]  # noqa: E731
                * (np.stack([-2 * np.sin(s), np.cos(s)], 1) / ddg32.speed("ellipse", s)[:, None]) @ ddg32.J.T)
dkn = (kn(tq + dt) - kn(tq - dt)) / (2 * dt)
fd2 = (variation(kn(tq), dkn, h) - variation(kn(tq), dkn, -h)) / (2 * h)
k2 = float((kq ** 2 * spq).sum() * 2 * np.pi / M)
close("Example 32.2.7: int kappa_s^2 ds = 6.6360 (ellipse)", round(k2, 4), 6.6360, 1e-12)
close("Example 32.2.7 / Fig 32.2.2(a): dL/deps for eta = kappa_s n_s is -int kappa^2 ds", fd2, -k2, 1e-6)
close("Exercise 32.2.4: sqrt(6.6360) = 2.5760", round(np.sqrt(k2), 4), 2.5760, 1e-12)

# ---------------------------------------------------------------- Proposition 32.2.4 (Steiner) and its proof step
# n_s' = -kappa_s t: d n_s/dt = -kappa_s |gamma'| t for the ellipse (symbolic)
gp = ddg32.ELLIPSE_EXPR.diff(t)
spd = sp.sqrt(gp.dot(gp))
Ts = gp / spd
Ns = sp.Matrix([-Ts[1], Ts[0]])
sym_equal("Prop 32.2.4 proof: n_s' = -kappa_s t (ellipse, in the parameter t)", list(Ns.diff(t)),
          list(-ke * spd * Ts), {t: (0, 2 * sp.pi)})
Lell = ddg32.smooth_length("ellipse")
close("Example 32.2.7: ellipse length L = 9.6884", round(Lell, 4), 9.6884, 1e-12)
vals = []
for eps in (-0.4, -0.2, 0.2, 0.4):
    Lo = float((np.abs(1 - eps * kq) * spq).sum() * 2 * np.pi / M)
    Lo2 = length_param(P + eps * N, d1 + eps * (-kq * spq)[:, None] * T)
    close(f"Prop 32.2.4: ellipse offset eps = {eps}: length = L - 2 pi eps (two quadratures)", [Lo, Lo2],
          [Lell - 2 * np.pi * eps] * 2, 1e-10)
    vals.append(round(Lo, 4))
close("Figure 32.2.2(b): offset lengths 12.2017, 10.9451, 8.4318, 7.1752", vals, [12.2017, 10.9451, 8.4318, 7.1752],
      1e-12)
L07 = float((np.abs(1 - 0.7 * kq) * spq).sum() * 2 * np.pi / M)
close("Example 32.2.7: eps = 0.7: parametric length 5.8909 vs L - 1.4 pi = 5.2902",
      [round(L07, 4), round(Lell - 1.4 * np.pi, 4)], [5.8909, 5.2902], 1e-12)
check("Example 32.2.7: 1 - 0.7 kappa_s < 0 somewhere (cusps)", np.any(1 - 0.7 * kq < 0))

# ---------------------------------------------------------------- Proposition 32.2.5 (three-point circles)
ok = True
for t0 in (0.0, 0.7, np.pi / 2):
    errs = []
    for hh in (0.1, 0.05, 0.025):
        Q = ddg32.point("ellipse", np.array([t0 - hh, t0, t0 + hh]))
        c = ddg32.circumcenters(Q)[1]
        R = np.linalg.norm(c - Q[1])
        k0 = ddg32.kappa("ellipse", np.array([t0]))[0]
        d = np.array([-2 * np.sin(t0), np.cos(t0)])
        n0 = ddg32.J @ (d / np.linalg.norm(d))
        errs.append(abs(R - 1 / k0) + np.linalg.norm(c - (Q[1] + n0 / k0)))
    ok &= errs[2] < errs[1] < errs[0] and errs[2] < 2e-2
check("Prop 32.2.5: circles through gamma(t0 - h), gamma(t0), gamma(t0 + h) -> osculating circle (ellipse)", ok)

# ---------------------------------------------------------------- Example 32.2.6 (circle, four ways)
rv = 1.7
s_ = np.linspace(0, 2 * np.pi * rv, 1001)
phi_c = np.unwrap(np.arctan2(np.cos(s_ / rv), -np.sin(s_ / rv)))
close("Example 32.2.6(A): phi(s) = s/r + pi/2", phi_c, s_ / rv + np.pi / 2, 1e-12)
close("Example 32.2.6(B),(C): circle of radius r - eps has length 2 pi r - 2 pi eps", 2 * np.pi * (rv - 0.3),
      2 * np.pi * rv - 0.3 * (2 * np.pi * rv) / rv, 1e-12)

# ---------------------------------------------------------------- Example 32.2.12 (regular polygon)
for n in (5, 12):
    X = ddg32.regular_ngon(n, rv)
    mi = np.pi * (2 * np.arange(n) + 1) / n
    close(f"Example 32.2.12: n = {n}: e_i = 2 r sin(pi/n)(-sin m_i, cos m_i)", dd.curve_edge_vectors(X, True),
          2 * rv * np.sin(np.pi / n) * np.stack([-np.sin(mi), np.cos(mi)], 1), 1e-12)
    close(f"Example 32.2.12: n = {n}: J t_i = -(cos m_i, sin m_i)", dd.curve_tangents(X, True) @ ddg32.J.T,
          -np.stack([np.cos(mi), np.sin(mi)], 1), 1e-12)

# ---------------------------------------------------------------- Implementation box and Exercise 32.2.5
check("Implementation 32.2: at most 30 lines", ddg32.code_lines(PAGE, "code-32-2-1") <= 30)
ns = ddg32.run_block(PAGE, "code-32-2-1")
tn = ns["tangents_normals"]
ok = True
for X, closed in ((ddg32.sample("ellipse", 10)[0], True), (ddg32.sample("flower", 37, 0.3, 1)[0], True),
                  (np.array([[0, 0], [2, 0], [3, 1.0]]), False)):
    E, ell, T, Nn = tn(X, closed)
    ok &= np.allclose(E, dd.curve_edge_vectors(X, closed)) and np.allclose(ell, dd.curve_edge_lengths(X, closed))
    ok &= np.allclose(T, dd.curve_tangents(X, closed)) and np.allclose(Nn, np.stack([-T[:, 1], T[:, 0]], 1))
check("Implementation 32.2: tangents_normals = dgdec (closed ellipse, jittered flower, open Example 32.2.9(c))", ok)
try:
    tn(np.array([[0, 0], [0, 0], [1, 0.0]]), False)
    check("Non-example 32.2.10(a): a repeated vertex is rejected", False)
except ValueError:
    check("Non-example 32.2.10(a): a repeated vertex is rejected", True)

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ddg32.run_block(PAGE, "code-ex-32-2-5", ns)
check("Exercise 32.2.5: the solution code prints exactly the output shown",
      buf.getvalue().strip() == ddg32.code_block(PAGE, "out-ex-32-2-5").strip(), f"got:\n{buf.getvalue()}")
X10 = ddg32.sample("ellipse", 10)[0]
th = dd.turning_angles(X10, True)
close("Figure 32.2.3: jumps 1.1525 (ends), 0.6461, 0.3484 (top/bottom); sum 2 pi",
      [round(th[0], 4), round(th[1], 4), round(th[2], 4), th.sum()], [1.1525, 0.6461, 0.3484, 2 * np.pi], 1e-12)

# ---------------------------------------------------------------- Exercises 32.2.1 - 32.2.3
sym_equal("Exercise 32.2.1: clockwise circle offset gamma + eps n_s has radius r + eps",
          sp.sqrt(((1 + sp.Symbol("e") / r) * g_cw[0]) ** 2 + ((1 + sp.Symbol("e") / r) * g_cw[1]) ** 2),
          r + sp.Symbol("e"), {r: (0.5, 2), sp.Symbol("e"): (0, 0.4), tt: (0, 6)})
close("Exercise 32.2.2: kappa_s(2 pi/5) = 5.207, kappa_s(pi/10) = 0.939",
      [round(float(kf.subs(t, 2 * sp.pi / 5)), 3), round(float(kf.subs(t, sp.pi / 10)), 3), round(5.5 / 3.25 ** 1.5, 3)],
      [5.207, 0.939, 0.939], 1e-12)
tz = np.linspace(0, 1, 1000001)
kz = ddg32.kappa("flower", tz)
z0 = tz[np.where(np.sign(kz[:-1]) != np.sign(kz[1:]))[0][0]]
close("Exercise 32.2.2: first inflection point at t = 0.44", round(z0, 2), 0.44, 1e-12)
close("Exercise 32.2.3: circle traversed twice, offset length 4 pi (r - eps)", 4 * np.pi * rv - 0.3 * 4 * np.pi,
      2 * (2 * np.pi * (rv - 0.3)), 1e-12)

summary()
