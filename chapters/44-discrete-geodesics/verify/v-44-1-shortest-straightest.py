"""Verification for Section 44.1 (Shortest and Straightest).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/verify/v-44-1-shortest-straightest.py
"""
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import ch44  # noqa: E402
import dgcode  # noqa: E402
import dgsym  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "44-1-shortest-straightest.html"
ex = dgsym.EXAMPLES

# ---------------------------------------------------------------- Example 44.1.2, Non-example 44.1.3 (symbolic)
sph = ex["sphere"]
th, ph = sph["coords"]
r = sph["params"][0]
X1 = sph["expr"].subs(r, 1)
s, th0 = sp.symbols("s theta0", positive=True)
N = X1  # outward unit normal of the unit sphere


def darboux_kappa_g(curve, tvar):
    """kappa_g = <gamma'', N x gamma'>/|gamma'|^3 for a curve on the unit sphere (any regular parameter)."""
    g1 = curve.diff(tvar)
    g2 = g1.diff(tvar)
    Nc = curve           # unit sphere: N = point
    return sp.simplify(g2.dot(Nc.cross(g1)) / sp.sqrt(g1.dot(g1)) ** 3)


lat = X1.subs({th: th0, ph: s})
sym_equal("Non-example 44.1.3: latitude kappa_g = cot(theta0)", darboux_kappa_g(lat, s), sp.cot(th0),
          {th0: (0.2, 2.9), s: (0, 6)})
eq = X1.subs({th: sp.pi / 2, ph: s})
sym_equal("Example 44.1.2(b): equator kappa_g = 0", darboux_kappa_g(eq, s), 0, {s: (0, 6)})
# a tilted great circle a cos s + b sin s
a = sp.Matrix([1, 0, 0])
b = sp.Matrix([0, sp.cos(sp.Rational(1, 3)), sp.sin(sp.Rational(1, 3))])
gc = a * sp.cos(s) + b * sp.sin(s)
sym_equal("Example 44.1.2(b): tilted great circle kappa_g = 0", darboux_kappa_g(gc, s), 0, {s: (0, 6)})
# helix on the cylinder of radius 1, (a, b) = (1, 0.3): speed sqrt(1.09), kappa_g = 0
hel = ex["helix"]
t = hel["coords"][0]
H = hel["expr"].subs({hel["params"][0]: 1, hel["params"][1]: sp.Rational(3, 10)})
H1, H2 = H.diff(t), H.diff(t, 2)
Ncyl = sp.Matrix([sp.cos(t), sp.sin(t), 0])
close("Example 44.1.2(c): helix speed sqrt(1.09) = 1.04403", float(sp.sqrt(sp.simplify(H1.dot(H1)))), 1.04403, 5e-6)
sym_equal("Example 44.1.2(c): helix kappa_g = 0 on the cylinder", H2.dot(Ncyl.cross(H1)), 0, {t: (0, 6)})
# numbers of Non-example 44.1.3 and Figure 44.1.1
p = ch44.sphere_point(np.pi / 3, -1.1)
q = ch44.sphere_point(np.pi / 3, 1.1)
close("Non-example 44.1.3: latitude arc length 1.9053", np.sin(np.pi / 3) * 2.2, 1.9053, 5e-5)
close("Non-example 44.1.3: great arc length 1.7634", np.arccos(p @ q), 1.7634, 5e-5)
close("Figure 44.1.1: long arc length 4.5198", 2 * np.pi - np.arccos(p @ q), 4.5198, 5e-5)

# ---------------------------------------------------------------- Example 44.1.4 (obstacle)
P0, Q0 = np.array([-2.0, 0.0]), np.array([1.0, 1.0])
dpq = Q0 - P0
tf = -(P0 @ dpq) / (dpq @ dpq)
close("Example 44.1.4: segment pq passes at distance 2/sqrt(10) = 0.632 from 0 (inside the segment)",
      [np.linalg.norm(P0 + tf * dpq), float(0 < tf < 1)], [2 / np.sqrt(10), 1.0], 1e-12)
t1 = np.array([np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)])
t2 = np.array([0.0, 1.0])
check("Example 44.1.4: <t1 - p, t1> = 0, <t2 - q, t2> = 0 (tangent points)",
      abs((t1 - P0) @ t1) < 1e-12 and abs((t2 - Q0) @ t2) < 1e-12)
close("Example 44.1.4: tangent segments sqrt(3) and 1", [np.linalg.norm(t1 - P0), np.linalg.norm(t2 - Q0)],
      [np.sqrt(3), 1.0], 1e-12)
total = np.sqrt(3) + np.pi / 6 + 1
close("Example 44.1.4: total sqrt 3 + pi/6 + 1 = 3.2556", total, 3.2556, 5e-5)
s1, s2 = np.array([np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)]), np.array([1.0, 0.0])
check("Example 44.1.4: lower tangent points", abs((s1 - P0) @ s1) < 1e-12 and abs((s2 - Q0) @ s2) < 1e-12)
lower = np.linalg.norm(s1 - P0) + 2 * np.pi / 3 + np.linalg.norm(s2 - Q0)
close("Example 44.1.4: the other side sqrt 3 + 2 pi/3 + 1 = 4.8264", lower, 4.8264, 5e-5)
# independent: visibility graph around a circumscribed polygon of the disk (a curve in Omega)
import heapq  # noqa: E402
Nb = 2000
angs = 2 * np.pi * np.arange(Nb) / Nb
Bd = np.stack([np.cos(angs), np.sin(angs)], 1) / np.cos(np.pi / Nb)
nodes = np.vstack([P0, Q0, Bd])


def _visible(A_, B_):
    d_ = B_ - A_
    tt_ = np.clip(-(A_ @ d_) / (d_ @ d_), 0, 1)
    return np.linalg.norm(A_ + tt_ * d_) >= 1 - 1e-12


adj = {i: [] for i in range(len(nodes))}
for k in range(Nb):
    i, j = 2 + k, 2 + (k + 1) % Nb
    w_ = np.linalg.norm(nodes[i] - nodes[j])
    adj[i].append((j, w_))
    adj[j].append((i, w_))
for s_ in (0, 1):
    for k in range(Nb):
        if _visible(nodes[s_], nodes[2 + k]):
            w_ = np.linalg.norm(nodes[s_] - nodes[2 + k])
            adj[s_].append((2 + k, w_))
            adj[2 + k].append((s_, w_))
dist_ = np.full(len(nodes), np.inf)
dist_[0] = 0.0
hp = [(0.0, 0)]
while hp:
    d0, i = heapq.heappop(hp)
    if d0 > dist_[i]:
        continue
    for j, w_ in adj[i]:
        if d0 + w_ < dist_[j]:
            dist_[j] = d0 + w_
            heapq.heappush(hp, (dist_[j], j))
close("Example 44.1.4: visibility graph around a 2000-gon outside the disk gives the same length", dist_[1], total, 1e-5)
# Exercise 44.1.4(a): m = intersection of the tangent lines at t1, t2
m_ = np.array([np.sqrt(3) - 2, 1.0])
check("Exercise 44.1.4(a): m lies on both tangent lines", abs(m_ @ t1 - 1) < 1e-12 and abs(m_ @ t2 - 1) < 1e-12)
check("Exercise 44.1.4(a): p - m - q avoids the open disk", _visible(P0, m_) and _visible(m_, Q0))
close("Exercise 44.1.4(a): |t1 - m| = |t2 - m| = tan(pi/12) = 2 - sqrt 3",
      [np.linalg.norm(t1 - m_), np.linalg.norm(t2 - m_)], [np.tan(np.pi / 12)] * 2, 1e-12)
poly_len = np.linalg.norm(m_ - P0) + np.linalg.norm(Q0 - m_)
close("Exercise 44.1.4(a): length 5 - sqrt 3 = 3.2679 > 3.2556", [poly_len, float(poly_len > total)],
      [3.2679, 1.0], 5e-5)
close("Exercise 44.1.4(a): 2 tan(pi/12) = 0.5359 vs pi/6 = 0.5236", [2 * np.tan(np.pi / 12), np.pi / 6],
      [0.5359, 0.5236], 5e-5)

# ---------------------------------------------------------------- Example 44.1.6 (cylinder), Exercise 44.1.1
Hh = 0.6 * np.pi
close("Example 44.1.6: vertical segment 0.6 pi = 1.885", Hh, 1.885, 5e-4)
close("Example 44.1.6: k = +-1 length 2 pi sqrt(1.09) = 6.560", np.hypot(2 * np.pi, Hh), 6.560, 5e-4)
close("Example 44.1.6: 2 pi sqrt(1.09) equals the helix length of one turn", np.hypot(2 * np.pi, Hh),
      2 * np.pi * np.sqrt(1 + 0.3 ** 2), 1e-12)
cyl = ex["cylinder"]
u, v = cyl["coords"]
rc = cyl["params"][0]
g_cyl = list(dgsym.first_ff(cyl["expr"], u, v))
sym_equal("Example 44.1.6: (s, v) = (r u, v) is a local isometry (E = r^2, F = 0, G = 1)", g_cyl,
          [rc ** 2, 0, 1], {rc: (0.5, 2)})
tt_c = sp.symbols("tc", real=True)
uf, vf = sp.Function("u")(tt_c), sp.Function("v")(tt_c)
gam_c = sp.Matrix([sp.cos(uf), sp.sin(uf), vf])
acc_c = gam_c.diff(tt_c, 2)
sym_equal("Example 44.1.6: <gamma'', x_u> = u'', <gamma'', x_v> = v'' on the cylinder (r = 1)",
          [sp.simplify(acc_c.dot(sp.Matrix([-sp.sin(uf), sp.cos(uf), 0]))), acc_c.dot(sp.Matrix([0, 0, 1]))],
          [uf.diff(tt_c, 2), vf.diff(tt_c, 2)])
rr, hh = 0.7, 1.3
lens = sorted(np.hypot(rr * np.pi * (2 * k + 1), hh) for k in range(-4, 4))
check("Exercise 44.1.1: exactly two shortest geodesics (k = 0, -1)", abs(lens[0] - lens[1]) < 1e-12 and lens[2] > lens[1] + 1e-6)

# ---------------------------------------------------------------- Lemma 44.1.7, Exercise 44.1.2
tt = sp.symbols("t", real=True)
E1 = sp.integrate(sp.Rational(1, 2) * 1, (tt, 0, 1))
E2 = sp.integrate(sp.Rational(1, 2) * (2 * tt) ** 2, (tt, 0, 1))
check("Exercise 44.1.2: E(gamma1) = 1/2, E(gamma2) = 2/3", E1 == sp.Rational(1, 2) and E2 == sp.Rational(2, 3))
check("Lemma 44.1.7: L^2 <= 2(b - a)E, equality for constant speed", 1 <= 2 * E2 and 1 == 2 * E1)
rng = np.random.default_rng(1)
for k in range(3):
    ts = np.linspace(0, 1, 2001)
    c = rng.normal(size=(4, 3))
    curve = np.stack([np.polyval(c[:, i], ts) for i in range(3)], 1)
    d = np.gradient(curve, ts, axis=0)
    sp_ = np.linalg.norm(d, axis=1)
    L_ = np.trapezoid(sp_, ts)
    E_ = 0.5 * np.trapezoid(sp_ ** 2, ts)
    check(f"Lemma 44.1.7: random cubic curve {k}: L^2 <= 2E", L_ ** 2 <= 2 * E_ + 1e-9)

# ---------------------------------------------------------------- (44.1.2) and Proposition 44.1.8 (symbolic)
eps = sp.symbols("epsilon", real=True)
f = sp.Function("f")
gam = sp.Matrix([2 * tt + 1, 3 - tt])                     # a line
eta = sp.Matrix([tt * (1 - tt), sp.sin(sp.pi * tt)])      # endpoints fixed
Eeps = sp.integrate(sp.Rational(1, 2) * ((gam + eps * eta).diff(tt)).dot((gam + eps * eta).diff(tt)), (tt, 0, 1))
sym_equal("Prop 44.1.8: line is critical, dE/deps = 0", sp.diff(Eeps, eps).subs(eps, 0), 0)
sym_equal("Prop 44.1.8: E(gamma + eta) - E(gamma) = (1/2) int |eta'|^2", sp.expand(Eeps - Eeps.subs(eps, 0)),
          eps ** 2 * sp.integrate(sp.Rational(1, 2) * eta.diff(tt).dot(eta.diff(tt)), (tt, 0, 1)))
gam2 = sp.Matrix([tt ** 2, tt])
E2e = sp.integrate(sp.Rational(1, 2) * ((gam2 + eps * eta).diff(tt)).dot((gam2 + eps * eta).diff(tt)), (tt, 0, 1))
lhs = sp.diff(E2e, eps).subs(eps, 0)
rhs = -sp.integrate(gam2.diff(tt, 2).dot(eta), (tt, 0, 1))
sym_equal("(44.1.2): dE/deps = -int <gamma'', eta> (non-line curve)", lhs, rhs)
check("(44.1.2): the non-line curve is not critical", sp.simplify(lhs) != 0)

# ---------------------------------------------------------------- Proposition 44.1.9, Exercise 44.1.3
lat_b = X1.subs({th: th0, ph: s})
acc = lat_b.diff(s, 2)
normal_part = acc.dot(lat_b) * lat_b
tang = sp.simplify(acc - normal_part)
NxT = sp.Matrix([-sp.cos(th0) * sp.cos(s), -sp.cos(th0) * sp.sin(s), sp.sin(th0)])
sym_equal("Exercise 44.1.3(b): tangential acceleration of the latitude = sin cos N x t", tang,
          sp.sin(th0) * sp.cos(th0) * NxT, {th0: (0.2, 2.9), s: (0, 6)})
mer = X1.subs({th: s, ph: sp.Rational(7, 10)})
sym_equal("Exercise 44.1.3(a): meridian gamma'' = -gamma", mer.diff(s, 2), -mer, {s: (0.1, 3)})
# numerical first variation on the sphere: dE/deps for the latitude is nonzero, for the equator zero


def energy_sphere(theta_fn, phi_fn, ts):
    th_, ph_ = theta_fn(ts), phi_fn(ts)
    dth, dph = np.gradient(th_, ts), np.gradient(ph_, ts)
    return 0.5 * np.trapezoid(dth ** 2 + np.sin(th_) ** 2 * dph ** 2, ts)


ts = np.linspace(0, 1, 20001)
bump = np.sin(np.pi * ts)
for name, th_c, crit in (("equator", np.pi / 2, True), ("latitude pi/3", np.pi / 3, False)):
    h = 1e-4
    Ep = energy_sphere(lambda x: th_c + h * np.sin(np.pi * x), lambda x: 2.2 * x, ts)
    Em = energy_sphere(lambda x: th_c - h * np.sin(np.pi * x), lambda x: 2.2 * x, ts)
    dE = (Ep - Em) / (2 * h)
    check(f"Prop 44.1.9: {name} is {'critical' if crit else 'not critical'} (dE/deps = {dE:.2e})",
          (abs(dE) < 1e-6) if crit else (abs(dE) > 1e-2))

# ---------------------------------------------------------------- Example 44.1.11, (44.1.3), Exercise 44.1.5
ell, m = sp.symbols("ell", positive=True), sp.symbols("m", positive=True, integer=True)
w = m * sp.pi / ell
integrand = sp.Rational(1, 2) * (eps ** 2 * w ** 2 * sp.cos(w * tt) ** 2 + sp.cos(eps * sp.sin(w * tt)) ** 2)
d2 = sp.diff(integrand, eps, 2).subs(eps, 0)
E2sym = sp.integrate(sp.expand(sp.simplify(d2)), (tt, 0, ell))
sym_equal("Exercise 44.1.5: E''(0) = (ell/2)(m^2 pi^2/ell^2 - 1)", sp.simplify(E2sym),
          ell / 2 * (m ** 2 * sp.pi ** 2 / ell ** 2 - 1), {ell: (1, 8)})
f3 = lambda L_, mm=1: L_ / 2 * (mm ** 2 * np.pi ** 2 / L_ ** 2 - 1)  # noqa: E731
close("(44.1.3): ell = 1.4 pi gives -1.0771", f3(1.4 * np.pi), -1.0771, 5e-5)
close("(44.1.3): ell = 0.6 pi gives +1.6755", f3(0.6 * np.pi), 1.6755, 5e-5)
check("Exercise 44.1.5: ell = 2.5 pi has exactly two negative directions (m = 1, 2)",
      [mm for mm in range(1, 10) if f3(2.5 * np.pi, mm) < 0] == [1, 2])
# Lemma 44.1.7 consequence in Example 44.1.11: E(eps) < E(0) implies shorter
L_eq = 1.4 * np.pi
tq = np.linspace(0, L_eq, 40001)
th_e = np.pi / 2 - 0.2 * np.sin(np.pi * tq / L_eq)
dth = np.gradient(th_e, tq)
speed = np.sqrt(dth ** 2 + np.sin(th_e) ** 2)
check("Example 44.1.11: the bumped long arc (eps = 0.2) is shorter than 1.4 pi", np.trapezoid(speed, tq) < L_eq - 1e-3)
for e_ in (0.35, -0.35):
    th_f = np.pi / 2 - e_ * np.sin(np.pi * tq / L_eq)
    sp_f = np.sqrt(np.gradient(th_f, tq) ** 2 + np.sin(th_f) ** 2)
    check(f"Figure 44.1.3(a): the bumped arc eps = {e_} is shorter than the long arc", np.trapezoid(sp_f, tq) < L_eq - 1e-3)
# the variation by reparametrization increases the energy
hfun = tq + 0.3 * np.sin(np.pi * tq / L_eq) * L_eq / np.pi * 0.5
E_rep = 0.5 * np.trapezoid(np.gradient(hfun, tq) ** 2, tq)
check("Example 44.1.11: reparametrizing increases the energy", E_rep > L_eq / 2 + 1e-4)

# ---------------------------------------------------------------- Exercise 44.1.6 (Coding)
check("Exercise 44.1.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-44-1-6") <= 30)
ns, out = ch44.run_printed(PAGE, "code-ex-44-1-6")
check("Exercise 44.1.6: printed output matches the page", out.strip() == ch44.printed_block(PAGE, "out-ex-44-1-6").strip(), out)
P = ns["P"]
close("Exercise 44.1.6: converged polygon lies on the great circle plane", np.abs(P @ np.cross(P[0], P[-1])).max(), 0, 1e-12)
check("Exercise 44.1.6: equal spacing (constant speed)", np.ptp(np.linalg.norm(np.diff(P, axis=0), axis=1)) < 1e-12)
summary()
