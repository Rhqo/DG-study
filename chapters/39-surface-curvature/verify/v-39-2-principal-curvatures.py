"""Verification for Section 39.2 (Normal and Principal Curvatures).

Run from the project root: PYTHONPATH=tools python3 chapters/39-surface-curvature/verify/v-39-2-principal-curvatures.py
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

PAGE = HERE.parents[1] / "39-2-principal-curvatures.html"
ex = dgsym.EXAMPLES

# ---------------------------------------------------------------- Implementation 39.2 and Exercise 39.2.6
check("Implementation 39.2: at most 30 lines", dgcode.code_lines(PAGE, "code-39-2-1") <= 30)
ns, out = ddg39.run_printed(PAGE, "code-39-2-1")
check("Implementation 39.2: printed output matches the page", out.strip() == ddg39.printed_block(PAGE, "out-39-2-1").strip(),
      out)
k, kn, kg = ns["darboux"](ns["sphere"], lambda s: (np.pi / 3, s), 0.4)
close("Implementation 39.2: latitude pi/3 (kappa, kappa_n, kappa_g)", [k, kn, kg], [2 / np.sqrt(3), -1, 1 / np.sqrt(3)], 1e-6)
check("Exercise 39.2.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-39-2-6") <= 30)
ns2, out2 = ddg39.run_printed(PAGE, "code-ex-39-2-6", dict(ns))
check("Exercise 39.2.6: printed output matches the page",
      out2.strip() == ddg39.printed_block(PAGE, "out-ex-39-2-6").strip(), out2)

# torus (2, 3) knot: Darboux frame positively oriented, kappa_n = g(W X, X) / g(X, X)
c = lambda s: (2 * s, 3 * s)  # noqa: E731
kap, kn, kg, t, N = ddg39.darboux(ddg39.torus, c, 0.4)
check("Def 39.2.1: Darboux frame (t, N x t, N) is positively oriented and orthonormal",
      abs(np.linalg.det(np.column_stack([t, np.cross(N, t), N])) - 1) < 1e-9 and abs(t @ N) < 1e-8)
X = np.array([2.0, 3.0])
g, hh, n_, xu, xv = ddg39.forms(ddg39.torus, 0.8, 1.2)
W = np.linalg.solve(g, hh)
close("Prop 39.2.5: kappa_n of the (2,3) knot = g(W X, X)/g(X, X)", kn, (W @ X) @ g @ X / (X @ g @ X), 1e-6)
close("Implementation 39.2: (2,3) knot values", [kap, kn, kg], [0.42415, 0.31318, -0.28604], 5e-6)
# Prop 39.2.5: two curves with the same tangent have the same kappa_n but different kappa_g
c2 = lambda s: (0.8 + 2 * (s - 0.4) + 3.0 * (s - 0.4) ** 2, 1.2 + 3 * (s - 0.4) - 2.0 * (s - 0.4) ** 2)  # noqa: E731
kap2, kn2, kg2, *_ = ddg39.darboux(ddg39.torus, c2, 0.4)
check("Prop 39.2.5: same direction, same kappa_n, different kappa_g", abs(kn2 - kn) < 1e-5 and abs(kg2 - kg) > 0.05,
      f"{kn2} {kn} {kg2} {kg}")

# ---------------------------------------------------------------- Example 39.2.2, (39.2.1), Example 39.2.6 (symbolic)
sp_ = ex["sphere"]
th, ph = sp_["coords"]
r = sp_["params"][0]
th0 = sp.symbols("theta0", positive=True)
Xs = sp_["expr"]
gam = Xs.subs(th, th0)
s = sp.symbols("s", real=True)
gs = gam.subs(ph, s / (r * sp.sin(th0)))                  # arc length
T = sp.simplify(gs.diff(s))
Tp = sp.simplify(T.diff(s))
Ns = (gs / r)
sym_equal("Example 39.2.6: |t| = 1", T.dot(T), 1, {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
NxT = sp.simplify(Ns.cross(T))
phi_s = s / (r * sp.sin(th0))
sym_equal("(39.2.1): N x t = (-cos th0 cos ph, -cos th0 sin ph, sin th0)", NxT,
          sp.Matrix([-sp.cos(th0) * sp.cos(phi_s), -sp.cos(th0) * sp.sin(phi_s), sp.sin(th0)]),
          {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
sym_equal("Example 39.2.6: kappa_n = -1/r", Tp.dot(Ns), -1 / r, {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
sym_equal("Example 39.2.6: kappa_g = cot(theta0)/r", Tp.dot(NxT), sp.cot(th0) / r, {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
sym_equal("Prop 39.2.4: kappa^2 = kappa_n^2 + kappa_g^2 (latitude)", Tp.dot(Tp), 1 / (r * sp.sin(th0)) ** 2,
          {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
# Darboux equations for the latitude (m = 0, Exercise 39.2.3)
Np = sp.simplify(Ns.diff(s))
sym_equal("(39.2.2)/Exercise 39.2.3: N' = -kappa_n t - m N x t with m = 0", Np, T / r,
          {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
sym_equal("(39.2.2): (N x t)' = -kappa_g t + m N with m = 0", NxT.diff(s), -sp.cot(th0) / r * T,
          {th0: (0.2, 2.9), r: (0.5, 2.0), s: (0.0, 3.0)})
# Non-example 39.2.3: Frenet n points to the center (0, 0, r cos th0), not +-N, +-N x t when th0 != pi/2
n_fr = sp.simplify(Tp / sp.sqrt(Tp.dot(Tp)))
check("Non-example 39.2.3: Frenet n = -(cos ph, sin ph, 0) differs from +-N and +-N x t (th0 = pi/3)",
      all(sp.simplify((n_fr - sgn * w).subs({th0: sp.pi / 3, r: 1, s: sp.Rational(1, 3)})) != sp.zeros(3, 1)
          for w in (Ns, NxT) for sgn in (1, -1)))
# Figure 39.2.1 numbers
close("Figure 39.2.1: kappa = 1.155, kappa_g = 0.577 at theta0 = pi/3", [1 / np.sin(np.pi / 3), 1 / np.tan(np.pi / 3)],
      [1.1547005, 0.5773503], 1e-7)

# ---------------------------------------------------------------- Example 39.2.7 (helix on the cylinder)
a, b, tt = sp.symbols("a b t", positive=True)
hel = sp.Matrix([a * sp.cos(tt), a * sp.sin(tt), b * tt])
sp_speed = sp.sqrt(hel.diff(tt).dot(hel.diff(tt)))
Th = sp.simplify(hel.diff(tt) / sp_speed)
Thp = sp.simplify(Th.diff(tt) / sp_speed)
Nc = sp.Matrix([sp.cos(tt), sp.sin(tt), 0])
sym_equal("Example 39.2.7: helix kappa_n = -a/(a^2 + b^2)", Thp.dot(Nc), -a / (a ** 2 + b ** 2))
sym_equal("Example 39.2.7: helix kappa_g = 0", Thp.dot(Nc.cross(Th)), 0)
close("Example 39.2.7: (a, b) = (1, 0.3) gives kappa_n = -0.91743", -1 / 1.09, -0.91743, 1e-5)
# Euler for the helix
vt = sp.atan(a / b)
sym_equal("Example 39.2.8: Euler's formula reproduces the helix kappa_n", -sp.sin(vt) ** 2 / a, -a / (a ** 2 + b ** 2))

# ---------------------------------------------------------------- Example 39.2.8 (principal curvatures, symbolic)
for key, expect in [("sphere", lambda P: [-1 / P[0], -1 / P[0]]), ("cylinder", lambda P: [0, -1 / P[0]])]:
    e = ex[key]
    Wm = dgsym.shape_operator(e["expr"], *e["coords"], e["positive"])
    ev = sorted(Wm.eigenvals().keys(), key=lambda z: -float(z.subs({e["params"][0]: 1})))
    full = [x for x in Wm.eigenvals() for _ in range(Wm.eigenvals()[x])]
    full = sorted(full, key=lambda z: -float(z.subs({e["params"][0]: 1})))
    sym_equal(f"Example 39.2.8: principal curvatures of the {key}", full, expect(e["params"]))
to = ex["torus"]
Rr, rr = to["params"]
uu = to["coords"][0]
Wt = dgsym.shape_operator(to["expr"], *to["coords"], to["positive"])
sym_equal("Example 39.2.8(c): torus W = diag(1/r, cos u/(R + r cos u))", Wt, sp.diag(1 / rr, sp.cos(uu) / (Rr + rr * sp.cos(uu))))
sym_equal("Section 39.2: torus kappa1 - kappa2 = R/(r(R + r cos u)) > 0", 1 / rr - sp.cos(uu) / (Rr + rr * sp.cos(uu)),
          Rr / (rr * (Rr + rr * sp.cos(uu))))
sad = ex["saddle"]
Wsd = dgsym.shape_operator(sad["expr"], *sad["coords"]).subs(sad["point"])
sym_equal("Example 39.2.8(d): saddle origin W = diag(2, -2)", Wsd, sp.diag(2, -2))

# ---------------------------------------------------------------- Example 39.2.9 (saddle at q)
k1, k2, X1, X2, gq = ddg39.principal(ddg39.saddle, *ddg39.SADDLE_Q)
close("Example 39.2.9: kappa1, kappa2", [k1, k2], [0.678284, -0.846136], 1e-5)
Wq = np.linalg.solve(gq, ddg39.forms(ddg39.saddle, *ddg39.SADDLE_Q)[1])
Hq, Kq = np.trace(Wq) / 2, np.linalg.det(Wq)
close("Example 39.2.9: kappa = H +- sqrt(H^2 - K)", [Hq + np.sqrt(Hq * Hq - Kq), Hq - np.sqrt(Hq * Hq - Kq)], [k1, k2], 1e-9)
sgn = lambda v: v if v[0] < 0 else -v  # noqa: E731
close("Example 39.2.9: X1, X2 (g-unit, sign chosen with negative first entry)", np.concatenate([sgn(X1), sgn(X2)]),
      [-0.7631, 0.1767, -0.1974, 0.8523], 1e-4)
ang = np.degrees(np.arccos(abs(X1 @ X2) / np.linalg.norm(X1) / np.linalg.norm(X2)))
close("Example 39.2.9: Euclidean angle 63.92 degrees", ang, 63.92, 5e-3)
D = np.column_stack(ddg39.partials(ddg39.saddle, *ddg39.SADDLE_Q))
check("Example 39.2.9: g(X1, X2) = 0 and dx(X1) . dx(X2) = 0", abs(X1 @ gq @ X2) < 1e-10 and abs((D @ X1) @ (D @ X2)) < 1e-8)
check("Figure 39.2.2: saddle has no umbilic (K < 0 on [-1, 1]^2)",
      max(ddg39.KH(ddg39.saddle, x, y)[0] for x in np.linspace(-1, 1, 11) for y in np.linspace(-1, 1, 11)) < 0)

# ---------------------------------------------------------------- Example 39.2.10 (ellipsoid umbilics)
def umbilic_params(A, B, Cc):
    th_u = np.arccos(np.sqrt((B * B - Cc * Cc) / (A * A - Cc * Cc)))
    return th_u


for (A, B, Cc) in [(2.0, 1.5, 1.0), (3.0, 2.0, 1.0)]:
    ell = lambda t_, p_: ddg39.ellipsoid(t_, p_, A, B, Cc)  # noqa: E731
    thu = umbilic_params(A, B, Cc)
    pts = [(thu, 0.0), (np.pi - thu, 0.0), (thu, np.pi), (np.pi - thu, np.pi)]
    ok = True
    for (t_, p_) in pts:
        kk1, kk2, *_ = ddg39.principal(ell, t_, p_)
        ok &= abs(kk1 - kk2) < 1e-6 and abs(kk1 + A * Cc / B ** 3) < 1e-6
    check(f"Example 39.2.10: four umbilics with kappa = -ac/b^3, ellipsoid {(A, B, Cc)}", ok)
P0 = ddg39.ellipsoid(umbilic_params(2, 1.5, 1), 0.0)
close("Example 39.2.10: umbilic position (1.5275, 0, 0.6455)", P0, [1.5275253, 0.0, 0.6454972], 1e-6)
# no other umbilics: on a grid away from the four points the gap k1 - k2 stays positive
thu = umbilic_params(2, 1.5, 1)
U = [(thu, 0.0), (np.pi - thu, 0.0), (thu, np.pi), (np.pi - thu, np.pi)]
gap_min = np.inf
for t_ in np.linspace(0.08, np.pi - 0.08, 60):
    for p_ in np.linspace(0, 2 * np.pi, 120, endpoint=False):
        if min(np.hypot(t_ - a_, (p_ - b_ + np.pi) % (2 * np.pi) - np.pi) for a_, b_ in U) < 0.15:
            continue
        kk1, kk2, *_ = ddg39.principal(ddg39.ellipsoid, t_, p_)
        gap_min = min(gap_min, kk1 - kk2)
check("Example 39.2.10: k1 - k2 > 0.01 on the grid away from the four umbilics", gap_min > 0.01, f"{gap_min}")
# index of each umbilic: +1/2 (sum 2 = chi(S^2))


def param_index(x, center, rad=0.02, n=720):
    angs = []
    for tt_ in np.linspace(0, 2 * np.pi, n + 1):
        X1_ = ddg39.principal(x, center[0] + rad * np.cos(tt_), center[1] + rad * np.sin(tt_))[2]
        angs.append(np.arctan2(X1_[1], X1_[0]))
    tot = sum((a1 - a0 + np.pi / 2) % np.pi - np.pi / 2 for a0, a1 in zip(angs[:-1], angs[1:]))
    return tot / (2 * np.pi)


idx = [param_index(ddg39.ellipsoid, pt) for pt in U]
close("Section 39.2: the four ellipsoid umbilics have index +1/2 (sum 2 = chi)", idx + [sum(idx)], [0.5] * 4 + [2.0], 1e-6)

# ---------------------------------------------------------------- Figure 39.2.3 (Monge umbilics)
for cc, idx_e, nsep in [(-1.0, -0.5, 3), (0.2, 0.5, 1)]:
    d = ddg39.monge(1.0, 0.0, cc, 0.0)
    k1o, k2o, *_ = ddg39.graph_principal(d, 0.0, 0.0)
    check(f"Figure 39.2.3: origin is an umbilic with k = 1 (c = {cc})", abs(k1o - 1) < 1e-12 and abs(k2o - 1) < 1e-12)
    close(f"Figure 39.2.3: line field indices (c = {cc})", [ddg39.line_field_index(d, which=w) for w in (0, 1)],
          [idx_e, idx_e], 1e-6)
    check(f"Figure 39.2.3: {nsep} separatrices per family (c = {cc})",
          all(len(ddg39.separatrix_directions(d, which=w, rad=0.004)) == nsep for w in (0, 1)))
x_, y_ = sp.symbols("x y", real=True)
for cc in (-1, sp.Rational(1, 5)):
    f = (x_ ** 2 + y_ ** 2) / 2 + (x_ ** 3 + 3 * cc * x_ * y_ ** 2) / 6
    Wm = dgsym.shape_operator(sp.Matrix([x_, y_, f]), x_, y_).subs({x_: 0, y_: 0})
    sym_equal(f"Section 39.2: graph umbilic W(0) = I (c = {cc})", Wm, sp.eye(2))

# ---------------------------------------------------------------- Exercises 39.2.2, 39.2.4
close("Exercise 39.2.2: Euler at the saddle origin vanishes at 45 degrees", 2 * np.cos(np.pi / 4) ** 2 - 2 * np.sin(np.pi / 4) ** 2, 0, 1e-15)
u0, v_, s2 = sp.symbols("u0 v s", real=True)
Xt = to["expr"]
ut, vtt = to["coords"]
par = Xt.subs(ut, u0)
rho = Rr + rr * sp.cos(u0)
gp = par.subs(vtt, s2 / rho)
Tt = sp.simplify(gp.diff(s2))
Ttp = sp.simplify(Tt.diff(s2))
Nt = dgsym.unit_normal(Xt, ut, vtt, to["positive"]).subs({ut: u0, vtt: s2 / rho})
dom = {u0: (0.1, 6.0), s2: (0.0, 5.0), Rr: (1.5, 2.5), rr: (0.2, 0.9)}
sym_equal("Exercise 39.2.4: torus parallel kappa_n = cos u0/(R + r cos u0)", Ttp.dot(Nt), sp.cos(u0) / rho, dom)
sym_equal("Exercise 39.2.4: torus parallel kappa_g = -sin u0/(R + r cos u0)", Ttp.dot(Nt.cross(Tt)), -sp.sin(u0) / rho, dom)
cases = [(0.0, 1 / 2.8, 0.0), (np.pi, -1 / 1.2, 0.0), (np.pi / 2, 0.0, -0.5)]
for u00, kn_e, kg_e in cases:
    kap_, kn_, kg_, *_ = ddg39.darboux(ddg39.torus, lambda s_: (u00, s_), 0.7)
    close(f"Exercise 39.2.4: u0 = {u00:.4f} (kappa_n, kappa_g)", [kn_, kg_], [kn_e, kg_e], 1e-6)

summary()
