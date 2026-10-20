"""Verification for Section 44.5 (The Geodesic Equation).

Run from the project root: PYTHONPATH=tools python3 chapters/44-discrete-geodesics/verify/v-44-5-geodesic-equation.py
"""
import sys
from pathlib import Path

import dgmesh as dm  # noqa: F401  (thread limit)
import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import ch44  # noqa: E402
import dgcode  # noqa: E402
import dgsym  # noqa: E402
from dgcheck import check, close, sym_equal, summary  # noqa: E402

PAGE = HERE.parents[1] / "44-5-geodesic-equation.html"
ex = dgsym.EXAMPLES

# ---------------------------------------------------------------- Implementation 44.5 (two blocks)
for bid in ("code-44-5-1", "code-44-5-2"):
    check(f"Implementation 44.5: {bid} at most 30 lines", dgcode.code_lines(PAGE, bid) <= 30)
check("Implementation 44.5: lambdify uses modules='numpy'", 'modules="numpy"' in dgcode.code_block(PAGE, "code-44-5-1"))
ns, _ = ch44.run_printed(PAGE, "code-44-5-1")
ns, out = ch44.run_printed(PAGE, "code-44-5-2", ns)
check("Implementation 44.5: printed output matches the page", out.strip() == ch44.printed_block(PAGE, "out-44-5-2").strip(), out)


# ---------------------------------------------------------------- Proposition 44.5.4: extrinsic Gamma = metric formula
def extrinsic_gamma(X, coords, positive=()):
    """tangential coefficients of x_ij in the basis x_1, x_2 (solve with the Gram matrix)"""
    D = [X.diff(q) for q in coords]
    g = sp.Matrix(2, 2, lambda i, j: D[i].dot(D[j]))
    ginv = g.inv()
    out = [[[0] * 2 for _ in range(2)] for _ in range(2)]
    for i in range(2):
        for j in range(2):
            xij = X.diff(coords[i]).diff(coords[j])
            rhs = sp.Matrix([xij.dot(D[l]) for l in range(2)])
            sol = ginv * rhs
            for k in range(2):
                out[k][i][j] = dgsym.simp(sol[k], positive)
    return out


for key in ("sphere", "cylinder", "torus", "saddle"):
    e = ex[key]
    X = e["expr"]
    coords = e["coords"]
    Gm = dgsym.christoffel(dgsym.induced_metric(X, coords, e["positive"]), coords, e["positive"])
    Ge = extrinsic_gamma(X, coords, e["positive"])
    sym_equal(f"Prop 44.5.4: {key}: tangential part of x_ij = metric formula (44.5.2)",
              [Ge[k][i][j] for k in range(2) for i in range(2) for j in range(2)],
              [Gm[k][i][j] for k in range(2) for i in range(2) for j in range(2)], e.get("domain"))
sph = ex["sphere"]
th, ph = sph["coords"]
Gs = dgsym.christoffel(dgsym.induced_metric(sph["expr"], (th, ph), sph["positive"]), (th, ph), sph["positive"])
for (k, i, j), val in sph["expected"]["christoffel"].items():
    sym_equal(f"Example 44.5.7: sphere Gamma^{k}_{i}{j} = §7 value", Gs[k][i][j], val, sph["domain"])
check("Exercise 44.5.1: sphere Gamma^theta_thetatheta = 0", sp.simplify(Gs[0][0][0]) == 0)
# Gamma_ijl identity used in the proof (sphere)
D = [sph["expr"].diff(q) for q in (th, ph)]
g = dgsym.induced_metric(sph["expr"], (th, ph), sph["positive"])
ok = True
for i in range(2):
    for j in range(2):
        for l_ in range(2):
            lhs = sp.Rational(1, 2) * (sp.diff(g[j, l_], (th, ph)[i]) + sp.diff(g[i, l_], (th, ph)[j]) - sp.diff(g[i, j], (th, ph)[l_]))
            ok &= sp.simplify(lhs - sph["expr"].diff((th, ph)[i]).diff((th, ph)[j]).dot(D[l_])) == 0
check("Prop 44.5.4 proof: <x_ij, x_l> = (1/2)(d_i g_jl + d_j g_il - d_l g_ij) on the sphere", ok)

# ---------------------------------------------------------------- Proposition 44.5.5: (44.5.3) for a general curve
t = sp.symbols("t", real=True)
r = sph["params"][0]
c1, c2 = sp.Rational(1, 2) + t / 3 + t ** 2 / 7, sp.sin(t) + t
Xs = sph["expr"].subs(r, 1)
curve = Xs.subs({th: c1, ph: c2})
acc = curve.diff(t, 2)
N = curve
tang = acc - acc.dot(N) * N
G1 = dgsym.christoffel(dgsym.induced_metric(Xs, (th, ph), sph["positive"]), (th, ph), sph["positive"])
dc = [c1.diff(t), c2.diff(t)]
ddc = [c1.diff(t, 2), c2.diff(t, 2)]
basis = [Xs.diff(th).subs({th: c1, ph: c2}), Xs.diff(ph).subs({th: c1, ph: c2})]
formula = sp.zeros(3, 1)
for k in range(2):
    coef = ddc[k] + sum(G1[k][i][j].subs({th: c1, ph: c2}) * dc[i] * dc[j] for i in range(2) for j in range(2))
    formula += coef * basis[k]
ft = sp.lambdify(t, list(tang), modules="numpy")
ff = sp.lambdify(t, list(formula), modules="numpy")
worst = max(np.abs(np.array(ft(tv), float) - np.array(ff(tv), float)).max() for tv in np.linspace(0.1, 1.0, 13))
check("Prop 44.5.5: (44.5.3) D_t gamma' = sum_k (gamma''^k + Gamma gamma'^i gamma'^j) x_k (sphere, general curve)",
      worst < 1e-12, f"{worst}")

# ---------------------------------------------------------------- Proposition 44.5.2(b): metric compatibility (numerical)
rng = np.random.default_rng(2)
tt = np.linspace(0, 1, 2001)
thc, phc = 0.7 + 0.3 * tt, 0.2 + 1.1 * tt ** 2
P = np.stack([np.sin(thc) * np.cos(phc), np.sin(thc) * np.sin(phc), np.cos(thc)], 1)
a1, a2 = np.cos(3 * tt), np.sin(2 * tt) + 1
xth = np.stack([np.cos(thc) * np.cos(phc), np.cos(thc) * np.sin(phc), -np.sin(thc)], 1)
xph = np.stack([-np.sin(thc) * np.sin(phc), np.sin(thc) * np.cos(phc), 0 * tt], 1)
Vf, Wf = a1[:, None] * xth + a2[:, None] * xph, (a2 - a1)[:, None] * xth + (2 * a1)[:, None] * xph
dV, dW = np.gradient(Vf, tt, axis=0), np.gradient(Wf, tt, axis=0)
DV = dV - np.einsum("ij,ij->i", dV, P)[:, None] * P
DW = dW - np.einsum("ij,ij->i", dW, P)[:, None] * P
lhs = np.gradient(np.einsum("ij,ij->i", Vf, Wf), tt)
rhs = np.einsum("ij,ij->i", DV, Wf) + np.einsum("ij,ij->i", Vf, DW)
check("Prop 44.5.2(b): d/dt <V, W> = <D_t V, W> + <V, D_t W> (sphere, finite differences)",
      np.abs(lhs - rhs)[5:-5].max() < 1e-5)

# ---------------------------------------------------------------- Example 44.5.7 (sphere)
T, Pf = sp.Function("theta")(t), sp.Function("phi")(t)
eq2 = Pf.diff(t, 2) + 2 * sp.cos(T) / sp.sin(T) * T.diff(t) * Pf.diff(t)
cons = sp.diff(sp.sin(T) ** 2 * Pf.diff(t), t)
sym_equal("Example 44.5.7: d/dt(sin^2 theta phi') = sin^2 theta * (second equation)", cons, sp.expand(sp.sin(T) ** 2 * eq2))
eq1 = lambda thf, phf: sp.simplify(thf.diff(t, 2) - sp.sin(thf) * sp.cos(thf) * phf.diff(t) ** 2)  # noqa: E731
eq2f = lambda thf, phf: sp.simplify(phf.diff(t, 2) + 2 * sp.cot(thf) * thf.diff(t) * phf.diff(t))  # noqa: E731
check("Example 44.5.7: equator (pi/2, t) and meridian (t, 0.3) solve both equations",
      eq1(sp.pi / 2 + 0 * t, t) == 0 and eq2f(sp.pi / 2 + 0 * t, t) == 0 and eq1(t, sp.Rational(3, 10) + 0 * t) == 0
      and eq2f(t, sp.Rational(3, 10) + 0 * t) == 0)
th0 = sp.Rational(1, 2)
check("Example 44.5.7: a circle of latitude theta_0 = 1/2 does not", sp.simplify(eq1(th0 + 0 * t, t)) != 0)

# ---------------------------------------------------------------- Example 44.5.8 (cylinder)
cy = ex["cylinder"]
Gc = dgsym.christoffel(dgsym.induced_metric(cy["expr"], cy["coords"]), cy["coords"])
check("Example 44.5.8: all Christoffel symbols of the cylinder vanish", all(sp.simplify(Gc[k][i][j]) == 0
                                                                          for k in range(2) for i in range(2) for j in range(2)))

# ---------------------------------------------------------------- Example 44.5.9 (torus), (44.5.5), Exercises 44.5.3-44.5.4
to = ex["torus"]
u, v = to["coords"]
Rt, rt = to["params"]
Gt = dgsym.christoffel(dgsym.induced_metric(to["expr"], (u, v), to["positive"]), (u, v), to["positive"])
rho = Rt + rt * sp.cos(u)
sym_equal("Example 44.5.9: Gamma^u_vv = rho sin u / r", Gt[0][1][1], rho * sp.sin(u) / rt, to["domain"])
sym_equal("Example 44.5.9: Gamma^v_uv = -r sin u / rho", Gt[1][0][1], -rt * sp.sin(u) / rho, to["domain"])
check("Example 44.5.9: the other Christoffel symbols vanish",
      all(sp.simplify(Gt[k][i][j]) == 0 for (k, i, j) in ((0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1))))
U, Vv = sp.Function("u")(t), sp.Function("v")(t)
rhoU = Rt + rt * sp.cos(U)
eqv = Vv.diff(t, 2) - 2 * rt * sp.sin(U) / rhoU * U.diff(t) * Vv.diff(t)
sym_equal("Example 44.5.9: d/dt(rho^2 v') = rho^2 * (second equation)", sp.diff(rhoU ** 2 * Vv.diff(t), t), sp.expand(rhoU ** 2 * eqv))
close("Example 44.5.9: c = 2.8 cos 45 deg = 1.9799 and 2.8 cos 75 deg = 0.7247", [2.8 * np.cos(np.pi / 4), 2.8 * np.cos(5 * np.pi / 12)],
      [1.9799, 0.7247], 5e-5)
close("Example 44.5.9: turning parallel cos u = (c - R)/r = -0.0251", (2.8 * np.cos(np.pi / 4) - 2) / 0.8, -0.0251, 5e-5)
close("Exercise 44.5.4: inner equator reached iff psi_0 > arccos(3/7) = 64.62 deg", np.degrees(np.arccos(3 / 7)), 64.62, 5e-3)
C, W, Pp = ch44.surface_geodesic(to["expr"].subs({Rt: 2, rt: 0.8}), (u, v), {}, np.array([0.0, 0.0]),
                                 np.array([np.sin(np.pi / 4) / 0.8, np.cos(np.pi / 4) / 2.8]), np.linspace(0, 40, 8001))
cl = (2 + 0.8 * np.cos(C[:, 0])) ** 2 * W[:, 1]
close("Example 44.5.9: RK4 keeps the Clairaut constant to 1e-11 (8000 steps, t <= 40)", np.ptp(cl), 0, 1e-11)
close("Example 44.5.9: the 45 deg geodesic turns at cos u = -0.0251", np.cos(C[:, 0]).min(), -0.0251, 5e-5)
# Exercise 44.5.3: surface of revolution
rv = ex["revolution"]
Gr = dgsym.christoffel(dgsym.induced_metric(rv["expr"], rv["coords"]), rv["coords"])
ur = rv["coords"][0]
rho_f = rv["functions"][0](ur)
sym_equal("Exercise 44.5.3: Gamma^v_uv = rho'/rho on a surface of revolution", Gr[1][0][1], sp.diff(rho_f, ur) / rho_f)
check("Exercise 44.5.3: Gamma^v_uu = Gamma^v_vv = 0", sp.simplify(Gr[1][0][0]) == 0 and sp.simplify(Gr[1][1][1]) == 0)
Er = sp.diff(rho_f, ur) ** 2 + sp.diff(rv["functions"][1](ur), ur) ** 2
sym_equal("Exercise 44.5.3: Gamma^u_uu = E_u/(2E)", Gr[0][0][0], sp.diff(Er, ur) / (2 * Er))
# Exercise 44.5.2: hyperbolic plane
hp = ex["hyperbolic_plane"]
xh, yh = hp["coords"]
Gh = dgsym.christoffel(hp["expr"], (xh, yh))
for (k, i, j), val in hp["expected"]["christoffel"].items():
    sym_equal(f"Exercise 44.5.2: hyperbolic Gamma^{k}_{i}{j}", Gh[k][i][j], val, hp["domain"])
yt = sp.exp(t)
eqy = sp.diff(yt, t, 2) + Gh[1][0][0].subs(yh, yt) * 0 + Gh[1][1][1].subs(yh, yt) * sp.diff(yt, t) ** 2
check("Exercise 44.5.2: x = x0, y = e^t solves the geodesic equation", sp.simplify(eqy) == 0)
check("Exercise 44.5.2: and has hyperbolic speed 1", sp.simplify(sp.diff(yt, t) ** 2 / yt ** 2 - 1) == 0)

# ---------------------------------------------------------------- Figure 44.5.2 drift numbers
drift = []
for n in (100, 1600):
    tq = np.linspace(0, 20, n + 1)
    C, W, _ = ch44.surface_geodesic(to["expr"].subs({Rt: 2, rt: 0.8}), (u, v), {}, np.array([0.0, 0.0]),
                                    np.array([np.sin(np.pi / 4) / 0.8, np.cos(np.pi / 4) / 2.8]), tq)
    c_ = (2 + 0.8 * np.cos(C[:, 0])) ** 2 * W[:, 1]
    drift.append(np.abs(c_ - c_[0]).max())
check("Figure 44.5.2: drift 1.1e-5 at h = 0.2 and 1.2e-10 at h = 0.0125",
      abs(drift[0] / 1.1e-5 - 1) < 0.1 and abs(drift[1] / 1.2e-10 - 1) < 0.15, str(drift))

# ---------------------------------------------------------------- Exercise 44.5.6 (Coding)
check("Exercise 44.5.6: at most 30 lines", dgcode.code_lines(PAGE, "code-ex-44-5-6") <= 30)
ns2, out2 = ch44.run_printed(PAGE, "code-ex-44-5-6", dict(ns))
check("Exercise 44.5.6: printed output matches the page", out2.strip() == ch44.printed_block(PAGE, "out-ex-44-5-6").strip(), out2)
summary()
