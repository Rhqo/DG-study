"""Verification for Section 36.3 (Interpolation: Whitney Forms).

Run from the project root:
``PYTHONPATH=tools python3 chapters/36-discrete-exterior-calculus/verify/v-36-3-whitney-forms.py``
"""

import pathlib
import sys

import dgcode
import dgdec as dd
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch36  # noqa: E402

from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE / "36-3-whitney-forms.html"
rng = np.random.default_rng(3603)
J = np.array([[0.0, -1.0], [1.0, 0.0]])

# ---------------------------------------------------------------- Implementation 36.3
ns, same, printed = ch36.run_printed(PAGE, "code-36-3-1", "out-36-3-1")
check("Implementation 36.3: code-36-3-1 prints exactly out-36-3-1", same, printed)
check("Implementation 36.3: code-36-3-1 is at most 30 lines", dgcode.code_lines(PAGE, "code-36-3-1") <= 30)

# ---------------------------------------------------------------- Proposition 36.3.2, Example 36.3.3
V, F = ch36.grid(4)
V = V + 0.05 * rng.normal(size=V.shape)
ok_lag = ok_pu = ok_grad = ok_cont = True
E, index = dd.edge_list(F)
ef = dm.edge_faces(F)
for f, tri in enumerate(F):
    T = V[tri]
    A = 0.5 * ch36.cross2(T[1] - T[0], T[2] - T[0])
    G = dd.barycentric_gradients(T)
    ok_lag &= np.allclose(dd.barycentric_coordinates(T, T), np.eye(3))
    P = rng.dirichlet(np.ones(3), 5) @ T
    ok_pu &= np.allclose(dd.barycentric_coordinates(P, T).sum(1), 1)
    for c in range(3):
        i, j, k = c, (c + 1) % 3, (c + 2) % 3
        ok_grad &= np.allclose(G[i], J @ (T[k] - T[j]) / (2 * A))
for (i, j), fs in ef.items():
    if len(fs) == 2:
        p = V[i] + 0.37 * (V[j] - V[i])
        lam = [dict(zip(F[f].tolist(), dd.barycentric_coordinates(p, V[F[f]]))) for f in fs]
        ok_cont &= all(abs(lam[0].get(v, 0) - lam[1].get(v, 0)) < 1e-12 for v in (i, j)) \
            and abs(lam[0][i] + lam[0][j] - 1) < 1e-12
check("Proposition 36.3.2(b): phi_i(x_j) = delta_ij on every face (jittered 4x4 grid)", ok_lag)
check("Proposition 36.3.2(c): barycentric coordinates sum to 1", ok_pu)
check("Proposition 36.3.2(d): grad phi_i = J(x_k - x_j) / (2 A) for counterclockwise (i, j, k), (36.3.1)", ok_grad)
check("Proposition 36.3.2(a): hat functions agree on shared edges from both faces (continuity)", ok_cont)
T33 = np.array([[0.0, 0.0], [1.0, 0.0], [0.35, 0.9]])
close("Example 36.3.3(b): gradients (-1, -0.7222), (1, -0.3889), (0, 1.1111) and their sum 0",
      np.concatenate([dd.barycentric_gradients(T33).ravel(), dd.barycentric_gradients(T33).sum(0)]),
      [-1, -0.65 / 0.9, 1, -0.35 / 0.9, 0, 1 / 0.9, 0, 0], tol=1e-12)

# ---------------------------------------------------------------- Example 36.3.6 (sympy, reference triangle)
x, y, t = sp.symbols("x y t", real=True)
ph = [1 - x - y, x, y]
grad = [sp.Matrix([sp.diff(p, x), sp.diff(p, y)]) for p in ph]
wh = {(a, b): ph[a] * grad[b] - ph[b] * grad[a] for a in range(3) for b in range(3) if a != b}
check("Example 36.3.6: phi_01 = (1-y)dx + x dy, phi_12 = x dy - y dx, phi_20 = -y dx + (x-1) dy",
      sp.simplify(wh[(0, 1)] - sp.Matrix([1 - y, x])) == sp.zeros(2, 1)
      and sp.simplify(wh[(1, 2)] - sp.Matrix([-y, x])) == sp.zeros(2, 1)
      and sp.simplify(wh[(2, 0)] - sp.Matrix([-y, x - 1])) == sp.zeros(2, 1))
pts = [sp.Matrix([0, 0]), sp.Matrix([1, 0]), sp.Matrix([0, 1])]


def line_int(form, a, b):
    gam = pts[a] + t * (pts[b] - pts[a])
    integrand = (form.subs({x: gam[0], y: gam[1]}).T * (pts[b] - pts[a]))[0]
    return sp.integrate(integrand, (t, 0, 1))


ok = all(line_int(wh[s], *s2) == (1 if s == s2 else 0) for s in ((0, 1), (1, 2), (2, 0))
         for s2 in ((0, 1), (1, 2), (2, 0)))
check("Theorem 36.3.10 / Example 36.3.6: the 9 edge integrals of the reference Whitney 1-forms are delta (sympy)", ok)
w2 = lambda a, b: grad[a][0] * grad[b][1] - grad[a][1] * grad[b][0]  # noqa: E731
phi012 = 2 * (ph[0] * w2(1, 2) + ph[1] * w2(2, 0) + ph[2] * w2(0, 1))
check("Example 36.3.6: each dphi^dphi = dx^dy, phi_012 = 2 dx^dy, its integral over the triangle is 1",
      all(w2(*p) == 1 for p in ((1, 2), (2, 0), (0, 1))) and sp.simplify(phi012 - 2) == 0
      and sp.integrate(sp.integrate(phi012, (x, 0, 1 - y)), (y, 0, 1)) == 1)

# ---------------------------------------------------------------- Non-example 36.3.7
dphi1 = grad[1]
check("Non-example 36.3.7(a): dphi_1 = dx integrates to 1, -1, 0 over [0,1], [1,2], [2,0]",
      [line_int(dphi1, *s) for s in ((0, 1), (1, 2), (2, 0))] == [1, -1, 0])

# ---------------------------------------------------------------- Proposition 36.3.9 and Theorem 36.3.10 on meshes
for name, (Vm, Fm) in (("jittered grid", (V, F)), ("icosphere(1)", dm.icosphere(1)), ("kite", ch36.kite())):
    Em, idxm = dd.edge_list(Fm)
    a = rng.normal(size=len(Em))
    worst_t, worst_int = 0.0, 0.0
    tq, wq = dd.gauss_legendre_01(4)
    for f, tri in enumerate(Fm):
        for c in range(3):
            i, j = int(tri[c]), int(tri[(c + 1) % 3])
            sgn = 1 if i < j else -1
            target = sgn * a[idxm[(min(i, j), max(i, j))]]
            for s in (0.0, 0.3, 0.8, 1.0):
                b = dd.barycentric_coordinates(Vm[i] + s * (Vm[j] - Vm[i]), Vm[tri])
                worst_t = max(worst_t, abs(dd.whitney_interpolate(Vm, Fm, a, f, b, idxm) @ (Vm[j] - Vm[i]) - target))
            val = sum(w * dd.whitney_interpolate(Vm, Fm, a, f, dd.barycentric_coordinates(
                Vm[i] + s * (Vm[j] - Vm[i]), Vm[tri]), idxm) @ (Vm[j] - Vm[i]) for s, w in zip(tq, wq))
            worst_int = max(worst_int, abs(val - target))
    check(f"Proposition 36.3.9: W_1 alpha (x_j - x_i) = alpha_e at every point of every edge, from every face ({name})",
          worst_t < 1e-12)
    check(f"Theorem 36.3.10: R_1 W_1 = id ({name})", worst_int < 1e-12)
# R_2 W_2 = id: phi_f is the constant 2-form with value 2 on (e1, e2)
ok = True
for f, tri in enumerate(F):
    T = V[tri]
    G = dd.barycentric_gradients(T)
    e1, e2 = T[1] - T[0], T[2] - T[0]
    for (a_, b_) in ((1, 2), (2, 0), (0, 1)):
        ok &= abs((G[a_] @ e1) * (G[b_] @ e2) - (G[a_] @ e2) * (G[b_] @ e1) - 1) < 1e-12
check("Theorem 36.3.10 (k = 2): (dphi_j ^ dphi_k)(e1, e2) = 1 for the three cyclic pairs, so int phi_f = 1", ok)

# ---------------------------------------------------------------- Example 36.3.11
WR = sp.Rational(1, 2) * wh[(1, 2)]
check("Example 36.3.11: R_1(x dy) = (0 on [0,1], 0 on [0,2], 1/2 on [1,2])",
      [line_int(sp.Matrix([0, x]), *s) for s in ((0, 1), (0, 2), (1, 2))] == [0, 0, sp.Rational(1, 2)])
diff = WR - sp.Matrix([0, x])
check("Example 36.3.11: W R(x dy) - x dy = -(1/2) d(xy), exact with zero edge integrals",
      sp.simplify(diff - (-sp.Rational(1, 2)) * sp.Matrix([y, x])) == sp.zeros(2, 1)
      and all(line_int(diff, *s) == 0 for s in ((0, 1), (1, 2), (2, 0))))

# ---------------------------------------------------------------- Proposition 36.3.12 (sympy, generic triangle)
X = sp.symbols("x0:3", real=True)
Y = sp.symbols("y0:3", real=True)
u = sp.symbols("u0:3", real=True)
al = sp.symbols("a01 a12 a20", real=True)
M = sp.Matrix([[X[0], X[1], X[2]], [Y[0], Y[1], Y[2]], [1, 1, 1]])
lam = M.inv() * sp.Matrix([x, y, 1])
lam = [sp.simplify(li) for li in lam]
gr = [sp.Matrix([sp.diff(li, x), sp.diff(li, y)]) for li in lam]
whit = lambda a, b: lam[a] * gr[b] - lam[b] * gr[a]  # noqa: E731
lhs = (u[1] - u[0]) * whit(0, 1) + (u[2] - u[1]) * whit(1, 2) + (u[0] - u[2]) * whit(2, 0)
rhs = sum((u[c] * gr[c] for c in range(3)), sp.zeros(2, 1))
check("Proposition 36.3.12: W_1(d_0 u) = d(W_0 u) on a generic triangle (sympy)",
      sp.simplify(lhs - rhs) == sp.zeros(2, 1))
W1 = al[0] * whit(0, 1) + al[1] * whit(1, 2) + al[2] * whit(2, 0)
curl = sp.diff(W1[1], x) - sp.diff(W1[0], y)
detE = (X[1] - X[0]) * (Y[2] - Y[0]) - (Y[1] - Y[0]) * (X[2] - X[0])     # = 2 A (signed)
check("Proposition 36.3.12: d(W_1 alpha) = (alpha_01 + alpha_12 + alpha_20) phi_012 with phi_012 = (2 / detE) dx^dy",
      sp.simplify(curl - (al[0] + al[1] + al[2]) * 2 / detE) == 0)

# ---------------------------------------------------------------- Corollary 36.3.13, Figure 36.3.3
Vg, Fg = ch36.grid(4)
Eg, idxg = dd.edge_list(Fg)
close("Corollary 36.3.13: W_1 R_1 reproduces the constant 1-form 2 dx - dy at all barycenters",
      dd.whitney_at_barycenters(Vg, Fg, dd.derham1(lambda p: np.array([2.0, -1.0]), Vg, Eg)),
      np.tile([2.0, -1.0], (len(Fg), 1)), tol=1e-12)
uaff = dd.derham0(lambda p: 3 * p[0] - 2 * p[1] + 1, Vg)
ok = all(abs(dd.barycentric_coordinates(q, Vg[tri]) @ uaff[tri] - (3 * q[0] - 2 * q[1] + 1)) < 1e-12
         for tri in Fg for q in rng.dirichlet(np.ones(3), 3) @ Vg[tri])
check("Corollary 36.3.13: W_0 R_0 g = g for affine g", ok)
rho = 2.5
ok = all(abs(dd.derham2(lambda p: rho, Vg, Fg)[f] / (0.5 * abs(ch36.cross2(Vg[t_[1]] - Vg[t_[0]],
         Vg[t_[2]] - Vg[t_[0]]))) - rho) < 1e-12 for f, t_ in enumerate(Fg))
check("Corollary 36.3.13 / Exercise 36.3.4: R_2(rho dx^dy) / area = rho for constant rho", ok)
for n in (2, 4, 8):
    Vn, Fn = ch36.grid(n)
    En, idxn = dd.edge_list(Fn)
    a = dd.derham1(lambda p: np.array([0.0, p[0]]), Vn, En)
    eb = max(np.linalg.norm(dd.whitney_interpolate(Vn, Fn, a, f, np.full(3, 1 / 3), idxn)
                            - np.array([0.0, Vn[tri].mean(0)[0]])) for f, tri in enumerate(Fn))
    ec = max(np.linalg.norm(dd.whitney_interpolate(Vn, Fn, a, f, b, idxn) - np.array([0.0, b @ Vn[tri][:, 0]]))
             for f, tri in enumerate(Fn) for b in np.eye(3))
    close(f"Figure 36.3.3 (n = {n}): largest errors sqrt(2)/6 h at barycenters, h/2 at corners", [eb, ec],
          [np.sqrt(2) / 6 / n, 0.5 / n], tol=1e-12)

# ---------------------------------------------------------------- Exercises
bc = {x: sp.Rational(1, 3), y: sp.Rational(1, 3)}
a_, b_, c_ = sp.symbols("a b c", real=True)
Wbc = (a_ * wh[(0, 1)] + b_ * wh[(1, 2)] + c_ * wh[(2, 0)]).subs(bc)
check("Exercise 36.3.1: value at the barycenter is ((2a - b - c)/3, (a + b - 2c)/3)",
      sp.simplify(Wbc - sp.Matrix([(2 * a_ - b_ - c_) / 3, (a_ + b_ - 2 * c_) / 3])) == sp.zeros(2, 1))
rot = wh[(0, 1)] + wh[(1, 2)] + wh[(2, 0)]
check("Exercise 36.3.1: phi_01 + phi_12 + phi_20 = 3((x - 1/3) dy - (y - 1/3) dx)",
      sp.simplify(rot - 3 * sp.Matrix([-(y - sp.Rational(1, 3)), x - sp.Rational(1, 3)])) == sp.zeros(2, 1))
hh = sp.symbols("h", positive=True)
err = hh * x - x ** 2
check("Exercise 36.3.3: max of h x - x^2 on [0, h] is h^2/4 at x = h/2",
      sp.solve(sp.diff(err, x), x) == [hh / 2] and sp.simplify(err.subs(x, hh / 2) - hh ** 2 / 4) == 0)
ok = True
for (p_, q_) in ((0, 1), (1, 2), (2, 0)):
    Xf = whit(p_, q_)
    div = sp.diff(Xf[0], x) + sp.diff(Xf[1], y)
    cu = sp.diff(Xf[1], x) - sp.diff(Xf[0], y)
    ok &= sp.simplify(div) == 0 and sp.simplify(cu - 2 / detE) == 0
check("Exercise 36.3.5: div phi_ab = 0 and curl phi_ab = 2/detE = 1/A on a counterclockwise triangle (sympy)", ok)
close("Exercise 36.3.5: curl = 1/0.45 = 2.22 on the triangle of Figure 36.3.1", 1 / 0.45, 2.2222, tol=1e-4)
ns6, same6, printed6 = ch36.run_printed(PAGE, "code-ex-36-3-6", "out-ex-36-3-6")
check("Exercise 36.3.6: code-ex-36-3-6 prints exactly out-ex-36-3-6", same6, printed6)

summary()
