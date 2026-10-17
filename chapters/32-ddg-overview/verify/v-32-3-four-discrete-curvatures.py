"""Verification for Section 32.3 (Four Discrete Curvatures).

Run from the project root: PYTHONPATH=tools python3 chapters/32-ddg-overview/verify/v-32-3-four-discrete-curvatures.py
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
import ddg32  # noqa: E402
from dgcheck import check, close, summary, sym_equal  # noqa: E402

PAGE = HERE.parents[1] / "32-3-four-discrete-curvatures.html"
rng = np.random.default_rng(32)
J = ddg32.J


def theta_552(X, closed=True):
    """turning angles by (5.6.2): 2 arctan(det/(1 + dot)) (independent of dgdec)."""
    T = dd.curve_tangents(X, closed)
    a, b = (np.roll(T, 1, 0), T) if closed else (T[:-1], T[1:])
    return 2 * np.arctan((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / (1 + (a * b).sum(1)))


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


# ---------------------------------------------------------------- Definition 32.3.1, Example 32.3.2, Non-example 32.3.3
ok = True
for _ in range(50):
    X = rng.normal(size=(int(rng.integers(3, 12)), 2))
    th = dd.turning_angles(X, True)
    T = dd.curve_tangents(X, True)
    ok &= np.allclose(th, theta_552(X)) and np.all(np.abs(th) < np.pi)
    ok &= all(np.allclose(rot(th[i]) @ T[i - 1], T[i]) for i in range(len(X)))       # t_i = R_theta t_{i-1}
check("Def 32.3.1: atan2 formula = (5.6.2) and t_i = R_theta_i t_{i-1} (50 random closed polygons)", ok)
close("Example 32.3.2(a): regular 7-gon theta = 2 pi/7, clockwise -2 pi/7",
      [dd.turning_angles(ddg32.regular_ngon(7), True), dd.turning_angles(ddg32.regular_ngon(7)[::-1], True)],
      [np.full(7, 2 * np.pi / 7), np.full(7, -2 * np.pi / 7)], 1e-12)
close("Example 32.3.2(a): square theta = pi/2", dd.turning_angles(np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), True),
      np.full(4, np.pi / 2), 1e-12)
Xo = np.array([[0, 0], [2, 0], [3, 1.0]])
close("Example 32.3.2(b): theta_1 = pi/4", dd.turning_angles(Xo, False)[1], np.pi / 4, 1e-12)
F40 = ddg32.sample("flower", 40)[0]
th40 = dd.turning_angles(F40, True)
close("Example 32.3.2(c): flower n = 40: theta in [-1.130, 0.994]", [round(th40.min(), 3), round(th40.max(), 3)],
      [-1.130, 0.994], 1e-12)
neg = np.where(th40 < 0)[0]
tneg = 2 * np.pi * neg / 40
check("Example 32.3.2(c): the negative turning angles sit near the waists t = pi/5 + 2 pi k/5",
      np.all(np.min(np.abs(((tneg[:, None] - (np.pi / 5 + 2 * np.pi * np.arange(5) / 5)) + np.pi) % (2 * np.pi) - np.pi),
                    axis=1) < 0.4))
Tf = dd.curve_tangents(np.array([[0, 0], [1, 0], [0.5, 0.0]]), False)
check("Non-example 32.3.3: the fold-back vertex has t_1 = -t_0 (turning angle undefined)", np.allclose(Tf[1], -Tf[0]))

# ---------------------------------------------------------------- Proposition 32.3.4
ok, ks = True, set()
for _ in range(300):
    X = rng.normal(size=(int(rng.integers(3, 15)), 2))
    s = dd.turning_angles(X, True).sum() / (2 * np.pi)
    ok &= abs(s - round(s)) < 1e-10
    ks.add(int(round(s)))
check(f"Prop 32.3.4: sum theta_i in 2 pi Z for 300 random closed polygons (k seen: {sorted(ks)})", ok and len(ks) > 2)
a = np.linspace(0, 2 * np.pi, 13)[:-1]
eight = np.stack([np.sin(a), np.sin(a) * np.cos(a)], 1)
twice = ddg32.regular_ngon(5)[[0, 2, 4, 1, 3]]          # pentagram: turns twice
close("Prop 32.3.4: figure-eight polygon k = 0, pentagram k = 2",
      [dd.turning_angles(eight, True).sum(), dd.turning_angles(twice, True).sum()], [0.0, 4 * np.pi], 1e-10)

# ---------------------------------------------------------------- Definition 32.3.5 (formulas, signs)
ok = True
for _ in range(50):
    X = rng.normal(size=(int(rng.integers(3, 12)), 2))
    th = theta_552(X)
    w = np.linalg.norm(np.roll(X, -1, 0) - np.roll(X, 1, 0), axis=1)
    mine = {"A": th, "B": 2 * np.sin(th / 2), "C": 2 * np.tan(th / 2), "D": 2 * np.sin(th) / w}
    for k, f in dd.DISCRETE_CURVATURES.items():
        ok &= np.allclose(f(X, True), mine[k]) and np.all(np.sign(mine[k]) == np.sign(th))
check("Def 32.3.5: (32.3.4) = dgdec.DISCRETE_CURVATURES, all with the sign of theta_i", ok)
check("(32.3.3): sum of dual lengths = L", all(abs(dd.curve_dual_lengths(X, True).sum() - dd.curve_length(X, True))
                                              < 1e-12 for X in [rng.normal(size=(9, 2)), F40]))

# ---------------------------------------------------------------- Proposition 32.3.6 (length gradient)
ok = True
for _ in range(20):
    X = rng.normal(size=(8, 2))
    G = np.zeros_like(X)
    h = 1e-6
    for i in range(len(X)):
        for c in range(2):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, c] += h
            Xm[i, c] -= h
            G[i, c] = (dd.curve_length(Xp, True) - dd.curve_length(Xm, True)) / (2 * h)
    T = dd.curve_tangents(X, True)
    kB = dd.curvature_length_variation(X, True)
    nrm = dd.curve_vertex_normals(X, True)
    tb = (np.roll(T, 1, 0) + T)
    tb /= np.linalg.norm(tb, axis=1, keepdims=True)
    ok &= np.allclose(G, np.roll(T, 1, 0) - T, atol=1e-7) and np.allclose(G, -kB[:, None] * nrm, atol=1e-7)
    ok &= np.allclose(nrm, tb @ J.T) and np.allclose(G, dd.length_gradient(X, True), atol=1e-7)
check("Prop 32.3.6: finite-difference grad L = t_{i-1} - t_i = -kappa^B n_i, n_i = J tbar_i (20 random polygons)", ok)
th_s = sp.symbols("theta", real=True)
R = lambda a: sp.Matrix([[sp.cos(a), -sp.sin(a)], [sp.sin(a), sp.cos(a)]])  # noqa: E731
Js = sp.Matrix([[0, -1], [1, 0]])
sym_equal("Prop 32.3.6 proof: R_a - R_-a = 2 sin a J", list(R(th_s) - R(-th_s)), list(2 * sp.sin(th_s) * Js))
sym_equal("Prop 32.3.6 proof: R_a + R_-a = 2 cos a I", list(R(th_s) + R(-th_s)), list(2 * sp.cos(th_s) * sp.eye(2)))

# ---------------------------------------------------------------- Proposition 32.3.7 (discrete Steiner)


def miter_by_intersection(X, eps):
    """vertices of the mitered offset by intersecting consecutive offset lines (independent of (32.3.8))."""
    T = dd.curve_tangents(X, True)
    N = T @ J.T
    Q = []
    for i in range(len(X)):
        p1, d1 = X[i] + eps * N[i - 1], T[i - 1]
        p2, d2 = X[i] + eps * N[i], T[i]
        lam = np.linalg.solve(np.c_[d1, -d2], p2 - p1)
        Q.append(p1 + lam[0] * d1)
    return np.array(Q)


def arc_join_length(X, eps, m=20000):
    """length of the offset edges joined by circular arcs, arcs integrated numerically."""
    T = dd.curve_tangents(X, True)
    N = T @ J.T
    total = dd.curve_length(X, True)
    for i in range(len(X)):
        a0 = np.arctan2(*(N[i - 1])[::-1])
        a1 = a0 + dd.turning_angles(X, True)[i]
        u = np.linspace(a0, a1, m)
        P = X[i] + abs(eps) * np.sign(eps) * np.stack([np.cos(u), np.sin(u)], 1)
        total += np.linalg.norm(np.diff(P, axis=0), axis=1).sum()
    return total


Pent = np.array([[0.0, 0.0], [2.0, -0.1], [2.6, 1.1], [1.3, 2.0], [-0.3, 1.1]])
for X, eps in ((Pent, -0.35), (ddg32.regular_ngon(9, 1.3), -0.2)):
    th = dd.turning_angles(X, True)
    L = dd.curve_length(X, True)
    check("Prop 32.3.7(a): hypothesis eps * theta_i < 0 at every vertex", np.all(eps * th < 0))
    close("Prop 32.3.7(a): arc join length = L - eps sum kappa^A (numerical arcs)", arc_join_length(X, eps),
          L - eps * th.sum(), 1e-6)
    T = dd.curve_tangents(X, True)
    chords = np.linalg.norm(eps * (T - np.roll(T, 1, 0)) @ J.T, axis=1).sum()
    close("Prop 32.3.7(a): chord join length = L - eps sum kappa^B", L + chords, L - eps * (2 * np.sin(th / 2)).sum(),
          1e-12)
ok = True
for X in (Pent, F40, ddg32.sample("ellipse", 12)[0], rng.normal(size=(7, 2)) * 0 + ddg32.regular_ngon(7)):
    th = dd.turning_angles(X, True)
    ell = dd.curve_edge_lengths(X, True)
    L = ell.sum()
    for eps in (0.004, -0.004, 0.02, -0.02):
        tb = np.tan(th / 2)
        if np.any(ell - eps * (tb + np.roll(tb, -1)) <= 0):
            continue
        Q = miter_by_intersection(X, eps)
        Qf = X + eps * dd.curve_vertex_normals(X, True) / np.cos(th / 2)[:, None]
        ok &= np.allclose(Q, Qf, atol=1e-10)
        ok &= abs(dd.curve_length(Q, True) - (L - eps * (2 * tb).sum())) < 1e-10
        ok &= np.allclose(dd.curve_tangents(Q, True), dd.curve_tangents(X, True), atol=1e-8)   # same directions
check("Prop 32.3.7(b): q_i = intersection of the offset lines, edges keep t_i, length L - eps sum kappa^C "
      "(convex and nonconvex, both signs of eps)", ok)
LA = dd.curve_length(Pent, True) + 0.35 * dd.turning_angles(Pent, True).sum()
close("Figure 32.3.2: L = 7.9012, L_A = 10.1003, L_B = 9.9552, L_C = 10.4567",
      [round(dd.curve_length(Pent, True), 4), round(LA, 4),
       round(dd.curve_length(Pent, True) + 0.35 * dd.curvature_length_variation(Pent, True).sum(), 4),
       round(dd.curve_length(Pent, True) + 0.35 * dd.curvature_steiner(Pent, True).sum(), 4)],
      [7.9012, 10.1003, 9.9552, 10.4567], 1e-12)

# ---------------------------------------------------------------- Proposition 32.3.8 (circumcircle)
ok = True
for _ in range(100):
    X = rng.normal(size=(6, 2))
    th = dd.turning_angles(X, True)
    c = ddg32.circumcenters(X)
    for i in range(6):
        A_ = np.array([2 * (X[i - 1] - X[i]), 2 * (X[(i + 1) % 6] - X[i])])
        b_ = np.array([np.sum(X[i - 1] ** 2 - X[i] ** 2), np.sum(X[(i + 1) % 6] ** 2 - X[i] ** 2)])
        ci = np.linalg.solve(A_, b_)                                             # independent circumcenter
        ok &= np.allclose(ci, c[i])
        r = np.linalg.norm(ci - X[i])
        ok &= abs(1 / r - abs(dd.curvature_osculating_circle(X, True)[i])) < 1e-9
        proj = np.dot(ci - X[i], dd.curve_vertex_normals(X, True)[i])
        ok &= abs(proj - dd.curve_dual_lengths(X, True)[i] / dd.curvature_length_variation(X, True)[i]) < 1e-8
        ok &= np.sign(proj) == np.sign(th[i])
check("Prop 32.3.8: 1/r_i = |kappa^D_i| and <c_i - gamma_i, n_i> = lbar_i / kappa^B_i, sign = sign theta_i "
      "(100 random hexagons)", ok)
Gf = np.array([[-1.1, -0.45], [0.0, 0.0], [0.05, 1.0]])
vals = [dd.turning_angles(Gf, False)[1], dd.curvature_length_variation(Gf, False)[1],
        1 / np.linalg.norm(ddg32.circumcenters(Gf)[1] - Gf[1]), dd.curvature_osculating_circle(Gf, False)[1],
        np.linalg.norm(Gf[1] - Gf[0]), np.linalg.norm(Gf[2] - Gf[1])]
close("Figure 32.3.1: theta 1.1325, kappa^B 1.0730, 1/r 0.9785 = kappa^D, edges 1.188 and 1.001",
      [round(vals[0], 4), round(vals[1], 4), round(1 / vals[2], 4), round(vals[3], 4), round(vals[4], 3),
       round(vals[5], 3)], [1.1325, 1.0730, 1.0219, 0.9785, 1.188, 1.001], 1e-12)
close("Figure 32.3.1: |grad L| = kappa^B", np.linalg.norm(dd.length_gradient(Gf, False)[1]), vals[1], 1e-12)

# ---------------------------------------------------------------- Proposition 32.3.9
th = np.linspace(-np.pi + 1e-6, np.pi - 1e-6, 200001)
check("Prop 32.3.9(a): |2 sin(theta/2) - theta| <= |theta|^3/24 on (-pi, pi)",
      np.all(np.abs(2 * np.sin(th / 2) - th) <= np.abs(th) ** 3 / 24 + 1e-15))
th2 = np.linspace(-np.pi / 2, np.pi / 2, 200001)
check("Prop 32.3.9(b): |2 tan(theta/2) - theta| <= |theta|^3/3 on [-pi/2, pi/2]",
      np.all(np.abs(2 * np.tan(th2 / 2) - th2) <= np.abs(th2) ** 3 / 3 + 1e-15))
nz = th[np.abs(th) > 1e-3]
check("Prop 32.3.9: |kappa^B| < |kappa^A| < |kappa^C| for 0 < |theta| < pi",
      np.all(np.abs(2 * np.sin(nz / 2)) < np.abs(nz)) and np.all(np.abs(nz) < np.abs(2 * np.tan(nz / 2))))
ok = True
for _ in range(100):
    ell, a = rng.uniform(0.2, 2), rng.uniform(-3, 3)
    p0 = np.array([0.0, 0.0])
    t0 = np.array([np.cos(a), np.sin(a)])
    t1 = rot(rng.uniform(-3, 3)) @ t0
    X = np.array([p0 - ell * t0, p0, p0 + ell * t1])
    ok &= abs(dd.curvature_osculating_circle(X, False)[1]
              - dd.curvature_length_variation(X, False)[1] / dd.curve_dual_lengths(X, False)[1]) < 1e-10
check("Prop 32.3.9(c): equal edges => kappa^D = kappa^B / lbar (100 random vertices)", ok)

# ---------------------------------------------------------------- Proposition 32.3.10, Figure 32.3.3
for name, ns_ in (("ellipse", (256, 512)), ("flower", (512, 1024))):
    errs = []
    for n in ns_:
        X, tv = ddg32.sample(name, n)
        lb = dd.curve_dual_lengths(X, True)
        ks = ddg32.kappa(name, tv)
        errs.append([np.max(np.abs(f(X, True) / (1 if k == "D" else lb) - ks)) for k, f in
                     dd.DISCRETE_CURVATURES.items()])
    orders = np.log2(np.array(errs[0]) / np.array(errs[1]))
    check(f"Prop 32.3.10: {name}, observed order of the four pointwise values = {np.round(orders, 2)}",
          np.all(np.abs(orders - 2) < 0.05))
# arc-length sampling (the hypothesis of Prop 32.3.10) of the ellipse by inverting the arc length numerically
m = 200000
tt = 2 * np.pi * np.arange(m + 1) / m
sp_ = ddg32.speed("ellipse", tt)
arc = np.r_[0, np.cumsum(0.5 * (sp_[1:] + sp_[:-1]) * np.diff(tt))]
errs = []
for n in (128, 256):
    s_i = arc[-1] * np.arange(n) / n
    t_i = np.interp(s_i, arc, tt)
    X = ddg32.point("ellipse", t_i)
    lb = dd.curve_dual_lengths(X, True)
    errs.append([np.max(np.abs(f(X, True) / (1 if k == "D" else lb) - ddg32.kappa("ellipse", t_i)))
                 for k, f in dd.DISCRETE_CURVATURES.items()])
orders = np.log2(np.array(errs[0]) / np.array(errs[1]))
check(f"Prop 32.3.10: ellipse sampled uniformly in arc length, orders {np.round(orders, 2)}",
      np.all(np.abs(orders - 2) < 0.05))
# text after Figure 32.3.3: irregular spacing t_i = 2 pi (i + u_i)/n, u_i uniform in [-0.25, 0.25] -> order 1
rj = np.random.default_rng(7)
errs = []
for n in (2048, 4096):
    tj = 2 * np.pi * (np.arange(n) + rj.uniform(-0.25, 0.25, n)) / n
    X = ddg32.point("ellipse", tj)
    lb = dd.curve_dual_lengths(X, True)
    errs.append([np.max(np.abs(f(X, True) / (1 if k == "D" else lb) - ddg32.kappa("ellipse", tj)))
                 for k, f in dd.DISCRETE_CURVATURES.items()])
orders = np.log2(np.array(errs[0]) / np.array(errs[1]))
check(f"Text after Fig 32.3.3: jittered sampling of the ellipse drops the order to about 1 ({np.round(orders, 2)}; "
      f"n * error at n = 4096: {np.round(np.array(errs[1]) * 4096, 2)})",
      np.all(np.abs(orders - 1) < 0.25) and np.all((np.array(errs[1]) * 4096 > 3) & (np.array(errs[1]) * 4096 < 8)))
X16, tv16 = ddg32.sample("ellipse", 16)
lb16 = dd.curve_dual_lengths(X16, True)
close("Figure 32.3.3(a): n = 16, at t = 0: A 1.84, B 1.80, C 1.93, D 1.80; theta_0 = 0.76",
      [round(dd.curvature_turning_angle(X16, True)[0] / lb16[0], 2),
       round(dd.curvature_length_variation(X16, True)[0] / lb16[0], 2),
       round(dd.curvature_steiner(X16, True)[0] / lb16[0], 2),
       round(dd.curvature_osculating_circle(X16, True)[0], 2), round(dd.turning_angles(X16, True)[0], 2)],
      [1.84, 1.80, 1.93, 1.80, 0.76], 1e-12)
e = dd.curve_edge_lengths(X16, True)
close("Figure 32.3.3(a): the two edges at t = 0 have equal length (so B = D there)", e[-1], e[0], 1e-12)

# ---------------------------------------------------------------- Example 32.3.11 (regular polygon)
n_, r_ = sp.symbols("n r", positive=True)
x = sp.symbols("x", positive=True)
A_pw = (sp.pi / n_) / (r_ * sp.sin(sp.pi / n_))
C_pw = 1 / (r_ * sp.cos(sp.pi / n_))
sym_equal("Example 32.3.11: kappa^A/lbar = (1/r)(1 + pi^2/(6 n^2) + ...)",
          sp.series(A_pw.subs(n_, 1 / x), x, 0, 3).removeO(), (1 + sp.pi ** 2 * x ** 2 / 6) / r_)
sym_equal("Example 32.3.11: kappa^C/lbar = (1/r)(1 + pi^2/(2 n^2) + ...)",
          sp.series(C_pw.subs(n_, 1 / x), x, 0, 3).removeO(), (1 + sp.pi ** 2 * x ** 2 / 2) / r_)
ok = True
for n in (3, 5, 8, 40):
    X = ddg32.regular_ngon(n, 1.7)
    lb = dd.curve_dual_lengths(X, True)
    ok &= np.allclose(dd.curvature_length_variation(X, True) / lb, 1 / 1.7)
    ok &= np.allclose(dd.curvature_osculating_circle(X, True), 1 / 1.7)
    ok &= np.allclose(dd.curvature_turning_angle(X, True) / lb, float(A_pw.subs({n_: n, r_: 1.7})))
    ok &= np.allclose(dd.curvature_steiner(X, True) / lb, float(C_pw.subs({n_: n, r_: 1.7})))
check("Example 32.3.11: closed forms (B and D exactly 1/r) for n = 3, 5, 8, 40", ok)
close("Example 32.3.11: r = 1, n = 8: A 1.0262, C 1.0824",
      [round(float(A_pw.subs({n_: 8, r_: 1})), 4), round(float(C_pw.subs({n_: 8, r_: 1})), 4)], [1.0262, 1.0824],
      1e-12)

# ---------------------------------------------------------------- Example 32.3.12 (uneven circle)
U = ddg32.uneven_circle()[:3]
lbU = dd.curve_dual_lengths(U, False)[1]
close("Example 32.3.12: theta = 25 deg (inscribed angle), lbar = 0.43207",
      [np.degrees(dd.turning_angles(U, False)[1]), round(lbU, 5)], [25.0, 0.43207], 1e-9)
close("Example 32.3.12: pointwise A 1.0099, B 1.0019, C 1.0262, D 1.0000",
      [round(dd.curvature_turning_angle(U, False)[1] / lbU, 4), round(dd.curvature_length_variation(U, False)[1] / lbU, 4),
       round(dd.curvature_steiner(U, False)[1] / lbU, 4), round(dd.curvature_osculating_circle(U, False)[1], 4)],
      [1.0099, 1.0019, 1.0262, 1.0], 1e-12)
check("Example 32.3.12: D is exact at every vertex of the uneven circle",
      np.allclose(dd.curvature_osculating_circle(ddg32.uneven_circle(), True), 1.0))

# ---------------------------------------------------------------- Proposition 32.3.13 and Exercise 32.3.1
ok = True
for _ in range(200):
    th = rng.dirichlet(np.ones(int(rng.integers(3, 12)))) * 2 * np.pi
    if np.any(th >= np.pi):
        continue
    ok &= (2 * np.sin(th / 2)).sum() < 2 * np.pi < (2 * np.tan(th / 2)).sum()
check("Prop 32.3.13: sum kappa^B < 2 pi < sum kappa^C for random convex turning angles", ok)
Sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
close("Exercise 32.3.1: square: A, B, C, D = pi/2, sqrt2, 2, sqrt2; sums 2 pi, 4 sqrt 2 = 5.657, 8",
      [dd.curvature_turning_angle(Sq, True)[0], dd.curvature_length_variation(Sq, True)[0],
       dd.curvature_steiner(Sq, True)[0], dd.curvature_osculating_circle(Sq, True)[0],
       dd.curvature_turning_angle(Sq, True).sum(), round(dd.curvature_length_variation(Sq, True).sum(), 3),
       dd.curvature_steiner(Sq, True).sum()],
      [np.pi / 2, np.sqrt(2), 2, np.sqrt(2), 2 * np.pi, 5.657, 8], 1e-12)
close("Exercise 32.3.1: circumradius sqrt2/2", np.linalg.norm(ddg32.circumcenters(Sq)[0] - Sq[0]), np.sqrt(2) / 2, 1e-12)

# ---------------------------------------------------------------- Exercise 32.3.2
c1 = ddg32.circumcenters(Xo)[1]
n1 = dd.curve_vertex_normals(Xo, False)[1]
close("Exercise 32.3.2: kappa A, B, C, D = 0.7854, 0.7654, 0.8284, 0.4472",
      [round(f(Xo, False)[1], 4) for f in dd.DISCRETE_CURVATURES.values()], [0.7854, 0.7654, 0.8284, 0.4472], 1e-12)
close("Exercise 32.3.2: center (1, 2), radius sqrt 5, n_1 = (-0.3827, 0.9239), projection 2.2304",
      [*c1, np.linalg.norm(c1 - Xo[1]), *np.round(n1, 4), round(np.dot(c1 - Xo[1], n1), 4),
       round(dd.curve_dual_lengths(Xo, False)[1] / dd.curvature_length_variation(Xo, False)[1], 4)],
      [1, 2, np.sqrt(5), -0.3827, 0.9239, 2.2304, 2.2304], 1e-12)

# ---------------------------------------------------------------- Exercise 32.3.3
ok = True
for _ in range(100):
    X = rng.normal(size=(3, 2))
    c = ddg32.circumcenters(X)[1]
    tb = dd.curve_tangents(X, False)
    tbar = (tb[0] + tb[1]) / np.linalg.norm(tb[0] + tb[1])
    e = dd.curve_edge_lengths(X, False)
    ok &= abs(np.dot(c - X[1], tbar) * np.linalg.norm(tb[0] + tb[1]) - (e[1] - e[0]) / 2) < 1e-9
check("Exercise 32.3.3: <c_i - gamma_i, t_{i-1} + t_i> = (l_i - l_{i-1})/2 (100 random vertices)", ok)

# ---------------------------------------------------------------- Exercise 32.3.4 (inward miter of a regular polygon)
ok = True
for n, r, eps in ((5, 1.0, 0.3), (8, 2.0, 1.1), (12, 1.0, -0.4)):
    X = ddg32.regular_ngon(n, r)
    Q = miter_by_intersection(X, eps)
    rho = r * np.cos(np.pi / n)
    ok &= np.allclose(np.linalg.norm(Q, axis=1), (rho - eps) / np.cos(np.pi / n))     # regular, inradius rho - eps
    ok &= abs(dd.curve_length(Q, True) - 2 * n * (rho - eps) * np.tan(np.pi / n)) < 1e-12
    ok &= abs(dd.curve_length(Q, True) - (dd.curve_length(X, True) - eps * 2 * n * np.tan(np.pi / n))) < 1e-12
check("Exercise 32.3.4: inward/outward miter of a regular n-gon is a regular n-gon of inradius rho - eps", ok)

# ---------------------------------------------------------------- Exercise 32.3.5 (Jensen)
ok = True
for _ in range(300):
    n = int(rng.integers(3, 10))
    th = rng.dirichlet(np.ones(n)) * 2 * np.pi
    if np.any(th >= np.pi):
        continue
    ok &= (2 * np.sin(th / 2)).sum() <= 2 * n * np.sin(np.pi / n) + 1e-12
    ok &= (2 * np.tan(th / 2)).sum() >= 2 * n * np.tan(np.pi / n) - 1e-12
check("Exercise 32.3.5: sum kappa^B <= 2n sin(pi/n), sum kappa^C >= 2n tan(pi/n) (random convex)", ok)
xs = sp.symbols("x", positive=True)
check("Exercise 32.3.5: f'' = -sin(x/2)/2, g'' = sec^2(x/2) tan(x/2)",
      sp.simplify(sp.diff(2 * sp.sin(xs / 2), xs, 2) + sp.sin(xs / 2) / 2) == 0
      and sp.simplify(sp.diff(2 * sp.tan(xs / 2), xs, 2) - sp.tan(xs / 2) / sp.cos(xs / 2) ** 2) == 0)

# ---------------------------------------------------------------- Implementation box and Exercise 32.3.6
check("Implementation 32.3: at most 30 lines", ddg32.code_lines(PAGE, "code-32-3-1") <= 30)
ns = ddg32.run_block(PAGE, "code-32-3-1")
ok = True
for X in (F40, ddg32.uneven_circle(), rng.normal(size=(11, 2)), Pent):
    kA, kB, kC, kD, lbar = ns["four_curvatures"](X)
    ok &= all(np.allclose(v, f(X, True)) for v, f in zip((kA, kB, kC, kD), dd.DISCRETE_CURVATURES.values()))
    ok &= np.allclose(lbar, dd.curve_dual_lengths(X, True))
check("Implementation 32.3: four_curvatures = dgdec (flower, uneven circle, random, pentagon)", ok)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    ddg32.run_block(PAGE, "code-ex-32-3-6", ns)
check("Exercise 32.3.6: the solution code prints exactly the output shown",
      buf.getvalue().strip() == ddg32.code_block(PAGE, "out-ex-32-3-6").strip(), f"got:\n{buf.getvalue()}")

summary()
