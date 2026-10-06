"""Verification for Section 0.8 (Riemannian Geometry: Measuring on a Manifold).

Run from the project root: ``PYTHONPATH=tools python3 chapters/00-tour/verify/v-0-8-riemannian.py``
"""

import numpy as np
import sympy as sp

import dgnum
import dgsym
from dgcheck import check, close, summary, sym_equal

# ---------------------------------------------------------------- Example 0.8.2: the three metrics
sph = dgsym.EXAMPLES["sphere"]
th, ph = sph["coords"]
r = sph["params"][0]
g_sph = dgsym.induced_metric(sph["expr"], sph["coords"], sph["positive"])
sym_equal("Example 0.8.2: induced metric on S^2(r) = diag(r^2, r^2 sin^2 theta)", g_sph,
          sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2), {th: (0, sp.pi)})
E, F, G = dgsym.first_ff(sph["expr"], th, ph, sph["positive"])
check("Example 0.8.2: (E, F, G) of the sphere as in §7",
      sp.simplify(E - sph["expected"]["E"]) == 0 and F == 0 and sp.simplify(G - sph["expected"]["G"]) == 0)
check("Section curvature: K(S^2(r)) = 1/r^2 from the metric",
      sp.simplify(dgsym.sectional(g_sph, (th, ph), [1, 0], [0, 1], sph["positive"]) - 1 / r ** 2) == 0)

hp = dgsym.EXAMPLES["hyperbolic_plane"]
x, y = hp["coords"]
g_h = hp["expr"]
Gm = dgsym.christoffel(g_h, (x, y))
ok = all(sp.simplify(Gm[k][i][j] - hp["expected"]["christoffel"].get((k, i, j), 0)) == 0
         for k in range(2) for i in range(2) for j in range(2))
check("Example 0.8.2: Christoffel symbols of U^2 as in §7", ok)
check("Section curvature: K(U^2) = -1 from the metric",
      sp.simplify(dgsym.sectional(g_h, (x, y), [1, 0], [0, 1]) + 1) == 0)
# Lee sign check: R(X,Y)Z = c(<Y,Z>X - <X,Z>Y) with c = -1
R = dgsym.riemann(g_h, (x, y))
Xv, Yv, Zv = [1, 0], [0, 1], [0, 1]
lhs = dgsym.R_apply(R, Xv, Yv, Zv)
ip = lambda a, b: sum(g_h[i, j] * a[i] * b[j] for i in range(2) for j in range(2))  # noqa: E731
rhs = [-(ip(Yv, Zv) * Xv[l] - ip(Xv, Zv) * Yv[l]) for l in range(2)]
check("Section curvature: R(X,Y)Z = c(<Y,Z>X - <X,Z>Y) with c = -1 on U^2 (Lee sign, (0.8.6))",
      all(sp.simplify(a - b) == 0 for a, b in zip(lhs, rhs)))

# gradient in U²: grad f = y² (f_x, f_y)
fsym = sp.Function("f")(x, y)
ginv = sp.Matrix(g_h).inv()
gradf = ginv * sp.Matrix([sp.diff(fsym, x), sp.diff(fsym, y)])
sym_equal("(0.8.3): grad f = g^{ij} d_j f = y^2 (f_x, f_y) on U^2", gradf,
          y ** 2 * sp.Matrix([sp.diff(fsym, x), sp.diff(fsym, y)]))

# ---------------------------------------------------------------- (0.8.2) lengths, Exercise 0.8.1, Figure 0.8.1
t = sp.symbols("t", positive=True)
eps = sp.symbols("epsilon", positive=True)
check("Exercise 0.8.1: L_g of (0,1)->(0,e) is 1", sp.integrate(1 / t, (t, 1, sp.E)) == 1)
check("Exercise 0.8.1: L_g of (0,eps)->(0,1) is -ln(eps)", sp.simplify(sp.integrate(1 / t, (t, eps, 1)) + sp.log(eps)) == 0)
s = sp.symbols("s", real=True)
arc = sp.integrate(1 / sp.sin(s), (s, sp.pi / 3, 2 * sp.pi / 3))
check("Figure 0.8.1: g-length of the geodesic arc A->B is ln 3", sp.simplify(arc - sp.log(3)) == 0)
check("Figure 0.8.1: g-length of the horizontal segment is 2/sqrt3 > ln 3",
      sp.simplify(2 / sp.sqrt(3) - sp.Rational(2, 1) / sp.sqrt(3)) == 0 and float(2 / sp.sqrt(3)) > float(sp.log(3)))

Gnum = dgnum.numeric_christoffel(g_h, (x, y))
A = np.array([-1.0, np.sqrt(3)])
v0 = np.array([np.sqrt(3), 1.0]) / 2 * A[1]
ts = np.linspace(0, np.log(3), 4001)
xs, vs = dgnum.integrate_geodesic(Gnum, A, v0, ts)
close("Figure 0.8.1: the geodesic from A stays on x^2 + y^2 = 4", np.max(np.abs(np.hypot(xs[:, 0], xs[:, 1]) - 2)), 0.0,
      tol=1e-7)
close("Figure 0.8.1: it reaches B = (1, sqrt3) at g-length ln 3", xs[-1], [1.0, np.sqrt(3)], tol=1e-6)
xs2, _ = dgnum.integrate_geodesic(Gnum, np.array([-2.2, 0.5]), np.array([0.0, 0.5]), np.linspace(0, 2, 2001))
close("Figure 0.8.1: vertical lines are geodesics", np.max(np.abs(xs2[:, 0] + 2.2)), 0.0, tol=1e-10)


# ---------------------------------------------------------------- (0.8.4), (0.8.5): exp and log on the unit sphere
def exp_p(p, v):
    nv = np.linalg.norm(v)
    return p.copy() if nv < 1e-15 else np.cos(nv) * p + np.sin(nv) * v / nv


def log_p(p, q):
    c = np.clip(p @ q, -1, 1)
    a = np.arccos(c)
    if a < 1e-15:
        return np.zeros(3)
    return a / np.sin(a) * (q - c * p)


rng = np.random.default_rng(8)
worst_log, worst_geo = 0.0, 0.0
for _ in range(50):
    p = rng.normal(size=3)
    p /= np.linalg.norm(p)
    v = rng.normal(size=3)
    v -= (v @ p) * p
    v *= rng.uniform(0.1, 3.0) / np.linalg.norm(v)
    q = exp_p(p, v)
    worst_log = max(worst_log, np.linalg.norm(log_p(p, q) - v))
    # t ↦ exp_p(tv) has acceleration normal to S² (a geodesic) and constant speed |v|
    h = 1e-4
    a = (exp_p(p, (1 + h) * v) - 2 * q + exp_p(p, (1 - h) * v)) / h ** 2
    worst_geo = max(worst_geo, np.linalg.norm(a - (a @ q) * q))
close("(0.8.5): log_p(exp_p(v)) = v for |v| < pi", worst_log, 0.0, tol=1e-9)
close("(0.8.4): t -> exp_p(tv) is a geodesic (tangential acceleration 0)", worst_geo, 0.0, tol=1e-5)
pN = np.array([0.0, 0.0, 1.0])
close("Exercise 0.8.2: exp_N((pi/2, 0, 0)) = (1, 0, 0)", exp_p(pN, np.array([np.pi / 2, 0, 0])), [1, 0, 0], tol=1e-12)
close("Exercise 0.8.2: every v with |v| = pi goes to S", exp_p(pN, np.pi * np.array([0.6, 0.8, 0])), [0, 0, -1],
      tol=1e-12)
qq = np.array([0.6, 0.0, 0.8])
close("(0.8.5): d(p, q) = |log_p q| = arccos<p, q>", np.linalg.norm(log_p(pN, qq)), np.arccos(0.8), tol=1e-12)

# ---------------------------------------------------------------- (0.8.7) Gauss–Bonnet for triangles, Figure 0.8.3
check("Figure 0.8.3 / Exercise 0.8.3: octant angle sum - pi = K * area (pi/2)",
      sp.simplify(3 * sp.pi / 2 - sp.pi - 1 * (4 * sp.pi / 8)) == 0)
Pa, Pb, Pc = np.array([-1.0, 1.0]), np.array([1.0, 1.0]), np.array([0.0, 3.0])
sides = [(Pa, Pb, np.array([0.0, 0.0])), (Pb, Pc, np.array([-3.5, 0.0])), (Pc, Pa, np.array([3.5, 0.0]))]
for P, Q, c in sides:
    check(f"Figure 0.8.3: {P}, {Q} lie on one semicircle centred at {c}",
          abs(np.linalg.norm(P - c) - np.linalg.norm(Q - c)) < 1e-12)


def tangent(P, c, Q):
    rv = P - c
    tv = np.array([-rv[1], rv[0]]) / np.linalg.norm(rv)
    return tv if tv @ (Q - P) > 0 else -tv


angs = [np.arccos(tangent(Pa, sides[0][2], Pb) @ tangent(Pa, sides[2][2], Pc)),
        np.arccos(tangent(Pb, sides[0][2], Pa) @ tangent(Pb, sides[1][2], Pc)),
        np.arccos(tangent(Pc, sides[2][2], Pa) @ tangent(Pc, sides[1][2], Pb))]
close("Figure 0.8.3: angles of the hyperbolic triangle (deg)", np.degrees(angs), [32.4712, 32.4712, 81.2026], tol=1e-3)
area = 0.0
for P, Q, c in sides:
    a0, a1 = np.arctan2(*(P - c)[::-1]), np.arctan2(*(Q - c)[::-1])
    rr = np.linalg.norm(P - c)
    tt = np.linspace(a0, a1, 200001)
    pts = np.stack([c[0] + rr * np.cos(tt), c[1] + rr * np.sin(tt)], 1)
    mid = (pts[1:] + pts[:-1]) / 2
    area += np.sum(np.diff(pts[:, 0]) / mid[:, 1])
close("(0.8.7): hyperbolic area (Stokes: oint dx/y) = pi - angle sum", area, np.pi - sum(angs), tol=1e-8)
close("Figure 0.8.3: hyperbolic area ~ 0.591", area, 0.5909, tol=1e-3)


# ---------------------------------------------------------------- (0.8.8): circumference of geodesic circles, Figure 0.8.4
def d_hyp(p, q):
    return np.arccosh(1 + ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) / (2 * p[1] * q[1]))


tt = np.linspace(0, 2 * np.pi, 20001)
for rho in (0.5, 1.0, 2.5):
    cy, Rr = np.cosh(rho), np.sinh(rho)
    pts = np.stack([Rr * np.cos(tt), cy + Rr * np.sin(tt)], 1)
    close(f"(0.8.8): points of the circle (rho = {rho}) are at distance rho from (0,1) in U^2",
          np.max(np.abs(d_hyp(np.array([0.0, 1.0]), pts.T) - rho)), 0.0, tol=1e-10)
    mid = (pts[1:] + pts[:-1]) / 2
    L = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1) / mid[:, 1])
    close(f"(0.8.8): hyperbolic circumference = 2 pi sinh rho (rho = {rho})", L / (2 * np.pi * np.sinh(rho)), 1.0,
          tol=1e-6)
rho_s = sp.symbols("rho", positive=True)
check("(0.8.8): hyperbolic disk area = 2 pi (cosh rho - 1) (polar coordinates, metric dr^2 + sinh^2 r dtheta^2)",
      sp.simplify(sp.integrate(2 * sp.pi * sp.sinh(t), (t, 0, rho_s)) - 2 * sp.pi * (sp.cosh(rho_s) - 1)) == 0)
sym_equal("(0.8.8): unit sphere circle of latitude at colatitude rho has length 2 pi sin rho",
          sp.sqrt(sph["expected"]["G"].subs({r: 1, th: rho_s})) * 2 * sp.pi, 2 * sp.pi * sp.sin(rho_s),
          {rho_s: (0, sp.pi)})


# ---------------------------------------------------------------- SO(3) with the bi-invariant metric, CV box (averaging)
def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(K):
    return np.array([K[2, 1], K[0, 2], K[1, 0]])


def expR(w):
    a = np.linalg.norm(w)
    if a < 1e-15:
        return np.eye(3)
    K = hat(w)
    return np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a ** 2 * K @ K


def logR(Rm):
    c = np.clip((np.trace(Rm) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-12:
        return np.zeros(3)
    return a / (2 * np.sin(a)) * vee(Rm - Rm.T)


def ipm(A_, B_):
    return 0.5 * np.trace(A_.T @ B_)


w = rng.normal(size=3)
close("Remark 0.8.4: |[w]_x|^2 = |w|^2 for <A,B> = tr(A^T B)/2", ipm(hat(w), hat(w)), w @ w, tol=1e-12)
R1 = expR(rng.normal(size=3))
A_, B_ = hat(rng.normal(size=3)), hat(rng.normal(size=3))
close("Remark 0.8.4: the metric is left invariant", ipm(R1 @ A_, R1 @ B_), ipm(A_, B_), tol=1e-12)
close("Remark 0.8.4: the metric is right invariant", ipm(A_ @ R1, B_ @ R1), ipm(A_, B_), tol=1e-12)
worst = 0.0
for _ in range(30):
    wv = rng.normal(size=3)
    wv *= rng.uniform(0.05, 3.0) / np.linalg.norm(wv)
    worst = max(worst, np.linalg.norm(logR(expR(wv)) - wv))
close("Remark 0.8.4: log(exp([w]_x)) = w for |w| < pi", worst, 0.0, tol=1e-9)
Ra, Rb = expR(np.array([0.2, -0.4, 0.1])), expR(np.array([-0.3, 0.5, 0.6]))
ang = np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))
close("Remark 0.8.4: d(R1, R2) = rotation angle of R1^T R2 = |log(R1^T R2)|", np.linalg.norm(logR(Ra.T @ Rb)), ang,
      tol=1e-12)
# the curve t ↦ R exp(t[w]) has constant speed |w| in this metric (checked numerically)
hh = 1e-6
Rt = lambda tt_: Ra @ expR(tt_ * np.array([0.3, 0.1, -0.2]))  # noqa: E731
spd = [np.sqrt(ipm((Rt(t0 + hh) - Rt(t0 - hh)) / (2 * hh), (Rt(t0 + hh) - Rt(t0 - hh)) / (2 * hh))) for t0 in (0, 0.7, 1.9)]
close("Remark 0.8.4: t -> R exp(t[w]) has constant speed |w|", spd, [np.linalg.norm([0.3, 0.1, -0.2])] * 3, tol=1e-6)


def karcher(Rs, iters=100):
    M = Rs[0].copy()
    for _ in range(iters):
        delta = np.mean([logR(M.T @ Ri) for Ri in Rs], axis=0)
        M = M @ expR(delta)
        if np.linalg.norm(delta) < 1e-14:
            break
    return M


center = expR(np.array([0.4, -0.2, 0.9]))
Rs = [center @ expR(0.3 * rng.normal(size=3)) for _ in range(12)]
Mk = karcher(Rs)
res = np.linalg.norm(np.sum([logR(Mk.T @ Ri) for Ri in Rs], axis=0))
close("CV box (averaging): first-order condition sum log(M^T R_i) = 0 at the Karcher mean", res, 0.0, tol=1e-10)


def cost(M):
    return sum(np.linalg.norm(logR(M.T @ Ri)) ** 2 for Ri in Rs)


check("CV box (averaging): the Karcher mean has lower cost than nearby rotations",
      all(cost(Mk) < cost(Mk @ expR(1e-3 * rng.normal(size=3))) for _ in range(10)))
angles = [0.1, 0.25, -0.05, 0.4]
Rz = lambda a: expR(np.array([0, 0, a]))  # noqa: E731
close("CV box (averaging): rotations about one axis average to the mean angle", karcher([Rz(a) for a in angles]),
      Rz(np.mean(angles)), tol=1e-10)

# ---------------------------------------------------------------- Riemannian optimization on a sphere (CV box)
Amat = rng.normal(size=(8, 9))
Q = Amat.T @ Amat
hvec = rng.normal(size=9)
hvec /= np.linalg.norm(hvec)
step = 0.5 / np.linalg.eigvalsh(Q).max()
for _ in range(20000):
    egrad = 2 * Q @ hvec
    rgrad = egrad - (egrad @ hvec) * hvec                     # projection onto T_h S^8
    hvec = hvec - step * rgrad
    hvec /= np.linalg.norm(hvec)                              # retraction
_, _, Vt = np.linalg.svd(Amat)
close("CV box (optimization): Riemannian GD on S^8 finds the smallest right singular vector (DLT)",
      abs(hvec @ Vt[-1]), 1.0, tol=1e-8)
check("CV box (optimization): Riemannian gradient is tangent", abs(rgrad @ hvec) < 1e-12)

# Exercise 0.8.4: f = z on S², grad = e3 − <e3, x> x
e3 = np.array([0, 0, 1.0])
gradS = lambda p: e3 - (e3 @ p) * p  # noqa: E731
close("Exercise 0.8.4: grad z at (1,0,0) is e3", gradS(np.array([1.0, 0, 0])), e3, tol=1e-15)
close("Exercise 0.8.4: grad z at N is 0", gradS(e3), np.zeros(3), tol=1e-15)
pt = np.array([0.6, 0.0, 0.8])
close("Exercise 0.8.4: grad z at (0.6, 0, 0.8) = (-0.48, 0, 0.36)", gradS(pt), [-0.48, 0, 0.36], tol=1e-12)

summary()
