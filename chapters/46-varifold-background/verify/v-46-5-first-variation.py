"""Verification for Section 46.5 (First Variation and Generalized Mean Curvature).

Run from the project root: PYTHONPATH=tools python3 chapters/46-varifold-background/verify/v-46-5-first-variation.py
"""
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgmesh as dm  # noqa: E402
import var46 as v  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(465)


def smooth_field(seed, n=2):
    """X(x) = c + B x + sin-terms, with its Jacobian (a smooth test field; compact support is irrelevant on compact sets)."""
    r = np.random.default_rng(seed)
    c = r.normal(size=n)
    B = r.normal(size=(n, n))
    K = r.normal(size=(n, n))
    a = r.normal(size=n)

    def X(x):
        x = np.asarray(x, float)
        return c + x @ B.T + a * np.sin(x @ K.T)

    def DX(x):
        x = np.asarray(x, float)
        return B + (a * np.cos(K @ x))[:, None] * K

    return X, DX


# ---------------------------------------------------------------- Proposition 46.5.2: delta V(X) is the rate of change of mass
G = v.ellipse(2 * np.pi * np.arange(12) / 12)
for seed in (1, 2, 3):
    X, DX = smooth_field(seed)
    dv = v.polyline_first_variation_quadrature(X, DX, G, closed=True)
    # moving the vertices by the flow of X to first order: lengths of the polygon with vertices x + h X(x)
    h = 1e-6
    Lp = v.polyline_length(G + h * X(G), closed=True)
    Lm = v.polyline_length(G - h * X(G), closed=True)
    close(f"Proposition 46.5.2: d/dt length = delta V(X) for a 12-gon (field {seed})", (Lp - Lm) / (2 * h), dv, 1e-6)
    W = v.polyline_first_variation(G, closed=True)
    close(f"Example 46.5.6: delta V(X) = sum <t_(k-1) - t_k, X(gamma_k)> (field {seed})",
          float(np.sum(W * X(G))), dv, 1e-10)

# ---------------------------------------------------------------- Proposition 46.5.3: smooth curves and surfaces
# circle of radius r: H = -x / r^2 (inward, |H| = 1/r)
r0 = 1.3
for seed in (4, 5):
    X, DX = smooth_field(seed)

    def integrand_div(t):
        x = v.circle(t, r0)
        P = v.line_proj(t + np.pi / 2)
        return np.array([np.trace(P[i] @ DX(x[i])) for i in range(len(t))]) * r0

    def integrand_H(t):
        x = v.circle(t, r0)
        H = -x / r0 ** 2
        return -np.sum(H * X(x), axis=1) * r0

    close(f"Proposition 46.5.3: circle r = {r0}: int div_C X = -int <H, X> with H = -x/r^2 (field {seed})",
          v.integrate_1d(integrand_div, 0, 2 * np.pi, n=32, pieces=32), v.integrate_1d(integrand_H, 0, 2 * np.pi, n=32, pieces=32),
          1e-10)
# ellipse: H = kappa_s n_s
for seed in (6,):
    X, DX = smooth_field(seed)

    def f_div(t):
        x = v.ellipse(t)
        P = v.dir_proj(v.ellipse_tangent(t))
        return np.array([np.trace(P[i] @ DX(x[i])) for i in range(len(t))]) * v.ellipse_speed(t)

    def f_H(t):
        H = v.ellipse_kappa(t)[:, None] * v.ellipse_normal(t)
        return -np.sum(H * X(v.ellipse(t)), axis=1) * v.ellipse_speed(t)

    close("Proposition 46.5.3: ellipse: int div X = -int <kappa n, X> ds",
          v.integrate_1d(f_div, 0, 2 * np.pi, n=32, pieces=32), v.integrate_1d(f_H, 0, 2 * np.pi, n=32, pieces=32), 1e-10)


# torus T_{2, 0.8} with outward N: H = -(R + 2 r cos u) / (2 r (R + r cos u)), H-vector = 2 H N
def torus_check(Rt, rt, X, DX, n=48):
    gx, gw = np.polynomial.legendre.leggauss(n)
    us = np.pi * (gx + 1)
    wu = np.pi * gw
    tot_div = tot_H = 0.0
    for u, wuu in zip(us, wu):
        for vv, wvv in zip(us, wu):
            x = np.array([(Rt + rt * math.cos(u)) * math.cos(vv), (Rt + rt * math.cos(u)) * math.sin(vv), rt * math.sin(u)])
            N = np.array([math.cos(u) * math.cos(vv), math.cos(u) * math.sin(vv), math.sin(u)])
            dA = rt * (Rt + rt * math.cos(u)) * wuu * wvv
            P = np.eye(3) - np.outer(N, N)
            H = -(Rt + 2 * rt * math.cos(u)) / (2 * rt * (Rt + rt * math.cos(u)))
            tot_div += np.trace(P @ DX(x)) * dA
            tot_H += -2 * H * (N @ X(x)) * dA
    return tot_div, tot_H


for seed in (7, 8):
    X, DX = smooth_field(seed, n=3)
    a, b = torus_check(2.0, 0.8, X, DX)
    close(f"Proposition 46.5.3: torus T(2, 0.8): int div_M X = -int <2HN, X> dA (field {seed})", a, b, 1e-9 * max(1, abs(b)))
# sphere: X = (0, 0, z): int div = 8 pi r^2 / 3 = int <2x/r^2, X>
close("Proposition 46.5.3: sphere r: -int <H, (0,0,z)> = int 2 z^2 / r^2 dA = 8 pi r^2 / 3",
      v.integrate_1d(lambda u: 2 * np.pi * r0 ** 2 * np.sin(u) * 2 * np.cos(u) ** 2, 0, np.pi), 8 * math.pi * r0 ** 2 / 3, 1e-10)
close("Example: |H| of the sphere of radius 1.3 is 2/r = 1.5385", 2 / r0, 1.5385, 5e-5)

# ---------------------------------------------------------------- Example 46.5.7: an open segment
a_, b_ = np.array([0.2, -0.1]), np.array([1.4, 0.7])
S = np.stack([a_, b_])
t = (b_ - a_) / np.linalg.norm(b_ - a_)
for seed in (9, 10):
    X, DX = smooth_field(seed)
    close(f"Example 46.5.7: segment: delta V(X) = <X(b) - X(a), t> (field {seed})",
          v.polyline_first_variation_quadrature(X, DX, S, closed=False), (X(b_) - X(a_)) @ t, 1e-10)

# ---------------------------------------------------------------- Example 46.5.8: junctions
def junction_dv(dirs, c, radius=0.6):
    """delta V(X) for half-lines from 0 (cut at length 1) with X = bump(|x|/radius) c."""
    def X(x):
        return v.bump(x, (0, 0), radius) * np.asarray(c, float)

    def DX(x):
        h = 1e-6
        return np.stack([(X(x + h * e) - X(x - h * e)) / (2 * h) for e in np.eye(2)], axis=1)

    tot = 0.0
    for u in dirs:
        seg = np.linspace(0, 1, 41)[:, None] * np.asarray(u, float)[None, :]   # the half-line cut at length 1
        tot += v.polyline_first_variation_quadrature(X, DX, seg, closed=False, nq=24)
    return tot


Y = [(math.cos(a), math.sin(a)) for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3)]
T = [(1, 0), (-1, 0), (0, 1)]
Xc = [(1, 0), (-1, 0), (0, 1), (0, -1)]
for c in ((1.0, 0.0), (0.3, -0.8)):
    close(f"Example 46.5.8: 120-degree Y junction is stationary (c = {c})", junction_dv(Y, c), 0.0, 1e-6)
    close(f"Example 46.5.8: two crossing lines are stationary (c = {c})", junction_dv(Xc, c), 0.0, 1e-6)
    close(f"Example 46.5.8: T junction: delta V(X) = -<u1+u2+u3, X(0)> (c = {c})", junction_dv(T, c), -c[1], 1e-6)
# Steiner: L(y) = 2 sqrt(1 + y^2) + 1 - y
close("Figure 46.5.2: T junction length 3", 3.0, 2 * 1 + 1, 0)
close("Figure 46.5.2: minimum 1 + sqrt3 = 2.7321 at y = 1/sqrt3 = 0.5774",
      min(2 * math.sqrt(1 + y * y) + 1 - y for y in np.linspace(0, 1, 100001)), 1 + math.sqrt(3), 1e-9)
close("Figure 46.5.2: slope of L at y = 0 is -1", (2 * math.sqrt(1 + 1e-12) - 2) / 1e-6 - 1, -1.0, 1e-5)
close("Figure 46.5.2: 1 + sqrt3 = 2.7321", 1 + math.sqrt(3), 2.7321, 5e-5)

# ---------------------------------------------------------------- Example 46.5.9: density on a curve
A = 0.5


def u_of_t(ts):
    u = np.asarray(ts, float) / (2 * np.pi)
    for _ in range(40):
        u = u - (2 * np.pi * u + A * np.sin(2 * np.pi * u) - ts) / (2 * np.pi * (1 + A * np.cos(2 * np.pi * u)))
    return u


def theta(ts):
    return 1 / (1 + A * np.cos(2 * np.pi * u_of_t(ts)))


def dtheta(ts):
    u = u_of_t(ts)
    return A * np.sin(2 * np.pi * u) / (1 + A * np.cos(2 * np.pi * u)) ** 3


hh = 1e-6
tt = np.linspace(0.1, 6.0, 7)
close("Example 46.5.9: d theta / dt = a sin(2 pi u) / (1 + a cos 2 pi u)^3",
      (theta(tt + hh) - theta(tt - hh)) / (2 * hh), dtheta(tt), 1e-7)
for seed in (11, 12):
    X, DX = smooth_field(seed)

    def f_div(ts):
        x = v.circle(ts)
        P = v.line_proj(ts + np.pi / 2)
        return np.array([np.trace(P[i] @ DX(x[i])) for i in range(len(ts))]) * theta(ts)

    def f_H(ts):
        x = v.circle(ts)
        Hv = -x + (dtheta(ts) / theta(ts))[:, None] * v.circle_tangent(ts)     # kappa n + (theta'/theta) t
        return -np.sum(Hv * X(x), axis=1) * theta(ts)

    close(f"Example 46.5.9: delta v(C, theta)(X) = -int <kappa n + (theta'/theta) t, X> theta ds (field {seed})",
          v.integrate_1d(f_div, 0, 2 * np.pi, n=32, pieces=64), v.integrate_1d(f_H, 0, 2 * np.pi, n=32, pieces=64), 1e-9)
us = np.linspace(0, 1, 200001)
tang = A * np.sin(2 * np.pi * us) / (1 + A * np.cos(2 * np.pi * us)) ** 2
k = int(np.argmax(tang))
close("Example 46.5.9: max of theta'/theta = 0.847", tang.max(), 0.847, 5e-4)
close("Example 46.5.9: attained at u = 0.381", us[k], 0.381, 1e-3)
close("Example 46.5.9: ... that is t = 2 pi u + a sin 2 pi u = 2.73 (156.6 degrees)", 2 * math.pi * us[k] + A * math.sin(2 * math.pi * us[k]), 2.733, 1e-3)

# ---------------------------------------------------------------- Example 46.5.11: the staircase limit has unbounded first variation
for n in (1, 4, 16):
    W = v.polyline_first_variation(v.staircase(n), closed=False)
    close(f"Example 46.5.11: total variation of delta v(S_{n}, 1) = (2n - 1) sqrt2 + 2",
          np.sum(np.linalg.norm(W, axis=1)), (2 * n - 1) * math.sqrt(2) + 2, 1e-12)
CEN, RAD = (0.5, 0.5), 0.4
psi_D = v.diagonal_integral(lambda X, P: v.bump(X, CEN, RAD), nq=256)


def Xk(k):
    def X(x):
        x = np.atleast_2d(x)
        val = v.bump(x, CEN, RAD) * np.sin(k * (x[:, 0] - x[:, 1]))
        return np.stack([val, np.zeros_like(val)], axis=1).squeeze()

    def DX(x):
        h = 1e-6
        return np.stack([(X(x + h * e) - X(x - h * e)) / (2 * h) for e in np.eye(2)], axis=1)
    return X, DX


for kk in (5, 20, 80):
    X, DX = Xk(kk)
    # V_inf: (sqrt2/2) int_D div X dH^1; on D: div X = k psi
    lim = math.sqrt(2) / 2 * kk * psi_D
    vinf = v.staircase_limit_integral(lambda Xs, Ps: np.array([np.trace(P @ DX(x)) for x, P in zip(Xs, Ps)]), nq=128)
    close(f"Example 46.5.11: delta V_inf(X_k) = (sqrt2/2) k int_D psi, k = {kk}", vinf, lim, 1e-4 * lim)
    vD = v.diagonal_integral(lambda Xs, Ps: np.array([np.trace(P @ DX(x)) for x, P in zip(Xs, Ps)]), nq=128)
    close(f"Example 46.5.11: delta v(D,1)(X_k) = 0, k = {kk}", vD, 0.0, 1e-6)
close("Example 46.5.11: int_D psi dH^1 = 0.4828", psi_D, 0.4828, 5e-5)
X, DX = Xk(20)
lim20 = math.sqrt(2) / 2 * 20 * psi_D
vals = [v.polyline_first_variation_quadrature(X, DX, v.staircase(n), closed=False, nq=16) for n in (50, 200, 800)]
check(f"Example 46.5.11: delta v(S_n,1)(X_20) -> delta V_inf(X_20) = {lim20:.3f} ({np.round(vals, 3)})",
      abs(vals[-1] - lim20) < abs(vals[0] - lim20) and abs(vals[-1] - lim20) < 0.02 * lim20)
close("Example 46.5.11: delta V_inf(X_20) = 6.827", lim20, 6.827, 5e-4)

# ---------------------------------------------------------------- Proposition 46.5.12: meshes, hat functions, cotan formula
def check_mesh(V, F, name, verts):
    L = dm.cotan_laplacian(V, F)
    LX = L @ V
    mb = dm.mass_barycentric(V, F)
    # edge form: sum over edges at i of (l/2)(eta_f + eta_g)
    fn = dm.face_normals(V, F)
    for i in verts:
        hv = v.hat_first_variation(V, F, i)
        close(f"Proposition 46.5.12: {name}, vertex {i}: delta V(phi_i) = -(L x)_i", hv, -LX[i], 1e-10)
        h = 1e-6
        gA = np.array([(v.mesh_area(V + h * np.outer(np.eye(len(V))[i], e), F)
                        - v.mesh_area(V - h * np.outer(np.eye(len(V))[i], e), F)) / (2 * h) for e in np.eye(3)])
        close(f"Proposition 46.5.12: {name}, vertex {i}: delta V(phi_i) = grad_(x_i) A (finite differences)", hv, gA, 1e-6)
        close(f"Proposition 46.5.12: {name}, vertex {i}: ||V||(phi_i) = barycentric area", v.hat_mass(V, F, i), mb[i], 1e-12)
        tot = np.zeros(3)
        for f, tri in enumerate(F):
            if i not in tri:
                continue
            c = list(tri).index(i)
            for j in (tri[(c + 1) % 3], tri[(c + 2) % 3]):
                k = [x for x in tri if x != i and x != j][0]
                e = V[j] - V[i]
                ell = np.linalg.norm(e)
                eta = np.cross(e, fn[f])
                eta /= np.linalg.norm(eta)
                if eta @ (V[k] - V[i]) > 0:   # outward conormal points away from the third vertex
                    eta = -eta
                tot += 0.5 * ell * eta
        close(f"Proposition 46.5.12: {name}, vertex {i}: edge form sum (l/2)(eta_f + eta_g)", tot, hv, 1e-10)


Vc, Fc = v.cone_one_ring(6, 0.4)
Vc = np.vstack([Vc, [[0, 0, -2.0]]])   # close it with a bottom apex so that all ring edges are interior
Fc = np.vstack([Fc, [[7, 1 + (j + 1) % 6, 1 + j] for j in range(6)]])
check_mesh(Vc, Fc, "closed bicone", [0, 1])
Vi, Fi = dm.icosphere(2)
Vp = Vi + 0.03 * rng.normal(size=Vi.shape)
check_mesh(Vp, Fi, "perturbed icosphere", [0, 5, 17])
Vo, Fo = v.cone_one_ring(6, 0.4)
hv = v.hat_first_variation(Vo, Fo, 0)
close("Example: cone apex (h = 0.4, k = 6): delta V(phi_0) is vertical", hv[:2], np.zeros(2), 1e-12)
check("Example: cone apex: delta V(phi_0) points up (raising the apex adds area)", hv[2] > 0)
close("Remark 46.5.13: cone apex: delta V(phi_0) = (0, 0, 1.2579)", hv[2], 1.2579, 5e-5)
close("Remark 46.5.13: cone apex: ||V||(phi_0) = 0.9539", v.hat_mass(Vo, Fo, 0), 0.9539, 5e-5)
close("Remark 46.5.13: cone apex: ratio 1.3187", np.linalg.norm(hv) / v.hat_mass(Vo, Fo, 0), 1.3187, 5e-5)
for n, r0_, r90 in ((12, 1.6653, 0.2632), (48, 1.9747, 0.2508)):
    tq = 2 * np.pi * np.arange(n) / n
    Pq = v.ellipse(tq)
    wq = v.polyline_first_variation(Pq, closed=True)
    lq = np.linalg.norm(np.roll(Pq, -1, axis=0) - Pq, axis=1)
    rq = np.linalg.norm(wq, axis=1) / (0.5 * (lq + np.roll(lq, 1)))
    close(f"Figure 46.5.1: N = {n}: ratio at t = 0 is {r0_} (kappa = 2)", rq[0], r0_, 5e-5)
    close(f"Figure 46.5.1: N = {n}: ratio at t = 90 deg is {r90} (kappa = 0.25)", rq[n // 4], r90, 5e-5)
G12 = v.ellipse(2 * np.pi * np.arange(12) / 12)
tl = np.linalg.norm(v.polyline_first_variation(G12, closed=True), axis=1)
close("Figure 46.5.1: turning vector lengths at k = 0, 1, 2, 3", tl[:4], [0.9447, 0.6055, 0.3290, 0.2656], 5e-5)
half = 2 * np.pi * np.arange(12) / 12 + np.pi / 12
sm = v.ellipse_tangent(half) - v.ellipse_tangent(half - np.pi / 6)
close("Figure 46.5.1: turning vectors equal T(t_(k+1/2)) - T(t_(k-1/2)) exactly (affine image of a circle)",
      -v.polyline_first_variation(G12, closed=True), sm, 1e-12)
close("Example 46.5.11: total variation for n = 16 is 45.84", 31 * math.sqrt(2) + 2, 45.84, 5e-3)
for level, lo, hi, v5 in ((3, None, None, None), (4, 1.9939, 2.0029, 2.2903)):
    V4, F4 = dm.icosphere(level)
    LX = dm.cotan_apply(V4, F4, V4)
    mb = dm.mass_barycentric(V4, F4)
    ratio = np.linalg.norm(LX, axis=1) / mb
    inward = np.all(np.sum(LX * V4, axis=1) < 0)
    val = np.bincount(np.asarray(F4).ravel(), minlength=len(V4))
    if level == 4:
        close("Remark: icosphere(4): min of |delta V(phi_i)|/||V||(phi_i) at valence 6 = 1.9939", ratio[val == 6].min(), lo, 5e-5)
        close("Remark: icosphere(4): max at valence 6 = 2.0029", ratio[val == 6].max(), hi, 5e-5)
        close("Remark: icosphere(4): valence-5 vertices = 2.2903", ratio[val == 5].mean(), v5, 5e-5)
        check("Remark: icosphere(4): -delta V(phi_i) = (L x)_i points inward", inward)

# ---------------------------------------------------------------- Exercises 46.5.1-46.5.3
close("Exercise 46.5.1: circle r = 1.3, X = x: delta V = 2 pi r",
      v.integrate_1d(lambda t: np.ones_like(t) * r0, 0, 2 * np.pi), 2 * math.pi * r0, 1e-12)
Sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
Wsq = v.polyline_first_variation(Sq, closed=True)
cen = Sq - 0.5
check("Exercise 46.5.2: square: each corner carries sqrt2 times the outward diagonal unit vector",
      np.allclose(Wsq, math.sqrt(2) * cen / np.linalg.norm(cen, axis=1, keepdims=True)))
close("Exercise 46.5.2: total variation 4 sqrt2", np.sum(np.linalg.norm(Wsq, axis=1)), 4 * math.sqrt(2), 1e-12)
U4 = np.array([[1, 0], [0, 1], [-1, 0], [-math.sqrt(2) / 2, -math.sqrt(2) / 2]])
close("Exercise 46.5.3: sum of the four directions = (-0.7071, 0.2929)", U4.sum(axis=0), [-0.7071, 0.2929], 5e-5)
close("Exercise 46.5.3: its length 0.7654", np.linalg.norm(U4.sum(axis=0)), 0.7654, 5e-5)
D4 = [tuple(u) for u in U4]
for c in ((1.0, 0.0), (0.0, 1.0)):
    close(f"Exercise 46.5.3: delta V(X) = -<sum u, X(0)> (c = {c})", junction_dv(D4, c), -float(U4.sum(axis=0) @ np.array(c)), 1e-6)

summary()
