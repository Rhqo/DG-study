"""Verification for Section 49.2 (Preliminaries: Rendering, Merged Local Manifolds, Varifolds).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-2-preliminaries.py
"""
import math

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, summary

rng = np.random.default_rng(49_2)

# ---------------------------------------------------------------- (49.2.2): blending weights
# sum_i beta_i prod_{j<i} (1 - beta_j) = 1 - prod_j (1 - beta_j)   (Exercise 49.2.1)
ok = True
for _ in range(200):
    beta = rng.uniform(0, 1, size=rng.integers(1, 30))
    T = np.concatenate([[1.0], np.cumprod(1 - beta)[:-1]])
    ok &= abs((beta * T).sum() - (1 - np.prod(1 - beta))) < 1e-12
check("Exercise 49.2.1: blending weights sum to 1 - prod(1 - beta_j)", ok)
b = sp.symbols("b1:5")
lhs = sum(b[i] * sp.prod([1 - b[j] for j in range(i)]) for i in range(4))
check("Exercise 49.2.1: identity symbolically for 4 Gaussians", sp.expand(lhs - (1 - sp.prod([1 - x for x in b]))) == 0)
close("Exercise 49.2.1: three Gaussians beta = 0.5: weights 0.5, 0.25, 0.125, total 0.875",
      0.5 + 0.25 + 0.125, 1 - 0.5 ** 3, tol=1e-15)

# ---------------------------------------------------------------- Figure 49.2.2: principal curvatures on T_{2,0.8}
u, v = sp.symbols("u v", real=True)
Rt, rt = sp.Integer(2), sp.Rational(4, 5)
Xt = sp.Matrix([(Rt + rt * sp.cos(v)) * sp.cos(u), (Rt + rt * sp.cos(v)) * sp.sin(u), rt * sp.sin(v)])
Wt = dgsym.shape_operator(Xt, u, v).subs({u: 0, v: 0})
Nt = dgsym.unit_normal(Xt, u, v).subs({u: 0, v: 0})
sgn = 1 if float(Nt[0]) > 0 else -1          # make the normal outward (+e_x at q = (2.8, 0, 0))
ev = sorted([sgn * float(e) for e in Wt.eigenvals(multiple=True)], key=abs, reverse=True)
close("Figure 49.2.2: tau_1 = -1.25 (outward normal, W = -dN)", ev[0], -1.25, tol=1e-12)
close("Figure 49.2.2: tau_2 = -1/2.8 = -0.357", ev[1], -1 / 2.8, tol=1e-12)
close("Figure 49.2.2: meridian section height at s = 0.5", -(0.8 - math.sqrt(0.64 - 0.25)), -0.1755, tol=1e-4)
close("Figure 49.2.2: quadratic approximation at s = 0.5 along w1", 0.5 * (-1.25) * 0.25, -0.15625, tol=1e-12)

# ---------------------------------------------------------------- (49.2.4) = paper (14): sign of the sectional curvature
th, ph = sp.symbols("theta phi", positive=True)
g = sp.diag(1, sp.sin(th) ** 2)                 # unit sphere
Rl = dgsym.riemann_lower(g, [th, ph])           # R_{ijkl} = <R(d_i, d_j) d_k, d_l>, Lee convention = (13)
Ksec = dgsym.sectional(g, [th, ph], [1, 0], [0, 1])
check("Lee convention (13): K = Rm(v,w,w,v)/... = +1 on the unit sphere", sp.simplify(Ksec - 1) == 0)
paper14 = sp.simplify(Rl[0][1][0][1] / (g[0, 0] * g[1, 1] - g[0, 1] ** 2))
check("paper (14) taken literally, g(R(V1,V2)V1,V2)/..., gives -1 on the unit sphere", sp.simplify(paper14 + 1) == 0)
# radius r sphere: literal (14) gives -1/r^2
r = sp.symbols("r", positive=True)
gr = sp.diag(r ** 2, r ** 2 * sp.sin(th) ** 2)
Rlr = dgsym.riemann_lower(gr, [th, ph])
check("paper (14) literally on the sphere of radius r: -1/r^2",
      sp.simplify(Rlr[0][1][0][1] / (gr[0, 0] * gr[1, 1]) + 1 / r ** 2) == 0)
# constant curvature c: <R(X,Y)X, Y> = -c(|X|^2|Y|^2 - <X,Y>^2)   (Exercise 49.2.3)
c = sp.symbols("c")
Xv, Yv = sp.Matrix(sp.symbols("x1:4")), sp.Matrix(sp.symbols("y1:4"))
Rxyx = c * ((Yv.dot(Xv)) * Xv - (Xv.dot(Xv)) * Yv)
check("Exercise 49.2.3: <R(X,Y)X,Y> = -c(|X|^2|Y|^2 - <X,Y>^2) for constant curvature c",
      sp.expand(Rxyx.dot(Yv) + c * (Xv.dot(Xv) * Yv.dot(Yv) - Xv.dot(Yv) ** 2)) == 0)

# ---------------------------------------------------------------- Example 49.2.2: measure zero is not negligible
for d, expected in ((0.05, 0.19), (0.02, 0.0784)):
    close(f"Example 49.2.2: fraction 4d - 4d^2 at d = {d}", 4 * d - 4 * d * d, expected, tol=1e-12)
close("Exercise 49.2.2: half of the surface when d = (1 - 1/sqrt 2)/2 = 0.146",
      4 * ((1 - 1 / math.sqrt(2)) / 2) - 4 * ((1 - 1 / math.sqrt(2)) / 2) ** 2, 0.5, tol=1e-12)
close("Exercise 49.2.2: (1 - 1/sqrt 2)/2 = 0.146", (1 - 1 / math.sqrt(2)) / 2, 0.146, tol=1e-3)
# Monte Carlo on the surface of the unit cube (area 6): fraction of points within d of an edge
n = 400_000
face = rng.integers(0, 6, n)
a, bb = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
P = np.zeros((n, 3))
ax = face // 2
for k in range(3):
    m = ax == k
    others = [j for j in range(3) if j != k]
    P[m, k] = (face[m] % 2).astype(float)
    P[m, others[0]] = a[m]
    P[m, others[1]] = bb[m]
d = 0.05
near = np.zeros(n, bool)
for k in range(3):
    m = ax == k
    others = [j for j in range(3) if j != k]
    near[m] = (np.minimum(P[m, others[0]], 1 - P[m, others[0]]) < d) | (np.minimum(P[m, others[1]], 1 - P[m, others[1]]) < d)
close("Example 49.2.2: Monte Carlo fraction near the 12 edges of the unit cube (d = 0.05)", near.mean(), 0.19, tol=3e-3)

# ---------------------------------------------------------------- Example 49.2.3: first variation at a corner
# V = H^1 restricted to floor [0,1] x {0} and wall {0} x [0,1]; X compactly supported near the corner.
def bump(p):
    s = np.sum(p ** 2, axis=-1) / 0.25
    return np.where(s < 1, np.exp(-1 / np.maximum(1 - s, 1e-300)), 0.0)


cvec = np.array([0.3, -0.7])


def Xfield(p):
    return bump(p)[..., None] * cvec[None, :] * (1 + p[..., :1] - 2 * p[..., 1:2])


def div_T(p, t, h=1e-6):
    # tangential divergence along the unit tangent t: <D_t X, t>
    return np.sum((Xfield(p + h * t) - Xfield(p - h * t)) / (2 * h) * t, axis=-1)


s = np.linspace(0, 1, 200_001)
ds = s[1] - s[0]
floor = np.stack([s, 0 * s], axis=1)
wall = np.stack([0 * s, s], axis=1)


def trap(f):
    return ds * (f.sum() - 0.5 * (f[0] + f[-1]))


dV = trap(div_T(floor, np.array([1.0, 0]))) + trap(div_T(wall, np.array([0, 1.0])))
X0 = Xfield(np.zeros((1, 2)))[0]
close("Example 49.2.3: delta V(X) = -<e_x + e_z, X(corner)>", dV, -(X0[0] + X0[1]), tol=1e-6)
close("Example 49.2.3: |e_x + e_z| = sqrt 2", math.hypot(1, 1), math.sqrt(2), tol=1e-15)

# smoothed first variation / smoothed mass at the corner with a Gaussian kernel exp(-|x|^2/eps^2):
# |delta V * rho| / (||V|| * rho) at the corner = sqrt(2) rho(0) / (2 int_0^inf rho(s) ds) = sqrt(2/pi) / eps
for eps in (0.05, 0.1, 0.2):
    ss = np.linspace(0, 1, 400_001)
    rho = np.exp(-ss ** 2 / eps ** 2)
    mass = 2 * (ss[1] - ss[0]) * (rho.sum() - 0.5 * (rho[0] + rho[-1]))
    ratio = math.sqrt(2) * 1.0 / mass
    close(f"Example 49.2.3 (remark): smoothed curvature at the corner = sqrt(2/pi)/eps, eps = {eps}",
          ratio * eps, math.sqrt(2 / math.pi), tol=1e-6)
close("Example 49.2.3 (remark): sqrt(2/pi) = 0.80", math.sqrt(2 / math.pi), 0.80, tol=3e-3)

# ---------------------------------------------------------------- Exercise 49.2.4: sphere of radius r, X(x) = x
rs = sp.symbols("r", positive=True)
lhs = 2 * 4 * sp.pi * rs ** 2                        # div_T x = tr P = 2, integrated over the area 4 pi r^2
Hvec_dot_X = (-2 / rs) * rs                          # <H, x> with H = 2 H N = -(2/r) N, x = r N
rhs = -Hvec_dot_X * 4 * sp.pi * rs ** 2
check("Exercise 49.2.4: (17) and (18) both give 8 pi r^2 for X(x) = x", sp.simplify(lhs - rhs) == 0)

summary()
