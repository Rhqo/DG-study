"""Verification for Section 49.3 (Geometry-Constrained Optimization).

Run from the project root: PYTHONPATH=tools python3 chapters/49-paper-geosplat/verify/v-49-3-geometry-constrained-optimization.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import dgcode  # noqa: E402
import dgsym  # noqa: E402
import geo49  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

PAGE = HERE.parents[1] / "49-3-geometry-constrained-optimization.html"

# ---------------------------------------------------------------- Proposition 49.3.1 (paper Prop. 3.1, App. C)
s1, s2, sn, a1, a2 = sp.symbols("s1 s2 s_nbr a1 a2", positive=True)   # a_d = |tau_d|
sol = sp.solve([sp.Eq(s1 / s2, a1 / a2), sp.Eq(s1 * s2, sn ** 2 / 4)], [s1, s2], dict=True)
check("Prop 49.3.1: unique positive solution", len(sol) == 1)
check("Prop 49.3.1: s1 = (s_nbr/2) sqrt(|tau1/tau2|)", sp.simplify(sol[0][s1] - sn / 2 * sp.sqrt(a1 / a2)) == 0)
check("Prop 49.3.1: s2 = (s_nbr/2) sqrt(|tau2/tau1|)", sp.simplify(sol[0][s2] - sn / 2 * sp.sqrt(a2 / a1)) == 0)
check("Prop 49.3.1: tangent ellipse area pi s1 s2 = pi (s_nbr/2)^2 (area invariance)",
      sp.simplify(sp.pi * sol[0][s1] * sol[0][s2] - sp.pi * (sn / 2) ** 2) == 0)
# (27)-(28): V = s1 s2 s3, A = V / s3 = s1 s2 (constants 4 pi / 3 and pi omitted in the paper)
s3 = sp.symbols("s3", positive=True)
check("(27)-(28): V/s3 = s1 s2", sp.simplify(sol[0][s1] * sol[0][s2] * s3 / s3 - sn ** 2 / 4) == 0)
# projection of the 1-sigma ellipsoid onto the tangent plane is the ellipse with semi-axes s1, s2 (r3 = normal)
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
Sg = Q @ np.diag([0.3 ** 2, 0.1 ** 2, 0.01 ** 2]) @ Q.T
P2 = Q[:, :2].T @ Sg @ Q[:, :2]                       # covariance of the projection to span(r1, r2)
close("(28): projected 1-sigma ellipse has area pi s1 s2", math.pi * math.sqrt(np.linalg.det(P2)), math.pi * 0.03, tol=1e-12)

# ---------------------------------------------------------------- Example 49.3.2 (torus warm-up numbers)
S = geo49.warmup_scales(-1.25, -1 / 2.8, 0.2)
close("Example 49.3.2: s1 = 0.1 sqrt(3.5) = 0.1871", S[0], 0.18708, tol=1e-5)
close("Example 49.3.2: s2 = 0.1 / sqrt(3.5) = 0.0535", S[1], 0.05345, tol=1e-5)
close("Example 49.3.2: s1 s2 = 0.01", S[0] * S[1], 0.01, tol=1e-15)
close("Example 49.3.2: |tau1/tau2| = 3.5", 1.25 * 2.8, 3.5, tol=1e-12)

# ---------------------------------------------------------------- Example 49.3.3 (sagitta: GeoSplat swaps, sqrt balances)
k1, k2, c = sp.symbols("kappa1 kappa2 c", positive=True)
l1, l2 = c * sp.sqrt(k1 / k2), c * sp.sqrt(k2 / k1)        # GeoSplat: long axis l1 along kappa2, short l2 along kappa1
check("Example 49.3.3: GeoSplat sagitta along low curvature = isotropic sagitta along high curvature",
      sp.simplify(k2 * l1 ** 2 / 2 - k1 * c ** 2 / 2) == 0)
check("Example 49.3.3: GeoSplat sagitta along high curvature = isotropic sagitta along low curvature",
      sp.simplify(k1 * l2 ** 2 / 2 - k2 * c ** 2 / 2) == 0)
r1, r2 = c * (k1 / k2) ** sp.Rational(1, 4), c * (k2 / k1) ** sp.Rational(1, 4)
check("Example 49.3.3: square-root ratio gives equal sagittas c^2 sqrt(k1 k2)/2",
      sp.simplify(k2 * r1 ** 2 / 2 - c ** 2 * sp.sqrt(k1 * k2) / 2) == 0
      and sp.simplify(k1 * r2 ** 2 / 2 - c ** 2 * sp.sqrt(k1 * k2) / 2) == 0)
# minimax: for l1 l2 = c^2, max(k2 l1^2, k1 l2^2)/2 >= c^2 sqrt(k1 k2)/2 (AM-GM: product of the two is c^4 k1 k2)
x = np.linspace(0.2, 5, 2001)              # x = l1 / l2
kk1, kk2, cc = 1.25, 1 / 2.8, 0.1
L1, L2 = cc * np.sqrt(x), cc / np.sqrt(x)
worst = np.maximum(kk2 * L1 ** 2, kk1 * L2 ** 2) / 2
close("Example 49.3.3: the best ratio is sqrt(k1/k2) = 1.871", x[np.argmin(worst)], math.sqrt(3.5), tol=3e-3)
q = np.array([2.8, 0.0, 0.0])
ez, ey = np.array([0.0, 0, 1]), np.array([0.0, 1, 0])
dist = lambda p: float(geo49.torus_distance(p)[0])  # noqa: E731
close("Example 49.3.3: isotropic, gap at the end along w2 (exact torus)", dist(q + 0.1 * ey), 0.00179, tol=5e-6)
close("Example 49.3.3: isotropic, gap at the end along w1", dist(q + 0.1 * ez), 0.00623, tol=5e-6)
close("Example 49.3.3: GeoSplat, gap at the end along w2", dist(q + S[0] * ey), 0.00624, tol=5e-6)
close("Example 49.3.3: GeoSplat, gap at the end along w1", dist(q + S[1] * ez), 0.00178, tol=5e-6)
rt = 0.1 * 3.5 ** 0.25
close("Example 49.3.3: square-root, gap along w2", dist(q + rt * ey), 0.00334, tol=5e-6)
close("Example 49.3.3: square-root, gap along w1", dist(q + 0.01 / rt * ez), 0.00333, tol=5e-6)
close("Example 49.3.3: square-root half-axes 0.137, 0.073", rt, 0.1368, tol=1e-4)
close("Example 49.3.3: quadratic model c^2 sqrt(k1 k2)/2", 0.01 * math.sqrt(1.25 / 2.8) / 2, 0.00334, tol=5e-6)

# ---------------------------------------------------------------- Example 49.3.5: helicoid, H = 0, MAC > 0
u, v = sp.symbols("u v", real=True)
cc_ = sp.symbols("c", positive=True)
Xh = sp.Matrix([v * sp.cos(u), v * sp.sin(u), cc_ * u])
Kh, Hh = dgsym.K_H(Xh, u, v)
check("Example 49.3.5: helicoid H = 0", sp.simplify(Hh) == 0)
check("Example 49.3.5: helicoid K = -c^2/(c^2 + v^2)^2", sp.simplify(Kh + cc_ ** 2 / (cc_ ** 2 + v ** 2) ** 2) == 0)
# principal curvatures +-sqrt(-K) = +-c/(c^2+v^2), so MAC = c/(c^2+v^2)
close("Example 49.3.5: MAC at v = 0 (c = 1) is 1", float(geo49.mac([1.0, -1.0])), 1.0, tol=1e-15)
close("Example 49.3.5: MAC at v = 2 (c = 1) is 0.2", float(geo49.mac([0.2, -0.2])), 0.2, tol=1e-15)
close("Exercise 49.3.4: saddle (2, -2): H = 0, MAC = 2", float(geo49.mac([2.0, -2.0])), 2.0, tol=1e-15)
close("Exercise 49.3.4: cylinder of radius 1: MAC = 0.5", float(geo49.mac([1.0, 0.0])), 0.5, tol=1e-15)
close("Exercise 49.3.4: sphere of radius 2000: MAC = 0.0005 < xi_min", float(geo49.mac([1 / 2000, 1 / 2000])), 5e-4, tol=1e-15)
close("Exercise 49.3.4: sphere of radius 500 m: MAC = 0.002 (not upsampled)", float(geo49.mac([1 / 500, 1 / 500])), 2e-3, tol=1e-15)
close("Exercise 49.3.4: the same sphere in cm (radius 50000): MAC = 0.00002", float(geo49.mac([1 / 5e4, 1 / 5e4])), 2e-5, tol=1e-15)
# Exercise 49.3.1
S1 = geo49.warmup_scales(-2.0, 0.5, 0.1)
check("Exercise 49.3.1: scales (0.1, 0.025, 0.001)", np.allclose(S1, [0.1, 0.025, 0.001]))
close("Exercise 49.3.1: s1 s2 = 0.05^2", S1[0] * S1[1], 0.0025, tol=1e-15)
close("Exercise 49.3.1: square-root ratio s1 = 0.0707", 0.05 * math.sqrt(2), 0.0707, tol=1e-4)
close("Exercise 49.3.1: square-root ratio s2 = 0.0354", 0.05 / math.sqrt(2), 0.0354, tol=1e-4)
close("Exercise 49.3.1: printed clamp on signed tau: ratio |max(-2, 0.001)/0.5| = 0.002",
      abs(min(max(-2.0, 1e-3), 1e9) / min(max(0.5, 1e-3), 1e9)), 0.002, tol=1e-15)

# ---------------------------------------------------------------- (49.3.5): midpoint of a chord (Exercise 49.3.2)
rho, d = sp.symbols("rho d", positive=True)
sag = rho - sp.sqrt(rho ** 2 - d ** 2 / 4)
check("Exercise 49.3.2: midpoint distance = d^2/(8 rho) + O(d^4)", sp.simplify(sp.series(sag, d, 0, 4).removeO() - d ** 2 / (8 * rho)) == 0)
close("Exercise 49.3.2: rho = 1, d = 0.1 gives 0.00125", 1 - math.sqrt(1 - 0.0025), 0.00125, tol=1e-6)

# ---------------------------------------------------------------- (49.3.6) truncation and Adam (Warning box)
def adam_normal_motion(g, n, trunc, steps=100, lr=1e-3):
    m = np.zeros(3)
    vv = np.zeros(3)
    xx = np.zeros(3)
    for t in range(1, steps + 1):
        gg = geo49.truncate_normal(g, n) if trunc else g
        m = 0.9 * m + 0.1 * gg
        vv = 0.999 * vv + 0.001 * gg ** 2
        xx -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(vv / (1 - 0.999 ** t)) + 1e-15)
    return abs(xx @ n)


nz = np.array([0.0, 0, 1])
g = np.array([0.02, 0, 0.01])
close("Warning (Adam): 100 steps, lr 1e-3, normal motion without truncation", adam_normal_motion(g, nz, False), 0.1, tol=1e-9)
close("Warning (Adam): the same with the gradient truncated by (7)", adam_normal_motion(g, nz, True), 0.1, tol=1e-9)
close("Warning (SGD): 100 steps, omega = 0.01, without truncation", 100 * 0.01 * 0.01, 0.01, tol=1e-15)
gt = geo49.truncate_normal(g, nz)
close("Warning (SGD): with truncation the normal part is xi_min = 0.001 per unit step", 100 * 0.01 * abs(gt @ nz), 0.001, tol=1e-15)

# ---------------------------------------------------------------- Implementation 49.3 (code-49-3-1)
ns = dgcode.run_block(PAGE, "code-49-3-1")
check("Implementation 49.3: at most 30 lines", dgcode.code_lines(PAGE, "code-49-3-1") <= 30)
close("Implementation 49.3: scales = (0.1871, 0.0535, 0.001)", np.abs(ns["S"] - [0.18708, 0.05345, 0.001]).max(), 0.0, tol=1e-5)
check("Implementation 49.3: R is a rotation (det +1, orthogonal)",
      abs(np.linalg.det(ns["R"]) - 1) < 1e-12 and np.allclose(ns["R"].T @ ns["R"], np.eye(3)))
ev = np.sort(np.linalg.eigvalsh(ns["Sigma"]))
check("Implementation 49.3: Sigma has eigenvalues s_d^2", np.allclose(ev, np.sort(ns["S"] ** 2)))
check("Implementation 49.3: Sigma's shortest axis is the normal", abs(abs(np.linalg.eigh(ns["Sigma"])[1][:, 0] @ ns["n"]) - 1) < 1e-12)
close("Implementation 49.3: the step moves 1e-5 along the normal", ns["q"][0] - ns["mu_new"][0], 1e-5, tol=1e-15)
check("Implementation 49.3: tangent part of the step is lr * (0.2, -0.1)", np.allclose(ns["mu_new"][1:], [-0.002, 0.001]))

# ---------------------------------------------------------------- (49.3.8) L_rot and the sign of principal directions
w = np.array([0.0, 1, 0])
close("Exercise 49.3.3: (1 - <r, w>)^2 = 4 when r = -w", (1 - (-w) @ w) ** 2, 4.0, tol=1e-15)
close("Exercise 49.3.3: (1 - <r, w>^2)^2 = 0 when r = -w", (1 - ((-w) @ w) ** 2) ** 2, 0.0, tol=1e-15)

# ---------------------------------------------------------------- (49.3.7) the hinge on flat and nearly cylindrical regions
def ratio_after_clamp(t1, t2, xi_min=1e-3):
    a1, a2 = max(abs(t1), xi_min), max(abs(t2), xi_min)
    return a1 / a2


close("(49.3.7): flat region, both clamped: allowed ratio 1 + xi_min", ratio_after_clamp(0.0, 0.0) + 1e-3, 1.001, tol=1e-15)
close("(49.3.7): cylinder of radius 1: allowed ratio 1000 + xi_min", ratio_after_clamp(1.0, 0.0) + 1e-3, 1000.001, tol=1e-9)

# ---------------------------------------------------------------- Example 49.3.6: eq. (10) versus the 3DGS split
qq = np.array([2.8, 0.0, 0.0])
u3, w1, w2 = np.array([1.0, 0, 0]), np.array([0.0, 0, 1]), np.array([0.0, 1, 0])
A = geo49.split_samples_eq10(qq, w1, w2, u3, -1.25, -1 / 2.8, 20000, seed=3)
Rm = np.stack([w2, w1, u3], axis=1)
B = qq + (np.random.default_rng(4).normal(size=(20000, 3)) * S) @ Rm.T
close("Example 49.3.6: eq. (10) median distance to the torus 0.83", np.median(geo49.torus_distance(A)), 0.83, tol=0.01)
close("Example 49.3.6: 3DGS split median distance 0.0050", np.median(geo49.torus_distance(B)), 0.0050, tol=3e-4)
close("Example 49.3.6: eq. (10) median displacement 2.06", np.median(np.linalg.norm(A - qq, axis=1)), 2.06, tol=0.01)
close("Example 49.3.6: 3DGS split median displacement 0.14", np.median(np.linalg.norm(B - qq, axis=1)), 0.14, tol=0.005)
close("Example 49.3.6: std along w1 = 1/|tau1| = 0.8", np.std((A - qq) @ w1), 0.8, tol=0.02)
close("Example 49.3.6: std along w2 = 1/|tau2| = 2.8", np.std((A - qq) @ w2), 2.8, tol=0.06)
close("Example 49.3.6: flat region clamped at xi_min: std 1/xi_min = 1000", 1 / 1e-3, 1000.0, tol=1e-9)

summary()
