"""Verification for Section 27.1 (The Camera as a Smooth Map).

Checks:
  - (27.1.1)-(27.1.2): J = D(phi) has rank 2 on t_z > 0, J t = 0, fibers are open rays (Proposition 27.1.1)
  - (27.1.3): D(phi o g)(x) = J R, kernel = x - c, c = -R^T T
  - (27.1.4): J = K_2 * Jbar (intrinsics are an affine change of chart)
  - Proposition 27.1.2 and (27.1.5): rank of d(phi|_S) drops exactly where <N, t> = 0;
    sigma_1 sigma_2 = f_x f_y |<N, t_hat>| / (t_z^2 cos alpha); sphere example numbers
  - (27.1.6)-(27.1.7): Brown-Conrady distortion, D(0) = 0, DD(0) = I, det DD = L (r L)' (radial factor L = 1 + k1 r^2 + k2 r^4), fold radii
  - Example 27.1.4 numbers (k1 = -0.3), folding of outside points
  - (27.1.8) and Example 27.1.5 table: gnomonic / stereographic / equidistant / equisolid / orthographic charts,
    radial and tangential stretch factors, conformal and equal-area cases
  - equidistant chart = log map of S^2 at e_z ((0.8.5)), its Jacobian = gsplat fisheye_proj J, kernel = ray
  - projective (x/z) models send t and -t to the same pixel
  - Exercises 27.1.1-27.1.5

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/verify/v-27-1-camera-smooth-map.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(271)

# ---------------------------------------------------------------- (27.1.1)-(27.1.2) J, rank, kernel, fibers
tx, ty = sp.symbols("t_x t_y", real=True)
tz = sp.symbols("t_z", positive=True)
fx, fy = sp.symbols("f_x f_y", positive=True)
cx, cy = sp.symbols("c_x c_y", real=True)
t = sp.Matrix([tx, ty, tz])
phi = sp.Matrix([fx * tx / tz + cx, fy * ty / tz + cy])
J = phi.jacobian(t)
J_expected = sp.Matrix([[fx / tz, 0, -fx * tx / tz**2], [0, fy / tz, -fy * ty / tz**2]])
sym_equal("(27.1.2): J = D phi (same as (29.3.3))", J, J_expected)
sym_equal("Prop 27.1.1: J t = 0", J * t, sp.zeros(2, 1))
minor = sp.simplify(J[:, :2].det())
sym_equal("Prop 27.1.1: 2x2 minor = f_x f_y / t_z^2 (nonzero, rank 2)", minor, fx * fy / tz**2)
s = sp.symbols("s", positive=True)
sym_equal("Prop 27.1.1: phi(s t) = phi(t) for s > 0 (fiber contains the open ray)",
          phi.subs({tx: s * tx, ty: s * ty, tz: s * tz}, simultaneous=True), phi)
# fiber is exactly the open ray: phi(t) = u  <=>  t = t_z (ubar, 1)
ux, uy = sp.symbols("u_x u_y", real=True)
sol = sp.solve([phi[0] - ux, phi[1] - uy], [tx, ty], dict=True)[0]
sym_equal("Prop 27.1.1: fiber phi^{-1}(u) = {t_z (ubar_x, ubar_y, 1) : t_z > 0}",
          sp.Matrix([sol[tx], sol[ty]]), sp.Matrix([tz * (ux - cx) / fx, tz * (uy - cy) / fy]))

# ---------------------------------------------------------------- (27.1.3) world -> camera
def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def Jnum(tp, f=(1.0, 1.0)):
    return np.array([[f[0] / tp[2], 0, -f[0] * tp[0] / tp[2] ** 2],
                     [0, f[1] / tp[2], -f[1] * tp[1] / tp[2] ** 2]])


R = rot([0.3, -0.5, 0.8], 0.7)
T = np.array([0.2, -0.4, 3.0])
c = -R.T @ T
x0 = np.array([0.4, 0.3, 1.1])
t0 = R @ x0 + T
check("(27.1.3): point in front of camera", t0[2] > 0)
Jw = Jnum(t0) @ R
check("(27.1.3): kernel of J R is x - c", np.linalg.norm(Jw @ (x0 - c)) < 1e-12)
check("(27.1.3): rank J R = 2", np.linalg.matrix_rank(Jw) == 2)
# finite-difference check of the chain rule
h = 1e-6
fd = np.column_stack([((lambda xx: (R @ xx + T)[:2] / (R @ xx + T)[2])(x0 + h * e)
                       - (lambda xx: (R @ xx + T)[:2] / (R @ xx + T)[2])(x0 - h * e)) / (2 * h) for e in np.eye(3)])
close("(27.1.3): D(phi o g) = J R (finite differences)", np.abs(fd - Jw).max(), 0.0, tol=1e-8)

# ---------------------------------------------------------------- (27.1.4) intrinsics
sk = sp.symbols("s_k", real=True)
Kmat = sp.Matrix([[fx, sk, cx], [0, fy, cy], [0, 0, 1]])
ubar = sp.Matrix([tx / tz, ty / tz])
Jbar = ubar.jacobian(t)
pix = (Kmat * sp.Matrix([ubar[0], ubar[1], 1]))[:2, :]
sym_equal("(27.1.4): D(A_K o phibar) = K_2 Jbar", pix.jacobian(t), Kmat[:2, :2] * Jbar)
sym_equal("(27.1.4): with zero skew A_K o phibar = phi of (29.3.2)", pix.subs(sk, 0), phi)
sym_equal("(27.1.4): kernel unchanged by intrinsics", (Kmat[:2, :2] * Jbar) * t, sp.zeros(2, 1))

# ---------------------------------------------------------------- Proposition 27.1.2: silhouette
def tangent_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


Z0, RHO = 3.0, 1.0
ctr = np.array([0.0, 0.0, Z0])
worst = 0.0
for _ in range(400):
    n = rng.normal(size=3)
    n /= np.linalg.norm(n)
    p = ctr + RHO * n
    if p[2] <= 0.1:
        continue
    e1, e2 = tangent_basis(n)
    M = Jnum(p) @ np.column_stack([e1, e2])
    sv = np.linalg.svd(M, compute_uv=False)
    that = p / np.linalg.norm(p)
    cosal = p[2] / np.linalg.norm(p)
    pred = abs(n @ that) / (p[2] ** 2 * cosal)
    worst = max(worst, abs(sv[0] * sv[1] - pred))
close("(27.1.5): sigma_1 sigma_2 = |<N, t_hat>| / (t_z^2 cos alpha) (f = 1, 400 random sphere points)", worst, 0.0, tol=1e-12)
# contour points: <N, t> = 0 -> rank 1
alpha_c = np.arcsin(RHO / Z0)
close("Example 27.1.3: silhouette half-angle asin(1/3) = 19.47 deg", np.degrees(alpha_c), 19.4712, tol=1e-3)
close("Example 27.1.3: silhouette radius tan(alpha) = 1/sqrt(8) = 0.354", np.tan(alpha_c), 1 / np.sqrt(8), tol=1e-12)
# a contour point in the xz-plane
pc = np.array([np.sqrt(Z0**2 - RHO**2) * np.sin(alpha_c), 0.0, np.sqrt(Z0**2 - RHO**2) * np.cos(alpha_c)])
nc = (pc - ctr) / RHO
close("Example 27.1.3: contour point lies on the sphere", np.linalg.norm(pc - ctr), RHO, tol=1e-12)
close("Example 27.1.3: <N, t> = 0 at the contour point", nc @ pc, 0.0, tol=1e-12)
e1, e2 = tangent_basis(nc)
svc = np.linalg.svd(Jnum(pc) @ np.column_stack([e1, e2]), compute_uv=False)
check("Prop 27.1.2: rank d(phi|_S) = 1 at the contour", svc[1] < 1e-12 and svc[0] > 0.1)
close("Example 27.1.3: contour distance sqrt(8) = 2.828", np.linalg.norm(pc), np.sqrt(8), tol=1e-12)
close("Example 27.1.3: contour depth t_z = 8/3 = 2.667", pc[2], 8 / 3, tol=1e-12)
# in-plane singular values along the meridian y = 0 (formula used in Figure 27.1.2)
for psi in np.radians([0, 25, 50, 70.53, 100, 150]):
    n = np.array([np.sin(psi), 0.0, -np.cos(psi)])     # psi = 0: point nearest the camera
    p = ctr + RHO * n
    e_y = np.array([0, 1.0, 0])
    e_th = np.array([np.cos(psi), 0.0, np.sin(psi)])
    M = Jnum(p) @ np.column_stack([e_th, e_y])
    sv = np.sort(np.linalg.svd(M, compute_uv=False))
    pr = sorted([1 / p[2], np.linalg.norm(p) * abs(n @ p / np.linalg.norm(p)) / p[2] ** 2])
    close(f"Figure 27.1.2: meridian singular values at psi = {np.degrees(psi):.2f} deg", np.abs(sv - pr).max(), 0.0, tol=1e-12)
psi_c = np.arccos(RHO / Z0)
close("Figure 27.1.2: contour at psi = acos(1/3) = 70.53 deg = 90 - 19.47", np.degrees(psi_c), 70.5288, tol=1e-3)
nn = np.array([np.sin(psi_c), 0.0, -np.cos(psi_c)])
close("Figure 27.1.2: <N, t> = 0 at psi_c", nn @ (ctr + RHO * nn), 0.0, tol=1e-12)

# ---------------------------------------------------------------- (27.1.6)-(27.1.7): distortion
X, Y = sp.symbols("x y", real=True)
k1, k2, p1, p2 = sp.symbols("k_1 k_2 p_1 p_2", real=True)
r2 = X**2 + Y**2
g = 1 + k1 * r2 + k2 * r2**2
Dmap = sp.Matrix([X * g + 2 * p1 * X * Y + p2 * (r2 + 2 * X**2),
                  Y * g + p1 * (r2 + 2 * Y**2) + 2 * p2 * X * Y])
# COLMAP OPENCV: x + du with du = x*(k1 r2 + k2 r4) + 2 p1 x y + p2 (r2 + 2 x^2), dv = y*radial + 2 p2 x y + p1 (r2 + 2 y^2)
du = X * (k1 * r2 + k2 * r2**2) + 2 * p1 * X * Y + p2 * (r2 + 2 * X**2)
dv = Y * (k1 * r2 + k2 * r2**2) + 2 * p2 * X * Y + p1 * (r2 + 2 * Y**2)
sym_equal("(27.1.6): model = COLMAP OPENCV x + du (opencv.h)", Dmap, sp.Matrix([X + du, Y + dv]))
JD = Dmap.jacobian(sp.Matrix([X, Y]))
sym_equal("(27.1.6): D(0) = 0", Dmap.subs({X: 0, Y: 0}), sp.zeros(2, 1))
sym_equal("(27.1.6): DD(0) = I (tangential terms are quadratic)", JD.subs({X: 0, Y: 0}), sp.eye(2))
# radial part: det = g (r g)'
rr = sp.symbols("r", positive=True)
gr = 1 + k1 * rr**2 + k2 * rr**4
JDr = JD.subs({p1: 0, p2: 0})
detr = sp.simplify(JDr.det().subs({X: rr, Y: 0}))
sym_equal("(27.1.7): det of radial distortion = L (r L)'", detr, gr * sp.diff(rr * gr, rr))
# rotational invariance of the determinant: check at an off-axis point
th0 = sp.Rational(3, 7)
sym_equal("(27.1.7): det at (r cos a, r sin a) equals det at (r, 0)",
          JDr.det().subs({X: rr * sp.cos(th0), Y: rr * sp.sin(th0)}), gr * sp.diff(rr * gr, rr))


def fold_radius(K1, K2):
    """smallest positive root of (r g)' = 1 + 3 k1 r^2 + 5 k2 r^4"""
    roots = np.roots([5 * K2, 3 * K1, 1.0]) if K2 != 0 else np.array([-1 / (3 * K1)])
    s2 = [z.real for z in np.atleast_1d(roots) if abs(z.imag) < 1e-12 and z.real > 0]
    return np.sqrt(min(s2)) if s2 else np.inf


rs = fold_radius(-0.3, 0.0)
close("Example 27.1.4: k1=-0.3: r* = 1/sqrt(0.9) = 1.054", rs, 1 / np.sqrt(0.9), tol=1e-12)
close("Example 27.1.4: theta* = atan(1.054) = 46.51 deg", np.degrees(np.arctan(rs)), 46.51, tol=0.01)
rdmax = rs * (1 - 0.3 * rs**2)
close("Example 27.1.4: max distorted radius 0.703", rdmax, 0.7027, tol=1e-4)
close("Example 27.1.4: r = 1.5 (56.3 deg) lands at r_d = 0.4875", 1.5 * (1 - 0.3 * 1.5**2), 0.4875, tol=1e-12)
close("Example 27.1.4: theta = 60 deg lands at r_d = 0.173", np.sqrt(3) * (1 - 0.3 * 3), 0.1732, tol=1e-4)
close("Example 27.1.4: g = 0 at r = 1/sqrt(0.3) = 1.826 (61.3 deg)", np.degrees(np.arctan(1 / np.sqrt(0.3))), 61.29, tol=0.01)
# the two preimages of r_d = 0.4875
pre = sorted(z.real for z in np.roots([-0.3, 0, 1, -0.4875]) if abs(z.imag) < 1e-12 and z.real > 0)
close("Example 27.1.4: r_d = 0.4875 has preimages r = 0.533 and 1.5", np.array(pre), np.array([0.5329, 1.5]), tol=1e-4)
rs2 = fold_radius(-0.3, 0.03)
close("Figure 27.1.3: k1=-0.3, k2=0.03: r* = 1.214 (50.5 deg)", rs2, 1.2137, tol=1e-3)
check("Figure 27.1.3: k1=-0.3, k2=0.1: monotone for all r (discriminant < 0)", fold_radius(-0.3, 0.1) == np.inf
      and 0.9**2 - 4 * 0.5 < 0)
# 90 deg HFOV, 4:3: corner radius 1.25 > r*
close("Example 27.1.4: corner of a 90 deg x 4:3 image at normalized radius 1.25", np.hypot(1.0, 0.75), 1.25, tol=1e-12)


# Newton undistortion converges to the inner preimage
def undistort_newton(xd, K1, K2, iters=100):
    xx = np.array(xd, float)
    for _ in range(iters):
        r2_ = xx @ xx
        gg = 1 + K1 * r2_ + K2 * r2_**2
        F = xx * gg - xd
        dg = 2 * K1 + 4 * K2 * r2_
        Jn = gg * np.eye(2) + dg * np.outer(xx, xx)
        xx = xx - np.linalg.solve(Jn, F)
    return xx


xu = undistort_newton(np.array([0.4875, 0.0]), -0.3, 0.0)
close("Example 27.1.4: Newton from the distorted point returns the inner preimage 0.533", xu[0], 0.5329, tol=1e-4)

# ---------------------------------------------------------------- Example 27.1.5 table: charts of S^2
th = sp.symbols("theta", positive=True)
charts = {
    "perspective (gnomonic)": sp.tan(th),
    "stereographic": 2 * sp.tan(th / 2),
    "equidistant": th,
    "equisolid": 2 * sp.sin(th / 2),
    "orthographic": sp.sin(th),
}
stretch = {name: (sp.simplify(sp.diff(rf, th)), sp.simplify(rf / sp.sin(th))) for name, rf in charts.items()}
sym_equal("Example 27.1.5 table: gnomonic radial stretch 1/cos^2", stretch["perspective (gnomonic)"][0], 1 / sp.cos(th) ** 2)
sym_equal("Example 27.1.5 table: gnomonic tangential stretch 1/cos", stretch["perspective (gnomonic)"][1], 1 / sp.cos(th))
sym_equal("Example 27.1.5 table: stereographic is conformal (radial = tangential)",
          stretch["stereographic"][0], stretch["stereographic"][1])
sym_equal("Example 27.1.5 table: equisolid is equal-area (radial * tangential = 1)",
          stretch["equisolid"][0] * stretch["equisolid"][1], 1)
sym_equal("Example 27.1.5 table: equidistant radial stretch = 1", stretch["equidistant"][0], 1)
sym_equal("Example 27.1.5 table: equidistant tangential stretch theta/sin theta", stretch["equidistant"][1], th / sp.sin(th))
sym_equal("Example 27.1.5 table: orthographic radial stretch cos theta (0 at 90 deg)", stretch["orthographic"][0], sp.cos(th))
# generic stretch formula for a radial chart: check numerically by finite differences on the sphere
for name, rf in charts.items():
    f_r = sp.lambdify(th, rf, "numpy")
    th_test = 0.7
    om = np.array([np.sin(th_test), 0.0, np.cos(th_test)])

    def chart(w):
        thw = np.arccos(np.clip(w[2], -1, 1))
        rho_ = np.hypot(w[0], w[1])
        return f_r(thw) * w[:2] / rho_

    eth = np.array([np.cos(th_test), 0, -np.sin(th_test)])
    eph = np.array([0, 1.0, 0])
    hh = 1e-6
    rad = np.linalg.norm(chart(np.cos(hh) * om + np.sin(hh) * eth) - chart(np.cos(hh) * om - np.sin(hh) * eth)) / (2 * hh)
    tan_ = np.linalg.norm(chart(np.cos(hh) * om + np.sin(hh) * eph) - chart(np.cos(hh) * om - np.sin(hh) * eph)) / (2 * hh)
    ex = [float(stretch[name][0].subs(th, th_test)), float(stretch[name][1].subs(th, th_test))]
    close(f"Example 27.1.5 table: {name} stretches by finite differences on S^2", np.abs(np.array([rad, tan_]) - ex).max(), 0.0, tol=1e-6)

# stereographic lens = 2 x (south-pole stereographic projection sigma~(w) = (w_x, w_y)/(1 + w_z))
thv = np.linspace(0.05, 3.0, 40)
close("Example 27.1.5: |2 sigma~(w)| = 2 sin th/(1 + cos th) = 2 tan(th/2)",
      2 * np.sin(thv) / (1 + np.cos(thv)), 2 * np.tan(thv / 2), tol=1e-12)
# numbers quoted in the text
close("Sec charts: tan 80 deg = 5.67", np.tan(np.radians(80)), 5.671, tol=1e-3)
close("Sec charts: 80 deg = 1.396 rad", np.radians(80), 1.396, tol=1e-3)
close("Sec charts: 95 deg = 1.658 rad", np.radians(95), 1.658, tol=1e-3)
close("Sec charts: gnomonic stretch at 60 deg: 4 (radial), 2 (tangential)",
      np.array([1 / np.cos(np.radians(60)) ** 2, 1 / np.cos(np.radians(60))]), np.array([4.0, 2.0]), tol=1e-12)
close("Sec charts: equidistant tangential stretch at 90 deg = pi/2 = 1.571", np.pi / 2, 1.5708, tol=1e-4)

# equidistant chart = log map at e_z, (0.8.5): log_p(q) = theta/sin theta (q - cos theta p)
worst_log = 0.0
for _ in range(20):
    w = rng.normal(size=3)
    w /= np.linalg.norm(w)
    if w[2] < -0.98:
        continue
    thw = np.arccos(w[2])
    logv = thw / np.sin(thw) * (w - np.cos(thw) * np.array([0, 0, 1.0]))
    eq = thw * w[:2] / np.hypot(w[0], w[1])
    worst_log = max(worst_log, np.abs(logv[:2] - eq).max(), abs(logv[2]))
check("(27.1.8): equidistant chart = log_{e_z} of (0.8.5) (20 random directions, incl. back hemisphere)", worst_log < 1e-12)

# gsplat fisheye_proj Jacobian = D(equidistant chart o normalization); kernel = ray
X3, Y3, Z3 = sp.symbols("X Y Z", real=True)
rho_xy = sp.sqrt(X3**2 + Y3**2)
theta_e = sp.atan2(rho_xy, Z3)
ueq = sp.Matrix([fx * theta_e * X3 / rho_xy + cx, fy * theta_e * Y3 / rho_xy + cy])
Jeq = ueq.jacobian(sp.Matrix([X3, Y3, Z3]))
x2y2 = X3**2 + Y3**2
R2 = x2y2 + Z3**2
a_ = Z3 / R2 / x2y2
b_ = theta_e / rho_xy / x2y2
Jgs = sp.Matrix([[fx * (X3**2 * a_ + Y3**2 * b_), fx * X3 * Y3 * (a_ - b_), -fx * X3 / R2],
                 [fy * X3 * Y3 * (a_ - b_), fy * (Y3**2 * a_ + X3**2 * b_), -fy * Y3 / R2]])
fJ = sp.lambdify((X3, Y3, Z3, fx, fy), Jeq - Jgs, "numpy")
fK = sp.lambdify((X3, Y3, Z3, fx, fy), Jeq * sp.Matrix([X3, Y3, Z3]), "numpy")
errJ = errK = 0.0
for _ in range(30):
    P3 = rng.normal(size=3)
    errJ = max(errJ, np.abs(np.array(fJ(*P3, 300.0, 310.0), float)).max())
    errK = max(errK, np.abs(np.array(fK(*P3, 300.0, 310.0), float)).max())
close("Sec 3DGS: gsplat fisheye_proj J equals D(equidistant chart) (30 points incl. z < 0)", errJ, 0.0, tol=1e-9)
close("Sec 3DGS: kernel of the equidistant J is the ray", errK, 0.0, tol=1e-9)

# projective (x/z) models: t and -t give the same normalized point
tp = np.array([0.3, -0.2, 2.0])
check("Sec RP2 vs S2: t and -t have the same x/z, y/z", np.allclose(tp[:2] / tp[2], (-tp)[:2] / (-tp)[2]))
# OpenCV fisheye formula on t and -t (a = x/z, theta = atan(r))
def ocv_fisheye(P):
    a, b = P[0] / P[2], P[1] / P[2]
    r = np.hypot(a, b)
    thd = np.arctan(r)
    return thd / r * np.array([a, b])


check("Sec RP2 vs S2: OpenCV fisheye formula maps t and -t to the same point", np.allclose(ocv_fisheye(tp), ocv_fisheye(-tp)))
thw = np.arccos(-tp[2] / np.linalg.norm(tp))
check("Sec RP2 vs S2: the true equidistant image of -t is at theta = 169.8 deg, far from that of t",
      abs(np.degrees(thw) - 169.78) < 0.01)

# ---------------------------------------------------------------- review additions (independent checks)
# Prop 27.1.2: the integral of sigma_1 sigma_2 over the visible cap equals the area of the silhouette disk (pi/8)
npsi, nph = 1200, 600
psi_g = (np.arange(npsi) + 0.5) / npsi * np.arccos(RHO / Z0)
ph_g = (np.arange(nph) + 0.5) / nph * 2 * np.pi
PS, PH = np.meshgrid(psi_g, ph_g, indexing="ij")
Ng = np.stack([np.sin(PS) * np.cos(PH), np.sin(PS) * np.sin(PH), -np.cos(PS)], -1)
tg = ctr + RHO * Ng
ntg = np.linalg.norm(tg, axis=-1)
s12 = np.abs((Ng * tg).sum(-1) / ntg) / (tg[..., 2] ** 2 * tg[..., 2] / ntg)
dAg = RHO**2 * np.sin(PS) * (np.arccos(RHO / Z0) / npsi) * (2 * np.pi / nph)
close("Prop 27.1.2 (review): integral of sigma1 sigma2 over the visible cap = silhouette area pi/8",
      (s12 * dAg).sum(), np.pi / 8, tol=1e-5)
# (27.1.5) is the reciprocal of the pixel footprint (30.4.3): planes through random points, f = 500 px
worst_fp = 0.0
for _ in range(100):
    p0 = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(2, 6)])
    nrm = rng.normal(size=3)
    nrm /= np.linalg.norm(nrm)
    if abs(nrm @ p0) / np.linalg.norm(p0) < 0.1:
        continue
    F = 500.0
    u0, v0 = F * p0[0] / p0[2], F * p0[1] / p0[2]

    def Xbp(u, v):
        rr_ = np.array([u / F, v / F, 1.0])
        return (nrm @ p0) / (nrm @ rr_) * rr_

    hh = 1e-3
    Xu = (Xbp(u0 + hh, v0) - Xbp(u0 - hh, v0)) / (2 * hh)
    Xv = (Xbp(u0, v0 + hh) - Xbp(u0, v0 - hh)) / (2 * hh)
    foot = np.linalg.norm(np.cross(Xu, Xv))
    cosal = p0[2] / np.linalg.norm(p0)
    foot_3043 = p0[2] ** 2 * cosal / (F * F * abs(nrm @ p0) / np.linalg.norm(p0))
    sig12 = F * F * abs(nrm @ p0) / np.linalg.norm(p0) / (p0[2] ** 2 * cosal)
    worst_fp = max(worst_fp, abs(foot / foot_3043 - 1), abs(foot * sig12 - 1))
close("Prop 27.1.2 (review): sigma1 sigma2 = 1 / sqrt(EG - F^2) of (30.4.3) (100 random planes)", worst_fp, 0.0, tol=1e-7)
# Figure 27.1.2: angle between ray and meridian tangent, and sigma_in at the two marked points
for deg, ang_exp, sin_exp in ((25, 53.6, 0.392), (50, 22.0, 0.167)):
    n_ = np.array([np.sin(np.radians(deg)), 0.0, -np.cos(np.radians(deg))])
    p_ = ctr + RHO * n_
    ang = np.degrees(np.arcsin(abs(n_ @ p_) / np.linalg.norm(p_)))
    s_in = np.linalg.norm(p_) * abs(n_ @ p_ / np.linalg.norm(p_)) / p_[2] ** 2
    close(f"Figure 27.1.2: ray-tangent angle at psi = {deg} deg is {ang_exp} deg", ang, ang_exp, tol=0.05)
    close(f"Figure 27.1.2: sigma_in at psi = {deg} deg is {sin_exp}", s_in, sin_exp, tol=5e-4)
# (27.1.7) holds only without tangential terms
detfull = sp.simplify(JD.det() - g * sp.diff(rr * gr, rr).subs(rr, sp.sqrt(r2)))
check("(27.1.7) (review): with p1, p2 != 0 the determinant is not L (r L)' (the formula is radial-only)",
      sp.simplify(detfull.subs({p1: 0, p2: 0})) == 0 and abs(float(detfull.subs({X: 0.3, Y: 0.2, k1: -0.3, k2: 0.0, p1: 0.01, p2: -0.02}))) > 1e-3)
# OpenCV-style fixed-point undistortion (x = x_d / L(x)) from the distorted point also lands on the inner preimage
xf = 0.4875
for _ in range(5):                       # OpenCV's default: 5 iterations
    xf = 0.4875 / (1 - 0.3 * xf**2)
close("Sec distortion (review): 5 fixed-point iterations from r_d = 0.4875 give 0.533 (inner preimage)", xf, 0.5329, tol=1e-3)
close("Example 27.1.4: the 90 deg x 4:3 corner direction is at theta = atan(1.25) = 51.3 deg", np.degrees(np.arctan(1.25)), 51.34, tol=0.01)
close("Sec charts: equidistant circle radii 45, 60, 90 deg = 0.79, 1.05, 1.57", np.radians([45, 60, 90]), np.array([0.785, 1.047, 1.571]), tol=1e-3)

# ---------------------------------------------------------------- Exercises
# Ex 27.1.1: R = rotation about y by 30 deg, T = (0,0,4), x = (1, 0, 1)
Rx = rot([0, 1, 0], np.radians(30))
Tx = np.array([0, 0, 4.0])
xe = np.array([1.0, 0.0, 1.0])
te = Rx @ xe + Tx
ce = -Rx.T @ Tx
close("Ex 27.1.1: camera coordinates t = (1.366, 0, 4.366)", te, np.array([1.3660, 0, 4.3660]), tol=1e-4)
close("Ex 27.1.1: camera center c = (2, 0, -3.464)", ce, np.array([2.0, 0, -3.4641]), tol=1e-4)
Je = Jnum(te, (500.0, 500.0)) @ Rx
check("Ex 27.1.1: D(phi o g)(x - c) = 0", np.linalg.norm(Je @ (xe - ce)) < 1e-9)
close("Ex 27.1.1: x - c = (-1, 0, 4.464) is parallel to R^T t", np.cross(xe - ce, Rx.T @ te), np.zeros(3), tol=1e-12)
# Ex 27.1.2: k1 = -0.2 -> r* = 1/sqrt(0.6) = 1.291, theta = 52.2 deg; max r_d = 0.861
rs3 = fold_radius(-0.2, 0.0)
close("Ex 27.1.2: r* = 1.291", rs3, 1.2910, tol=1e-4)
close("Ex 27.1.2: theta* = 52.24 deg", np.degrees(np.arctan(rs3)), 52.24, tol=0.01)
close("Ex 27.1.2: max r_d = 2/3 r* = 0.861", rs3 * (1 - 0.2 * rs3**2), 0.8607, tol=1e-4)
close("Ex 27.1.2: FOV of the valid disk = 104.5 deg", 2 * np.degrees(np.arctan(rs3)), 104.48, tol=0.01)
# Ex 27.1.3: equidistant, f = 300 px, 190 deg FOV -> radius at 95 deg = 497.4 px
close("Ex 27.1.3: image radius f theta at 95 deg = 497.4 px", 300 * np.radians(95), 497.42, tol=0.01)
close("Ex 27.1.3: pinhole f tan(80 deg) = 1701 px", 300 * np.tan(np.radians(80)), 1701.4, tol=0.1)
close("Ex 27.1.3: pinhole f tan(89 deg) = 17187 px", 300 * np.tan(np.radians(89)), 17187, tol=1)
# Ex 27.1.4: sphere center (0,0,5), radius 2: silhouette half-angle asin(0.4) = 23.58 deg, image radius f tan = 0.436 f
close("Ex 27.1.4: silhouette half-angle 23.58 deg", np.degrees(np.arcsin(0.4)), 23.578, tol=1e-3)
close("Ex 27.1.4: silhouette image radius tan = 2/sqrt(21) = 0.436", np.tan(np.arcsin(0.4)), 2 / np.sqrt(21), tol=1e-12)
close("Ex 27.1.4: contour depth = 21/5 = 4.2", (25 - 4) / 5, 4.2, tol=1e-12)
# Ex 27.1.5: J for pixel = K2 Jbar with skew: kernel the same; with K2 = [[500, 0],[0, 500]] scaling
tq = np.array([0.5, -0.25, 2.0])
Jb = Jnum(tq)
K2 = np.array([[500.0, 1.0], [0.0, 505.0]])
check("Ex 27.1.5: kernel of K2 Jbar = kernel of Jbar = ray", np.linalg.norm(K2 @ Jb @ tq) < 1e-12)

summary()
