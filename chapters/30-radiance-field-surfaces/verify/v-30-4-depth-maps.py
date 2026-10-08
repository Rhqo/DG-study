"""Verification for Section 30.4 (Depth Maps as Parametrized Surfaces).

Notation: pixel (u, v), intrinsics f_x, f_y, c_x, c_y, r(u, v) = K^{-1}(u, v, 1)^T = (xb, yb, 1) with
xb = (u - c_x)/f_x, yb = (v - c_y)/f_y; z-depth d(u, v); back-projection X = d r; inverse depth rho = 1/d.

Run from the project root:
``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/verify/v-30-4-depth-maps.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(304)
u, v = sp.symbols("u v", real=True)
fx, fy = sp.symbols("f_x f_y", positive=True)
cx, cy = sp.symbols("c_x c_y", real=True)
d = sp.Function("d", positive=True)(u, v)
r = sp.Matrix([(u - cx) / fx, (v - cy) / fy, 1])
X = d * r
Xu, Xv = X.diff(u), X.diff(v)
cr = Xu.cross(Xv)

# ---------------------------------------------------------------- Proposition 30.4.1: <X_u x X_v, r> = d^2/(f_x f_y)
sym_equal("Proposition 30.4.1: <X_u x X_v, r> = d^2/(f_x f_y) for every depth map d(u, v)", (cr.T * r)[0],
          d ** 2 / (fx * fy))
sym_equal("Proposition 30.4.1: r_u x r_v = e_3/(f_x f_y)", r.diff(u).cross(r.diff(v)), sp.Matrix([0, 0, 1]) / (fx * fy))

# ---------------------------------------------------------------- Proposition 30.4.2: pixel footprint
def random_depth_fn():
    a = rng.normal(size=6) * np.array([0.0, 0.002, -0.002, 2e-5, -1e-5, 2e-5])
    a[0] = rng.uniform(2.0, 5.0)
    return lambda U, V: a[0] + a[1] * U + a[2] * V + a[3] * U * U + a[4] * U * V + a[5] * V * V


Kn = dict(fx=520.0, fy=540.0, cx=320.0, cy=240.0)
okfp = True
for trial in range(50):
    dfn = random_depth_fn()
    U0, V0 = rng.uniform(0, 640), rng.uniform(0, 480)
    h = 1e-3

    def Xn(U, V):
        return dfn(U, V) * np.array([(U - Kn["cx"]) / Kn["fx"], (V - Kn["cy"]) / Kn["fy"], 1.0])

    xu = (Xn(U0 + h, V0) - Xn(U0 - h, V0)) / (2 * h)
    xv = (Xn(U0, V0 + h) - Xn(U0, V0 - h)) / (2 * h)
    E, F, G = xu @ xu, xu @ xv, xv @ xv
    nrm = np.cross(xu, xv)
    nrm /= np.linalg.norm(nrm)
    rr = np.array([(U0 - Kn["cx"]) / Kn["fx"], (V0 - Kn["cy"]) / Kn["fy"], 1.0])
    cos_alpha = 1 / np.linalg.norm(rr)
    cos_theta = abs(nrm @ rr) / np.linalg.norm(rr)
    dd = dfn(U0, V0)
    pred = dd ** 2 * cos_alpha / (Kn["fx"] * Kn["fy"] * cos_theta)
    ell = dd * np.linalg.norm(rr)
    pred2 = ell ** 2 * cos_alpha ** 3 / (Kn["fx"] * Kn["fy"] * cos_theta)
    okfp &= np.isclose(np.sqrt(E * G - F * F), pred, rtol=1e-6) and np.isclose(pred, pred2, rtol=1e-12)
check("Proposition 30.4.2: sqrt(EG - F^2) = d^2 cos(alpha)/(f_x f_y |cos theta|) = l^2 cos^3(alpha)/(f_x f_y |cos theta|) "
      "(50 random depth maps and pixels)", okfp)
# fronto-parallel plane: E = d^2/f_x^2, G = d^2/f_y^2, F = 0
dd0 = sp.symbols("d0", positive=True)
Xp = dd0 * r
sym_equal("Proposition 30.4.2: fronto-parallel plane: E = d^2/f_x^2", (Xp.diff(u).T * Xp.diff(u))[0], dd0 ** 2 / fx ** 2)
sym_equal("Proposition 30.4.2: fronto-parallel plane: F = 0", (Xp.diff(u).T * Xp.diff(v))[0], 0)

# flatland: plane through (0, d0) tilted by theta, f pixels: footprint per pixel d0/(f cos theta), depth spread d0 tan(theta)/f
for th_deg in (0.0, 45.0, 60.0, 80.0, 85.0):
    th = np.radians(th_deg)
    f, d0 = 500.0, 3.0
    dep = lambda uu: d0 * f * np.cos(th) / (f * np.cos(th) - uu * np.sin(th))
    pts = lambda uu: dep(uu) * np.array([uu / f, 1.0])
    seg = np.linalg.norm(pts(0.5) - pts(-0.5))
    close(f"Figure 30.4.2: footprint of one pixel at tilt {th_deg} = (d/f)/cos(theta) (to 1e-3)", seg,
          d0 / (f * np.cos(th)), tol=1e-3 * d0 / (f * np.cos(th)))
    close(f"Figure 30.4.2: depth spread in one pixel at tilt {th_deg} = (d/f) tan(theta)", dep(0.5) - dep(-0.5),
          d0 * np.tan(th) / f, tol=2e-3 * max(d0 * np.tan(th) / f, 1e-3) + 1e-12)
close("Figure 30.4.2: at 85 deg one pixel spans 11.4 x (d/f) along the surface and 11.4 x (d/f) in depth",
      [1 / np.cos(np.radians(85)), np.tan(np.radians(85))], [11.474, 11.430], tol=1e-3)

# ---------------------------------------------------------------- Proposition 30.4.3: planes <=> affine inverse depth
nx, ny, nz, c = sp.symbols("n_x n_y n_z c", real=True)
n = sp.Matrix([nx, ny, nz])
dplane = c / (n.T * r)[0]
rho_plane = sp.simplify(1 / dplane)
check("Proposition 30.4.3: inverse depth of the plane <n, x> = c is affine in (u, v)",
      all(sp.simplify(sp.diff(rho_plane, *vv)) == 0 for vv in [(u, 2), (u, v), (v, 2)]))
check("Proposition 30.4.3: the depth itself is not affine unless n_x = n_y = 0 (d_uu != 0 in general)",
      sp.simplify(sp.diff(dplane, u, 2)) != 0 and sp.simplify(sp.diff(dplane, u, 2).subs({nx: 0, ny: 0})) == 0)
# converse: if rho = a u + b v + e then X = r/rho lies on the plane <n, X> = 1 with n = (a f_x, b f_y, e + a c_x + b c_y)
a_, b_, e_ = sp.symbols("a b e", real=True)
rho_aff = a_ * u + b_ * v + e_
Xa = r / rho_aff
nconv = sp.Matrix([a_ * fx, b_ * fy, e_ + a_ * cx + b_ * cy])
sym_equal("Proposition 30.4.3 (converse): affine inverse depth => <n, X> = 1, n = (a f_x, b f_y, e + a c_x + b c_y)",
          sp.simplify((nconv.T * Xa)[0]), 1)
# 1D: affine depth = parabola in 3D (Exercise 30.4.3)
uu = sp.symbols("uu", real=True)
p_, q_ = sp.symbols("p q", real=True)
dd1 = p_ * uu + q_
X1 = sp.Matrix([uu * dd1, dd1])                           # flatland, f = 1, c_x = 0
check("Exercise 30.4.3: depth affine in u gives x = u(pu + q), z = pu + q: the curve x = (z^2 - q z)/p is a parabola (p != 0)",
      sp.simplify(X1[0] - (X1[1] ** 2 - q_ * X1[1]) / p_) == 0)
close("Figure 30.4.4: plane at 45 deg through (0, 3), f = 1: d = 3/(1 - u), 1/d = (1 - u)/3",
      [3 / (1 - 0.5), (1 - 0.5) / 3], [6.0, 1 / 6], tol=1e-15)

# ---------------------------------------------------------------- Proposition 30.4.4: second fundamental form = -d <N, X> Hess(1/d)
okII = True
okN = True
for trial in range(30):
    co = rng.normal(size=6) * np.array([0.0, 1e-4, 1e-4, 3e-7, 2e-7, 3e-7])
    co[0] = 1 / rng.uniform(2.0, 5.0)
    rho = lambda U, V: co[0] + co[1] * U + co[2] * V + co[3] * U * U + co[4] * U * V + co[5] * V * V
    U0, V0 = rng.uniform(0, 640), rng.uniform(0, 480)

    def Xn(U, V):
        return np.array([(U - Kn["cx"]) / Kn["fx"], (V - Kn["cy"]) / Kn["fy"], 1.0]) / rho(U, V)

    h = 1e-2
    xu = (Xn(U0 + h, V0) - Xn(U0 - h, V0)) / (2 * h)
    xv = (Xn(U0, V0 + h) - Xn(U0, V0 - h)) / (2 * h)
    xuu = (Xn(U0 + h, V0) - 2 * Xn(U0, V0) + Xn(U0 - h, V0)) / h ** 2
    xvv = (Xn(U0, V0 + h) - 2 * Xn(U0, V0) + Xn(U0, V0 - h)) / h ** 2
    xuv = (Xn(U0 + h, V0 + h) - Xn(U0 + h, V0 - h) - Xn(U0 - h, V0 + h) + Xn(U0 - h, V0 - h)) / (4 * h * h)
    N = np.cross(xu, xv)
    N /= np.linalg.norm(N)
    Lc, Mc, Nc = xuu @ N, xuv @ N, xvv @ N
    dd = 1 / rho(U0, V0)
    fac = -dd * (N @ Xn(U0, V0))
    pred = fac * np.array([2 * co[3], co[4], 2 * co[5]])
    okII &= np.allclose([Lc, Mc, Nc], pred, rtol=2e-4, atol=1e-12)
    # normal from inverse depth: N ∝ (f_x rho_u, f_y rho_v, rho - xb f_x rho_u - yb f_y rho_v)
    ru = co[1] + 2 * co[3] * U0 + co[4] * V0
    rv = co[2] + co[4] * U0 + 2 * co[5] * V0
    xb, yb = (U0 - Kn["cx"]) / Kn["fx"], (V0 - Kn["cy"]) / Kn["fy"]
    Nr = np.array([Kn["fx"] * ru, Kn["fy"] * rv, rho(U0, V0) - xb * Kn["fx"] * ru - yb * Kn["fy"] * rv])
    Nr /= np.linalg.norm(Nr)
    okN &= np.isclose(abs(Nr @ N), 1.0, atol=1e-9)
check("Proposition 30.4.4: (L, M, N) in pixel coordinates = -d <N, X> (rho_uu, rho_uv, rho_vv) (30 random surfaces)", okII)
check("(30.4.5): normal from inverse depth N ∝ (f_x rho_u, f_y rho_v, rho - xb f_x rho_u - yb f_y rho_v) (30 random surfaces)",
      okN)
# symbolic version of the key identity: d^2/du^2 (<n0, X> rho) = 0 because <n0, X> rho = <n0, r> is affine
rho_s = sp.Function("rho", positive=True)(u, v)
Xs = r / rho_s
n0 = sp.Matrix(sp.symbols("m1 m2 m3", real=True))
check("Proposition 30.4.4 (derivation): <n0, X> rho = <n0, r> is affine in (u, v)",
      sp.simplify(sp.diff((n0.T * Xs)[0] * rho_s, u, 2)) == 0 and sp.simplify(sp.diff((n0.T * Xs)[0] * rho_s, u, v)) == 0)

# ---------------------------------------------------------------- graph surfaces: H = div(grad h/W)/2 and Delta h
ex = dgsym.EXAMPLES["graph"]
(gu, gv) = ex["coords"]
hfun = ex["functions"][0](gu, gv)
Kg, Hg = dgsym.K_H(ex["expr"], gu, gv)
W = sp.sqrt(1 + sp.diff(hfun, gu) ** 2 + sp.diff(hfun, gv) ** 2)
divform = (sp.diff(sp.diff(hfun, gu) / W, gu) + sp.diff(sp.diff(hfun, gv) / W, gv)) / 2
hsamp = gu ** 2 / 3 - gu * gv / 5 + sp.sin(gv)
sym_equal("(30.4.7): graph z = h(x, y), upward N: H = (1/2) div(grad h/sqrt(1 + |grad h|^2))",
          sp.simplify((Hg - divform).subs(hfun, hsamp).doit()), 0, {gu: (-1, 1), gv: (-1, 1)})
# small slope: H ~ Delta h / 2; numbers for h = 0.5 x^2 + 0.3 y^2 + slope * x at the origin
for slope in (0.1, 1.0):
    hs = 0.5 * gu ** 2 + 0.3 * gv ** 2 + slope * gu
    Hs = float(sp.simplify(divform.subs(hfun, hs).doit()).subs({gu: 0, gv: 0}))
    expect = (1.0 + (1 + slope ** 2) * 0.6) / (2 * (1 + slope ** 2) ** 1.5)
    close(f"(30.4.7): h = x^2/2 + 0.3 y^2 + {slope} x at 0: H = {expect:.4f}, H/(Delta h/2) = {expect / 0.8:.3f}", Hs, expect,
          tol=1e-12)
# perspective: a tilted plane has Delta d != 0 but H = 0 (and Hess(1/d) = 0)
dpl = 3 * 500 * np.cos(np.radians(45)) / (500 * np.cos(np.radians(45)) - np.array([-1.0, 0.0, 1.0]) * np.sin(np.radians(45)))
close("(30.4.7) contrast: 45-degree plane, f = 500 px: d_uu = 2 d tan^2/f^2 = 2.4e-5 per px^2 at the center, (1/d)_uu = 0",
      [dpl[0] - 2 * dpl[1] + dpl[2], (1 / dpl[0] - 2 / dpl[1] + 1 / dpl[2])], [2 * 3 / 500 ** 2, 0.0], tol=1e-9)

# ---------------------------------------------------------------- Figure 30.4.5: smoothness priors on a slanted plane
n_ = 161
ug = np.linspace(-0.5, 0.5, n_)
dtrue = 3 / (1 - ug)
noise_rng = np.random.default_rng(3044)
dn = dtrue + 0.06 * noise_rng.normal(size=n_)


def irls_l1(y, w, D, lam, it=300, eps=1e-6):
    x_ = y.copy()
    Wm = np.diag(w)
    for _ in range(it):
        res = np.abs(D @ x_)
        x_ = np.linalg.solve(Wm + lam * D.T @ np.diag(1 / np.maximum(res, eps)) @ D, Wm @ y)
    return x_


D1 = np.diff(np.eye(n_), axis=0)
D2 = np.diff(np.eye(n_), n=2, axis=0)


def plane_rms(dd):
    return np.sqrt(np.mean(((ug * dd) - dd + 3) ** 2 / 2))


lam = 100.0
rms = dict(noisy=plane_rms(dn), tv=plane_rms(irls_l1(dn, np.ones(n_), D1, lam)),
           d2=plane_rms(irls_l1(dn, np.ones(n_), D2, lam)), rho2=plane_rms(1 / irls_l1(1 / dn, dn ** 4, D2, 9.0 * lam)))
print("   Figure 30.4.5 (lambda = 100): " + ", ".join(f"{k} {v_:.4f}" for k, v_ in rms.items()))
check("Figure 30.4.5: TV on depth flattens the slanted plane (rms distance > 10x noise)", rms["tv"] > 10 * rms["noisy"])
check("Figure 30.4.5: second-order L1 on inverse depth beats second-order L1 on depth by > 5x", rms["d2"] > 5 * rms["rho2"])
check("Figure 30.4.5: second-order L1 on inverse depth is well below the noise level (< 1/4)", rms["rho2"] < 0.25 * rms["noisy"])
close("Figure 30.4.5 (lambda = 100, inverse-depth prior scaled by d_ref^2 = 9): rms 0.712, 0.053, 0.008",
      [rms["tv"], rms["d2"], rms["rho2"]], [0.712, 0.053, 0.008], tol=6e-4)
# the objective solved by the IRLS step (W + lam D^T diag(1/|Dx|) D) x = W y is (1/2) sum w (x - y)^2 + lam sum |Dx|
x_tv = irls_l1(dn, np.ones(n_), D2, 3.0, it=400)
obj = lambda x: 0.5 * np.sum((x - dn) ** 2) + 3.0 * np.sum(np.abs(D2 @ x))
pert = [obj(x_tv + 1e-4 * noise_rng.normal(size=n_)) - obj(x_tv) for _ in range(20)]
check("Figure 30.4.5: the IRLS fixed point minimizes (1/2) sum (d - d_obs)^2 + lambda R (random perturbations increase it)",
      min(pert) > -1e-9)
r_small = [plane_rms(irls_l1(dn, np.ones(n_), D2, 0.3)), plane_rms(1 / irls_l1(1 / dn, dn ** 4, D2, 9.0 * 0.3))]
close("Figure 30.4.5(b): at lambda = 0.3 both second-order priors give rms 0.012", r_small, [0.0118, 0.0119], tol=3e-4)
close("Figure 30.4.5: zero cost of the truth: second differences of 1/d_true vanish", np.abs(D2 @ (1 / dtrue)).max(), 0.0,
      tol=1e-14)
# the best depth-affine fit (lambda -> infinity for the d-prior) is a parabola 0.197 rms away from the plane
A_ = np.stack([ug, np.ones_like(ug)], 1)
coef = np.linalg.lstsq(A_, dtrue, rcond=None)[0]
close("Figure 30.4.5: the best depth-affine profile is 0.197 rms away from the plane", plane_rms(A_ @ coef), 0.1971,
      tol=1e-3)

# ---------------------------------------------------------------- Figure 30.4.3: finite-difference normal error vs tilt (3D)
f_px, d0, sig_rel = 500.0, 3.0, 2e-4
NBU = np.array([0.0, 0.0, -1.0, 1.0])
NBV = np.array([-1.0, 1.0, 0.0, 0.0])


def fd_normal_err3(th, model, trials=40000, seed=0, k=1):
    """central-difference normal of a plane tilted about the image x-axis (depth_to_normal stencil, spacing k px);
    returns rms (total, along the tilt, across the tilt) angle errors in rad"""
    rg = np.random.default_rng(seed)
    dep = lambda uu, vv: d0 * np.cos(th) / (np.cos(th) - vv / f_px * np.sin(th))
    nu, nv = k * NBU, k * NBV
    if model == "const":
        dd = dep(nu, nv)[None] + sig_rel * d0 * rg.normal(size=(trials, 4))
    else:
        dd = dep(nu[None] + rg.uniform(-0.5, 0.5, size=(trials, 4)), nv[None] + rg.uniform(-0.5, 0.5, size=(trials, 4)))
    r_ = np.stack([nu / f_px, nv / f_px, np.ones(4)], -1)
    P = dd[..., None] * r_[None]
    N = np.cross(P[:, 1] - P[:, 0], P[:, 3] - P[:, 2])
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    n = np.array([0.0, np.sin(th), -np.cos(th)])
    t_in = np.array([0.0, np.cos(th), np.sin(th)])
    tot = np.arccos(np.clip(np.abs(N @ n), -1, 1))
    along = np.arctan2(N @ t_in, N @ n)
    across = np.arcsin(np.clip(N[:, 0], -1, 1))
    return [np.sqrt(np.mean(a_ ** 2)) for a_ in (tot, along, across)]


kk = sig_rel * f_px / np.sqrt(2)                        # k = sigma f/(sqrt2 d), sigma = sig_rel d
for th_deg in (0.0, 30.0, 60.0, 80.0):
    th = np.radians(th_deg)
    c_ = np.cos(th)
    tot, al, ac = fd_normal_err3(th, "const")
    close(f"Figure 30.4.3(a): constant noise, tilt {th_deg}: total = k sqrt(cos^4 + cos^2), along = k cos^2, across = k cos",
          [tot, al, ac], [kk * np.sqrt(c_ ** 4 + c_ ** 2), kk * c_ ** 2, kk * c_], tol=0.03 * kk)
    s_ = np.sin(th)
    tot, al, ac = fd_normal_err3(th, "jitter")
    close(f"Figure 30.4.3(b): sub-pixel jitter, tilt {th_deg}: total = sin sqrt(1 + cos^2)/sqrt24, along = sin cos/sqrt24, "
          "across = sin/sqrt24 (first order, 10 % of the maximum)",
          [tot, al, ac], [s_ * np.sqrt(1 + c_ ** 2) / np.sqrt(24), s_ * c_ / np.sqrt(24), s_ / np.sqrt(24)],
          tol=0.1 / np.sqrt(24))
close("Figure 30.4.3(a): constant noise 0.02 % of depth at f = 500 px, facing the camera: sigma f/d = 5.73 deg total",
      np.degrees(fd_normal_err3(0.0, "const")[0]), 5.730, tol=0.08)
close("Figure 30.4.3(a): 80 deg: 0.71 deg total", np.degrees(fd_normal_err3(np.radians(80), "const")[0]), 0.714, tol=0.02)
close("Figure 30.4.3(b): jitter model saturates at 1/sqrt24 rad = 11.7 deg (first order), 12.0 deg Monte Carlo at 85 deg",
      [np.degrees(1 / np.sqrt(24)), np.degrees(fd_normal_err3(np.radians(85), "jitter")[0])], [11.70, 12.02], tol=0.06)
close("Figure 30.4.3(b): the 2D cross-section component peaks at 45 deg with 1/(2 sqrt 24) rad = 5.85 deg",
      np.degrees(0.5 / np.sqrt(24)), 5.848, tol=1e-3)
jt = [fd_normal_err3(np.radians(t), "jitter")[0] for t in (0, 30, 60, 85)]
ct = [fd_normal_err3(np.radians(t), "const")[0] for t in (0, 30, 60, 85)]
check("Figure 30.4.3: constant noise is worst facing the camera, sub-pixel jitter is worst near grazing (monotone)",
      np.all(np.diff(ct) < 0) and np.all(np.diff(jt) > 0))

# ---------------------------------------------------------------- Figure 30.4.1: silhouette = fold of the projection
# on a sphere seen from the origin, cos(theta) -> 0 at the rim and |grad d| -> infinity there
Cs, Rs = np.array([0.0, 0.0, 3.0]), 1.0
fpx = 120.0
for ang_frac in (0.5, 0.9, 0.99, 0.999):
    tan_rim = Rs / np.sqrt(9 - 1)                        # tangent of the rim angle
    xb = ang_frac * tan_rim
    rr_ = np.array([xb, 0.0, 1.0])
    a2, b2, c2 = rr_ @ rr_, -2 * rr_ @ Cs, Cs @ Cs - Rs ** 2
    t_ = (-b2 - np.sqrt(b2 * b2 - 4 * a2 * c2)) / (2 * a2)
    P_ = t_ * rr_
    costh = abs((P_ - Cs) @ rr_) / (Rs * np.linalg.norm(rr_))
    print(f"   sphere: at {ang_frac:.3f} of the rim angle, |cos theta| = {costh:.4f}")
check("Figure 30.4.1: |cos theta| -> 0 at the silhouette of a sphere (rim)", costh < 0.06)

# ---------------------------------------------------------------- exercises and the curvature bias at grazing angles
# Exercise 30.4.1: fronto-parallel plane at 2 m, f = 500 px: footprint (4 mm)^2; tilted by 60 deg: twice
close("Exercise 30.4.1: footprint at d = 2 m, f = 500 px: 16 mm^2; at 60 deg tilt: 32 mm^2",
      [(2.0 / 500) ** 2 * 1e6, (2.0 / 500) ** 2 * 1e6 / np.cos(np.radians(60))], [16.0, 32.0], tol=1e-9)
# Exercise 30.4.2: fronto-parallel plane, off-axis pixel: theta = alpha, footprint d^2/(f_x f_y) (independent of alpha)
for alpha in (0.0, 20.0, 40.0):
    al = np.radians(alpha)
    close(f"Exercise 30.4.2: fronto-parallel plane, alpha = {alpha}: d^2 cos(alpha)/(f^2 cos(theta = alpha)) = d^2/f^2",
          np.cos(al) / np.cos(al), 1.0, tol=1e-15)
# Exercise 30.4.4: plane rho = a u + b v + e gives N ∝ (a f_x, b f_y, e + a c_x + b c_y) by (30.4.5)
aa, bb, ee = 2e-4, -1e-4, 0.3
U0, V0 = 100.0, 300.0
xb, yb = (U0 - Kn["cx"]) / Kn["fx"], (V0 - Kn["cy"]) / Kn["fy"]
rho0 = aa * U0 + bb * V0 + ee
N5 = np.array([Kn["fx"] * aa, Kn["fy"] * bb, rho0 - xb * Kn["fx"] * aa - yb * Kn["fy"] * bb])
N3 = np.array([aa * Kn["fx"], bb * Kn["fy"], ee + aa * Kn["cx"] + bb * Kn["cy"]])
close("Exercise 30.4.4: (30.4.5) on a plane gives the normal of Proposition 30.4.3 (same at every pixel)",
      N5 / np.linalg.norm(N5), N3 / np.linalg.norm(N3), tol=1e-12)
# Exercise 30.4.5: noise level for 1 deg, and the k-pixel baseline
close("Exercise 30.4.5: 1 deg rms facing the camera needs sigma/d = 0.01745/500 = 3.49e-5 (0.0035 %, 0.10 mm at 3 m)",
      [np.radians(1.0) / 500, 3.0 * np.radians(1.0) / 500 * 1e3], [3.49e-5, 0.105], tol=1e-3)
for kpx in (1, 3):
    err = np.degrees(fd_normal_err3(0.0, "const", k=kpx, seed=5)[0])
    close(f"Exercise 30.4.5: stencil +-{kpx} px: rms error = 5.73/{kpx} deg (first order)", err, 5.730 / kpx,
          tol=0.04 * 5.730 / kpx)
# noise-free sphere: finite-difference normals are biased most near the silhouette (insight box)
fs = 500.0
uu_, vv_ = np.meshgrid(np.arange(601) - 300.0, np.arange(601) - 300.0)
rr_ = np.stack([uu_ / fs, vv_ / fs, np.ones_like(uu_)], -1)
Cq = np.array([0.0, 0.0, 3.0])
a3 = (rr_ * rr_).sum(-1)
b3 = -2 * rr_ @ Cq
c3 = Cq @ Cq - 1.0
disc3 = b3 * b3 - 4 * a3 * c3
t3 = np.where(disc3 > 0, (-b3 - np.sqrt(np.maximum(disc3, 0))) / (2 * a3), np.nan)
P3 = t3[..., None] * rr_
ntrue = P3 - Cq
costh3 = np.abs((ntrue * rr_).sum(-1)) / np.linalg.norm(rr_, axis=-1)
dx3 = P3[2:, 1:-1] - P3[:-2, 1:-1]
dy3 = P3[1:-1, 2:] - P3[1:-1, :-2]
N3d = np.cross(dx3, dy3)
N3d /= np.linalg.norm(N3d, axis=-1, keepdims=True)
ang3 = np.degrees(np.arccos(np.clip(np.abs((N3d * ntrue[1:-1, 1:-1]).sum(-1)), 0, 1)))
th3 = np.degrees(np.arccos(costh3[1:-1, 1:-1]))
ok3 = np.isfinite(ang3)
m_lo = ok3 & (th3 < 30)
m_hi = ok3 & (th3 >= 75)
print(f"   sphere FD bias: median {np.median(ang3[m_lo]):.4f} deg (theta < 30), median {np.median(ang3[m_hi]):.3f} deg, "
      f"max {ang3[m_hi].max():.2f} deg (theta >= 75)")
check("insight box: noise-free FD normals on a sphere: bias < 0.001 deg facing the camera, > 0.05 deg median beyond 75 deg",
      np.median(ang3[m_lo]) < 1e-3 and np.median(ang3[m_hi]) > 0.05 and ang3[m_hi].max() > 1.0)

summary()
