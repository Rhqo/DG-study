"""Verification for Section 29.6 (View-Dependent Color: Spherical Harmonics).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/verify/v-29-6-spherical-harmonics.py``
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(296)

# ---------------------------------------------------------------- the real spherical harmonics of Table (29.6.3)
x, y, z = sp.symbols("x y z", real=True)
pi = sp.pi
SH = {  # (l, m): polynomial in (x, y, z), evaluated on the unit sphere
    (0, 0): 1 / (2 * sp.sqrt(pi)) + 0 * x,
    (1, -1): sp.sqrt(3 / (4 * pi)) * y, (1, 0): sp.sqrt(3 / (4 * pi)) * z, (1, 1): sp.sqrt(3 / (4 * pi)) * x,
    (2, -2): sp.sqrt(15 / pi) / 2 * x * y, (2, -1): sp.sqrt(15 / pi) / 2 * y * z,
    (2, 0): sp.sqrt(5 / pi) / 4 * (2 * z ** 2 - x ** 2 - y ** 2), (2, 1): sp.sqrt(15 / pi) / 2 * x * z,
    (2, 2): sp.sqrt(15 / pi) / 4 * (x ** 2 - y ** 2),
    (3, -3): sp.sqrt(35 / (2 * pi)) / 4 * y * (3 * x ** 2 - y ** 2), (3, -2): sp.sqrt(105 / pi) / 2 * x * y * z,
    (3, -1): sp.sqrt(21 / (2 * pi)) / 4 * y * (4 * z ** 2 - x ** 2 - y ** 2),
    (3, 0): sp.sqrt(7 / pi) / 4 * z * (2 * z ** 2 - 3 * x ** 2 - 3 * y ** 2),
    (3, 1): sp.sqrt(21 / (2 * pi)) / 4 * x * (4 * z ** 2 - x ** 2 - y ** 2),
    (3, 2): sp.sqrt(105 / pi) / 4 * z * (x ** 2 - y ** 2), (3, 3): sp.sqrt(35 / (2 * pi)) / 4 * x * (x ** 2 - 3 * y ** 2),
}
check("table: 16 functions for l <= 3", len(SH) == 16)
th, ph = sp.symbols("theta phi", real=True)
on_sphere = {x: sp.sin(th) * sp.cos(ph), y: sp.sin(th) * sp.sin(ph), z: sp.cos(th)}


def lap_s2(f):
    """(29.6.1): Laplace-Beltrami of the round unit sphere, Delta = div grad (non-positive spectrum)."""
    return sp.diff(sp.sin(th) * sp.diff(f, th), th) / sp.sin(th) + sp.diff(f, ph, 2) / sp.sin(th) ** 2


for (l, m), P in SH.items():
    sym_equal(f"(29.6.2): Delta_S2 Y_{l},{m} = -{l * (l + 1)} Y_{l},{m}", lap_s2(P.subs(on_sphere)),
              -l * (l + 1) * P.subs(on_sphere), {th: (0, sp.pi), ph: (0, 2 * sp.pi)})
    check(f"(29.6.2): the polynomial for Y_{l},{m} is harmonic in R^3 and homogeneous of degree {l}",
          sp.simplify(sp.diff(P, x, 2) + sp.diff(P, y, 2) + sp.diff(P, z, 2)) == 0 and
          (l == 0 or sp.Poly(sp.expand(P), x, y, z).is_homogeneous and sp.Poly(sp.expand(P), x, y, z).total_degree() == l))
# the radial formula: Delta_R3 (r^l Y) = r^(l-2) (l(l+1) Y + Delta_S2 Y)
r_, l_ = sp.symbols("r l", positive=True)
Yf = sp.Function("Y")
sym_equal("derivation: Delta_R3 = d_r^2 + (2/r) d_r + Delta_S2 / r^2 applied to r^l Y gives r^(l-2) l (l+1) Y (radial part)",
          sp.diff(r_ ** l_, r_, 2) + 2 / r_ * sp.diff(r_ ** l_, r_), l_ * (l_ + 1) * r_ ** (l_ - 2))
check("table: multiplicity 2l + 1 for l = 0..3", [sum(1 for k in SH if k[0] == l) for l in range(4)] == [1, 3, 5, 7])
check("text: degree <= 3 gives 16 per channel, 48 for RGB", 16 * 3 == 48 and sum(2 * l + 1 for l in range(4)) == 16)
check("Exercise 29.6.1: (L+1)^2 functions up to degree L: L = 4 gives 25 per channel, 75 for RGB",
      sum(2 * l + 1 for l in range(5)) == 25 and 25 * 3 == 75)
check("Exercise 29.6.1: parameters with degree 4: 3 + 4 + 3 + 1 + 75 = 86, 46% more than 59",
      3 + 4 + 3 + 1 + 75 == 86 and round(86 / 59 - 1, 2) == 0.46)
check("Exercise 29.6.5: degree 21 needs 22^2 = 484 coefficients per channel", 22 ** 2 == 484)
check("text: degree 9 needs 100 coefficients per channel", 10 ** 2 == 100)

# numeric versions
keys = list(SH.keys())
fns = [sp.lambdify((x, y, z), SH[k], "numpy") for k in keys]


def Ymat(D):
    return np.stack([np.broadcast_to(f(D[:, 0], D[:, 1], D[:, 2]), (len(D),)) for f in fns], 1)


# orthonormality with exact quadrature (Gauss-Legendre in cos theta, uniform in phi)
tq, wq = np.polynomial.legendre.leggauss(12)
phq = np.linspace(0, 2 * np.pi, 24, endpoint=False)
T, PH = np.meshgrid(tq, phq, indexing="ij")
Wq = (wq[:, None] * np.ones_like(PH) * (2 * np.pi / 24)).ravel()
Dq = np.stack([np.sqrt(1 - T ** 2) * np.cos(PH), np.sqrt(1 - T ** 2) * np.sin(PH), T], -1).reshape(-1, 3)
Yq = Ymat(Dq)
close("orthonormality: int Y_a Y_b dA = delta_ab over S^2 (16 x 16)", Yq.T @ (Yq * Wq[:, None]), np.eye(16), tol=1e-12)
# compare with the 3DGS constants (absolute values)
c3dgs = [0.28209479177387814, 0.4886025119029199, 1.0925484305920792, 0.31539156525252005, 0.5462742152960396,
         0.5900435899266435, 2.890611442640554, 0.4570457994644658, 0.3731763325901154, 1.445305721320277]
mine = [1 / (2 * np.sqrt(np.pi)), np.sqrt(3 / (4 * np.pi)), np.sqrt(15 / np.pi) / 2, np.sqrt(5 / np.pi) / 4,
        np.sqrt(15 / np.pi) / 4, np.sqrt(35 / (2 * np.pi)) / 4, np.sqrt(105 / np.pi) / 2, np.sqrt(21 / (2 * np.pi)) / 4,
        np.sqrt(7 / np.pi) / 4, np.sqrt(105 / np.pi) / 4]
close("table: normalization constants agree in absolute value with the usual real-SH constants", mine, c3dgs, tol=1e-12)

# ---------------------------------------------------------------- rotation: D^l(R) by least squares
def rot(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def Dl(R, l, N=400):
    d = rng.normal(size=(N, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    idx = [i for i, k in enumerate(keys) if k[0] == l]
    A = Ymat(d)[:, idx]
    B = Ymat(d @ R)[:, idx]          # rows: Y(R^{-1} d) since (R^{-1} d)^T = d^T R
    M, res, _, _ = np.linalg.lstsq(A, B, rcond=None)
    return M, np.max(np.abs(A @ M - B)), idx


R1 = rot([1, 1, 0], np.radians(60))
R2 = rot([0.2, -1, 0.5], np.radians(110))
for l in range(4):
    M1, res1, _ = Dl(R1, l)
    M2, _, _ = Dl(R2, l)
    M12, _, _ = Dl(R1 @ R2, l)
    close(f"(29.6.4): degree-{l} space is invariant: Y_l o R^-1 is exactly a combination of Y_l (residual)", res1, 0.0,
          tol=1e-12)
    close(f"(29.6.4): D^{l}(R) is orthogonal", M1.T @ M1, np.eye(2 * l + 1), tol=1e-11)
    close(f"(29.6.4): D^{l}(R1 R2) = D^{l}(R1) D^{l}(R2)", M12, M1 @ M2, tol=1e-11)
# coefficient transformation: f = Y c  ->  f o R^-1 = Y (D c)
cvec = rng.normal(size=16)
d = rng.normal(size=(50, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
Dfull = np.zeros((16, 16))
for l in range(4):
    M, _, idx = Dl(R1, l)
    Dfull[np.ix_(idx, idx)] = M
close("(29.6.4): f(R^-1 d) = sum (D c)_lm Y_lm(d)", Ymat(d @ R1) @ cvec, Ymat(d) @ (Dfull @ cvec), tol=1e-11)
# l = 1: in the order (x, y, z) the block is R itself; in the table order (y, z, x) it is P R P^T
M1, _, _ = Dl(R1, 1)
P = np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])     # (y, z, x) = P (x, y, z)
close("Example 29.6.3: D^1(R) = P R P^T in the order (y, z, x), i.e. R in the order (x, y, z)", M1, P @ R1 @ P.T, tol=1e-12)
S = np.diag([-1.0, 1, -1])                                # official 3DGS l = 1 basis (-y, z, -x)
close("Example 29.6.3: with the basis (-y, z, -x) the block is S P R P^T S", S @ M1 @ S, S @ P @ R1 @ P.T @ S, tol=1e-12)
# Exercise 29.6.3 (symbolic l = 1): <c, R^{-1} d> = <R c, d>
Rs = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"R{i}{j}"))
cs = sp.Matrix(sp.symbols("c1:4")); ds = sp.Matrix(sp.symbols("d1:4"))
sym_equal("Exercise 29.6.3: c^T (R^T d) = (R c)^T d", (cs.T * Rs.T * ds)[0], ((Rs * cs).T * ds)[0])
# Exercise 29.6.4: rotation by 90 deg about z, l = 1 coefficients (a_x, a_y, a_z) in the (x, y, z) order
Rz = rot([0, 0, 1], np.pi / 2)
close("Exercise 29.6.4: R_z(90) (a_x, a_y, a_z) = (-a_y, a_x, a_z)", Rz @ np.array([1.0, 2.0, 3.0]), [-2.0, 1.0, 3.0], tol=1e-12)
# Exercise 29.6.2: z^2 restricted to S^2 = 1/3 + (4 sqrt(pi/5)/3) Y_20
expr = (sp.Rational(1, 3) + 4 * sp.sqrt(pi / 5) / 3 * SH[(2, 0)]).subs(on_sphere)
sym_equal("Exercise 29.6.2: z^2 = 1/3 + (4/3) sqrt(pi/5) Y_20 on the unit sphere", expr, sp.cos(th) ** 2,
          {th: (0, sp.pi), ph: (0, 2 * sp.pi)})
sym_equal("Exercise 29.6.2: Delta_S2 z^2 = 2 - 6 z^2 (not a multiple of z^2)", lap_s2(sp.cos(th) ** 2), 2 - 6 * sp.cos(th) ** 2,
          {th: (0, sp.pi)})

# ---------------------------------------------------------------- Figure 29.6.2: rotating the color function
def lobe(dirs, axis, kappa):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    return np.exp(kappa * (dirs @ a - 1))


r0 = np.array([np.cos(np.radians(60)), np.sin(np.radians(60)), 0.3]); r0 /= np.linalg.norm(r0)
cf = Yq.T @ (lobe(Dq, r0, 8.0) * Wq)                      # L2 projection of a lobe onto l <= 3 (exact quadrature of a smooth fn)
Rf = rot([0, 0, 1], np.radians(100))
Dfull_f = np.zeros((16, 16))
for l in range(4):
    M, _, idx = Dl(Rf, l)
    Dfull_f[np.ix_(idx, idx)] = M
dd = rng.normal(size=(2000, 3)); dd /= np.linalg.norm(dd, axis=1, keepdims=True)
truth = Ymat(dd @ Rf) @ cf
close("Figure 29.6.2(c): rotated coefficients reproduce f o R^-1 exactly", Ymat(dd) @ (Dfull_f @ cf), truth, tol=1e-10)
err_wrong = np.sqrt(np.mean((Ymat(dd) @ cf - truth) ** 2)) / np.sqrt(np.mean(truth ** 2))
print(f"relative RMS error without rotating the SH: {err_wrong:.3f}")
check("Figure 29.6.2(b): keeping the coefficients gives a relative RMS error above 100%", err_wrong > 1.0)
# the bright direction moves with the object: argmax of the rotated function is near R r0
grid = rng.normal(size=(200000, 3)); grid /= np.linalg.norm(grid, axis=1, keepdims=True)
amax_c = grid[np.argmax(Ymat(grid) @ (Dfull_f @ cf))]
amax_a = grid[np.argmax(Ymat(grid) @ cf)]
check("Figure 29.6.2: the maximum of f is within 3 degrees of r0", np.degrees(np.arccos(amax_a @ r0)) < 3)
check("Figure 29.6.2: the maximum of the rotated f is within 3 degrees of R r0", np.degrees(np.arccos(amax_c @ (Rf @ r0))) < 3)

# ---------------------------------------------------------------- Figure 29.6.3: band limit for zonal lobes
tg, wg = np.polynomial.legendre.leggauss(400)


def zonal_proj(kappa, L, t):
    f = np.exp(kappa * (tg - 1))
    out = np.zeros_like(t)
    for l in range(L + 1):
        Pl = np.polynomial.legendre.Legendre.basis(l)
        a = (2 * l + 1) / 2 * np.sum(wg * f * Pl(tg))
        out += a * Pl(t)
    return out


def rel_err(kappa, L):
    f = np.exp(kappa * (tg - 1))
    return np.sqrt(np.sum(wg * (zonal_proj(kappa, L, tg) - f) ** 2) / np.sum(wg * f ** 2))


vals = {k: (zonal_proj(k, 3, np.array([1.0]))[0], rel_err(k, 3)) for k in (5, 20, 100)}
for k, (pk, er) in vals.items():
    print(f"kappa={k}: degree-3 peak {pk:.3f}, relative L2 error {er:.3f}")
close("Figure 29.6.3: kappa = 5: degree-3 peak 0.80, error 0.19", [round(vals[5][0], 2), round(vals[5][1], 2)], [0.80, 0.19],
      tol=1e-9)
close("Figure 29.6.3: kappa = 20: degree-3 peak 0.33, error 0.66", [round(vals[20][0], 2), round(vals[20][1], 2)], [0.33, 0.66],
      tol=1e-9)
close("Figure 29.6.3: kappa = 100: degree-3 peak 0.08, error 0.92", [round(vals[100][0], 2), round(vals[100][1], 2)],
      [0.08, 0.92], tol=1e-9)
# angular half width at half maximum of the lobe: cos(t) = 1 + ln(1/2)/kappa
for k, hw in ((5, 30.5), (20, 15.1), (100, 6.7)):
    close(f"Figure 29.6.3: lobe half width at half maximum for kappa = {k} is {hw} degrees",
          round(np.degrees(np.arccos(1 + np.log(0.5) / k)), 1), hw, tol=1e-9)
Lneed = {k: next(L for L in range(60) if rel_err(k, L) < 0.1) for k in (5, 20, 100)}
print("degree needed for 10% error:", Lneed)
check("Figure 29.6.3(b): degrees needed for 10% error are 4, 9, 21", (Lneed[5], Lneed[20], Lneed[100]) == (4, 9, 21))

# ---------------------------------------------------------------- R2 review: independent checks
from math import factorial as _fact

# (a) official 3DGS basis: constants copied from graphdeco-inria/gaussian-splatting utils/sh_utils.py (eval_sh) and
#     diff-gaussian-rasterization forward.cu (computeColorFromSH). Claim (Example 29.6.3): it is (-1)^m times the table.
C1g = 0.4886025119029199
C2g = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3g = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658,
       1.445305721320277, -0.5900435899266435]


def Y3dgs(D, l):
    X_, Y_, Z_ = D[:, 0], D[:, 1], D[:, 2]
    if l == 1:
        return np.stack([-C1g * Y_, C1g * Z_, -C1g * X_], 1)
    if l == 2:
        return np.stack([C2g[0] * X_ * Y_, C2g[1] * Y_ * Z_, C2g[2] * (2 * Z_ * Z_ - X_ * X_ - Y_ * Y_), C2g[3] * X_ * Z_,
                         C2g[4] * (X_ * X_ - Y_ * Y_)], 1)
    return np.stack([C3g[0] * Y_ * (3 * X_ * X_ - Y_ * Y_), C3g[1] * X_ * Y_ * Z_, C3g[2] * Y_ * (4 * Z_ * Z_ - X_ * X_ - Y_ * Y_),
                     C3g[3] * Z_ * (2 * Z_ * Z_ - 3 * X_ * X_ - 3 * Y_ * Y_), C3g[4] * X_ * (4 * Z_ * Z_ - X_ * X_ - Y_ * Y_),
                     C3g[5] * Z_ * (X_ * X_ - Y_ * Y_), C3g[6] * X_ * (X_ * X_ - 3 * Y_ * Y_)], 1)


dch = rng.normal(size=(300, 3)); dch /= np.linalg.norm(dch, axis=1, keepdims=True)
for l in (1, 2, 3):
    idx = [i for i, k in enumerate(keys) if k[0] == l]
    Sl = np.diag([(-1.0) ** m for m in range(-l, l + 1)])
    close(f"Example 29.6.3: official 3DGS degree-{l} basis = (-1)^m times the table", Y3dgs(dch, l), Ymat(dch)[:, idx] @ Sl,
          tol=1e-12)
    G_ = np.linalg.lstsq(Y3dgs(dch, l), Y3dgs(dch @ R1, l), rcond=None)[0]
    close(f"Example 29.6.3: in the official basis the degree-{l} block is S_l D^l S_l", G_, Sl @ Dl(R1, l)[0] @ Sl, tol=1e-10)

# (b) D^l from the complex Wigner D-matrix (explicit small-d formula) converted to the real basis, compared with the
#     least-squares D^l of Remark 29.6.4. Complex SH: sympy Ynm (Condon-Shortley phase).
thv, phv = sp.symbols("theta_v phi_v")


def Ycx(l, D):
    T_ = np.arccos(np.clip(D[:, 2], -1, 1)); P_ = np.arctan2(D[:, 1], D[:, 0])
    out = []
    for m in range(-l, l + 1):
        f = sp.lambdify((thv, phv), sp.Ynm(l, m, thv, phv).expand(func=True), modules="numpy")
        out.append(np.broadcast_to(f(T_, P_), T_.shape).astype(complex))
    return np.stack(out, 1)


def wig_small(j, mp, m, b):
    tot = 0.0
    for k in range(2 * j + 1):
        if j + m - k < 0 or j - k - mp < 0 or k - m + mp < 0:
            continue
        tot += ((-1) ** (k - m + mp) * np.sqrt(_fact(j + mp) * _fact(j - mp) * _fact(j + m) * _fact(j - m))
                / (_fact(j + m - k) * _fact(k) * _fact(j - k - mp) * _fact(k - m + mp))
                * np.cos(b / 2) ** (2 * j - 2 * k + m - mp) * np.sin(b / 2) ** (2 * k - m + mp))
    return tot


def U_real(l):   # real_m = sum_k U[m, k] Y_l^k  (standard real SH from complex SH with Condon-Shortley phase)
    U = np.zeros((2 * l + 1, 2 * l + 1), complex)
    for m in range(-l, l + 1):
        if m == 0:
            U[l, l] = 1
        elif m > 0:
            U[m + l, m + l] = (-1) ** m / np.sqrt(2); U[m + l, -m + l] = 1 / np.sqrt(2)
        else:
            U[m + l, -m + l] = (-1) ** m / (1j * np.sqrt(2)); U[m + l, m + l] = -1 / (1j * np.sqrt(2))
    return U


al, be, ga = 0.7, 1.1, -0.4
Rzyz = rot([0, 0, 1], al) @ rot([0, 1, 0], be) @ rot([0, 0, 1], ga)
for l in range(4):
    idx = [i for i, k in enumerate(keys) if k[0] == l]
    Cc = Ycx(l, dch)
    close(f"Table: degree-{l} rows are the standard real SH built from complex SH", (U_real(l) @ Cc.T).T,
          Ymat(dch)[:, idx], tol=1e-12)
    Dc = np.array([[np.exp(-1j * mp * al) * wig_small(l, mp, m, be) * np.exp(-1j * m * ga) for m in range(-l, l + 1)]
                   for mp in range(-l, l + 1)])
    close(f"Wigner D: Y_l^m(R^-1 d) = sum_k Y_l^k(d) D_km (degree {l}, z-y-z Euler angles)", Cc @ Dc, Ycx(l, dch @ Rzyz),
          tol=1e-12)
    Dreal = np.linalg.solve(U_real(l).T, Dc @ U_real(l).T)
    close(f"Remark 29.6.4: least-squares D^{l} equals the real form of the Wigner D-matrix", Dreal, Dl(Rzyz, l)[0], tol=1e-10)

# (c) nodal lines (Figure 29.6.1 caption): |m| great circles through the poles and l - |m| circles of latitude
tline = np.linspace(1e-3, np.pi - 1e-3, 4001)
pline = np.linspace(0, 2 * np.pi, 8001, endpoint=False) + 1e-3   # avoid sampling exactly on a node
ok_nodal = True
for (l, m), f in zip(keys, fns):
    Dt = np.stack([np.sin(tline) * np.cos(0.37), np.sin(tline) * np.sin(0.37), np.cos(tline)], 1)
    vt = np.broadcast_to(f(Dt[:, 0], Dt[:, 1], Dt[:, 2]), tline.shape)
    Dp = np.stack([np.sin(1.234) * np.cos(pline), np.sin(1.234) * np.sin(pline), np.cos(1.234) * np.ones_like(pline)], 1)
    vp = np.broadcast_to(f(Dp[:, 0], Dp[:, 1], Dp[:, 2]), pline.shape)
    lat = np.sum(np.sign(vt[1:]) != np.sign(vt[:-1]))
    mer = np.sum(np.sign(np.roll(vp, -1)) != np.sign(vp))
    ok_nodal &= (lat == l - abs(m)) and (mer == 2 * abs(m))
check("Figure 29.6.1: Y_lm has |m| meridian great circles and l - |m| latitude circles as nodal lines", ok_nodal)

summary()
