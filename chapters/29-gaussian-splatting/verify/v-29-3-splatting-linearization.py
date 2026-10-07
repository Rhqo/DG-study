"""Verification for Section 29.3 (Splatting Is Linearization).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/verify/v-29-3-splatting-linearization.py``
(Monte Carlo with a fixed seed, at most a few seconds; set OMP_NUM_THREADS <= 4.)
"""

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(293)

# ---------------------------------------------------------------- (29.3.2)-(29.3.3): the perspective map and its Jacobian
tx, ty, tz, fx, fy, cx, cy = sp.symbols("t_x t_y t_z f_x f_y c_x c_y", real=True)
phi = sp.Matrix([fx * tx / tz + cx, fy * ty / tz + cy])
Jsym = phi.jacobian(sp.Matrix([tx, ty, tz]))
J_text = sp.Matrix([[fx / tz, 0, -fx * tx / tz ** 2], [0, fy / tz, -fy * ty / tz ** 2]])
sym_equal("(29.3.3): J = D phi(t)", Jsym, J_text)
sym_equal("(29.3.3): J t = 0 (kernel = ray direction)", Jsym * sp.Matrix([tx, ty, tz]), sp.zeros(2, 1))
# Hessian of phi_x (second-order term)
Hx = sp.hessian(phi[0], (tx, ty, tz))
sym_equal("second-order term: Hessian of phi_x", Hx,
          sp.Matrix([[0, 0, -fx / tz ** 2], [0, 0, 0], [-fx / tz ** 2, 0, 2 * fx * tx / tz ** 3]]))
check("second-order term: every entry of H_x involves the t_z direction", Hx[0, 0] == 0 and Hx[0, 1] == 0 and Hx[1, 1] == 0)


def phi_np(T, f=1.0):
    return f * T[:, :2] / T[:, 2:3]


def J_np(t0, f=1.0):
    x, y, z = t0
    return f * np.array([[1 / z, 0, -x / z ** 2], [0, 1 / z, -y / z ** 2]])


# ---------------------------------------------------------------- (29.3.1): world -> camera is exact
def random_rotation():
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


Wm = random_rotation()
bvec = np.array([0.2, -0.1, 3.0])
mu = np.array([0.1, 0.3, 0.5])
Rg = random_rotation()
Sig = Rg @ np.diag([0.04, 0.01, 0.0025]) @ Rg.T
X = mu + rng.normal(size=(300000, 3)) @ np.linalg.cholesky(Sig).T
T = X @ Wm.T + bvec
close("(29.3.1): camera-frame samples have covariance W Sigma W^T (MC, tol 2e-4)", np.cov(T.T), Wm @ Sig @ Wm.T, tol=2e-4)
# linearized pushforward
t0 = Wm @ mu + bvec
Ulin = phi_np(t0[None])[0] + (T - t0) @ J_np(t0).T
close("(29.3.4): linearized pushforward has covariance J W Sigma W^T J^T (MC, tol 1e-4)", np.cov(Ulin.T),
      J_np(t0) @ Wm @ Sig @ Wm.T @ J_np(t0).T, tol=1e-4)
# marginalization along lines parallel to t0 = integrating G over ker J: J Sigma J^T is the covariance of the marginal
Jm = J_np(t0)
N = np.cross(Jm[0], Jm[1])
close("ker J is spanned by t0", abs(N @ t0) / (np.linalg.norm(N) * np.linalg.norm(t0)), 1.0, tol=1e-12)

# ---------------------------------------------------------------- EWA = oblique projection along t0 onto t_z = t0_z, then exact projection
v = rng.normal(size=3)
lam = -v[2] / t0[2]
onplane = t0 + v + lam * t0
close("EWA = slide along t0 to the depth plane, then project exactly", phi_np(onplane[None])[0],
      phi_np(t0[None])[0] + Jm @ v, tol=1e-12)

# ---------------------------------------------------------------- flat fronto-parallel Gaussians project exactly
Sflat = np.diag([0.3, 0.2, 0.0])
Yf = rng.normal(size=(1000, 3)) @ np.sqrt(Sflat)
t0o = np.array([0.8, -0.3, 2.0])
close("exactness: a Gaussian flat in the plane t_z = const projects exactly (off axis)", phi_np(t0o + Yf),
      phi_np(t0o[None])[0] + Yf @ J_np(t0o).T, tol=1e-12)

# ---------------------------------------------------------------- Error experiment (Figure 29.3.3, table)
Nmc = 100000
Y = rng.normal(size=(4 * Nmc, 3))
Y = Y[(Y ** 2).sum(1) <= 9][:Nmc]                  # truncated at the 3-sigma ellipsoid
Y = np.concatenate([Y * np.array(sg) for sg in ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1))])  # reflect x, y
c_tr = (Y ** 2).mean()
# exact value: E[r^2; r <= 3] / (3 P(r <= 3)) for a standard normal in R^3 = P(chi2_5 <= 9) / P(chi2_3 <= 9)
xs = sp.symbols("x", positive=True)
chi_cdf = lambda k: sp.N(sp.integrate(xs ** (sp.Rational(k, 2) - 1) * sp.exp(-xs / 2), (xs, 0, 9))
                         / (2 ** sp.Rational(k, 2) * sp.gamma(sp.Rational(k, 2))))
c_exact = float(chi_cdf(5) / chi_cdf(3))
close("truncation: per-axis variance of the 3-sigma truncated normal is 0.918 (exact)", c_exact, 0.918, tol=0.0005)
close("truncation: Monte Carlo agrees with the exact value", c_tr, c_exact, tol=0.005)


def errors(s, th_deg, Yu=Y, A=None):
    """relative center offset and relative covariance error of exact vs linearized pushforward (same samples)."""
    t0 = np.array([np.tan(np.radians(th_deg)), 0.0, 1.0])
    D = s * Yu if A is None else Yu @ A.T
    Tm = t0 + D
    u = phi_np(Tm)
    ul = phi_np(t0[None])[0] + D @ J_np(t0).T
    Cl = np.cov(ul.T)
    e_mean = np.linalg.norm(u.mean(0) - ul.mean(0)) / np.sqrt(np.linalg.eigvalsh(Cl).max())
    e_cov = np.linalg.norm(np.cov(u.T) - Cl) / np.linalg.norm(Cl)
    return e_mean, e_cov


# Deterministic quadrature of the same expectations (Gauss-Legendre in r and cos(polar), uniform in azimuth).
# This is what Figure 29.3.3 and the table of Example 29.3.4 use; Monte Carlo above is an independent cross-check.
def quad_nodes(nr=120, nc=60, nph=120):
    xr, wr = np.polynomial.legendre.leggauss(nr)
    rr = 1.5 * (xr + 1.0)
    wr = 1.5 * wr * rr ** 2 * np.exp(-rr ** 2 / 2)
    xc, wc = np.polynomial.legendre.leggauss(nc)
    ph = np.linspace(0, 2 * np.pi, nph, endpoint=False)
    R_, CT, PH = np.meshgrid(rr, xc, ph, indexing="ij")
    ST = np.sqrt(1 - CT ** 2)
    Yq = np.stack([R_ * ST * np.cos(PH), R_ * ST * np.sin(PH), R_ * CT], -1).reshape(-1, 3)
    Wq = (wr[:, None, None] * wc[None, :, None] * np.ones(nph)[None, None, :]).ravel()
    return Yq, Wq / Wq.sum()


YQ, WQ = quad_nodes()


def quad_errors(s, th_deg, Yq=YQ, Wq=WQ, A=None):
    t0 = np.array([np.tan(np.radians(th_deg)), 0.0, 1.0])
    D = s * Yq if A is None else Yq @ A.T
    u = phi_np(t0 + D)
    ul = phi_np(t0[None])[0] + D @ J_np(t0).T

    def mom(v):
        m = Wq @ v
        d = v - m
        return m, (Wq[:, None] * d).T @ d

    m, Cv = mom(u)
    ml, Cl = mom(ul)
    return (np.linalg.norm(m - ml) / np.sqrt(np.linalg.eigvalsh(Cl).max()),
            np.linalg.norm(Cv - Cl) / np.linalg.norm(Cl))


c2 = float(WQ @ YQ[:, 0] ** 2)                                   # E[y_1^2]
c4 = float(WQ @ (YQ[:, 0] ** 2 * YQ[:, 2] ** 2)) / c2            # E[y_1^2 y_3^2] / E[y_1^2]
close("truncation: quadrature gives c_2 = E[y_1^2] = 0.918", c2, c_exact, tol=1e-6)
# exact value of c_4: E[r^4]/(5 E[r^2]) on the 3-ball = P(chi2_7 <= 9) / P(chi2_5 <= 9)
c4_exact = float(chi_cdf(7) / chi_cdf(5))
close("truncation: c_4 = E[y_1^2 y_3^2]/E[y_1^2] = P(chi2_7 <= 9)/P(chi2_5 <= 9) = 0.839", c4, c4_exact, tol=1e-6)
close("truncation: c_4 = 0.839", c4_exact, 0.839, tol=0.0005)
Yf, Wf = quad_nodes(200, 80, 160)
close("quadrature converged (s/t_z = 0.3, theta = 60)", np.array(quad_errors(0.3, 60)),
      np.array(quad_errors(0.3, 60, Yf, Wf)), tol=1e-6)

table = {}
for th in (0, 30, 60):
    for s in (0.01, 0.1, 0.2, 0.3):
        table[(th, s)] = quad_errors(s, th)
        print(f"theta={th:2d} s/t_z={s:4.2f}  center {100 * table[(th, s)][0]:7.4f}%  cov {100 * table[(th, s)][1]:8.4f}%")
check("table: on axis the center offset is 0 (symmetry)", all(table[(0, s)][0] < 1e-9 for s in (0.01, 0.1, 0.2, 0.3)))
# printed values of the table in Example 29.3.4 (fractions); tolerance = half a unit of the last printed digit
expected_cov = {(0, 0.01): (0.0003, 5e-5), (0, 0.1): (0.026, 5e-4), (0, 0.2): (0.12, 5e-3), (0, 0.3): (0.37, 5e-3),
                (30, 0.01): (0.0003, 5e-5), (30, 0.1): (0.034, 5e-4), (30, 0.2): (0.16, 5e-3), (30, 0.3): (0.59, 5e-3),
                (60, 0.01): (0.0005, 5e-5), (60, 0.1): (0.058, 5e-4), (60, 0.2): (0.29, 5e-3), (60, 0.3): (1.25, 5e-3)}
for key, (val, tol) in expected_cov.items():
    close(f"table: covariance error at theta={key[0]}, s/t_z={key[1]} is {val}", table[key][1], val, tol=tol)
expected_center = {(30, 0.01): (0.0048, 5e-5), (30, 0.1): (0.049, 5e-4), (30, 0.2): (0.11, 5e-3), (30, 0.3): (0.20, 5e-3),
                   (60, 0.01): (0.0083, 5e-5), (60, 0.1): (0.085, 5e-4), (60, 0.2): (0.19, 5e-3), (60, 0.3): (0.34, 5e-3)}
for key, (val, tol) in expected_center.items():
    close(f"table: center offset at theta={key[0]}, s/t_z={key[1]} is {val}", table[key][0], val, tol=tol)
# Monte Carlo (same expectations, different method) agrees within its sampling error
for key in ((0, 0.1), (30, 0.2), (60, 0.1), (60, 0.3)):
    mc = errors(key[1], key[0])
    close(f"cross-check: Monte Carlo cov error at theta={key[0]}, s/t_z={key[1]} within 5% of quadrature", mc[1],
          table[key][1], tol=0.05 * table[key][1])
# leading-order predictions (truncated normal): center offset ~ sqrt(c_2) (s/t_z) sin(theta),
# covariance error ~ 3 c_4 (s/t_z)^2 on axis (for the untruncated normal both constants are 1)
for th in (15, 30, 45, 60):
    close(f"prediction: center offset ~ sqrt(c_2) (s/t_z) sin(theta) for s/t_z = 0.01, theta = {th}",
          quad_errors(0.01, th)[0], np.sqrt(c2) * 0.01 * np.sin(np.radians(th)), tol=0.002 * 0.01)
close("prediction: on-axis covariance error ~ 3 c_4 (s/t_z)^2 at s/t_z = 0.01 (1%)", quad_errors(0.01, 0)[1],
      3 * c4 * 0.01 ** 2, tol=0.01 * 3 * c4 * 0.01 ** 2)
check("prediction: the coefficient is 3 c_4, not 3 c_2 (they differ by 9%)",
      abs(quad_errors(0.01, 0)[1] / (3 * c2 * 1e-4) - 1) > 0.05)
# series: E[(1 + e)^-2] = 1 + 3 r^2 + O(r^4) for e ~ N(0, r^2)
r, e = sp.symbols("r e", positive=True)
series = sp.series((1 + e) ** -2, e, 0, 4).removeO()
Ee2 = sp.integrate(series * sp.exp(-e ** 2 / (2 * r ** 2)) / sp.sqrt(2 * sp.pi * r ** 2), (e, -sp.oo, sp.oo))
sym_equal("derivation: E[(1 + eps)^-2] = 1 + 3 r^2 up to O(r^4)", sp.expand(Ee2), 1 + 3 * r ** 2)
# center offset formula: (1/2) E[H(delta, delta)] = f s^2 t_x / t_z^3
s_ = sp.symbols("s", positive=True)
off = sp.Rational(1, 2) * (s_ ** 2 * Hx.trace())
sym_equal("derivation: (1/2) E[H_x(delta,delta)] = f s^2 t_x / t_z^3 for Sigma = s^2 I", off, fx * s_ ** 2 * tx / tz ** 3)

# which extent matters: depth-elongated vs laterally elongated (on axis)
_, e_depth = quad_errors(None, 0, A=np.diag([0.01, 0.01, 0.2]))
_, e_lat = quad_errors(None, 0, A=np.diag([0.2, 0.01, 0.01]))
print(f"depth-elongated cov error {e_depth:.4f}, laterally elongated {e_lat:.6f}")
close("extent along the optical axis matters: depth-elongated s = (0.01, 0.01, 0.2) error 12%", e_depth, 0.12, tol=0.005)
close("extent along the optical axis matters: laterally elongated s = (0.2, 0.01, 0.01) error 0.03%", e_lat, 0.0003,
      tol=0.00005)

# ---------------------------------------------------------------- Figure 29.3.2: exact image of the k-sigma ellipsoid
def silhouette(t0, Sc, k):
    A = np.linalg.inv(Sc) / k ** 2
    Q = np.zeros((4, 4))
    Q[:3, :3] = A; Q[:3, 3] = -A @ t0; Q[3, :3] = -t0 @ A; Q[3, 3] = t0 @ A @ t0 - 1
    Cd = np.linalg.inv(Q)[:3, :3]
    Cm = np.linalg.inv(Cd)
    M, m, c = Cm[:2, :2], Cm[:2, 2], Cm[2, 2]
    if np.linalg.eigvalsh(M)[0] < 0:
        M, m, c = -M, -m, -c
    x0 = -np.linalg.solve(M, m)
    kap = m @ np.linalg.solve(M, m) - c
    return x0, kap * np.linalg.inv(M) / k ** 2


for th, s in ((0, 0.03), (45, 0.03), (0, 0.25), (45, 0.25)):
    t0 = np.array([np.tan(np.radians(th)), 0, 1.0])
    x0, B = silhouette(t0, s * s * np.eye(3), 2)
    d = rng.normal(size=(20000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    u = phi_np(t0 + 2 * s * d)
    val = np.einsum("ni,ij,nj->n", u - x0, np.linalg.inv(B), u - x0)
    close(f"Figure 29.3.2: silhouette touches the projected 2-sigma sphere (theta={th}, s={s})", val.max(), 4.0, tol=1e-4)
    check(f"Figure 29.3.2: all projected surface points are inside (theta={th}, s={s})", val.max() <= 4.0 + 1e-6)
# on axis the image of a sphere of radius rho at depth z is a circle of radius f rho / sqrt(z^2 - rho^2)
x0, B = silhouette(np.array([0, 0, 1.0]), 0.0625 * np.eye(3), 2)
close("Figure 29.3.2: on axis, s/t_z = 0.25: 2-sigma image radius / EWA radius = 1/sqrt(1 - 0.25) = 1.155",
      np.sqrt(B[0, 0] / 0.0625), 1 / np.sqrt(1 - 0.25), tol=1e-9)
x0, B = silhouette(np.array([1.0, 0, 1.0]), 0.0625 * np.eye(3), 2)
E45 = J_np(np.array([1.0, 0, 1.0])) @ (0.0625 * np.eye(3)) @ J_np(np.array([1.0, 0, 1.0])).T
shift = np.linalg.norm(x0 - np.array([1.0, 0])) / np.sqrt(np.linalg.eigvalsh(E45).max())
close("Figure 29.3.2 (d): outline center shifted by 0.94 EWA std (outward)", shift, 0.943, tol=1e-3)
check("Figure 29.3.2 (d): the shift is outward (larger u)", x0[0] > 1.0)
emd, ecd = errors(0.25, 45)
print(f"panel (d) MC: center {emd:.3f}, cov {ecd:.3f}")
close("Figure 29.3.2 (d): MC center offset 0.21 EWA std", emd, 0.21, tol=0.01)
close("Figure 29.3.2 (d): quadrature center offset 0.21 EWA std", quad_errors(0.25, 45)[0], 0.205, tol=0.001)

# ---------------------------------------------------------------- Figure 29.3.1(b): 1D exact pushforward density
z0, th1, s1 = 2.5, np.radians(30), 0.5
x0_ = z0 * np.tan(th1)
uu = np.linspace(-0.6, 1.8, 2401)
zz = np.linspace(0.05, 6.0, 6000)
UU, ZZ = np.meshgrid(uu, zz, indexing="ij")
dens = (np.exp(-((UU * ZZ - x0_) ** 2 + (ZZ - z0) ** 2) / (2 * s1 ** 2)) / (2 * np.pi * s1 ** 2) * ZZ).sum(1) * (zz[1] - zz[0])
close("Figure 29.3.1(b): exact 1D density integrates to 1", dens.sum() * (uu[1] - uu[0]), 1.0, tol=2e-3)
u0 = x0_ / z0
sd = s1 * np.sqrt(1 + np.tan(th1) ** 2) / z0
close("Figure 29.3.1(b): EWA std = s sqrt(1 + tan^2) / z0 = 0.231", sd, 0.2309, tol=1e-4)
mean_exact = (uu * dens).sum() / dens.sum()
print(f"1D: u0 = {u0:.4f}, exact mean = {mean_exact:.4f}, mode = {uu[np.argmax(dens)]:.4f}")
mode_exact = uu[np.argmax(dens)]
close("Figure 29.3.1(b): mode of the exact density is 0.54 (< u0 = 0.58)", round(mode_exact, 2), 0.54, tol=1e-9)
close("Figure 29.3.1(b): mean of the exact density is 0.60", round(mean_exact, 2), 0.60, tol=1e-9)
beta = np.arcsin(2 * s1 / np.hypot(x0_, z0))
close("Figure 29.3.1(a): exact interval tan(30 deg +- asin(2s/|t0|)) = [0.17, 1.20]",
      np.round(np.tan([th1 - beta, th1 + beta]), 2), [0.17, 1.20], tol=1e-9)
close("Figure 29.3.1(a): EWA interval u0 +- 2 sd = [0.12, 1.04]", np.round([u0 - 2 * sd, u0 + 2 * sd], 2), [0.12, 1.04],
      tol=1e-9)
close("Figure 29.3.1(b): exact mean is shifted outward by about 0.02 (= f s^2 tan / z0^2 = 0.023)", mean_exact - u0,
      0.023, tol=0.004)

# ---------------------------------------------------------------- (29.3.5): convolution adds covariances; dilation
S1 = np.array([[2.0, 0.3], [0.3, 0.5]])
S2 = 0.3 * np.eye(2)
g = np.linspace(-8, 8, 321)
GX, GY = np.meshgrid(g, g, indexing="ij")
P = np.stack([GX, GY], -1)


def gauss(Sm, normalized):
    val = np.exp(-0.5 * np.einsum("...i,ij,...j->...", P, np.linalg.inv(Sm), P))
    return val / (2 * np.pi * np.sqrt(np.linalg.det(Sm))) if normalized else val


h = g[1] - g[0]
conv = np.real(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(gauss(S1, False))) *
                                            np.fft.fft2(np.fft.ifftshift(gauss(S2, True)))))) * h * h
close("(29.3.5): G_Sigma1 * N(0, Sigma2) = sqrt(det S1 / det(S1 + S2)) G_{Sigma1 + Sigma2}", conv,
      np.sqrt(np.linalg.det(S1) / np.linalg.det(S1 + S2)) * gauss(S1 + S2, False), tol=1e-6)
# Exercise 29.3.4 and Figure 29.3.4
Sd = np.diag([0.01, 4.0])
close("Exercise 29.3.4: area ratio after +0.3 I is sqrt(det(S+0.3I)/det S) = 5.77",
      np.sqrt(np.linalg.det(Sd + 0.3 * np.eye(2)) / np.linalg.det(Sd)), 5.77, tol=0.005)
close("Exercise 29.3.4: normalization factor sqrt(det S / det(S + 0.3 I)) = 0.173",
      np.sqrt(np.linalg.det(Sd) / np.linalg.det(Sd + 0.3 * np.eye(2))), 0.173, tol=0.0005)
close("Figure 29.3.4: thin direction std 0.22 -> sqrt(0.05 + 0.3) = 0.59 px", np.sqrt(0.05 + 0.3), 0.59, tol=0.002)
close("Figure 29.3.4: 1D area ratio sqrt(0.35/0.05) = 2.65", np.sqrt(0.35 / 0.05), 2.65, tol=0.005)
close("Figure 29.3.4: normalized peak sqrt(0.05/0.35) = 0.378", np.sqrt(0.05 / 0.35), 0.378, tol=0.0005)
a20 = np.radians(20)
R20 = np.array([[np.cos(a20), -np.sin(a20)], [np.sin(a20), np.cos(a20)]])
S34 = R20 @ np.diag([100.0, 0.05]) @ R20.T
close("Figure 29.3.4 / text: 2D area ratio sqrt(det(S + 0.3 I)/det S) = 2.65 (same as the 1D cross-section)",
      np.sqrt(np.linalg.det(S34 + 0.3 * np.eye(2)) / np.linalg.det(S34)), 2.65, tol=0.005)
pxx, pyy = np.meshgrid(np.arange(8) + 0.5, np.arange(5) + 0.5)
P34 = np.stack([pxx, pyy], -1) - np.array([4.0, 2.5])
for Sm, lo, hi, lab in ((S34, 0.23, 0.90, "original"), (S34 + 0.3 * np.eye(2), 1.45, 1.58, "dilated")):
    cs = np.exp(-0.5 * np.einsum("...i,ij,...j->...", P34, np.linalg.inv(Sm), P34)).sum(0)
    close(f"Figure 29.3.4 ({lab}): per-column sums of pixel values range {lo}-{hi}", [cs.min(), cs.max()], [lo, hi],
          tol=0.005)

# Figure 29.3.2 (b): off-axis stretch 1/cos^2(45) = 2 in variance
E45b = J_np(np.array([1.0, 0, 1.0])) @ np.eye(3) @ J_np(np.array([1.0, 0, 1.0])).T
close("Figure 29.3.2(b): J stretches the variance by 1/cos^2 45 = 2 along u^1", E45b[0, 0], 2.0, tol=1e-12)
# pinhole radial magnification: perpendicular-to-ray unit vector maps to f/(z cos theta)
thq = np.radians(50)
tq = np.array([np.tan(thq), 0, 1.0])
vq = np.array([np.cos(thq), 0, -np.sin(thq)])
close("other cameras: radial magnification at distance d is f/(d cos^2 theta)", np.linalg.norm(J_np(tq) @ vq),
      1 / (np.linalg.norm(tq) * np.cos(thq) ** 2), tol=1e-12)

# ---------------------------------------------------------------- Exercises
# 29.3.1: J at t0 = (1, 0.5, 4), f = 500
Je = J_np(np.array([1.0, 0.5, 4.0]), f=500.0)
close("Exercise 29.3.1: J = [[125, 0, -31.25], [0, 125, -15.625]]", Je, [[125, 0, -31.25], [0, 125, -15.625]], tol=1e-12)
close("Exercise 29.3.1: J t0 = 0", Je @ np.array([1.0, 0.5, 4.0]), [0, 0], tol=1e-12)
# 29.3.2: s = 0.2 m at depth 2 m on axis, f = 1000 px
Je2 = J_np(np.array([0.0, 0.0, 2.0]), f=1000.0)
close("Exercise 29.3.2: Sigma_2D = 10^4 I px^2 (std 100 px)", Je2 @ (0.04 * np.eye(3)) @ Je2.T, 1e4 * np.eye(2), tol=1e-9)
close("Exercise 29.3.2: covariance error at s/t_z = 0.1 on axis is 2.6%", table[(0, 0.1)][1], 0.026, tol=0.0005)
close("Exercise 29.3.2: leading term 3 (s/t_z)^2 = 3% for the untruncated normal", 3 * 0.1 ** 2, 0.03, tol=1e-12)
# 29.3.5: d Sigma_2D / d t_z on axis = -2 f^2 s^2 / t_z^3 I
fz, sz, zz_ = 1000.0, 0.2, 2.0
hh = 1e-6
num = (J_np(np.array([0, 0, zz_ + hh]), fz) @ (sz ** 2 * np.eye(3)) @ J_np(np.array([0, 0, zz_ + hh]), fz).T
       - J_np(np.array([0, 0, zz_ - hh]), fz) @ (sz ** 2 * np.eye(3)) @ J_np(np.array([0, 0, zz_ - hh]), fz).T) / (2 * hh)
close("Exercise 29.3.5: dSigma_2D/dt_z = -2 f^2 s^2 / t_z^3 I", num, -2 * fz ** 2 * sz ** 2 / zz_ ** 3 * np.eye(2), tol=1e-2)

summary()
