"""Verification for Section 30.3 (Surfels and 2D Gaussian Splatting).

Run from the project root:
``OMP_NUM_THREADS=4 PYTHONPATH=tools python3 chapters/30-radiance-field-surfaces/verify/v-30-3-surfels-2dgs.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(303)

# ---------------------------------------------------------------- Proposition 30.3.1: the 2DGS parametrization
u, v = sp.symbols("u v", real=True)
su, sv = sp.symbols("s_u s_v", positive=True)
# an exact rational rotation (Cayley transform of a rational skew matrix): fast, deterministic symbolic checks
Ask = sp.Matrix([[0, -sp.Rational(1, 3), sp.Rational(1, 2)], [sp.Rational(1, 3), 0, -sp.Rational(2, 5)],
                 [-sp.Rational(1, 2), sp.Rational(2, 5), 0]])
Rm = (sp.eye(3) - Ask).inv() * (sp.eye(3) + Ask)
check("Proposition 30.3.1 setup: the Cayley rotation is in SO(3)", sp.simplify(Rm.T * Rm - sp.eye(3)) == sp.zeros(3, 3)
      and sp.simplify(Rm.det()) == 1)
tu, tv, tw = Rm[:, 0], Rm[:, 1], Rm[:, 2]
pk = sp.Matrix(sp.symbols("p1 p2 p3", real=True))
P = pk + su * tu * u + sv * tv * v
Pu, Pv = P.diff(u), P.diff(v)
E, F, G = (Pu.T * Pu)[0], (Pu.T * Pv)[0], (Pv.T * Pv)[0]
sym_equal("Proposition 30.3.1: E = s_u^2", E, su ** 2)
sym_equal("Proposition 30.3.1: F = 0", F, 0)
sym_equal("Proposition 30.3.1: G = s_v^2", G, sv ** 2)
cr = Pu.cross(Pv)
sym_equal("Proposition 30.3.1: P_u x P_v = s_u s_v t_w (so N = t_w = t_u x t_v)", cr, su * sv * tw)
sym_equal("Proposition 30.3.1: t_u x t_v = t_w for R in SO(3)", tu.cross(tv), tw)
sym_equal("Proposition 30.3.1: sqrt(EG - F^2) = s_u s_v", sp.sqrt(E * G - F ** 2), su * sv)
# area of the 1-sigma ellipse u^2 + v^2 <= 1
rr, th = sp.symbols("rr th", positive=True)
check("Proposition 30.3.1: area of the 1-sigma ellipse = pi s_u s_v",
      sp.simplify(sp.integrate(sp.integrate(su * sv * rr, (rr, 0, 1)), (th, 0, 2 * sp.pi)) - sp.pi * su * sv) == 0)
# the Gaussian metric: |P - p_k|_Sigma^2 = u^2 + v^2 with Sigma = s_u^2 t_u t_u^T + s_v^2 t_v t_v^T (pseudo-inverse in plane)
Sig_pinv = tu * tu.T / su ** 2 + tv * tv.T / sv ** 2
sym_equal("Proposition 30.3.1: Mahalanobis (in-plane) |P - p_k|_Sigma^2 = u^2 + v^2", ((P - pk).T * Sig_pinv * (P - pk))[0],
          u ** 2 + v ** 2)
sym_equal("Proposition 30.3.1: Euclidean |P - p_k|^2 = s_u^2 u^2 + s_v^2 v^2 (first fundamental form)",
          ((P - pk).T * (P - pk))[0], su ** 2 * u ** 2 + sv ** 2 * v ** 2)

# ---------------------------------------------------------------- (30.3.2)-(30.3.4): ray-splat intersection as a homography
def random_rotation():
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


Kpix = np.array([[800.0, 0, 320.0], [0, 820.0, 240.0], [0, 0, 1]])
ok_all, ok_depth, ok_paper = True, True, True
for trial in range(200):
    Rw = random_rotation()                                  # world -> camera rotation
    tcam = rng.normal(size=3) * 0.3
    Rs = random_rotation()
    s_uv = rng.uniform(0.05, 0.5, size=2)
    pw = rng.normal(size=3) * 0.5
    # splat center in front of the camera
    pc = Rw @ pw + tcam
    tcam = tcam + np.array([0, 0, 3.0 - pc[2]])
    Hs = np.column_stack([s_uv[0] * Rs[:, 0], s_uv[1] * Rs[:, 1], pw])          # (u, v, 1) -> world point
    M = Kpix @ np.column_stack([Rw, tcam]) @ np.vstack([Hs, [0, 0, 1]])        # (u, v, 1) -> (X, Y, W), W = cam z
    Tu, Tv, Tw = M[0], M[1], M[2]
    uv_true = rng.normal(size=2) * 1.2
    q = M @ np.array([uv_true[0], uv_true[1], 1.0])
    if q[2] <= 0.1:
        continue
    x, y = q[0] / q[2], q[1] / q[2]
    k = x * Tw - Tu                                  # forward.cu: k = pix.x * Tw - Tu
    l = y * Tw - Tv
    pp = np.cross(k, l)
    uv = pp[:2] / pp[2]
    ok_all &= np.allclose(uv, uv_true, atol=1e-8)
    depth = uv[0] * Tw[0] + uv[1] * Tw[1] + Tw[2]
    world_pt = Hs @ np.array([uv[0], uv[1], 1.0])
    ok_depth &= abs(depth - (Rw @ world_pt + tcam)[2]) < 1e-8
    # paper (10) with 4D planes h = (k1, k2, 0, k3)
    den = k[0] * l[1] - k[1] * l[0]
    u10 = (k[1] * l[2] - k[2] * l[1]) / den
    v10 = (k[2] * l[0] - k[0] * l[2]) / den
    ok_paper &= np.allclose([u10, v10], uv_true, atol=1e-8)
check("(30.3.3): (u, v) from the cross product of the two lines k = xT_w - T_u, l = yT_w - T_v recovers the splat point (200 trials)",
      ok_all)
check("(30.3.4): depth = u T_w,1 + v T_w,2 + T_w,3 is the camera z of the intersection (200 trials)", ok_depth)
check("[Huang24] eq. (10) is the same cross product written out", ok_paper)

# Exercise 30.3.2: fronto-parallel splat, K = I
Q = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 2.0]])
k = 0.03 * Q[2] - Q[0]
l = -0.01 * Q[2] - Q[1]
pp = np.cross(k, l)
close("Exercise 30.3.2: k x l = (0.006, -0.002, 0.01)", pp, [0.006, -0.002, 0.01], tol=1e-15)
close("Exercise 30.3.2: (u, v) = (0.6, -0.2), depth 2", [pp[0] / pp[2], pp[1] / pp[2], Q[2] @ np.array([0.6, -0.2, 1])],
      [0.6, -0.2, 2.0], tol=1e-12)
close("Exercise 30.3.1: 1-sigma and 2-sigma ellipse areas 0.18 pi, 0.72 pi", [0.18 * np.pi, 0.72 * np.pi], [0.5655, 2.2619],
      tol=1e-4)

# ---------------------------------------------------------------- Figure 30.3.2: flatland exact vs affine footprint
def flat_splat(cxz, phi_deg):
    cc = np.array(cxz, float)
    to_cam = -cc / np.linalg.norm(cc)
    ph = np.radians(phi_deg)
    Rr = np.array([[np.cos(ph), -np.sin(ph)], [np.sin(ph), np.cos(ph)]])
    n = Rr @ to_cam
    return cc, n, np.array([n[1], -n[0]])


def xi_of_u(uu, cc, t, s):
    p = cc[:, None] + s * np.atleast_1d(uu)[None] * t[:, None]
    return p[0] / p[1]


def u_of_xi(xi, cc, t, s):
    return -(cc[0] - cc[1] * xi) / (s * (t[0] - t[1] * xi))


cc, nn, tt = flat_splat((0.8, 2.0), 60.0)
S = 0.5
xi0 = cc[0] / cc[1]
dxi = (xi_of_u(1e-6, cc, tt, S) - xi_of_u(-1e-6, cc, tt, S))[0] / 2e-6
close("Figure 30.3.2: center pixel xi_0 = 0.4, |d xi/du| = 0.1346 (affine footprint std)", [xi0, abs(dxi)], [0.4, 0.13463],
      tol=1e-5)
img = xi_of_u(np.array([-2, -1, 1, 2.0]), cc, tt, S)
close("Figure 30.3.2: exact images of u = -2,-1,1,2: 0.606, 0.517, 0.241, 0.010", img, [0.6057, 0.5166, 0.2408, 0.0102],
      tol=1e-4)
xi = np.linspace(0.0, 0.9, 90001)
Gex = np.exp(-0.5 * u_of_xi(xi, cc, tt, S) ** 2)
Gaf = np.exp(-0.5 * ((xi - xi0) / dxi) ** 2)
close("Figure 30.3.2: max |G_exact - G_affine| = 0.176", np.max(np.abs(Gex - Gaf)), 0.1761, tol=1e-3)
close("Figure 30.3.2: the exact profile still peaks at xi_0 (the center maps to the center)", xi[np.argmax(Gex)], xi0, tol=2e-5)
# fronto-parallel splat: the homography is affine, exact = affine
cc0, nn0, tt0 = flat_splat((0.8, 2.0), 0.0)
cc1 = np.array([0.8, 2.0])
t1 = np.array([1.0, 0.0])
d1 = (xi_of_u(1e-6, cc1, t1, S) - xi_of_u(-1e-6, cc1, t1, S))[0] / 2e-6
check("Figure 30.3.2 (remark): a splat parallel to the image plane projects affinely (exact = affine)",
      np.allclose(xi_of_u(np.array([-2, -1, 1, 2.0]), cc1, t1, S), cc1[0] / cc1[1] + d1 * np.array([-2, -1, 1, 2.0])))

# ---------------------------------------------------------------- Figure 30.3.3: edge-on degeneration and the low-pass filter
fpx = 1000.0
s_small = 0.004
for phi, expect in [(0.0, 2.0), (60.0, 1.0), (85.0, 0.1743), (88.0, 0.0698)]:
    cc, nn, tt = flat_splat((0.0, 2.0), phi)
    w = np.abs(xi_of_u(np.array([1.0]), cc, tt, s_small) - xi_of_u(np.array([-1.0]), cc, tt, s_small))[0] * fpx / 2
    close(f"Figure 30.3.3(a): 1-sigma half-width at tilt {phi} deg = {expect} px (~ (s f/z) cos phi)", w, expect, tol=2e-4)
close("Figure 30.3.3: filter sigma = sqrt(2)/2 px (FilterSize = 0.707106, FilterInvSquare = 2 in auxiliary.h)",
      [np.sqrt(2) / 2, 1 / (np.sqrt(2) / 2) ** 2], [0.707106, 2.0], tol=1e-6)
# 88 deg, pixel centre 0.4 px from the projected center
cc, nn, tt = flat_splat((0.0, 2.0), 88.0)
xc = 0.0
pix = np.arange(-3, 4) + 0.4                    # pixel centres, offset from the splat center by 0.4 px
u_pix = u_of_xi(pix / fpx, cc, tt, s_small)
G_raw = np.exp(-0.5 * u_pix ** 2)
G_hat = np.maximum(G_raw, np.exp(-0.5 * 2.0 * (pix - xc) ** 2))     # rho = min(rho3d, 2 |x - c|^2)
print(f"   Figure 30.3.3(b): raw G at pixel centres: max {G_raw.max():.2e}; filtered max {G_hat.max():.3f}")
check("Figure 30.3.3(b): at 88 deg the raw splat value at every pixel centre is < 1e-6", G_raw.max() < 1e-6)
close("Figure 30.3.3(b): with the filter the nearest pixel (0.4 px away) gets exp(-0.4^2) = 0.852", G_hat.max(), np.exp(-0.16),
      tol=1e-9)

# ---------------------------------------------------------------- (30.3.6)-(30.3.7): depth distortion
rho_ = rng.uniform(0.05, 0.4, size=12)
T = np.cumprod(np.concatenate([[1.0], 1 - rho_[:-1]]))
wts = rho_ * T
zs = np.sort(rng.uniform(1.0, 6.0, size=12))
near, far = 0.2, 100.0
m = far / (far - near) * (1 - near / zs)
A = wts.sum()
double = sum(wts[i] * wts[j] * (m[i] - m[j]) ** 2 for i in range(12) for j in range(12))
half = sum(wts[i] * wts[j] * (m[i] - m[j]) ** 2 for i in range(12) for j in range(i))
mbar = (wts * m).sum() / A
var = (wts * (m - mbar) ** 2).sum() / A
close("(30.3.7): sum_{i,j} w_i w_j (m_i - m_j)^2 = 2 A^2 Var_w(m)", double, 2 * A ** 2 * var, tol=1e-12)
close("(30.3.7): the one-sided sum over j < i (paper App. A, forward.cu) = A^2 Var_w(m)", half, A ** 2 * var, tol=1e-12)
# forward.cu recursion: distortion += (m^2 A + M2 - 2 m M1) w with A, M1, M2 accumulated before i
dist, Aacc, M1, M2 = 0.0, 0.0, 0.0, 0.0
for i in range(12):
    dist += (m[i] ** 2 * Aacc + M2 - 2 * m[i] * M1) * wts[i]
    Aacc += wts[i]
    M1 += m[i] * wts[i]
    M2 += m[i] ** 2 * wts[i]
close("forward.cu: single-pass recursion = one-sided double sum", dist, half, tol=1e-12)
zz = sp.symbols("z", positive=True)
nn_, ff_ = sp.symbols("n f", positive=True)
mz = ff_ / (ff_ - nn_) * (1 - nn_ / zz)
check("(30.3.6): m = f/(f - n) (1 - n/z) is affine in 1/z", sp.simplify(sp.diff(mz.subs(zz, 1 / sp.Symbol("w")), sp.Symbol("w"), 2)) == 0)
close("(30.3.6): m(near) = 0, m(far) = 1", [float(mz.subs({nn_: 0.2, ff_: 100, zz: 0.2})), float(mz.subs({nn_: 0.2, ff_: 100, zz: 100}))],
      [0, 1], tol=1e-12)
# Exercise 30.3.3: two equal weights 0.5 at z = 2 and 4
m2 = far / (far - near) * (1 - near / np.array([2.0, 4.0]))
close("Exercise 30.3.3: m(2) = 0.9018, m(4) = 0.9519", m2, [0.90180, 0.95190], tol=1e-4)
close("Exercise 30.3.3: one-sided L2 distortion 0.25 (m_2 - m_1)^2 = 6.28e-4", 0.25 * (m2[1] - m2[0]) ** 2, 6.28e-4, tol=2e-6)
close("Exercise 30.3.3: the same pair at z = 20, 40: m-difference shrinks 10x", (far / (far - near)) * near * (1 / 20 - 1 / 40),
      (m2[1] - m2[0]) / 10, tol=1e-12)
# Mip-NeRF 360: continuous loss of a step function = pair term + (1/3) sum w_i^2 (s_{i+1} - s_i)
sb = np.sort(rng.uniform(0, 1, 7))
ww = rng.uniform(0, 0.3, 6)
grid = np.linspace(0, 1, 6001)
wfun = np.zeros_like(grid)
for i in range(6):
    wfun[(grid >= sb[i]) & (grid < sb[i + 1])] = ww[i] / (sb[i + 1] - sb[i])
dx = grid[1] - grid[0]
cont = np.sum(wfun[:, None] * wfun[None, :] * np.abs(grid[:, None] - grid[None, :])) * dx * dx
mid = (sb[1:] + sb[:-1]) / 2
disc = np.sum(ww[:, None] * ww[None, :] * np.abs(mid[:, None] - mid[None, :])) + np.sum(ww ** 2 * (sb[1:] - sb[:-1])) / 3
close("[Barron22] eq. (15) = eq. (14) for a step-function weight (w_i = weight of interval i)", cont, disc, tol=2e-3)

# ---------------------------------------------------------------- normal consistency: depth normal orientation in the code
# depth_to_normal: dx = points[2:, 1:-1] - points[:-2, 1:-1] (along image rows = v), dy along columns (= u); cross(dx, dy)
Kc = np.array([[500.0, 0, 50], [0, 500.0, 40], [0, 0, 1]])
Hh, Ww = 81, 101
gx, gy = np.meshgrid(np.arange(Ww, dtype=float), np.arange(Hh, dtype=float), indexing="xy")
rays = np.stack([gx, gy, np.ones_like(gx)], -1) @ np.linalg.inv(Kc).T
nplane = np.array([0.3, -0.2, -1.0])
nplane /= np.linalg.norm(nplane)                      # plane facing the camera (normal toward the camera: n_z < 0)
d0 = -3.0
zmap = d0 / (rays @ nplane)                            # z-depth of the plane <n, X> = d0
pts = zmap[..., None] * rays
ddx = pts[2:, 1:-1] - pts[:-2, 1:-1]
ddy = pts[1:-1, 2:] - pts[1:-1, :-2]
Nc = np.cross(ddx, ddy)
Nc /= np.linalg.norm(Nc, axis=-1, keepdims=True)
check("point_utils.depth_to_normal: cross(d/dv, d/du) gives the plane normal facing the camera (n_z < 0)",
      np.allclose(Nc, nplane, atol=1e-9))
close("depth_to_normal: depthmap * K^-1 (u, v, 1) uses z-depth (third coordinate of the point = depth)", pts[..., 2], zmap,
      tol=1e-12)

# code (train.py, gaussian_renderer): per-pixel normal loss = 1 - <sum_i w_i n_i, A N>, A = sum_i w_i; equals the paper's
# sum_i w_i (1 - <n_i, N>) when A = 1 (and differs by (1 - A)(1 + ...) otherwise)
wq = rng.uniform(0.05, 0.3, size=5)
nq = rng.normal(size=(5, 3)); nq /= np.linalg.norm(nq, axis=1, keepdims=True)
Nq = rng.normal(size=3); Nq /= np.linalg.norm(Nq)
for scale_A in (1.0, 0.7):
    wv = wq / wq.sum() * scale_A
    Aq = wv.sum()
    code_loss = 1 - (wv[:, None] * nq).sum(0) @ (Aq * Nq)
    paper_loss = np.sum(wv * (1 - nq @ Nq))
    if scale_A == 1.0:
        close("normal consistency: code 1 - <sum w n, A N> = paper sum w (1 - n.N) when A = 1", code_loss, paper_loss, tol=1e-12)
    else:
        check("normal consistency: code and paper differ when A < 1", abs(code_loss - paper_loss) > 1e-3)

# ---------------------------------------------------------------- Figure 30.3.4 / Exercise 30.3.4: shingles vs tangent splats
slope = 0.8
cosang = 1 / np.hypot(1, slope)
close("Exercise 30.3.4: fronto-parallel splats on a surface of slope 0.8: 1 - cos(38.66 deg) = 0.219", 1 - cosang, 0.2191,
      tol=1e-4)
close("Exercise 30.3.4: 45-degree surface: 1 - cos 45 = 0.293", 1 - np.cos(np.pi / 4), 0.2929, tol=1e-4)

# ---------------------------------------------------------------- SuGaR ideal SDF (Exercise 30.3.5)
s3 = sp.symbols("s_3", positive=True)
h = sp.symbols("h", real=True)                      # h = <p - mu, n>
dbar = sp.exp(-h ** 2 / (2 * s3 ** 2))
fbar = s3 * sp.sqrt(-2 * sp.log(dbar))
sym_equal("Exercise 30.3.5: SuGaR f_bar = s sqrt(-2 log d_bar) = |<p - mu, n>| (distance to the Gaussian's plane)",
          sp.simplify(fbar), sp.Abs(h))
# flat Gaussian: (p-mu)^T Sigma^-1 (p-mu) ~ <p-mu, n>^2 / s_3^2 when s_1, s_2 >> s_3 (SuGaR eq. 4)
s1v, s3v = 1.0, 0.01
pq = np.array([0.01, -0.01, 0.002])
mah = pq[0] ** 2 / s1v ** 2 + pq[1] ** 2 / s1v ** 2 + pq[2] ** 2 / s3v ** 2
close("SuGaR eq. (4): Mahalanobis distance ~ <p - mu, n>^2/s_3^2 near a flat Gaussian (relative error < 1%)",
      mah / (pq[2] ** 2 / s3v ** 2), 1.0, tol=1e-2)
# SuGaR code sign: density_grad = sum w Sigma^-1 (p - mu) = -grad d; normals = -normalize(density_grad) = +grad d/|grad d|
Sig = np.diag([1.0, 0.5, 0.01]) ** 2
muv = np.zeros(3)
dens = lambda p: np.exp(-0.5 * (p - muv) @ np.linalg.solve(Sig, p - muv))
p0 = np.array([0.2, 0.1, 0.008])
gnum = np.array([(dens(p0 + 1e-7 * e) - dens(p0 - 1e-7 * e)) / 2e-7 for e in np.eye(3)])
density_grad = dens(p0) * np.linalg.solve(Sig, p0 - muv)
code_normal = -density_grad / np.linalg.norm(density_grad)
close("sugar_model.py: -normalize(density_grad) = +grad d/|grad d| (toward increasing density)", code_normal,
      gnum / np.linalg.norm(gnum), tol=1e-6)

summary()
