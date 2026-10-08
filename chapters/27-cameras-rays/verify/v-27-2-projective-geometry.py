"""Verification for Section 27.2 (Projective Geometry for Multi-View Vision).

Checks:
  - (27.2.1) P = K[R | T] has rank 3 and kernel (c, 1); dehomogenized P X = phi(R x + T) for t_z > 0;
    x and 2c - x give the same image point (front/back is lost); the induced map is a submersion
  - (27.2.2) vanishing points K R d, vanishing line K^{-T} R n, orthogonal directions and (K K^T)^{-1}
  - PGL(3) = GL(3)/R^* has dimension 8 and PGL(3) ~ SL(3) (odd n)
  - (27.2.3) plane-induced homography K2 (R + T n^T / d) K1^{-1}, rotation homography K R K^{-1}
  - Proposition 27.2.2: 4 points in general position fix H (DLT null space of dimension 1), 3 points do not
  - Proposition 27.2.3 and (27.2.4): points by H, lines by H^{-T}, conics by H^{-T} C H^{-1}, dual conics by H C* H^T;
    joins and meets are cross products
  - Example 27.2.4: the 1-sigma ellipse of a Gaussian: dual conic contains Sigma - mu mu^T, transforms like (29.1.5)
  - (27.2.5) camera P: lines pull back to planes P^T l through the center; dual quadric outline P Q* P^T
  - Table of the hierarchy: invariants of Euclidean / similarity / affine / projective maps
  - Affine maps = stabilizer of the line at infinity
  - (27.2.6) NeRF NDC is a projective map: det M != 0, lines -> lines, center -> point at infinity, rays -> lines
    parallel to z', t' = 1 + n/z for o on the near plane, agreement with bmild/nerf ndc_rays and paper eqs (25)-(26);
    Mip-NeRF 360 contraction bends lines
  - Exercises 27.2.1-27.2.5
  - review additions: P(2c - x, 1) = -P(x, 1); Schur complement; tangency-cone check of (27.2.5); NDC on random lines
    and parallel pairs; Figure 27.2.1 numbers

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/verify/v-27-2-projective-geometry.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import sympy as sp

from dgcheck import check, close, sym_equal, summary

rng = np.random.default_rng(272)


def rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def dehom(p):
    return p[:-1] / p[-1]


def parallel(a, b, tol=1e-10):
    return np.linalg.norm(np.cross(a / np.linalg.norm(a), b / np.linalg.norm(b))) < tol


K = np.array([[800.0, 0, 320], [0, 780.0, 240], [0, 0, 1]])
R = rot([0.2, 1.0, -0.3], 0.5)
T = np.array([0.3, -0.2, 2.0])
c = -R.T @ T
P = K @ np.hstack([R, T[:, None]])

# ---------------------------------------------------------------- (27.2.1)
check("(27.2.1): rank P = 3", np.linalg.matrix_rank(P) == 3)
check("(27.2.1): P (c, 1) = 0 (kernel = camera center)", np.linalg.norm(P @ np.append(c, 1)) < 1e-9)
x = np.array([0.4, -0.3, 1.5])
t = R @ x + T
phi = np.array([K[0, 0] * t[0] / t[2] + K[0, 2], K[1, 1] * t[1] / t[2] + K[1, 2]])
close("(27.2.1): dehom(P (x,1)) = phi(R x + T)", dehom(P @ np.append(x, 1)), phi, tol=1e-9)
xm = 2 * c - x
close("(27.2.1): R(2c - x) + T = -t", R @ xm + T, -t, tol=1e-12)
close("(27.2.1): x and 2c - x have the same image point", dehom(P @ np.append(xm, 1)), phi, tol=1e-9)
# the induced map is a submersion on RP^3 minus the center: in the chart w = 1, D(dehom o P) has rank 2
h = 1e-6
Df = np.column_stack([(dehom(P @ np.append(x + h * e, 1)) - dehom(P @ np.append(x - h * e, 1))) / (2 * h) for e in np.eye(3)])
check("(27.2.1): differential of the induced map has rank 2", np.linalg.matrix_rank(Df, tol=1e-3) == 2)
check("(27.2.1): its kernel is x - c", np.linalg.norm(Df @ (x - c)) < 1e-4)

# ---------------------------------------------------------------- (27.2.2) vanishing points and lines
d1 = np.array([1.0, 0, 0])
d2 = np.array([0, 0, 1.0])
d3 = np.array([0, 1.0, 0])
v = [K @ R @ d for d in (d1, d2, d3)]
for d, vv in zip((d1, d2, d3), v):
    # parallel lines x0 + s d: images converge to v as s -> infinity
    x0 = rng.normal(size=3)
    far = P @ np.append(x0 + 1e8 * d, 1)
    check(f"(27.2.2): image of x0 + s d tends to the vanishing point (d = {d})", parallel(far, vv, tol=1e-6))
n_ground = np.array([0, 1.0, 0])            # ground plane normal (world y axis)
l_h = np.linalg.inv(K).T @ R @ n_ground
for d in (d1, d2, np.array([1.0, 0, 1.0])):
    check(f"(27.2.2): vanishing point of d = {d} (in the plane) lies on K^-T R n", abs(l_h @ (K @ R @ d)) < 1e-9)
omega = np.linalg.inv(K @ K.T)
for i in range(3):
    for j in range(i + 1, 3):
        close(f"(27.2.2): orthogonal directions: v{i+1}^T (K K^T)^-1 v{j+1} = 0", v[i] @ omega @ v[j], 0.0, tol=1e-9)

# ---------------------------------------------------------------- PGL(3) ~ SL(3)
for _ in range(5):
    H = rng.normal(size=(3, 3))
    lam = rng.uniform(0.3, 3) * rng.choice([-1, 1])
    S1 = H / np.cbrt(np.linalg.det(H))
    S2 = (lam * H) / np.cbrt(np.linalg.det(lam * H))
    close("PGL(3) ~ SL(3): H / det(H)^(1/3) does not depend on the scale of H", S1, S2, tol=1e-10)
    close("PGL(3) ~ SL(3): det = 1", np.linalg.det(S1), 1.0, tol=1e-10)
check("PGL(3): dim GL(3) - 1 = 8", 9 - 1 == 8)
check("PGL(2): -I has det 1, so SL(2) does not meet every scalar class only once", np.isclose(np.linalg.det(-np.eye(2)), 1.0))

# ---------------------------------------------------------------- (27.2.3) homographies
K1, K2 = K, np.array([[700.0, 0, 300], [0, 700.0, 250], [0, 0, 1]])
R12 = rot([0.1, 0.9, 0.2], 0.3)
T12 = np.array([0.5, 0.1, -0.1])
npl = np.array([0.1, -0.2, 1.0])
npl /= np.linalg.norm(npl)
dpl = 3.0
Hpl = K2 @ (R12 + np.outer(T12, npl) / dpl) @ np.linalg.inv(K1)
worst = 0
for _ in range(10):
    a, b = rng.normal(size=2)
    e1 = np.cross(npl, [1, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(npl, e1)
    t1 = dpl * npl + a * e1 + b * e2           # <n, t1> = d
    t2 = R12 @ t1 + T12
    worst = max(worst, np.abs(dehom(K2 @ t2) - dehom(Hpl @ (K1 @ t1))).max())
close("(27.2.3): plane homography K2 (R + T n^T/d) K1^-1 maps image 1 to image 2", worst, 0.0, tol=1e-9)
Hrot = K2 @ R12 @ np.linalg.inv(K1)
t1 = rng.normal(size=3) + [0, 0, 5]
close("(27.2.3): pure rotation homography K2 R K1^-1", dehom(K2 @ (R12 @ t1)), dehom(Hrot @ (K1 @ t1)), tol=1e-9)


def dlt_nullity(src, dst):
    rows = []
    for (x_, y_), (u_, v_) in zip(src, dst):
        p = np.array([x_, y_, 1.0])
        rows.append(np.concatenate([np.zeros(3), -p, v_ * p]))
        rows.append(np.concatenate([p, np.zeros(3), -u_ * p]))
    A = np.array(rows)
    s = np.linalg.svd(A, compute_uv=False)
    return 9 - np.sum(s > 1e-9 * s[0]), A


Htrue = np.array([[1.1, 0.2, 3.0], [-0.1, 0.9, 1.0], [0.001, 0.002, 1.0]])
src4 = np.array([[0, 0], [100, 0], [100, 80], [0, 80.0]])
dst4 = np.array([dehom(Htrue @ np.append(q, 1)) for q in src4])
nul4, A4 = dlt_nullity(src4, dst4)
check("Prop 27.2.2: 4 points in general position: DLT null space has dimension 1", nul4 == 1)
Hest = np.linalg.svd(A4)[2][-1].reshape(3, 3)
close("Prop 27.2.2: the null vector is H (up to scale)", Hest / Hest[2, 2], Htrue / Htrue[2, 2], tol=1e-8)
nul3, _ = dlt_nullity(src4[:3], dst4[:3])
check("Prop 27.2.2: 3 points leave a 3-dimensional null space", nul3 == 3)
src_col = np.array([[0, 0], [50, 0], [100, 0], [0, 80.0]])           # three collinear
dst_col = np.array([dehom(Htrue @ np.append(q, 1)) for q in src_col])
nulc, _ = dlt_nullity(src_col, dst_col)
check("Prop 27.2.2: 4 points with three collinear do not fix H (null space > 1)", nulc > 1)

# ---------------------------------------------------------------- duality: points vs lines
H = np.array([[1.2, 0.3, -0.4], [0.1, 0.8, 0.5], [0.3, -0.2, 1.0]])
p1, p2 = np.array([0.2, 0.1, 1.0]), np.array([1.1, 0.7, 1.0])
l = np.cross(p1, p2)
check("(27.2.4): l = p1 x p2 passes through p1, p2", abs(l @ p1) < 1e-12 and abs(l @ p2) < 1e-12)
p3 = 0.3 * p1 + 0.7 * p2
lH = np.linalg.inv(H).T @ l
check("Prop 27.2.3: H p_i lie on H^-T l", max(abs(lH @ (H @ q)) for q in (p1, p2, p3)) < 1e-12)
wrong = H @ l
check("Prop 27.2.3: H l does not pass through H p_i (lines are not points)", max(abs(wrong @ (H @ q)) for q in (p1, p2, p3)) > 0.05)
l2 = np.array([1.0, -2.0, 0.5])
q = np.cross(l, l2)
check("(27.2.4): meet of two lines = l1 x l2", abs(l @ q) < 1e-12 and abs(l2 @ q) < 1e-12)
# conics
Cc = np.array([[1.0, 0.2, -0.3], [0.2, 2.0, 0.1], [-0.3, 0.1, -1.0]])
th = np.linspace(0, 2 * np.pi, 7)
# points on the conic: solve for a few points
pts = []
for ang in th:
    dvec = np.array([np.cos(ang), np.sin(ang), 0.0])
    # p = o + s d with o = (0,0,1): quadratic in s
    o = np.array([0, 0, 1.0])
    a2, a1, a0 = dvec @ Cc @ dvec, 2 * o @ Cc @ dvec, o @ Cc @ o
    s = (-a1 + np.sqrt(a1**2 - 4 * a2 * a0)) / (2 * a2)
    pts.append(o + s * dvec)
Cp = np.linalg.inv(H).T @ Cc @ np.linalg.inv(H)
check("(27.2.4): conic C -> H^-T C H^-1", max(abs((H @ p) @ Cp @ (H @ p)) for p in pts) < 1e-9)
Cs = np.linalg.inv(Cc)
for p in pts[:3]:
    tl = Cc @ p                       # tangent line at p
    check("(27.2.4): tangent line l = C p satisfies l^T C^-1 l = 0 (dual conic)", abs(tl @ Cs @ tl) < 1e-9)
    tlH = np.linalg.inv(H).T @ tl
    close("(27.2.4): dual conic C* -> H C* H^T", tlH @ (H @ Cs @ H.T) @ tlH, 0.0, tol=1e-9)

# ---------------------------------------------------------------- Example 27.2.4: Gaussian ellipse as a conic
mu = np.array([1.0, 0.5])
Sig = np.array([[3.12, 1.52], [1.52, 1.37]])
Si = np.linalg.inv(Sig)
Cg = np.block([[Si, -(Si @ mu)[:, None]], [-(Si @ mu)[None, :], np.array([[mu @ Si @ mu - 1]])]])
Cg_star = np.linalg.inv(Cg)
Cg_star = Cg_star / (-Cg_star[2, 2])        # normalize so that the (3,3) entry is -1
close("Example 27.2.4: dual conic = [[Sigma - mu mu^T, -mu], [-mu^T, -1]]",
      Cg_star, np.block([[Sig - np.outer(mu, mu), -mu[:, None]], [-mu[None, :], np.array([[-1.0]])]]), tol=1e-9)
A = np.array([[0.7, -0.4], [0.3, 1.5]])
bvec = np.array([2.0, -1.0])
Haff = np.block([[A, bvec[:, None]], [np.zeros((1, 2)), np.ones((1, 1))]])
Cs2 = Haff @ Cg_star @ Haff.T
mu2 = A @ mu + bvec
Sig2 = A @ Sig @ A.T
close("Example 27.2.4: H C* H^T reproduces mu' = A mu + b and Sigma' = A Sigma A^T (29.1.5)",
      Cs2, np.block([[Sig2 - np.outer(mu2, mu2), -mu2[:, None]], [-mu2[None, :], np.array([[-1.0]])]]), tol=1e-9)
Cg2 = np.linalg.inv(Haff).T @ Cg @ np.linalg.inv(Haff)
close("Example 27.2.4: conic block transforms like (29.1.6): A^-T Sigma^-1 A^-1", Cg2[:2, :2],
      np.linalg.inv(A).T @ Si @ np.linalg.inv(A), tol=1e-9)

# ---------------------------------------------------------------- (27.2.5) camera: pullback of lines, pushforward of dual quadrics
lim = np.array([0.002, -0.001, 0.3])
plane = P.T @ lim
check("(27.2.5): back-projected plane P^T l contains the camera center", abs(plane @ np.append(c, 1)) < 1e-9)
# a point whose image lies on l lies on the plane
u0 = np.array([100.0, 0, 1])
u0[1] = -(lim[0] * u0[0] + lim[2]) / lim[1]
ray_pt = np.append(c + 2.5 * R.T @ np.linalg.inv(K) @ u0, 1)
check("(27.2.5): points projecting onto l lie on P^T l", abs(plane @ ray_pt) < 1e-8 and abs(lim @ (P @ ray_pt)) < 1e-8)
# ellipsoid outline: P Q* P^T
Sig3 = np.array([[0.30, 0.05, 0.02], [0.05, 0.20, -0.03], [0.02, -0.03, 0.10]])
mu3 = np.array([0.5, -0.2, 1.0])
S3i = np.linalg.inv(Sig3)
Q = np.block([[S3i, -(S3i @ mu3)[:, None]], [-(S3i @ mu3)[None, :], np.array([[mu3 @ S3i @ mu3 - 1]])]])
Qs = np.linalg.inv(Q)
Cout_star = P @ Qs @ P.T
Cout = np.linalg.inv(Cout_star)
# contour points: x on the ellipsoid with <grad, x - c> = 0
L = np.linalg.cholesky(Sig3)
worst = 0.0
cnt = 0
for ang in np.linspace(0, 2 * np.pi, 400, endpoint=False):
    for el in np.linspace(-np.pi / 2, np.pi / 2, 200):
        y = np.array([np.cos(el) * np.cos(ang), np.cos(el) * np.sin(ang), np.sin(el)])
        xe = mu3 + L @ y
        g = S3i @ (xe - mu3)
        if abs(g @ (xe - c)) / (np.linalg.norm(g) * np.linalg.norm(xe - c)) < 2e-3:
            pimg = P @ np.append(xe, 1)
            pimg = pimg / np.linalg.norm(pimg)
            worst = max(worst, abs(pimg @ Cout @ pimg) / np.linalg.norm(Cout))
            cnt += 1
check(f"(27.2.5): {cnt} sampled contour points of an ellipsoid lie on the conic (P Q* P^T)^-1", cnt > 20 and worst < 5e-3)

# ---------------------------------------------------------------- hierarchy table
def cross_ratio(a, b, c_, d):
    return ((c_ - a) * (d - b)) / ((c_ - b) * (d - a))


s4 = np.array([0.0, 1.0, 2.5, 4.0])
line_pts = np.array([[s, 0.5 * s + 1, 1.0] for s in s4])
Hp = np.array([[1.0, 0.4, 0.2], [-0.3, 1.2, 0.1], [0.2, 0.1, 1.0]])
img = np.array([dehom(Hp @ p) for p in line_pts])
dirv = img[-1] - img[0]
sp_img = (img - img[0]) @ dirv / (dirv @ dirv)
close("Table: projective maps preserve the cross-ratio", cross_ratio(*sp_img), cross_ratio(*s4), tol=1e-10)
ratio_before = (s4[1] - s4[0]) / (s4[2] - s4[0])
ratio_after = (sp_img[1] - sp_img[0]) / (sp_img[2] - sp_img[0])
check("Table: projective maps do not preserve length ratios on a line", abs(ratio_after - ratio_before) > 1e-3)
Aff = np.array([[1.3, 0.6, 0.2], [-0.2, 0.7, -0.4], [0, 0, 1.0]])
imgA = np.array([dehom(Aff @ p) for p in line_pts])
dA = imgA[-1] - imgA[0]
spA = (imgA - imgA[0]) @ dA / (dA @ dA)
close("Table: affine maps preserve length ratios on a line", (spA[1] - spA[0]) / (spA[2] - spA[0]), ratio_before, tol=1e-12)
u1, u2 = np.array([1.0, 0]), np.array([0.6, 0.8])
ang = lambda a, b: np.arccos(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
check("Table: affine maps do not preserve angles", abs(ang(Aff[:2, :2] @ u1, Aff[:2, :2] @ u2) - ang(u1, u2)) > 1e-2)
Sim = 1.7 * rot([0, 0, 1], 0.4)[:2, :2]
close("Table: similarities preserve angles", ang(Sim @ u1, Sim @ u2), ang(u1, u2), tol=1e-12)
check("Table: similarities scale lengths", abs(np.linalg.norm(Sim @ u1) - 1.0) > 0.5)
Euc = rot([0, 0, 1], 0.4)[:2, :2]
close("Table: Euclidean maps preserve lengths", np.linalg.norm(Euc @ u2), 1.0, tol=1e-12)
check("Table: dof 3, 4, 6, 8 in 2D; dim PGL(4) = 15, dim Sim(3) = 7", (3, 4, 6, 8, 16 - 1, 3 + 3 + 1) == (3, 4, 6, 8, 15, 7))
# affine = stabilizer of l_inf
linf = np.array([0, 0, 1.0])
check("Affine maps fix the line at infinity: Aff^-T l_inf ~ l_inf", parallel(np.linalg.inv(Aff).T @ linf, linf))
check("A general homography moves the line at infinity", not parallel(np.linalg.inv(Hp).T @ linf, linf))
# the vanishing line of the ground plane is the image of l_inf of that plane: rectification
Hrect = np.array([[1, 0, 0], [0, 1, 0], [*(l_h / l_h[2])]])
check("Affine rectification: H with last row l^T sends l to l_inf", parallel(np.linalg.inv(Hrect).T @ (l_h / l_h[2]), linf))

# ---------------------------------------------------------------- (27.2.6) NeRF NDC
W_, H_, fcam, n = 504.0, 378.0, 400.0, 1.0
ax_, ay_ = -fcam / (W_ / 2), -fcam / (H_ / 2)
M = np.array([[-ax_, 0, 0, 0], [0, -ay_, 0, 0], [0, 0, -1, -2 * n], [0, 0, -1, 0]])   # -a_x = 2 f / W


def ndc_point(X):
    return np.array([ax_ * X[0] / X[2], ay_ * X[1] / X[2], 1 + 2 * n / X[2]])


Xs = np.array([0.3, -0.2, -4.0])
close("(27.2.6): M (x,y,z,1) dehomogenizes to the NDC formula", dehom(M @ np.append(Xs, 1)), ndc_point(Xs), tol=1e-12)
close("(27.2.6): det M = -2 n a_x a_y != 0", np.linalg.det(M), -2 * n * ax_ * ay_, tol=1e-9)
check("(27.2.6): camera center (0,0,0,1) -> point at infinity (0,0,-2n,0)", np.allclose(M @ [0, 0, 0, 1.0], [0, 0, -2 * n, 0]))
img_inf = M @ np.array([0.3, -0.1, -1.0, 0.0])
close("(27.2.6): plane at infinity -> plane z' = 1", img_inf[2] / img_inf[3], 1.0, tol=1e-12)
close("(27.2.6): near plane z = -n -> z' = -1", ndc_point(np.array([0.1, 0.1, -n]))[2], -1.0, tol=1e-12)
# lines -> lines (generic line)
o_ = np.array([0.5, 0.2, -1.5])
d_ = np.array([-0.3, 0.4, -1.0])
Ls = np.array([ndc_point(o_ + s * d_) for s in np.linspace(0, 20, 30)])
res = np.linalg.svd(Ls - Ls.mean(0), compute_uv=False)
check("(27.2.6): a generic line stays a line in NDC (second singular value ~ 0)", res[1] < 1e-10 * res[0])
# rays through the center become parallel to z'
dray = np.array([0.2, -0.1, -1.0])
Lr = np.array([ndc_point(s * dray) for s in np.linspace(1.5, 50, 20)])
check("(27.2.6): a ray through the center becomes a line parallel to the z' axis", np.ptp(Lr[:, 0]) < 1e-12 and np.ptp(Lr[:, 1]) < 1e-12)
# NeRF's ndc_rays (bmild/nerf run_nerf_helpers.py) vs paper eqs (25)-(26) and the parameter t'
ro = np.array([0.1, -0.05, 0.0])
rd = np.array([0.15, 0.1, -1.0])
tn = -(n + ro[2]) / rd[2]
on = ro + tn * rd
o0 = -1. / (W_ / (2. * fcam)) * on[0] / on[2]
o1 = -1. / (H_ / (2. * fcam)) * on[1] / on[2]
o2 = 1. + 2. * n / on[2]
dd0 = -1. / (W_ / (2. * fcam)) * (rd[0] / rd[2] - on[0] / on[2])
dd1 = -1. / (H_ / (2. * fcam)) * (rd[1] / rd[2] - on[1] / on[2])
dd2 = -2. * n / on[2]
oN, dN = np.array([o0, o1, o2]), np.array([dd0, dd1, dd2])
worst = 0.0
for tt in np.linspace(0.0, 50.0, 26):
    X = on + tt * rd
    tprime = 1 - on[2] / (on[2] + tt * rd[2])                 # paper eq (15)
    worst = max(worst, np.abs(ndc_point(X) - (oN + tprime * dN)).max())
    close_ok = abs(tprime - (1 + n / X[2])) < 1e-12           # o on the near plane: t' = 1 + n/z = 1 - n/|z|
    if not close_ok:
        worst = 1.0
close("(27.2.6): ndc_rays origin/direction and t' = 1 - o_z/(o_z + t d_z) reproduce the NDC image of the ray", worst, 0.0, tol=1e-10)
close("(27.2.6): z' = 2 t' - 1 on the ray", ndc_point(on + 7 * rd)[2], 2 * (1 - on[2] / (on[2] + 7 * rd[2])) - 1, tol=1e-12)
# uniform t' in [0, 1] is uniform in disparity 1/|z| from 1/n to 0
tps = np.linspace(0, 0.99, 12)
zs = -n / (1 - tps)                       # from t' = 1 + n/z
check("(27.2.6): uniform t' = uniform disparity 1/|z| (linear in t')", np.allclose(np.diff(1 / np.abs(zs)), np.diff(1 / np.abs(zs))[0]))


def contract(xv):
    r_ = np.linalg.norm(xv)
    return xv if r_ <= 1 else (2 - 1 / r_) * xv / r_


Lc = np.array([contract(np.array([1.5, -3.0, 0.5]) + s * np.array([0.3, 1.0, 0.2])) for s in np.linspace(0, 10, 40)])
resc = np.linalg.svd(Lc - Lc.mean(0), compute_uv=False)
check("(27.2.6) remark: the Mip-NeRF 360 contraction bends a line (second singular value clearly > 0)", resc[1] > 1e-2 * resc[0])

# ---------------------------------------------------------------- Exercises
# Ex 27.2.1: K = [[500,0,320],[0,500,240],[0,0,1]], R = I: vanishing point of d = (1, 0, 1) is (820, 240)
K5 = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
close("Ex 27.2.1: vanishing point of (1,0,1) = (820, 240)", dehom(K5 @ np.array([1.0, 0, 1])), np.array([820.0, 240.0]), tol=1e-12)
check("Ex 27.2.1: d = (1, 0, 0) has a vanishing point at infinity [500 : 0 : 0]", np.allclose(K5 @ [1.0, 0, 0], [500, 0, 0]))
lh5 = np.linalg.inv(K5).T @ np.array([0, 1.0, 0])
close("Ex 27.2.1: horizon of the plane with normal (0,1,0) is the row u_y = 240", lh5 / lh5[1], np.array([0, 1.0, -240.0]), tol=1e-12)
# Ex 27.2.2: H = diag(2, 1, 1): the line u_x = 1, i.e. l = (1, 0, -1), goes to (1/2, 0, -1) ~ u_x = 2
H2 = np.diag([2.0, 1.0, 1.0])
l1 = np.array([1.0, 0, -1.0])
close("Ex 27.2.2: H^-T l = (0.5, 0, -1) (the line u_x = 2)", np.linalg.inv(H2).T @ l1, np.array([0.5, 0, -1.0]), tol=1e-12)
close("Ex 27.2.2: H l = (2, 0, -1) is the wrong line u_x = 1/2", H2 @ l1, np.array([2.0, 0, -1.0]), tol=1e-12)
# Ex 27.2.4: NDC of points on the optical axis at z = -1, -2, -4, -infinity: z' = -1, 0, 0.5, 1
close("Ex 27.2.4: NDC z' of z = -1, -2, -4", np.array([1 + 2 * n / z for z in (-1.0, -2.0, -4.0)]), np.array([-1.0, 0.0, 0.5]), tol=1e-12)
# Ex 27.2.5: dof counts
check("Ex 27.2.3: correspondences needed: Euclidean 2 (3 dof, 4 eq), similarity 2, affine 3, projective 4",
      [int(np.ceil(k / 2)) for k in (3, 4, 6, 8)] == [2, 2, 3, 4])
# Ex 27.2.5: affine H keeps the (3,3) entry -1 of the dual conic; a non-affine H does not
Hna = np.array([[1.0, 0.1, 0.2], [0.0, 1.1, -0.1], [0.15, -0.1, 1.0]])
Cna = Hna @ Cg_star @ Hna.T
check("Ex 27.2.5: H C* H^T keeps (3,3) = -1 for affine H", abs(Cs2[2, 2] + 1) < 1e-12)
check("Ex 27.2.5: a non-affine H changes the (3,3) entry", abs(Cna[2, 2] + 1) > 1e-2)
# the image conic's center is not the image of the old center for non-affine H
Cn_ = Cna / (-Cna[2, 2])
mu_new = -Cn_[:2, 2]
check("Ex 27.2.5: for non-affine H the conic center is not H(mu)", np.linalg.norm(mu_new - dehom(Hna @ np.append(mu, 1))) > 1e-3)

# ---------------------------------------------------------------- review additions (independent checks)
# Prop 27.2.1: P(2c - x, 1) = -P(x, 1): same point of RP^2, opposite sign of the third coordinate t_z
X1 = P @ np.append(x, 1)
X2 = P @ np.append(2 * c - x, 1)
close("Prop 27.2.1 (review): P(2c - x, 1) = -P(x, 1)", X2, -X1, tol=1e-9)
check("Prop 27.2.1 (review): the sign of the third coordinate (t_z) is what [P X] forgets", X1[2] > 0 > X2[2])
# Schur complement of the dual conic recovers Sigma (Example 27.2.4)
Cs_n = Cg_star / (-Cg_star[2, 2])
close("Example 27.2.4 (review): Schur complement C*_11 - C*_12 C*_21 / C*_22 = Sigma",
      Cs_n[:2, :2] - np.outer(Cs_n[:2, 2], Cs_n[2, :2]) / Cs_n[2, 2], Sig, tol=1e-9)
# PGL(3) -> SL(3), H -> H / det(H)^(1/3) is a homomorphism
worst_h = 0.0
for _ in range(5):
    A1, A2 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    nz = lambda Z: Z / np.cbrt(np.linalg.det(Z))
    worst_h = max(worst_h, np.abs(nz(A1 @ A2) - nz(A1) @ nz(A2)).max())
close("PGL(3) ~ SL(3) (review): the normalization map preserves products", worst_h, 0.0, tol=1e-10)
# (27.2.5) independently: the outline conic equals the tangency condition of rays from c (no dual quadric)
Cc4 = np.append(c, 1)
qv = Q[:3, :] @ Cc4
Mcone = np.outer(qv, qv) - (Cc4 @ Q @ Cc4) * Q[:3, :3]        # ray c + s d tangent iff d^T Mcone d = 0
Aimg = R.T @ np.linalg.inv(K)                                   # pixel p -> ray direction d
Cimg = Aimg.T @ Mcone @ Aimg
Cn1, Cn2 = Cout / np.linalg.norm(Cout), Cimg / np.linalg.norm(Cimg)
close("(27.2.5) (review): (P Q* P^T)^-1 = tangency-cone conic (up to sign)", min(np.abs(Cn1 - Cn2).max(), np.abs(Cn1 + Cn2).max()), 0.0, tol=1e-10)
# NDC: 300 random lines stay lines; generic parallel pairs stop being parallel; pairs with d_z = 0 stay parallel
worst_l, kept, tot = 0.0, 0, 0
for _ in range(300):
    o3 = rng.normal(size=3) + np.array([0, 0, -5.0])
    d3 = rng.normal(size=3)
    pts3 = np.array([o3 + ss * d3 for ss in np.linspace(-1, 1, 25)])
    pts3 = pts3[pts3[:, 2] < -0.2]
    if len(pts3) < 5:
        continue
    Y3 = np.array([ndc_point(q3) for q3 in pts3])
    sv3 = np.linalg.svd(Y3 - Y3.mean(0), compute_uv=False)
    worst_l = max(worst_l, sv3[1] / sv3[0])
    o4 = o3 + rng.normal(size=3)
    a3, b3 = ndc_point(o3 + 0.3 * d3) - ndc_point(o3), ndc_point(o4 + 0.3 * d3) - ndc_point(o4)
    if np.all(np.isfinite(a3)) and np.all(np.isfinite(b3)) and o3[2] + 0.3 * d3[2] < 0 and o4[2] + 0.3 * d3[2] < 0 and o4[2] < 0:
        tot += 1
        kept += parallel(a3, b3, tol=1e-6)
check("Prop 27.2.5 (review): 300 random lines stay lines in NDC", worst_l < 1e-12)
check(f"Sec NDC (review): parallel lines are generally not parallel in NDC ({kept} of {tot} pairs stay parallel)", kept == 0 and tot > 100)
o5, d5, o6 = np.array([0.2, 0.1, -3.0]), np.array([1.0, 0.5, 0.0]), np.array([-0.4, 0.3, -6.0])
check("Sec NDC (review): lines parallel to the image plane (d_z = 0) stay parallel",
      parallel(ndc_point(o5 + d5) - ndc_point(o5), ndc_point(o6 + d5) - ndc_point(o6)))

# ---------------------------------------------------------------- Figure 27.2.1 numbers (world y down, camera at height 1.5)
yaw, pitch = np.radians(30), np.radians(10)
fwd = np.array([np.sin(yaw) * np.cos(pitch), np.sin(pitch), np.cos(yaw) * np.cos(pitch)])
right = np.cross([0, 1.0, 0], fwd)
right /= np.linalg.norm(right)
down = np.cross(fwd, right)
Rf = np.vstack([right, down, fwd])
check("Figure 27.2.1: R is a rotation (no mirror)", np.isclose(np.linalg.det(Rf), 1.0) and np.allclose(Rf @ Rf.T, np.eye(3)))
vAf, vBf = dehom(K5 @ Rf @ np.array([1.0, 0, 0])), dehom(K5 @ Rf @ np.array([0, 0, 1.0]))
close("Figure 27.2.1: v_A = K R e_1 = (1199, 152)", vAf, np.array([1199.4, 151.8]), tol=0.1)
close("Figure 27.2.1: v_B = K R e_3 = (27, 152)", vBf, np.array([26.9, 151.8]), tol=0.1)
close("Figure 27.2.1: horizon 500 tan 10 deg = 88 px above the center", 240 - vAf[1], 500 * np.tan(np.radians(10)), tol=1e-9)
close("Figure 27.2.1: half horizontal FOV atan(320/500) = 32.6 deg", np.degrees(np.arctan(320 / 500)), 32.62, tol=0.01)

summary()
