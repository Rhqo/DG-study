"""Verification for Section 28.3 (Bundle Adjustment on a Manifold).

Run from the project root: ``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/verify/v-28-3-bundle-adjustment.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np  # noqa: E402

import dgba as B  # noqa: E402
import dglie as L  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(283)
cams, Xs = B.make_ba()
N, M = len(cams), len(Xs)
check("BA example: all points in front of all cameras",
      all(((T[:3, :3] @ Xs.T).T + T[:3, 3])[:, 2].min() > 1 for T in cams))
check("BA example: dim M = 6N + 3M = 150", 6 * N + 3 * M == 150)

# ---------------------------------------------------------------- Proposition 28.3.1
s, Q, u = 1.4, L.expSO3(np.radians(25) * np.array([0, 1.0, 0])), np.array([0.6, 0.2, -0.5])
c2, X2 = B.apply_similarity(cams, Xs, s, Q, u)
for T, T2 in zip(cams, c2):
    close("Prop 28.3.1: T' X' = s T X", (T2[:3, :3] @ X2.T).T + T2[:3, 3], s * ((T[:3, :3] @ Xs.T).T + T[:3, 3]),
          tol=1e-12)
close("Prop 28.3.1 / Figure 28.3.1: all projections unchanged", B.project_all(c2, X2), B.project_all(cams, Xs),
      tol=1e-13)
check("Figure 28.3.1: moving only the points changes the image",
      np.abs(B.project_all(cams, X2) - B.project_all(cams, Xs)).max() > 0.1)
s2, Q2, u2 = 0.7, L.expSO3([0.3, -0.2, 0.5]), np.array([-1.0, 0.4, 0.2])
ca, Xa = B.apply_similarity(*B.apply_similarity(cams, Xs, s2, Q2, u2), s, Q, u)
cb, Xb = B.apply_similarity(cams, Xs, s * s2, Q @ Q2, s * Q @ u2 + u)
close("Prop 28.3.1: it is a group action (S1 . (S2 . x) = (S1 S2) . x), points", Xa, Xb, tol=1e-12)
close("Prop 28.3.1: it is a group action, cameras", np.array(ca), np.array(cb), tol=1e-12)

# ---------------------------------------------------------------- generators (28.3.4) and the null space
J = B.jacobian(cams, Xs)
G = B.gauge_generators(cams, Xs)
close("(28.3.4): J G = 0", np.abs(J @ G).max(), 0.0, tol=1e-14)
check("(28.3.4): the 7 generators are linearly independent", np.linalg.matrix_rank(G) == 7)
eps = 1e-6
basis = [(e, np.zeros(3), 0.0) for e in np.eye(3)] + [(np.zeros(3), e, 0.0) for e in np.eye(3)] + \
        [(np.zeros(3), np.zeros(3), 1.0)]
Gnum = np.zeros_like(G)
for k, (w, uu, sg) in enumerate(basis):
    cp, Xp = B.apply_similarity(cams, Xs, np.exp(eps * sg), L.expSO3(eps * w), eps * uu)
    cm, Xm = B.apply_similarity(cams, Xs, np.exp(-eps * sg), L.expSO3(-eps * w), -eps * uu)
    for i in range(N):
        Gnum[6 * i:6 * i + 6, k] = (L.logSE3(cp[i] @ L.inv_T(cams[i])) - L.logSE3(cm[i] @ L.inv_T(cams[i]))) / (2 * eps)
    Gnum[6 * N:, k] = ((Xp - Xm) / (2 * eps)).ravel()
close("(28.3.4): generators = derivative of the action in the left-perturbation coordinates", Gnum, G, tol=1e-8)
sv = np.linalg.svd(J, compute_uv=False)
_, _, Vt = np.linalg.svd(J)
Qn = Vt[-7:].T
rel = np.linalg.norm(G - Qn @ (Qn.T @ G)) / np.linalg.norm(G)
check(f"Figure 28.3.2: null space of J = span of the generators (relative residual {rel:.1e} <= 1e-14)", rel <= 1e-14)
check("Figure 28.3.2: exactly 7 zero singular values (monocular)", int(np.sum(sv < 1e-10 * sv[0])) == 7)
close("Figure 28.3.2: the 143rd singular value is 0.039", sv[142], 0.0391, tol=2e-4)
check("Figure 28.3.2: singular values 144-150 are ~3e-16 or less", sv[143] < 4e-16)
check("Figure 28.3.2: all of the first 143 singular values >= 0.039", sv[:143].min() > 0.039)
Jd = B.jacobian(cams, Xs, depth=True)
svd_ = np.linalg.svd(Jd, compute_uv=False)
check("Figure 28.3.2 / Exercise 28.3.1: with metric depth, exactly 6 zero singular values",
      int(np.sum(svd_ < 1e-10 * svd_[0])) == 6)
check("with metric depth: the scale generator is no longer in the kernel", np.linalg.norm(Jd @ G[:, 6]) > 1)
close("with metric depth: rotation/translation generators still in the kernel", np.abs(Jd @ G[:, :6]).max(), 0.0,
      tol=1e-13)
Jf = J[:, 7:]
svf = np.linalg.svd(Jf, compute_uv=False)
check("Figure 28.3.2: gauge fixed (camera 1 + first translation coordinate of camera 2): full rank", svf[-1] > 1e-3)
close("Figure 28.3.2: smallest singular value with the gauge fixed = 0.0036", svf[-1], 0.00356, tol=5e-5)
check("Exercise 28.3.1: quotient dimension 143 (= number of nonzero singular values)",
      int(np.sum(sv > 1e-10 * sv[0])) == 143)

# Exercise 28.3.2: D phi(x) x = 0
for _ in range(5):
    x = rng.normal(size=3) + np.array([0, 0, 4.0])
    close("Exercise 28.3.2: Dphi(x_c) x_c = 0 (the ray is the kernel)", B.Dphi(x) @ x, np.zeros(2), tol=1e-15)
# Exercise 28.3.1: three non-collinear fixed points remove the gauge
fixed_pts = [0, 1, 2]
cols = [c for c in range(J.shape[1]) if not any(6 * N + 3 * j <= c < 6 * N + 3 * j + 3 for j in fixed_pts)]
svp = np.linalg.svd(J[:, cols], compute_uv=False)
check("Exercise 28.3.1: fixing three non-collinear points (COLMAP THREE_POINTS) leaves no gauge",
      svp[-1] > 1e-4 * svp[0])

# ---------------------------------------------------------------- LM with and without gauge fixing (Section 28.3)
r2 = np.random.default_rng(3)
obs = B.project_all(cams, Xs) + 0.002 * r2.normal(size=(N, M, 2))
c0 = [L.expSE3(np.r_[0.05 * r2.normal(size=3), 0.02 * r2.normal(size=3)]) @ T for T in cams]
X0 = Xs + 0.05 * r2.normal(size=Xs.shape)


def solve(cams_, Xs_, lam, fix, iters=10):
    cams_ = [T.copy() for T in cams_]
    Xs_ = Xs_.copy()
    hist, orth = [], []
    for _ in range(iters):
        r = B.residuals(cams_, Xs_, obs)
        hist.append(np.sqrt(np.mean(r ** 2)))
        Jm = B.jacobian(cams_, Xs_)
        if fix:
            keep = np.arange(7, Jm.shape[1])
            Jk = Jm[:, keep]
            d = np.zeros(Jm.shape[1])
            d[keep] = -np.linalg.solve(Jk.T @ Jk + lam * np.eye(len(keep)), Jk.T @ r)
        else:
            d = -np.linalg.solve(Jm.T @ Jm + lam * np.eye(Jm.shape[1]), Jm.T @ r)
            Qg, _ = np.linalg.qr(B.gauge_generators(cams_, Xs_))
            orth.append(np.linalg.norm(Qg.T @ d) / np.linalg.norm(d))
        for i in range(N):
            cams_[i] = L.expSE3(d[6 * i:6 * i + 6]) @ cams_[i]
        Xs_ = Xs_ + d[6 * N:].reshape(-1, 3)
    return cams_, Xs_, np.array(hist), np.array(orth)


cf, Xf, hf, _ = solve(c0, X0, 1e-12, True)
cl, Xl, hl, orth = solve(c0, X0, 1e-3, False)
close("Section 28.3: gauge-fixed GN converges to RMS 0.001557", hf[-1], 0.001557, tol=5e-7)
close("Section 28.3: free LM converges to the same RMS", hl[-1], hf[-1], tol=1e-9)
check(f"Section 28.3: LM steps are orthogonal to the gauge directions (max ratio {orth.max():.1e} <= 1e-10)",
      orth.max() <= 1e-10)
sc, R, t = B.umeyama(Xl, Xf)
close("Section 28.3: the two solutions differ by a similarity (alignment residual 3e-7)",
      B.rms((sc * (R @ Xl.T)).T + t, Xf), 3e-7, tol=2e-7)
close("Section 28.3: ... with scale 1.063", sc, 1.063, tol=1e-3)
# Exercise 28.3.3
rr = B.residuals(c0, X0, obs)
J0 = B.jacobian(c0, X0)
Qg, _ = np.linalg.qr(B.gauge_generators(c0, X0))
for lam in [1.0, 1e-3, 1e-5]:
    d = -np.linalg.solve(J0.T @ J0 + lam * np.eye(J0.shape[1]), J0.T @ rr)
    check(f"Exercise 28.3.3: LM step orthogonal to ker J (lambda = {lam:g})",
          np.linalg.norm(Qg.T @ d) < 1e-9 * np.linalg.norm(d))
Gg0 = B.gauge_generators(c0, X0)
Dm = np.diag(np.diag(J0.T @ J0))
for lam in [1e-3, 1.0]:
    d = -np.linalg.solve(J0.T @ J0 + lam * Dm, J0.T @ rr)
    check(f"Exercise 28.3.3 / Section 28.3: with D = diag(J^T J) the LM step is D-orthogonal to ker J (lambda = {lam:g})",
          np.linalg.norm(Gg0.T @ Dm @ d) < 1e-12 * np.linalg.norm(Gg0) * np.linalg.norm(Dm @ d))
d = -np.linalg.solve(J0.T @ J0 + 1e-3 * Dm, J0.T @ rr)
close("Section 28.3: ... but not Euclidean-orthogonal (gauge component ratio 0.41 at lambda = 1e-3)",
      np.linalg.norm(Qg.T @ d) / np.linalg.norm(d), 0.41, tol=0.006)
d_pinv = -np.linalg.pinv(J0, rcond=1e-12) @ rr
d_small = -np.linalg.solve(J0.T @ J0 + 1e-10 * np.eye(J0.shape[1]), J0.T @ rr)
close("Exercise 28.3.3: lambda -> 0 gives the minimum-norm (pseudo-inverse) step", d_small, d_pinv, tol=1e-6)

# ---------------------------------------------------------------- Schur complement (Exercise 28.3.4)
Jk = J0[:, 7:]
H = Jk.T @ Jk + 1e-9 * np.eye(Jk.shape[1])
g = Jk.T @ rr
nc = 6 * N - 7
Bm, E, Cm = H[:nc, :nc], H[:nc, nc:], H[nc:, nc:]
offdiag = Cm.copy()
for j in range(M):
    offdiag[3 * j:3 * j + 3, 3 * j:3 * j + 3] = 0
close("Exercise 28.3.4: the point block C is block-diagonal (3x3 blocks)", np.abs(offdiag).max(), 0.0, tol=0)
Cinv = np.zeros_like(Cm)
for j in range(M):
    Cinv[3 * j:3 * j + 3, 3 * j:3 * j + 3] = np.linalg.inv(Cm[3 * j:3 * j + 3, 3 * j:3 * j + 3])
Sm = Bm - E @ Cinv @ E.T
dc = np.linalg.solve(Sm, -g[:nc] + E @ Cinv @ g[nc:])
dp = Cinv @ (-g[nc:] - E.T @ dc)
close("Exercise 28.3.4: Schur complement solution = full solution", np.r_[dc, dp], np.linalg.solve(H, -g), tol=1e-8)
check("Exercise 28.3.4: full system 150 x 150, reduced camera system 30 x 30 (before fixing the gauge)",
      6 * N + 3 * M == 150 and 6 * N == 30)
check("Exercise 28.3.4: N = 1000, M = 1e5: 306000 parameters, 6000 x 6000 reduced system",
      6 * 1000 + 3 * 100000 == 306000)

# ---------------------------------------------------------------- scale drift (Figure 28.3.3, Exercise 28.3.5)
Ex = B.drift_experiment()
P = Ex["P"]
dr = np.array([T[:3, 3] for T in Ex["DR"]])
Xse, cse = B.pose_graph(Ex, "se3")
Xsim, csim = B.pose_graph(Ex, "sim")
pse = np.array([X[:3, 3] for X in Xse])
psim = np.array([X[:3, 3] for X in Xsim])
close("Figure 28.3.3: s_60 = 0.67", Ex["s"][-1], 0.674, tol=1e-3)
close("Figure 28.3.3: odometry end gap 3.8", np.linalg.norm(dr[-1] - dr[0]), 3.76, tol=0.01)
close("Figure 28.3.3: RMS error odometry 2.41", B.rms(dr, P), 2.41, tol=0.006)
close("Figure 28.3.3: RMS error SE(3) pose graph 2.73", B.rms(pse, P), 2.73, tol=0.006)
close("Figure 28.3.3: RMS error Sim(3) pose graph 0.27", B.rms(psim, P), 0.27, tol=0.006)
check("Figure 28.3.3: both pose graphs close the loop", np.linalg.norm(pse[-1] - pse[0]) < 0.05
      and np.linalg.norm(psim[-1] - psim[0]) < 0.05)
for name, A, val in [("odometry", dr, 1.12), ("SE(3)", pse, 0.86), ("Sim(3)", psim, 0.15)]:
    sc_, R_, t_ = B.umeyama(A, P)
    close(f"Exercise 28.3.5: similarity-aligned RMS, {name}", B.rms((sc_ * (R_ @ A.T)).T + t_, P), val, tol=0.006)
check("Figure 28.3.3: pose graph costs decrease", cse[-1] < cse[0] and csim[-1] < csim[0])
# consistency of (28.3.5): with noise-free data the true similarities give zero residuals
Ex0 = B.drift_experiment(rot_noise_deg=0.0, trans_noise=0.0)
Xtrue = [B.to_sim(L.make_T(Ex0["Rs"][k], Ex0["P"][k]), 1.0 / Ex0["s"][k]) for k in range(Ex0["K"] + 1)]
Zs = [B.to_sim(z) for z in Ex0["Z"]]
res = [L.logSim3(np.linalg.inv(Zs[k]) @ np.linalg.inv(Xtrue[k]) @ Xtrue[k + 1]) for k in range(Ex0["K"])]
# the odometry edges carry scale 1, so only the relative-scale component log(s_k / s_{k+1}) is left
close("(28.3.5): noise-free odometry residuals are zero except the scale component log(s_k/s_{k+1})",
      np.array([r_[:6] for r_ in res]), np.zeros((Ex0["K"], 6)), tol=1e-10)
close("(28.3.5): ... whose value is log(s_k/s_{k+1})", np.array([r_[6] for r_ in res]),
      np.log(Ex0["s"][:-1] / Ex0["s"][1:]), tol=1e-10)
Zl = B.to_sim(np.eye(4), 1.0 / Ex0["s"][-1])
close("(28.3.5): the loop edge (scale ratio s_0/s_60) has zero residual at the truth",
      L.logSim3(np.linalg.inv(Zl) @ np.linalg.inv(Xtrue[0]) @ Xtrue[-1]), np.zeros(7), tol=1e-10)


# ---------------------------------------------------------------- nerfstudio and BARF (Section 28.3)
def ns_exp_map_SE3(tv):
    lin, ang = tv[:3], tv[3:]
    th = np.linalg.norm(ang)
    if th < 1e-2:                        # nerfstudio's near_zero branch (Taylor series)
        a, b, c = 1 - th ** 2 / 6, 0.5 - th ** 2 / 24, 1.0 / 6 - th ** 2 / 120
        cs = 1 - th ** 2 / 2
    else:
        a, b, c = np.sin(th) / th, (1 - np.cos(th)) / th ** 2, (th - np.sin(th)) / th ** 3
        cs = np.cos(th)
    Rm = cs * np.eye(3) + a * L.hat3(ang) + b * np.outer(ang, ang)
    tr = a * lin + b * np.cross(ang, lin) + c * ang * (ang @ lin)
    return Rm, tr


def ns_exp_map_SO3xR3(tv):
    return L.expSO3(tv[3:]), tv[:3]


def ns_apply_to_camera(adj, c2w):
    dR, dt = adj
    return L.make_T(dR @ c2w[:3, :3], c2w[:3, 3] + dt)


def ns_apply_to_raybundle(adj, origin, direction):     # origins + dt, directions rotated by dR
    dR, dt = adj
    return origin + dt, dR @ direction


for _ in range(3):
    adj = ns_exp_map_SE3(rng.normal(size=6) * 0.4)
    c2w = L.expSE3(rng.normal(size=6))
    xc = rng.normal(size=3)
    o, dvec = ns_apply_to_raybundle(adj, c2w[:3, 3], c2w[:3, :3] @ xc)
    T2 = ns_apply_to_camera(adj, c2w)
    close("nerfstudio: apply_to_raybundle gives the rays of the camera from apply_to_camera (same product update)",
          np.r_[o, dvec], np.r_[T2[:3, 3], T2[:3, :3] @ xc], tol=1e-13)


for _ in range(4):
    tv = rng.normal(size=6) * 0.5
    Rm, tr = ns_exp_map_SE3(tv)
    close("nerfstudio exp_map_SE3 = (Exp(w), V(w) v) of (28.1.4)", L.make_T(Rm, tr), L.expSE3(tv), tol=1e-12)
    Rs_, ts_ = ns_exp_map_SO3xR3(tv)
    close("nerfstudio exp_map_SO3xR3 = (Exp(w), v)", ts_, tv[:3], tol=0)
    c2w = L.expSE3(rng.normal(size=6))
    out = ns_apply_to_camera((Rm, tr), c2w)
    check("nerfstudio apply_to_camera != group product Exp(xi) T_wc when the center is not the origin",
          np.abs(out - L.expSE3(tv) @ c2w).max() > 1e-3)
    close("nerfstudio apply_to_camera keeps the product structure (R, t) -> (dR R, t + dt)", out[:3, 3],
          c2w[:3, 3] + tr, tol=1e-14)
h = 1e-6
c2w = L.expSE3(rng.normal(size=6))
for k in range(6):
    e = np.zeros(6)
    e[k] = h
    dA = (ns_apply_to_camera(ns_exp_map_SE3(e), c2w) - ns_apply_to_camera(ns_exp_map_SE3(-e), c2w)) / (2 * h)
    dB = (ns_apply_to_camera(ns_exp_map_SO3xR3(e), c2w) - ns_apply_to_camera(ns_exp_map_SO3xR3(-e), c2w)) / (2 * h)
    close(f"nerfstudio: SE3 and SO3xR3 modes have the same differential at 0 (direction {k})", dA, dB, tol=1e-8)
tv = np.r_[0.3, 0.0, 0.0, 0.0, 0.0, 1.0]
check("nerfstudio: the two modes differ at second order (V(w) v != v)",
      np.abs(ns_exp_map_SE3(tv)[1] - ns_exp_map_SO3xR3(tv)[1]).max() > 0.05)


def barf_weight(alpha, k):
    if alpha < k:
        return 0.0
    if alpha - k < 1:
        return 0.5 * (1 - np.cos((alpha - k) * np.pi))
    return 1.0


for k in range(4):
    close(f"BARF (14): w_{k} continuous at alpha = k", barf_weight(k, k), 0.0, tol=0)
    close(f"BARF (14): w_{k} continuous at alpha = k + 1", barf_weight(k + 1 - 1e-12, k), 1.0, tol=1e-10)
    close(f"BARF (14): w_{k} = 1/2 at alpha = k + 1/2", barf_weight(k + 0.5, k), 0.5, tol=1e-15)


def barf_compose_pair(pa, pb):          # camera.py: pose_new(x) = pose_b o pose_a(x)
    return L.make_T(pb[:3, :3] @ pa[:3, :3], pb[:3, :3] @ pa[:3, 3] + pb[:3, 3])


Tcw = L.expSE3(rng.normal(size=6))
xi = rng.normal(size=6) * 0.3
wu = np.r_[xi[3:], xi[:3]]               # BARF orders (w, u)
refine = L.expSE3(np.r_[wu[3:], wu[:3]])
close("BARF: compose([pose_refine, pose]) = T_cw Exp(xi) (world-frame perturbation)",
      barf_compose_pair(refine, Tcw), Tcw @ L.expSE3(xi), tol=1e-13)
# a ray sample x_c (camera coordinates) sits at the world point (T Exp(xi))^{-1} x_c = Exp(-xi) x_w
xc = np.array([0.2, -0.1, 3.0])
xw = L.inv_T(Tcw)[:3, :3] @ xc + L.inv_T(Tcw)[:3, 3]
Jn = np.column_stack([(L.inv_T(Tcw @ L.expSE3(h * e)) @ np.r_[xc, 1] - L.inv_T(Tcw @ L.expSE3(-h * e)) @ np.r_[xc, 1])[:3]
                      / (2 * h) for e in np.eye(6)])
close("Section 28.3 (BARF): d(sample point)/d xi = -(I, -[x_w]_x), the (28.2.2) form with the world point",
      Jn, -np.hstack([np.eye(3), -L.hat3(xw)]), tol=1e-7)

summary()
