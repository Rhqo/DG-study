"""Verification for Section 28.2 (Calculus on Lie Groups).

Run from the project root: ``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/verify/v-28-2-calculus-on-lie-groups.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np  # noqa: E402

import dglie as L  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(282)
H = 1e-6


def num_jac(f, n, h=H):
    return np.column_stack([(f(h * e) - f(-h * e)) / (2 * h) for e in np.eye(n)])


# ---------------------------------------------------------------- (28.2.1) left/right relation
for _ in range(5):
    R = L.random_rotation(rng)
    d = rng.normal(size=3) * 0.4
    close("(28.2.1) SO(3): Exp(R d) R = R Exp(d)", L.expSO3(R @ d) @ R, R @ L.expSO3(d), tol=1e-13)
    T = L.expSE3(rng.normal(size=6))
    xi = rng.normal(size=6) * 0.4
    close("(28.2.1) SE(3): Exp(Ad_T xi) T = T Exp(xi)", L.expSE3(L.Ad_SE3(T) @ xi) @ T, T @ L.expSE3(xi), tol=1e-12)
    S = L.expSim3(rng.normal(size=7) * 0.7)
    z = rng.normal(size=7) * 0.3
    close("(28.2.1) Sim(3): Exp(Ad_S z) S = S Exp(z)", L.expSim3(L.Ad_Sim3(S) @ z) @ S, S @ L.expSim3(z), tol=1e-11)
    f = lambda Tm: np.concatenate([Tm[:3, :3] @ np.array([0.3, -1, 2]) + Tm[:3, 3], [Tm[0, 1] ** 2]])  # noqa: E731
    JR = num_jac(lambda e: f(T @ L.expSE3(e)), 6)
    JL = num_jac(lambda e: f(L.expSE3(e) @ T), 6)
    close("(28.2.1): right Jacobian = left Jacobian Ad_T (numerical)", JR, JL @ L.Ad_SE3(T), tol=1e-7)

# ---------------------------------------------------------------- Proposition 28.2.2
for _ in range(5):
    R = L.random_rotation(rng)
    p = rng.normal(size=3)
    close("(28.2.2): d(R Exp(d) p)/dd = -R[p]_x", num_jac(lambda e: R @ L.expSO3(e) @ p, 3), -R @ L.hat3(p), tol=1e-8)
    close("(28.2.2): d(Exp(d) R p)/dd = -[Rp]_x", num_jac(lambda e: L.expSO3(e) @ R @ p, 3), -L.hat3(R @ p), tol=1e-8)
    T = L.expSE3(rng.normal(size=6))
    x = rng.normal(size=3)
    act = lambda Tm, y: Tm[:3, :3] @ y + Tm[:3, 3]  # noqa: E731
    JR = np.hstack([T[:3, :3], -T[:3, :3] @ L.hat3(x)])
    JL = np.hstack([np.eye(3), -L.hat3(act(T, x))])
    close("(28.2.2): d(T Exp(xi) x)/dxi = R(I, -[x]_x)", num_jac(lambda e: act(T @ L.expSE3(e), x), 6), JR, tol=1e-8)
    close("(28.2.2): d(Exp(xi) T x)/dxi = (I, -[Tx]_x)", num_jac(lambda e: act(L.expSE3(e) @ T, x), 6), JL, tol=1e-8)
    close("Derivation of Prop 28.2.2: R(I, -[x]_x) = (I, -[Tx]_x) Ad_T", JR, JL @ L.Ad_SE3(T), tol=1e-12)
    close("Prop 28.2.2: d(Tx)/dx = R", num_jac(lambda e: act(T, x + e), 3), T[:3, :3], tol=1e-8)
R = L.expSO3([0, 0, np.pi / 2])
p = np.array([1.0, 0, 0])
close("Exercise 28.2.1: -R[p]_x", -R @ L.hat3(p), [[0, 0, -1], [0, 0, 0], [0, -1, 0]], tol=1e-15)
close("Exercise 28.2.1: -[Rp]_x", -L.hat3(R @ p), [[0, 0, -1], [0, 0, 0], [1, 0, 0]], tol=1e-15)
close("Exercise 28.2.1: e3 x e2 = -e1", np.cross([0, 0, 1.0], [0, 1.0, 0]), [-1, 0, 0], tol=0)
A1, B1 = -R @ L.hat3(p), -L.hat3(R @ p)
check("Exercise 28.2.1: the two matrices differ in columns 1 and 2 and agree in column 3 (R e3 = e3)",
      np.abs(A1[:, 0] - B1[:, 0]).max() > 0.5 and np.abs(A1[:, 1] - B1[:, 1]).max() > 0.5
      and np.allclose(A1[:, 2], B1[:, 2]) and np.allclose(R[:, 2], [0, 0, 1]))


# ---------------------------------------------------------------- reprojection Jacobian (28.2.3), (28.2.4), g2o
def phi(xc):
    return xc[:2] / xc[2]


def Dphi(xc):
    x, y, z = xc
    return np.array([[1 / z, 0, -x / z ** 2], [0, 1 / z, -y / z ** 2]])


for _ in range(5):
    T = L.expSE3(rng.normal(size=6) * 0.5)
    X = rng.normal(size=3) + np.array([0, 0, 0.0])
    xc = T[:3, :3] @ X + T[:3, 3]
    if xc[2] < 0.5:
        X = X + T[:3, :3].T @ np.array([0, 0, 2.0 - xc[2]])
        xc = T[:3, :3] @ X + T[:3, 3]
    Jxi = Dphi(xc) @ np.hstack([np.eye(3), -L.hat3(xc)])
    close("(28.2.3): residual Jacobian wrt left (camera-frame) perturbation of T_cw",
          num_jac(lambda e: phi(L.expSE3(e)[:3, :3] @ xc + L.expSE3(e)[:3, 3]), 6), Jxi, tol=1e-8)
    close("(28.2.3): residual Jacobian wrt the point = Dphi R",
          num_jac(lambda e: phi(T[:3, :3] @ (X + e) + T[:3, 3]), 3), Dphi(xc) @ T[:3, :3], tol=1e-8)
    x, y, z = xc
    expanded = np.array([[1 / z, 0, -x / z ** 2, -x * y / z ** 2, 1 + x ** 2 / z ** 2, -y / z],
                         [0, 1 / z, -y / z ** 2, -(1 + y ** 2 / z ** 2), x * y / z ** 2, x / z]])
    close("(28.2.4): expanded 2x6 matrix", expanded, Jxi, tol=1e-13)
    fx, fy = 1.0, 1.0
    g2o = np.array([[x * y / z ** 2 * fx, -(1 + x * x / z ** 2) * fx, y / z * fx, -1 / z * fx, 0, x / z ** 2 * fx],
                    [(1 + y * y / z ** 2) * fy, -x * y / z ** 2 * fy, -x / z * fy, 0, -1 / z * fy, y / z ** 2 * fy]])
    close("(28.2.4) = -(g2o EdgeSE3ProjectXYZ::linearizeOplus), g2o columns ordered (w, v)",
          -g2o, expanded[:, [3, 4, 5, 0, 1, 2]], tol=1e-13)

# ---------------------------------------------------------------- Figure 28.2.1 and Exercise 28.2.2
c = np.array([3.0, 0.0, 1.2])
f = -c / np.linalg.norm(c)
r = np.cross(f, [0, 0, 1.0])
r /= np.linalg.norm(r)
Rwc = np.column_stack([r, np.cross(f, r), f])
T = L.make_T(Rwc, c)
xi = np.array([0, 0, 0, 0, 0, 0.6])
close("Figure 28.2.1: right perturbation keeps the center", (T @ L.expSE3(xi))[:3, 3], c, tol=1e-14)
close("Figure 28.2.1: right perturbation keeps the optical axis", (T @ L.expSE3(xi))[:3, 2], f, tol=1e-14)
close("Figure 28.2.1: left perturbation moves the center to (2.48, 1.69, 1.2)", (L.expSE3(xi) @ T)[:3, 3],
      [2.476, 1.694, 1.2], tol=1e-3)
xl = L.Ad_SE3(T) @ xi
close("Exercise 28.2.2: xi_l = (0, 0.6 f)", xl, np.concatenate([np.zeros(3), 0.6 * f]), tol=1e-14)
close("Exercise 28.2.2: f = (-0.928, 0, -0.371), 0.6 f = (-0.557, 0, -0.223)", [f, 0.6 * f],
      [[-0.9285, 0, -0.3714], [-0.5571, 0, -0.2228]], tol=1e-4)
close("Exercise 28.2.2: Exp(xi_l) T = T Exp(xi_r)", L.expSE3(xl) @ T, T @ L.expSE3(xi), tol=1e-13)
# Figure 28.2.1 (a): seen from behind along the optical axis (basis x_c, -y_c), the right perturbation turns the
# image axes clockwise by 0.6 rad about the unchanged center
Tr = T @ L.expSE3(xi)
view = lambda v3: np.array([v3 @ Rwc[:, 0], -(v3 @ Rwc[:, 1])])  # noqa: E731
close("Figure 28.2.1 (a): x_c of the rolled camera = (cos 0.6, -sin 0.6) in the view (clockwise)", view(Tr[:3, 0]),
      [np.cos(0.6), -np.sin(0.6)], tol=1e-14)
close("Figure 28.2.1: the left perturbation moves the center by 2 * 3 sin(0.3) = 1.77",
      np.linalg.norm((L.expSE3(xi) @ T)[:3, 3] - c), 2 * 3 * np.sin(0.3), tol=1e-13)
close("Figure 28.2.1: 2 * 3 sin(0.3) = 1.77", 2 * 3 * np.sin(0.3), 1.773, tol=1e-3)


# ---------------------------------------------------------------- Figure 28.2.2 (GN with mixed perturbations)
def make_problem(angle, center, seed=0):
    rr = np.random.default_rng(seed)
    Xc = np.c_[rr.uniform(-2, 2, 40), rr.uniform(-1.5, 1.5, 40), rr.uniform(4, 8, 40)]
    ax = np.array([0.3, 1.0, 0.2])
    ax /= np.linalg.norm(ax)
    Rw = L.expSO3(angle * ax)
    return (Rw @ Xc.T).T + center, Xc[:, :2] / Xc[:, 2:], L.inv_T(L.make_T(Rw, center))


def residual(Tm, Xw, u):
    xcs = (Tm[:3, :3] @ Xw.T).T + Tm[:3, 3]
    return (xcs[:, :2] / xcs[:, 2:] - u).ravel(), xcs


def jacobian(Tm, Xw, xcs, mode):
    rows = []
    for Xp, xp in zip(Xw, xcs):
        if mode == "L":
            rows.append(Dphi(xp) @ np.hstack([np.eye(3), -L.hat3(xp)]))
        else:
            rows.append(Dphi(xp) @ Tm[:3, :3] @ np.hstack([np.eye(3), -L.hat3(Xp)]))
    return np.vstack(rows)


def run(Xw, u, T0, jm, um, iters=20, lam=0.0):
    Tm = T0.copy()
    hist, Ts = [], []
    for _ in range(iters + 1):
        rr, xcs = residual(Tm, Xw, u)
        if np.any(xcs[:, 2] <= 0):
            return np.array(hist), Ts, True
        hist.append(np.sqrt(np.mean(rr ** 2)))
        Ts.append(Tm.copy())
        J = jacobian(Tm, Xw, xcs, jm)
        d = -np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ rr)
        Tm = L.expSE3(d) @ Tm if um == "L" else Tm @ L.expSE3(d)
    return np.array(hist), Ts, False


dT = L.expSE3(np.r_[0.15, -0.1, 0.2, np.radians(10) * np.array([0.6, -0.5, 0.62])])
res = {}
for key, ang, cen in [("a", np.radians(3), np.zeros(3)), ("b", np.radians(120), np.array([4.0, -2.0, 3.0]))]:
    Xw, u, Tcw = make_problem(ang, cen)
    for jm, um in [("L", "L"), ("R", "R"), ("R", "L"), ("L", "R")]:
        res[(key, jm, um)] = run(Xw, u, dT @ Tcw, jm, um)
    hL, TsL, _ = res[(key, "L", "L")]
    hR, TsR, _ = res[(key, "R", "R")]
    close(f"Figure 28.2.2 ({key}): consistent left and right GN give identical iterates (first 4)",
          np.array(TsL[:4]), np.array(TsR[:4]), tol=1e-10)
    check(f"Figure 28.2.2 ({key}): consistent GN below 1e-14 at iteration 4", hL[4] < 1e-14 and hL[3] > 1e-14)
    close(f"Figure 28.2.2 ({key}): initial RMS = 0.0997", hL[0], 0.0997, tol=1e-4)
for jm, um in [("R", "L"), ("L", "R")]:
    h, _, behind = res[("a", jm, um)]
    first = int(np.argmax(h < 1e-14))
    check(f"Figure 28.2.2 (a): mixed {jm}/{um} converges, first below 1e-14 at iteration 11", not behind and first == 11)
    h, _, behind = res[("b", jm, um)]
    check(f"Figure 28.2.2 (b): mixed {jm}/{um} puts points behind the camera within 5 iterations",
          behind and len(h) <= 6 and h[1] > h[0])
close("Figure 28.2.2 (b): mixed right-Jacobian/left-update peaks at RMS 13.7", res[("b", "R", "L")][0].max(), 13.72,
      tol=0.01)
for jm, um in [("R", "L"), ("L", "R")]:
    h = res[("a", jm, um)][0]
    rate = (h[10] / h[0]) ** (1 / 10)
    check(f"Figure 28.2.2 (a): mixed {jm}/{um} shrinks by about 1/20 per step (geometric mean {rate:.3f})",
          0.03 < rate < 0.08)
# Exercise 28.2.5: LM equivariance holds iff Ad_T is orthogonal (t = 0)
Xw, u, Tcw = make_problem(np.radians(40), np.array([1.0, 0.5, -0.3]))
_, TsL, _ = run(Xw, u, dT @ Tcw, "L", "L", iters=3, lam=1e-2)
_, TsR, _ = run(Xw, u, dT @ Tcw, "R", "R", iters=3, lam=1e-2)
check("Exercise 28.2.5: with damping and t != 0, left and right LM iterates differ",
      np.abs(TsL[2] - TsR[2]).max() > 1e-6)
Xw, u, Tcw = make_problem(np.radians(40), np.zeros(3))
T0 = L.make_T(L.expSO3([0.1, -0.2, 0.15]), np.zeros(3)) @ Tcw
T0[:3, 3] = 0.0
A = L.Ad_SE3(T0)
close("Exercise 28.2.5: Ad_T is orthogonal when t = 0", A.T @ A, np.eye(6), tol=1e-13)

# ---------------------------------------------------------------- Definition 28.2.3, Proposition 28.2.4
for _ in range(6):
    w = rng.normal(size=3)
    w *= 2.8 / np.linalg.norm(w) * rng.uniform(0.1, 1)
    Jn = num_jac(lambda e: L.logSO3(L.expSO3(w).T @ L.expSO3(w + e)), 3)
    close("(28.2.5)/(28.2.6): J_r closed form = numerical derivative", Jn, L.Jr(w), tol=1e-8)
    Jln = num_jac(lambda e: L.logSO3(L.expSO3(w + e) @ L.expSO3(w).T), 3)
    close("(28.2.5)/(28.2.6): J_l closed form = numerical derivative", Jln, L.Jl(w), tol=1e-8)
    close("(28.2.6): J_l(w) = J_r(-w)", L.Jl(w), L.Jr(-w), tol=1e-14)
    close("(28.2.6): J_l = J_r^T", L.Jl(w), L.Jr(w).T, tol=1e-14)
    Vcols = np.column_stack([L.expSE3(np.r_[e, w])[:3, 3] for e in np.eye(3)])   # translation of Exp((e_k, w))
    close("(28.2.6): J_l = V of (28.1.4)", L.Jl(w), Vcols, tol=1e-13)
    close("(28.2.6): J_r^{-1} closed form", L.Jr_inv(w) @ L.Jr(w), np.eye(3), tol=1e-12)
    close("Prop 28.2.4: J_l = Exp(w) J_r", L.Jl(w), L.expSO3(w) @ L.Jr(w), tol=1e-13)
    us = (np.arange(3000) + 0.5) / 3000
    close("Derivation: J_r = int_0^1 Exp(-u w) du", np.mean([L.expSO3(-uu * w) for uu in us], axis=0), L.Jr(w),
          tol=1e-7)
    # BCH errors
    m = rng.normal(size=3)
    m /= np.linalg.norm(m)
    errs = []
    for e in [1e-3, 1e-4]:
        tr = L.logSO3(L.expSO3(w) @ L.expSO3(e * m))
        errs.append(np.linalg.norm(tr - w - L.Jr_inv(w) @ (e * m)))
    check("(28.2.7): right BCH error is O(|d|^2) (ratio ~100 for 10x smaller d)", 50 < errs[0] / errs[1] < 200)
    tr = L.logSO3(L.expSO3(1e-4 * m) @ L.expSO3(w))
    check("(28.2.7): left BCH error is O(|d|^2)", np.linalg.norm(tr - w - L.Jl_inv(w) @ (1e-4 * m)) < 1e-7)
    # gradient relation dL/dw = J_r^T dL/dd
    a = rng.normal(size=(3, 3))
    loss = lambda Rm: np.sum(a * Rm)  # noqa: E731
    gw = num_jac(lambda e: np.array([loss(L.expSO3(w + e))]), 3)[0]
    gd = num_jac(lambda e: np.array([loss(L.expSO3(w) @ L.expSO3(e))]), 3)[0]
    close("Section 28.2: dL/dw = J_r(w)^T dL/dd", gw, L.Jr(w).T @ gd, tol=1e-7)
n = np.array([0.36, 0.48, 0.8])
for th in [0.3, 1.0, np.pi, 5.0, 2 * np.pi - 1e-3]:
    sv = np.sort(np.linalg.svd(L.Jr(th * n), compute_uv=False))
    s_perp = 2 * abs(np.sin(th / 2)) / th
    close(f"Figure 28.2.3 (b): singular values of J_r at theta = {th:.3f}", sv, np.sort([1, s_perp, s_perp]), tol=1e-12)
close("Figure 28.2.3 (b): 2/pi = 0.64", 2 / np.pi, 0.6366, tol=1e-4)
ths = np.linspace(1e-3, np.pi, 300)
check("Figure 28.2.3 (b): on [0, pi] the singular values stay in [2/pi, 1]",
      np.all(2 * np.sin(ths / 2) / ths >= 2 / np.pi - 1e-12))
w = np.array([0, 0, 2.0])
dl = np.array([0, 0, 0.1])
close("Exercise 28.2.3: J_r(w) w = w", L.Jr(w) @ w, w, tol=1e-14)
close("Exercise 28.2.3: parallel w, d: Exp(w + d) = Exp(w) Exp(d)", L.expSO3(w + dl), L.expSO3(w) @ L.expSO3(dl),
      tol=1e-14)

# ---------------------------------------------------------------- Proposition 28.2.5 (double cover)
for _ in range(10):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    close("Prop 28.2.5: R(q) = R(-q)", L.q_to_R(q), L.q_to_R(-q), tol=1e-15)
    Rq = L.q_to_R(q)
    q2 = L.R_to_q(Rq)
    check("Prop 28.2.5: R(q) = R(q') iff q' = +-q (recover q from R up to sign)",
          np.allclose(q2, q) or np.allclose(q2, -q))
    th = rng.uniform(0, np.pi)
    nn = rng.normal(size=3)
    nn /= np.linalg.norm(nn)
    close("(28.2.8): q = (cos th/2, sin th/2 n) -> Exp(th n)",
          L.q_to_R(np.r_[np.cos(th / 2), np.sin(th / 2) * nn]), L.expSO3(th * nn), tol=1e-14)
    q1, q2 = rng.normal(size=4), rng.normal(size=4)
    q1, q2 = q1 / np.linalg.norm(q1), q2 / np.linalg.norm(q2)
    close("(28.2.8): arccos|<q1, q2>| = d(R1, R2)/2", np.arccos(min(1, abs(q1 @ q2))),
          L.rot_angle(L.q_to_R(q1).T @ L.q_to_R(q2)) / 2, tol=1e-7)
    close("Derivation of Prop 28.2.5: R(q1)^T R(q2) = R(conj(q1) q2)", L.q_to_R(q1).T @ L.q_to_R(q2),
          L.q_to_R(L.qmul(L.qconj(q1), q2)), tol=1e-14)


# ---------------------------------------------------------------- Ceres QuaternionManifold (internal/ceres/manifold.cc)
def ceres_plus(x, d):
    nd = np.linalg.norm(d)
    if nd == 0:
        return x.copy()
    qd = np.r_[np.cos(nd), np.sin(nd) / nd * d]
    return L.qmul(qd, x)                          # QuaternionProduct(q_delta, x)


def ceres_minus(y, x):
    a = L.qmul(y, L.qconj(x))
    un = np.linalg.norm(a[1:])
    if un == 0:
        return np.zeros(3)
    return np.arctan2(un, a[0]) * a[1:] / un


def ceres_plus_jacobian(x):
    w, xx, yy, zz = x
    return np.array([[-xx, -yy, -zz], [w, zz, -yy], [-zz, w, xx], [yy, -xx, w]])


for _ in range(6):
    x = rng.normal(size=4)
    x /= np.linalg.norm(x)
    d = rng.normal(size=3) * 0.4
    close("Ceres QuaternionManifold::Plus = left multiplication by Exp(2 delta)",
          L.q_to_R(ceres_plus(x, d)), L.expSO3(2 * d) @ L.q_to_R(x), tol=1e-13)
    nd = np.linalg.norm(d)
    close("Ceres Plus = great circle of S^3: cos|d| x + sin|d| ((0, d/|d|) * x)", ceres_plus(x, d),
          np.cos(nd) * x + np.sin(nd) * L.qmul(np.r_[0, d / nd], x), tol=1e-14)
    close("Ceres PlusJacobian (as in manifold.cc) = numerical dPlus/ddelta at 0",
          num_jac(lambda e: ceres_plus(x, e), 3), ceres_plus_jacobian(x), tol=1e-9)
    y = rng.normal(size=4)
    y /= np.linalg.norm(y)
    close("Ceres: Plus(x, Minus(y, x)) = y", ceres_plus(x, ceres_minus(y, x)), y, tol=1e-12)
    close("Ceres: Minus(y, x) for y close to -x", np.linalg.norm(ceres_minus(-ceres_plus(x, 0.05 * np.array([1, 0, 0.])),
                                                                             x)), np.pi - 0.05, tol=1e-12)
for _ in range(4):
    x = rng.normal(size=4)
    x /= np.linalg.norm(x)
    d = rng.normal(size=3) * 0.3
    Rx = L.q_to_R(x)
    close("Ceres Plus in right-perturbation coordinates: R(Plus(x, d)) = R Exp(2 R^T d)", L.q_to_R(ceres_plus(x, d)),
          Rx @ L.expSO3(2 * Rx.T @ d), tol=1e-13)
    Sig = rng.normal(size=(3, 3))
    Sig = Sig @ Sig.T
    A2 = 2 * Rx.T                                   # delta_r = 2 R^T delta  (exact, linear)
    close("Ceres tangent covariance -> right radians: Sigma_r = 4 R^T Sigma R", A2 @ Sig @ A2.T, 4 * Rx.T @ Sig @ Rx,
          tol=1e-12)
check("Ceres: |Minus| > pi/2 (rotation angle > pi) when the scalar part of y x^{-1} is negative",
      np.linalg.norm(ceres_minus(-ceres_plus(np.r_[1.0, 0, 0, 0], np.array([0.0, 0.3, 0])), np.r_[1.0, 0, 0, 0]))
      > np.pi / 2)


# ---------------------------------------------------------------- retractions (Section 28.2)
def cayley(d):
    K = L.hat3(d)
    return np.linalg.solve(np.eye(3) - K / 2, np.eye(3) + K / 2)


def gtsam_cayley(omega):
    x, y, z = omega
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    f = 1.0 / (4.0 + x2 + y2 + z2)
    f2 = 2 * f
    return np.array([[(4 + x2 - y2 - z2) * f, (xy - 2 * z) * f2, (xz + 2 * y) * f2],
                     [(xy + 2 * z) * f2, (4 - x2 + y2 - z2) * f, (yz - 2 * x) * f2],
                     [(xz - 2 * y) * f2, (yz + 2 * x) * f2, (4 - x2 - y2 + z2) * f]])


for _ in range(4):
    d = rng.normal(size=3)
    close("GTSAM Rot3::CayleyChart::Retract = (I - K/2)^{-1}(I + K/2)", gtsam_cayley(d), cayley(d), tol=1e-13)
rs = np.random.default_rng(3)
R0 = L.random_rotation(rs)
m = rs.normal(size=3)
m /= np.linalg.norm(m)
diffs = {}
for e in [0.1, 0.01]:
    dd = e * m
    A = R0 @ L.expSO3(dd)
    q = L.R_to_q(R0)
    retr = {"cayley": R0 @ cayley(dd), "svd": L.project_SO3(R0 @ (np.eye(3) + L.hat3(dd))),
            "quat": L.q_to_R(q + 0.5 * L.qmul(q, np.r_[0, dd]))}
    for k, Rr in retr.items():
        diffs[(k, e)] = L.rot_angle(A.T @ Rr)
        check(f"retraction {k}: R0 at delta = 0 and first-order agreement", diffs[(k, e)] < 2 * e ** 2)
close("Section 28.2: |delta| = 0.1: Cayley differs from R Exp by 8.3e-5", diffs[("cayley", 0.1)], 8.32e-5, tol=1e-6)
close("Section 28.2: |delta| = 0.1: quaternion normalization differs by 8.3e-5", diffs[("quat", 0.1)], 8.32e-5, tol=1e-6)
close("Section 28.2: |delta| = 0.1: SVD projection differs by 3.3e-4", diffs[("svd", 0.1)], 3.31e-4, tol=2e-6)
for e in [0.1, 0.01]:
    dd = e * m
    for k, Rr, expected in [("cayley", R0 @ cayley(dd), 2 * np.arctan(e / 2)),
                            ("svd", L.project_SO3(R0 @ (np.eye(3) + L.hat3(dd))), np.arctan(e)),
                            ("quat", L.q_to_R(L.R_to_q(R0) + 0.5 * L.qmul(L.R_to_q(R0), np.r_[0, dd])),
                             2 * np.arctan(e / 2))]:
        close(f"Section 28.2: retraction {k} = R Exp(angle * delta/|delta|), angle as stated (|delta| = {e})",
              Rr, R0 @ L.expSO3(expected * m), tol=1e-13)
close("Section 28.2: |delta|^3/12 at 0.1 = 8.3e-5 and |delta|^3/3 = 3.3e-4", [0.1 ** 3 / 12, 0.1 ** 3 / 3],
      [8.33e-5, 3.33e-4], tol=1e-6)
for k in ["cayley", "svd", "quat"]:
    close(f"Section 28.2: retraction {k} differs at order |delta|^3 (ratio 1000)", diffs[(k, 0.1)] / diffs[(k, 0.01)],
          1000, tol=10)


# ---------------------------------------------------------------- Exercise 28.2.4 (over-parametrization)
def g(qv, pp):
    return L.q_to_R(qv) @ pp


q = rng.normal(size=4)
p1, p2 = np.array([1.0, 0.2, -0.4]), np.array([-0.3, 1.1, 0.5])
Jg = np.vstack([num_jac(lambda e: g(q + np.r_[e], pp), 4) for pp in [p1, p2]])
close("Exercise 28.2.4: Dg(q) q = 0", Jg @ q, np.zeros(6), tol=1e-8)
check("Exercise 28.2.4: one point gives rank 2", np.linalg.matrix_rank(num_jac(lambda e: g(q + e, p1), 4), 1e-7) == 2)
check("Exercise 28.2.4: two points give rank 3", np.linalg.matrix_rank(Jg, 1e-7) == 3)


# small monocular BA, N = 4 cameras: rank deficiency with quaternion-4 parameters vs 3-parameter retraction
def ba_problem(seed=7, N=4, M=12):
    rr = np.random.default_rng(seed)
    Xs = rr.uniform(-1, 1, (M, 3)) + np.array([0, 0, 5.0])
    cams = []
    for i in range(N):
        cen = np.array([1.5 * np.cos(i), 0.5 * np.sin(2 * i), 0.3 * i])
        z = Xs.mean(0) - cen
        z /= np.linalg.norm(z)
        xax = np.cross([0, 1.0, 0], z)
        xax /= np.linalg.norm(xax)
        Rwc_ = np.column_stack([xax, np.cross(z, xax), z])
        cams.append(L.inv_T(L.make_T(Rwc_, cen)))
    return cams, Xs


cams, Xs = ba_problem()


def residuals_quat(params, N, M):
    out = []
    for i in range(N):
        qv, tv = params[7 * i:7 * i + 4], params[7 * i + 4:7 * i + 7]
        Rm = L.q_to_R(qv)
        for j in range(M):
            X = params[7 * N + 3 * j:7 * N + 3 * j + 3]
            out.append(phi(Rm @ X + tv))
    return np.concatenate(out)


N, M = len(cams), len(Xs)
x0 = np.concatenate([np.r_[L.R_to_q(T[:3, :3]) * (1 + 0.3 * k), T[:3, 3]] for k, T in enumerate(cams)] + [Xs.ravel()])
Jq = num_jac(lambda e: residuals_quat(x0 + e, N, M), len(x0), h=1e-6)
svq = np.linalg.svd(Jq, compute_uv=False)
nullq = int(np.sum(svq < 1e-6 * svq[0]))


def residuals_tangent(dv):
    out = []
    for i, T in enumerate(cams):
        Tn = L.expSE3(dv[6 * i:6 * i + 6]) @ T
        for j in range(M):
            X = Xs[j] + dv[6 * N + 3 * j:6 * N + 3 * j + 3]
            out.append(phi(Tn[:3, :3] @ X + Tn[:3, 3]))
    return np.concatenate(out)


Jt = num_jac(residuals_tangent, 6 * N + 3 * M, h=1e-6)
svt = np.linalg.svd(Jt, compute_uv=False)
nullt = int(np.sum(svt < 1e-6 * svt[0]))
check(f"Exercise 28.2.4: BA with 4-parameter quaternions has null space 7 + N = 11 (found {nullq})", nullq == 7 + N)
check(f"Exercise 28.2.4: BA with the 3-parameter retraction has null space 7 (found {nullt})", nullt == 7)

summary()
