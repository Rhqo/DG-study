"""Verification for Section 28.4 (Interpolating and Averaging Poses).

Run from the project root: ``PYTHONPATH=tools python3 chapters/28-lie-groups-poses/verify/v-28-4-interpolating-averaging.py``
"""

import os
import pathlib
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))

import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

import dglie as L  # noqa: E402
from dgcheck import check, close, summary  # noqa: E402

rng = np.random.default_rng(284)


# ---------------------------------------------------------------- SLERP (Proposition 28.4.1, Figure 28.4.1)
def eigen_slerp(q0, q1, t):
    """Eigen/src/Geometry/Quaternion.h QuaternionBase::slerp, reimplemented."""
    one = 1 - np.finfo(float).eps
    d = q0 @ q1
    absD = abs(d)
    if absD >= one:
        s0, s1 = 1 - t, t
    else:
        th = np.arccos(absD)
        sth = np.sqrt(1 - absD * absD)
        s0, s1 = np.sin((1 - t) * th) / sth, np.sin(t * th) / sth
    if d < 0:
        s1 = -s1
    return s0 * q0 + s1 * q1


for _ in range(8):
    q0, q1 = rng.normal(size=4), rng.normal(size=4)
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    if q0 @ q1 < 0:
        q1 = -q1
    R0, R1 = L.q_to_R(q0), L.q_to_R(q1)
    Om = np.arccos(q0 @ q1)
    for s in [0.2, 0.5, 0.9]:
        qs = L.slerp(q0, q1, s)
        close("(28.4.1): SLERP stays on S^3", np.linalg.norm(qs), 1.0, tol=1e-14)
        close("(28.4.1): angle from q0 = s Omega (great circle at constant speed)", np.arccos(np.clip(qs @ q0, -1, 1)),
              s * Om, tol=1e-7)
        close("Prop 28.4.1: R(SLERP) = R0 Exp(s Log(R0^T R1))", L.q_to_R(qs), R0 @ L.expSO3(s * L.logSO3(R0.T @ R1)),
              tol=1e-12)
        close("Prop 28.4.1: Eigen's slerp (with the sign flip) = SLERP to the near representative",
              L.q_to_R(eigen_slerp(q0, -q1, s)), L.q_to_R(qs), tol=1e-12)
    close("Prop 28.4.1: d(R0, R1) = 2 Omega", L.rot_angle(R0.T @ R1), 2 * Om, tol=1e-9)
n = np.array([0.3, -0.5, 0.81])
n /= np.linalg.norm(n)
q0 = np.array([1.0, 0, 0, 0])
q1 = L.expq(np.radians(150) * n)
close("Figure 28.4.1: <q0, q1> = cos 75 deg", q0 @ q1, np.cos(np.radians(75)), tol=1e-15)
h = 1e-6
for s in [0.1, 0.5, 0.8]:
    sp_ = L.rot_angle(L.q_to_R(L.slerp(q0, q1, s)).T @ L.q_to_R(L.slerp(q0, q1, s + h))) / h
    close("Figure 28.4.1: SLERP speed 150 deg", np.degrees(sp_), 150, tol=1e-4)
    spw = L.rot_angle(L.q_to_R(L.slerp(q0, -q1, s, False)).T @ L.q_to_R(L.slerp(q0, -q1, s + h, False))) / h
    close("Figure 28.4.1: wrong-sign SLERP speed 210 deg", np.degrees(spw), 210, tol=1e-4)
close("Figure 28.4.1: wrong-sign SLERP ends at R1", L.q_to_R(L.slerp(q0, -q1, 1.0, False)), L.q_to_R(q1), tol=1e-12)
# NLERP speed: rotation angle 2 phi(s), phi = atan2(s sin Om, 1 - s + s cos Om)
s_ = sp.symbols("s")
Om = sp.rad(75)
phi = sp.atan2(s_ * sp.sin(Om), 1 - s_ + s_ * sp.cos(Om))
dphi = sp.diff(phi, s_)
v0 = float(sp.deg(2 * dphi.subs(s_, 0)))
vh = float(sp.deg(2 * dphi.subs(s_, sp.Rational(1, 2))))
close("Figure 28.4.1: NLERP speed at the ends = 2 sin 75 deg rad = 110.7 deg", v0, 110.69, tol=0.01)
close("Figure 28.4.1: NLERP speed at s = 1/2 = 4 tan 37.5 deg rad = 175.9 deg", vh, 175.86, tol=0.01)
close("Section 28.4: speed ratio 2 tan 37.5 / sin 75 = 1.59", vh / v0, 2 * np.tan(np.radians(37.5)) / np.sin(np.radians(75)),
      tol=1e-9)
close("Section 28.4: ... = 1.589", vh / v0, 1.589, tol=1e-3)
for s in [0.13, 0.5, 0.77]:
    qn = (1 - s) * q0 + s * q1
    w = L.logSO3(L.q_to_R(qn / np.linalg.norm(qn)))
    close("Figure 28.4.1: NLERP passes through rotations about the same axis", np.cross(w, n), np.zeros(3), tol=1e-12)
# Exercise 28.4.1
qz = L.expq([0, 0, np.radians(150)])
for s, slerp_deg, nlerp_deg in [(0.5, 75.0, 75.0), (0.25, 37.5, 33.02)]:
    close(f"Exercise 28.4.1: SLERP rotation angle at s = {s}", np.degrees(L.rot_angle(L.q_to_R(L.slerp(q0, qz, s)))),
          slerp_deg, tol=1e-9)
    qn = (1 - s) * q0 + s * qz
    close(f"Exercise 28.4.1: NLERP rotation angle at s = {s}", np.degrees(L.rot_angle(L.q_to_R(qn))), nlerp_deg, tol=0.01)
# Exercise 28.4.2
q1b = np.r_[np.cos(np.radians(120)), np.sin(np.radians(120)) * np.array([0, 0, 1.0])]
close("Exercise 28.4.2: q1 is the rotation by -120 deg about z", L.q_to_R(q1b), L.expSO3([0, 0, -np.radians(120)]),
      tol=1e-14)
tot = sum(L.rot_angle(L.q_to_R(L.slerp(q0, q1b, a, False)).T @ L.q_to_R(L.slerp(q0, q1b, a + 0.01, False)))
          for a in np.arange(0, 1, 0.01))
close("Exercise 28.4.2: without the sign check SLERP turns 240 deg", np.degrees(tot), 240, tol=1e-6)
tot = sum(L.rot_angle(L.q_to_R(L.slerp(q0, q1b, a)).T @ L.q_to_R(L.slerp(q0, q1b, a + 0.01))) for a in np.arange(0, 1, 0.01))
close("Exercise 28.4.2: with the sign check it turns 120 deg", np.degrees(tot), 120, tol=1e-6)

# ---------------------------------------------------------------- Proposition 28.4.2 (no bi-invariant metric)


def invariant_forms(Ads, dim):
    """Dimension of the space of symmetric G with A^T G A = G for all A in Ads."""
    idx = [(i, j) for i in range(dim) for j in range(i, dim)]
    rows = []
    for A in Ads:
        for (a, b) in idx:
            row = np.zeros(len(idx))
            for k, (i, j) in enumerate(idx):
                E = np.zeros((dim, dim))
                E[i, j] = E[j, i] = 1.0
                row[k] = (A.T @ E @ A - E)[a, b]
            rows.append(row)
    sv = np.linalg.svd(np.array(rows), compute_uv=False)
    return int(np.sum(sv < 1e-10 * sv[0])) + max(0, len(idx) - len(sv))


Ts = [L.expSE3(rng.normal(size=6)) for _ in range(4)]


def invariant_basis(Ads, dim):
    """Basis of the symmetric G with A^T G A = G for all A in Ads."""
    idx = [(i, j) for i in range(dim) for j in range(i, dim)]
    Es = []
    for (i, j) in idx:
        E = np.zeros((dim, dim))
        E[i, j] = E[j, i] = 1.0
        Es.append(E)
    M = np.vstack([np.column_stack([(A.T @ E @ A - E).ravel() for E in Es]) for A in Ads])
    _, sv, Vt = np.linalg.svd(M)
    null = Vt[np.sum(sv > 1e-10 * sv[0]):]
    return [sum(c * E for c, E in zip(vec, Es)) for vec in null]


Gse = invariant_basis([L.Ad_SE3(T) for T in Ts], 6)
check("Prop 28.4.2: Ad-invariant symmetric forms on se(3) form a 2-dim space (both indefinite or degenerate)",
      len(Gse) == 2)
check("Prop 28.4.2: every Ad-invariant form vanishes on the translation block, so none is positive definite",
      all(np.abs(G[:3, :3]).max() < 1e-10 for G in Gse))
check("Prop 28.4.2 (comparison): on so(3) the invariant forms are multiples of I (dimension 1)",
      len(invariant_basis([T[:3, :3] for T in Ts], 3)) == 1)
Gk = np.zeros((6, 6))
Gk[:3, 3:] = Gk[3:, :3] = np.eye(3)
check("Prop 28.4.2: the form <w, v'> + <v, w'> is Ad-invariant (but indefinite)",
      all(np.allclose(L.Ad_SE3(T).T @ Gk @ L.Ad_SE3(T), Gk) for T in Ts) and np.linalg.eigvalsh(Gk).min() < 0)
Gab = np.zeros((6, 6))
Gab[3:, 3:] = 1.3 * np.eye(3)
Gab[:3, 3:] = Gab[3:, :3] = 0.6 * np.eye(3)
check("Prop 28.4.2: a<w, w'> + b(<w, v'> + <v, w'>) is Ad-invariant (a = 1.3, b = 0.6)",
      all(np.allclose(L.Ad_SE3(T).T @ Gab @ L.Ad_SE3(T), Gab) for T in Ts))
ev = np.linalg.eigvalsh(Gab)
check("Prop 28.4.2: with b != 0 it is nondegenerate of signature (3, 3): a bi-invariant pseudo-Riemannian metric",
      int(np.sum(ev > 1e-12)) == 3 and int(np.sum(ev < -1e-12)) == 3)
check("Prop 28.4.2: every invariant form is a combination of <w, w'> and <w, v'> + <v, w'> (span check)",
      all(np.linalg.matrix_rank(np.vstack([G.ravel(), Gk.ravel(),
                                           np.block([[np.zeros((3, 3)), np.zeros((3, 3))],
                                                     [np.zeros((3, 3)), np.eye(3)]]).ravel()]), tol=1e-8) == 2
          for G in Gse))
Ss = [L.expSim3(rng.normal(size=7)) for _ in range(4)]
Gsim = invariant_basis([L.Ad_Sim3(S) for S in Ss], 7)
check("Prop 28.4.2: on sim(3) every invariant form also vanishes on the translation block (no bi-invariant metric)",
      all(np.abs(G[:3, :3]).max() < 1e-10 for G in Gsim))
w = np.array([0, 0, 1.0])
for tl in [1.0, 10.0, 100.0]:
    v = L.Ad_SE3(L.make_T(np.eye(3), [tl, 0, 0])) @ np.r_[0, 0, 0, w]
    close(f"Prop 28.4.2: Ad_(I,t) (0, w) = (t x w, w), |t| = {tl}", v, np.r_[np.cross([tl, 0, 0], w), w], tol=1e-12)


# ---------------------------------------------------------------- Figure 28.4.2, Proposition 28.4.3
def cam_pose(center, yaw):
    f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    r = np.cross(f, [0, 0, 1.0])
    return L.make_T(np.column_stack([r, np.cross(f, r), f]), center)


T0 = cam_pose(np.zeros(3), 0.0)
T1 = cam_pose(np.array([4.0, 2.0, 1.0]), np.radians(120))
ss = np.linspace(0, 1, 2001)
xi = L.logSE3(L.inv_T(T0) @ T1)
lie = [T0 @ L.expSE3(s * xi) for s in ss]
dR = L.logSO3(T0[:3, :3].T @ T1[:3, :3])
prod = [L.make_T(T0[:3, :3] @ L.expSO3(s * dR), (1 - s) * T0[:3, 3] + s * T1[:3, 3]) for s in ss]
C0, C1 = L.inv_T(T0), L.inv_T(T1)
dRc = L.logSO3(C0[:3, :3].T @ C1[:3, :3])
prodcw = [L.inv_T(L.make_T(C0[:3, :3] @ L.expSO3(s * dRc), (1 - s) * C0[:3, 3] + s * C1[:3, 3])) for s in ss]


def length(Tl):
    tot = 0.0
    for A, B in zip(Tl[:-1], Tl[1:]):
        tw = L.logSE3(L.inv_T(A) @ B)            # body twist over the step
        wv = L.logSO3(A[:3, :3].T @ B[:3, :3])
        tot += np.sqrt(wv @ wv + np.sum((B[:3, 3] - A[:3, 3]) ** 2))
        assert np.isclose(np.linalg.norm(tw[3:]), np.linalg.norm(wv))
    return tot


lens = [length(lie), length(prod), length(prodcw)]
close("Figure 28.4.2: length of (ii) = sqrt((2 pi/3)^2 + 21) = 5.04", lens[1], np.sqrt((2 * np.pi / 3) ** 2 + 21), tol=1e-6)
close("Figure 28.4.2: lengths 5.04 (ii), 5.88 (i), 7.19 (iii)", lens, [5.885, 5.038, 7.186], tol=2e-3)
check("Prop 28.4.3: (ii) is the shortest of the three", lens[1] < lens[0] < lens[2])
for P_, val in [(lie, 1.29), (prodcw, 2.01)]:
    c = np.array([T[:3, 3] for T in P_])
    u = (c[-1] - c[0]) / np.linalg.norm(c[-1] - c[0])
    close("Figure 28.4.2: max deviation of the center path from the chord", np.max(np.linalg.norm(np.cross(c - c[0], u), axis=1)),
          val, tol=6e-3)
# Prop 28.4.3: the speed in the left-invariant metric equals |omega_body|^2 + |c'|^2
for s in [0.2, 0.7]:
    A, B = prod[int(s * 2000)], prod[int(s * 2000) + 1]
    tw = L.logSE3(L.inv_T(A) @ B)
    TinvTdot = np.r_[A[:3, :3].T @ (B[:3, 3] - A[:3, 3]), L.logSO3(A[:3, :3].T @ B[:3, :3])]
    close("Prop 28.4.3: T^{-1} dT = (R^T dc, body omega) to first order", tw, TinvTdot, tol=1e-5)
    close("Prop 28.4.3: |R^T dc| = |dc|", np.linalg.norm(A[:3, :3].T @ (B[:3, 3] - A[:3, 3])),
          np.linalg.norm(B[:3, 3] - A[:3, 3]), tol=1e-15)
# product metric: the geodesic is the pair (geodesic of SO(3), straight line); perturbing increases the length
for amp in [0.05, 0.2]:
    bump = [L.make_T(T[:3, :3], T[:3, 3] + amp * np.sin(np.pi * s) * np.array([0, 0, 1.0])) for T, s in zip(prod, ss)]
    check(f"Prop 28.4.3: a perturbed center path is longer (amplitude {amp})", length(bump) > lens[1])
# Lie-group interpolation: invariance under left/right multiplication and inversion
A_, B_ = L.expSE3(rng.normal(size=6)), L.expSE3(rng.normal(size=6))
for s in [0.0, 0.3, 0.8, 1.0]:
    Ts_ = T0 @ L.expSE3(s * xi)
    left = L.expSE3(s * L.logSE3(T1 @ L.inv_T(T0))) @ T0
    close("Section 28.4: T0 Exp(s Log(T0^{-1} T1)) = Exp(s Log(T1 T0^{-1})) T0", left, Ts_, tol=1e-11)
    X0, X1 = A_ @ T0 @ B_, A_ @ T1 @ B_
    close("Section 28.4: the Lie interpolation of (A T0 B, A T1 B) is A T(s) B",
          X0 @ L.expSE3(s * L.logSE3(L.inv_T(X0) @ X1)), A_ @ Ts_ @ B_, tol=1e-10)
    inv = C0 @ L.expSE3(s * L.logSE3(L.inv_T(C0) @ C1))
    close("Section 28.4: interpolating T_cw gives the inverse curve", inv, L.inv_T(Ts_), tol=1e-11)
# Exercise 28.4.3: body-origin dependence of the product interpolation
Tbc = L.make_T(L.expSO3([0.1, -0.2, 0.3]), np.array([0.3, -0.1, 0.2]))
cam_from_prod = np.array([(T @ Tbc)[:3, 3] for T in prod])
u = (cam_from_prod[-1] - cam_from_prod[0]) / np.linalg.norm(cam_from_prod[-1] - cam_from_prod[0])
check("Exercise 28.4.3: IMU moved straight, camera center curved",
      np.max(np.linalg.norm(np.cross(cam_from_prod - cam_from_prod[0], u), axis=1)) > 0.05)
close("Exercise 28.4.3: c(s) = p(s) + R(s) t_bc", cam_from_prod,
      np.array([T[:3, 3] + T[:3, :3] @ Tbc[:3, 3] for T in prod]), tol=1e-13)
Wb0, Wb1 = T0, T1
for s in [0.25, 0.6]:
    lhs = Wb0 @ L.expSE3(s * L.logSE3(L.inv_T(Wb0) @ Wb1)) @ Tbc
    Wc0, Wc1 = Wb0 @ Tbc, Wb1 @ Tbc
    close("Exercise 28.4.3: Lie interpolation of IMU poses times T_bc = Lie interpolation of camera poses", lhs,
          Wc0 @ L.expSE3(s * L.logSE3(L.inv_T(Wc0) @ Wc1)), tol=1e-11)

# ---------------------------------------------------------------- cumulative B-spline (Definition 28.4.4, Figure 28.4.3)
k = 4
m = np.zeros((k, k))
for s_i in range(k):
    for n_ in range(k):
        m[s_i, n_] = sp.binomial(k - 1, n_) / sp.factorial(k - 1) * sum(
            (-1) ** (l_ - s_i) * sp.binomial(k, l_ - s_i) * (k - 1 - l_) ** (k - 1 - n_) for l_ in range(s_i, k))
mcum = np.array([[m[j:, n_].sum() for n_ in range(k)] for j in range(k)])
close("(28.4.3): cumulative blending matrix from [Sommer20, (18)-(19)]", mcum, L.M_CUM, tol=1e-14)
close("(28.4.3): lambda_0 = 1", [L.bspline_lambdas(u_)[0] for u_ in [0, 0.3, 1]], [1, 1, 1], tol=1e-15)
close("(28.4.3): lambda(0) = (1, 5/6, 1/6, 0)", L.bspline_lambdas(0.0), [1, 5 / 6, 1 / 6, 0], tol=1e-15)
close("(28.4.3): lambda(1) = (1, 1, 5/6, 1/6) (continuity across segments)", L.bspline_lambdas(1.0), [1, 1, 5 / 6, 1 / 6],
      tol=1e-15)
# in R^3 (translations only) the cumulative form equals sum B_i p_i of the uniform cubic B-spline
ps = rng.normal(size=(6, 3))
Tp = [L.make_T(np.eye(3), p) for p in ps]
for t in [0.0, 0.4, 1.7, 2.99]:
    i = int(np.floor(t))
    u_ = t - i
    B = np.array([(1 - u_) ** 3, 3 * u_ ** 3 - 6 * u_ ** 2 + 4, -3 * u_ ** 3 + 3 * u_ ** 2 + 3 * u_ + 1, u_ ** 3]) / 6
    close("(28.4.3): in R^3 the cumulative form = sum B_j(u) p_{i+j}", L.bspline_SE3(Tp, t)[:3, 3], B @ ps[i:i + 4],
          tol=1e-12)
yaws = np.array([0, 0.6, 1.4, 1.0, 0.2, -0.4, 0.3, 1.0])
cent = np.array([[0, 0, 0], [1.0, 0.6, 0.1], [1.7, 1.8, 0.2], [2.4, 2.9, 0.1], [3.6, 3.1, 0.0], [4.8, 2.5, 0.1],
                 [5.7, 2.9, 0.2], [6.3, 4.0, 0.1]])
Tc = [L.make_T(L.expSO3([0, 0, y]) @ L.expSO3([0.05 * np.sin(3 * y), 0, 0]), c) for c, y in zip(cent, yaws)]


def spl(t, Tl=Tc):
    return L.bspline_SE3(Tl, t)


def d_body(f, t, e):
    return L.logSE3(L.inv_T(f(t)) @ f(t + e)) / e


for knot in [1.0, 2.0, 3.0, 4.0]:
    e = 1e-3
    close(f"Definition 28.4.4: continuity at knot {knot}", spl(knot - 1e-12), spl(knot + 1e-12), tol=1e-8)
    vL = L.logSE3(L.inv_T(spl(knot - e)) @ spl(knot)) / e
    vR = L.logSE3(L.inv_T(spl(knot)) @ spl(knot + e)) / e
    close(f"Definition 28.4.4: C^1 at knot {knot} (one-sided body velocities)", vL, vR, tol=4e-3)
    aL = (L.logSE3(L.inv_T(spl(knot - e)) @ spl(knot)) - L.logSE3(L.inv_T(spl(knot - 2 * e)) @ spl(knot - e))) / e ** 2
    aR = (L.logSE3(L.inv_T(spl(knot + e)) @ spl(knot + 2 * e)) - L.logSE3(L.inv_T(spl(knot)) @ spl(knot + e))) / e ** 2
    close(f"Definition 28.4.4: C^2 at knot {knot} (one-sided body accelerations)", aL, aR, tol=2e-2)
# third derivative jumps (the spline is C^2 but not C^3)
knot, e = 2.0, 2e-3


def jerk(t0, sgn):
    vs = [L.logSE3(L.inv_T(spl(t0 + sgn * j * e)) @ spl(t0 + sgn * (j + 1) * e)) / e for j in range(4)]
    return sgn * (vs[2] - 2 * vs[1] + vs[0]) / e ** 2


check("Definition 28.4.4: the third derivative jumps at a knot",
      np.linalg.norm(jerk(knot, 1) - jerk(knot, -1)) > 0.1 * np.linalg.norm(jerk(knot, 1)))
Tmod = list(Tc)
Tmod[7] = L.make_T(L.expSO3([0, 0, 2.5]), cent[7] + np.array([2.0, -3.0, 1.0]))
for t in [0.2, 1.5, 2.9]:
    close("Definition 28.4.4: local support (T_7 does not affect t < 4 in spline time)", spl(t, Tmod), spl(t), tol=1e-12)
check("Definition 28.4.4: the spline does not pass through the control poses",
      min(np.linalg.norm(spl(float(i))[:3, 3] - cent[i + 1]) for i in range(5)) > 1e-3)
rates = [L.logSO3(Tc[kk][:3, :3].T @ Tc[kk + 1][:3, :3])[2] for kk in range(7)]
close("Figure 28.4.3: piecewise yaw rates (0.6, 0.8, -0.4, -0.8, -0.6, 0.7, 0.7)", rates,
      [0.6, 0.8, -0.4, -0.8, -0.6, 0.7, 0.7], tol=2e-3)
hh = 1e-4
wsp = [L.logSO3(spl(t)[:3, :3].T @ spl(t + hh)[:3, :3])[2] / hh for t in np.linspace(0, 5 - hh, 1001)]
close("Figure 28.4.3: spline yaw rate range [-0.733, 0.714]", [min(wsp), max(wsp)], [-0.733, 0.714], tol=2e-3)


# ---------------------------------------------------------------- rotation averaging (Figure 28.4.4)
def make_data(kout, seed, nin=20):
    r = np.random.default_rng(seed)
    Rs = L.random_rotation(r)
    ins = [Rs @ L.expSO3(np.radians(10) * r.normal(size=3)) for _ in range(nin)]
    outs = []
    for _ in range(kout):
        nn = r.normal(size=3)
        nn /= np.linalg.norm(nn)
        outs.append(Rs @ L.expSO3(np.radians(r.uniform(90, 170)) * nn))
    return Rs, ins, outs


errs = {0: [], 8: []}
for kout in [0, 8]:
    for t in range(30):
        Rs, ins, outs = make_data(kout, t)
        Rl = ins + outs
        Rc = L.mean_chordal(Rl)
        Rk, _ = L.mean_geodesic_L2(Rl, R0=Rc)
        R1, _ = L.mean_geodesic_L1(Rl, R0=Rk)
        errs[kout].append([np.degrees(L.rot_angle(Rs.T @ M)) for M in (Rc, Rk, R1)])
        if t < 3:
            close("(28.4.4): chordal SVD mean = quaternion eigenvector mean", Rc, L.mean_quat_eig(Rl), tol=1e-12)
            cost = lambda R: sum(np.linalg.norm(R - Ri) ** 2 for Ri in Rl)  # noqa: E731
            check("(28.4.4): the chordal mean minimizes sum |R - R_i|_F^2 (vs. 20 random perturbations)",
                  all(cost(Rc @ L.expSO3(0.05 * rng.normal(size=3))) > cost(Rc) for _ in range(20)))
            close("(28.4.4): Karcher mean satisfies sum Log(R^T R_i) = 0",
                  np.sum([L.logSO3(Rk.T @ Ri) for Ri in Rl], axis=0), np.zeros(3), tol=1e-9)
            vs = [L.logSO3(R1.T @ Ri) for Ri in Rl]
            close("(28.4.5): L1 mean is a Weiszfeld fixed point (sum v_i/|v_i| = 0)",
                  np.sum([v / np.linalg.norm(v) for v in vs], axis=0), np.zeros(3), tol=1e-8)
            for Rm, pw, nm in [(Rk, 2, "Karcher"), (R1, 1, "L1")]:
                def fcost(R, pw=pw):
                    return sum(L.rot_angle(R.T @ Ri) ** pw for Ri in Rl)
                c0 = fcost(Rm)
                perts = [L.expSO3(0.03 * rng.normal(size=3)) for _ in range(10)]
                check(f"(28.4.4): the {nm} mean is a local minimum of its cost",
                      all(fcost(Rm @ P) > c0 for P in perts))
Rs6, ins6, outs6 = make_data(6, 0)
Rl6 = ins6 + outs6
Rc6 = L.mean_chordal(Rl6)
Rk6, _ = L.mean_geodesic_L2(Rl6, R0=Rc6)
R16, _ = L.mean_geodesic_L1(Rl6, R0=Rk6)
close("Figure 28.4.4 (a): errors of chordal, Karcher, L1 = (4.0, 13.7, 2.4) deg, inlier-only Karcher 2.6",
      [np.degrees(L.rot_angle(Rs6.T @ M)) for M in (Rc6, Rk6, R16, L.mean_geodesic_L2(ins6)[0])], [3.98, 13.66, 2.35, 2.61],
      tol=0.006)
wk = L.logSO3(Rs6.T @ Rk6)
pull = np.sum([L.logSO3(Rs6.T @ R) for R in outs6], axis=0) / len(Rl6)
check("Figure 28.4.4 (a): the Karcher offset points along the outliers' net pull (angle < 15 deg)",
      np.degrees(np.arccos(wk @ pull / np.linalg.norm(wk) / np.linalg.norm(pull))) < 15)
e0, e8 = np.mean(errs[0], axis=0), np.mean(errs[8], axis=0)
close("Figure 28.4.4: k = 0 mean errors (chordal, Karcher, L1) = (3.77, 3.76, 4.13)", e0, [3.77, 3.76, 4.13], tol=0.006)
close("Figure 28.4.4: k = 8 mean errors = (7.56, 17.01, 5.12)", e8, [7.56, 17.01, 5.12], tol=0.006)
close("Figure 28.4.4: inlier noise scale 10 sqrt3 / sqrt20 = 3.87 deg", 10 * np.sqrt(3) / np.sqrt(20), 3.87, tol=0.005)
# quaternion sign problem (insight box)
er_rand, er_cons, er_eig = [], [], []
for t in range(30):
    Rs, ins, _ = make_data(0, t)
    qs = np.array([L.R_to_q(R) for R in ins])
    signs = np.where(np.random.default_rng(100 + t).random(20) < 0.5, -1, 1)
    mq = (qs * signs[:, None]).mean(0)
    er_rand.append(np.degrees(L.rot_angle(Rs.T @ L.q_to_R(mq))))
    qc = qs * np.sign(qs @ qs[0])[:, None]
    er_cons.append(np.degrees(L.rot_angle(Rs.T @ L.q_to_R(qc.mean(0)))))
    er_eig.append(np.degrees(L.rot_angle(Rs.T @ L.mean_quat_eig(ins))))
close("Insight box: random signs: mean error 46 deg, worst 177 deg", [np.mean(er_rand), np.max(er_rand)], [46.35, 176.54],
      tol=0.01)
close("Insight box: hemisphere-consistent signs and the eigenvector method: 3.8 deg", [np.mean(er_cons), np.mean(er_eig)],
      [3.77, 3.77], tol=0.006)
Rs, ins, _ = make_data(0, 12)
close("Insight box: R* rotation angle 179.5 deg (seed 12)", np.degrees(L.rot_angle(Rs)), 179.5, tol=0.05)
qs = np.array([L.R_to_q(R) for R in ins])
check("Insight box: R_to_q returns q0 >= 0", np.all(qs[:, 0] >= 0))
close("Insight box: q0 >= 0 rule, componentwise mean: error 42.0 deg", np.degrees(L.rot_angle(Rs.T @ L.q_to_R(qs.mean(0)))),
      42.04, tol=0.01)
close("Insight box: eigenvector method: 2.9 deg", np.degrees(L.rot_angle(Rs.T @ L.mean_quat_eig(ins))), 2.91, tol=0.01)
check("Insight box: sum q q^T is invariant under q -> -q",
      np.allclose(sum(np.outer(q, q) for q in qs), sum(np.outer(-q, -q) for q in qs)))
# Insight box: the chordal L2 "force" d/dd [8 sin^2(d/2)] = 4 sin d, largest at pi/2 and 0 at pi
dd = sp.symbols("d", positive=True)
check("Insight box: d/dd 8 sin^2(d/2) = 4 sin d", sp.simplify(sp.diff(8 * sp.sin(dd / 2) ** 2, dd) - 4 * sp.sin(dd)) == 0)
check("Insight box: the chordal force 4 sin d is maximal at d = pi/2 and vanishes at d = pi",
      sp.solve(sp.diff(4 * sp.sin(dd), dd), dd)[0] == sp.pi / 2 and 4 * sp.sin(sp.pi) == 0)
# Exercise 28.4.4
for th, val in [(10, 0.2465), (90, 2.0), (180, 2.8284)]:
    R = L.expSO3([0, 0, np.radians(th)])
    close(f"Exercise 28.4.4: |I - R|_F at {th} deg", np.linalg.norm(np.eye(3) - R), val, tol=1e-4)
    close(f"Exercise 28.4.4: = 2 sqrt2 sin(theta/2) at {th} deg", np.linalg.norm(np.eye(3) - R),
          2 * np.sqrt(2) * np.sin(np.radians(th) / 2), tol=1e-14)
close("Exercise 28.4.4: ratio 8 sin^2(th/2)/th^2 -> 2 and 8/pi^2 = 0.81", [8 * np.sin(1e-4 / 2) ** 2 / 1e-8, 8 / np.pi ** 2],
      [2.0, 0.8106], tol=1e-4)
# Exercise 28.4.5 and (28.4.6) gauge
R12, R23 = L.expSO3([0, 0, np.radians(30)]), L.expSO3([0, 0, np.radians(40)])
R31 = L.expSO3([np.radians(2), 0, 0]) @ L.expSO3([0, 0, np.radians(-70)])
close("Exercise 28.4.5: cycle error 2 deg", np.degrees(L.rot_angle(R31 @ R23 @ R12)), 2.0, tol=1e-12)
Rabs = [L.random_rotation(rng) for _ in range(3)]
Q = L.random_rotation(rng)
pairs = [(0, 1, R12), (1, 2, R23), (2, 0, R31)]
cost = lambda Rl: sum(L.rot_angle(Rij.T @ Rl[j] @ Rl[i].T) ** 2 for i, j, Rij in pairs)  # noqa: E731
close("Exercise 28.4.5 / (28.4.6): the cost is invariant under R_i -> R_i Q (3-dim gauge)", cost([R @ Q for R in Rabs]),
      cost(Rabs), tol=1e-10)
Rt = [L.expSO3(rng.normal(size=3)) for _ in range(3)]
close("Section 28.4: exact relative rotations satisfy cycle consistency",
      (Rt[0] @ Rt[2].T) @ (Rt[2] @ Rt[1].T) @ (Rt[1] @ Rt[0].T), np.eye(3), tol=1e-13)

summary()
