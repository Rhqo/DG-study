"""Verification for Section 27.3 (Epipolar Geometry and the Essential Manifold).

Checks:
  - (27.3.1) epipolar constraint pbar2^T [T]x R pbar1 = 0, coplanarity of t2, T, R t1; epipolar lines; epipoles
  - (27.3.2) F = K2^-T E K1^-1 has rank 2 and p2^T F p1 = 0
  - Proposition 27.3.1: E = [T]x R has singular values (|T|, |T|, 0); conversely U diag(s,s,0) V^T = [T]x R with
    T = s u3, R = U W V^T (W = R_z(-90 deg), as in COLMAP)
  - Proposition 27.3.2: the normalized essential space is 5-dimensional (Jacobian rank of (R, T) -> [T]x R on
    SO(3) x S^2; tangent space of the Demazure equations); Tron-Daniilidis representation E(Q) = R1^T [e_z]x R2 and its
    invariance under H_z; the H_pi elements give +-E
  - Proposition 27.3.3: four decompositions (R, T), (R, -T), (R', T), (R', -T) with R' = R_T(pi) R; cheirality signs
  - 5-point: 5 x 9 system has a 4-dimensional null space; 7-point: det(a F1 + (1 - a) F2) is a cubic with 1 or 3 real
    roots, one of which is the true F; 8-point: linear; nearest essential matrix U diag(m, m, 0) V^T
  - Hartley normalization: condition numbers and errors (numbers used in the text and Figure 27.3.3)
  - (27.3.4) triangulation depth error: fronto-parallel stereo sigma_z = sqrt(2) sigma_u z^2 / (f b), Monte Carlo;
    |J2 r1| = f b / z^2; values at the COLMAP thresholds 1.5 and 16 deg
  - Exercises 27.3.1-27.3.5
  - review additions: the H_pi signs (quotient is E/{+-1}), the SVD parametrization's rank and kernel, rank-3
    matrices rejected by the cubic equations, a 200-point cheirality vote, R_b = R_u3(pi) R_a, the 1.9-2.4 ratio range,
    COLMAP's forward-motion threshold as an angle, and the 2-dim family of solutions under pure rotation

Run from the project root: ``PYTHONPATH=tools python3 chapters/27-cameras-rays/verify/v-27-3-essential-manifold.py``
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

from dgcheck import check, close, summary

rng = np.random.default_rng(273)


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expso3(w):
    th = np.linalg.norm(w)
    if th < 1e-14:
        return np.eye(3) + hat(w)
    K = hat(w / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def rand_rot():
    return expso3(rng.normal(size=3) * 1.0)


def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


# ---------------------------------------------------------------- (27.3.1)
R = expso3(np.array([0.05, -0.3, 0.02]))
T = np.array([-1.0, 0.1, 0.2])
T = T / np.linalg.norm(T)
E = hat(T) @ R
worst = 0
for _ in range(50):
    t1 = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(3, 10)])
    t2 = R @ t1 + T
    p1, p2 = t1 / t1[2], t2 / t2[2]
    worst = max(worst, abs(p2 @ E @ p1), abs(np.linalg.det(np.column_stack([t2, T, R @ t1]))))
close("(27.3.1): pbar2^T E pbar1 = 0 and det(t2, T, R t1) = 0 for 50 points", worst, 0.0, tol=1e-12)
l2 = E @ p1
check("(27.3.1): epipolar line l2 = E pbar1 passes through pbar2", abs(l2 @ p2) < 1e-12)
e2 = T
e1 = R.T @ T
close("(27.3.1): epipoles: E e1 = 0, e2^T E = 0", np.abs(E @ e1).max() + np.abs(e2 @ E).max(), 0.0, tol=1e-12)
c2_in_1 = -R.T @ T
check("(27.3.1): e1 is the image of camera 2's center in camera 1", np.linalg.norm(np.cross(e1, c2_in_1)) < 1e-12)
# ray 1 projects onto l2: points at depth s along pbar1 land on l2, from e2 (s->0) to H_inf pbar1 (s->inf)
for s in (0.5, 2, 5, 1e6):
    q = R @ (s * p1) + T
    check(f"(27.3.1): point at depth {s:g} on ray 1 projects onto l2", abs(l2 @ (q / q[2])) < 1e-9)
check("(27.3.1): s -> infinity gives R pbar1 (vanishing point, infinite homography)", abs(l2 @ (R @ p1)) < 1e-12)

# ---------------------------------------------------------------- (27.3.2)
K1 = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
K2 = np.array([[700.0, 0, 300], [0, 710.0, 250], [0, 0, 1]])
F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
sF = np.linalg.svd(F, compute_uv=False)
check("(27.3.2): F has rank 2", sF[2] < 1e-12 * sF[0] and sF[1] > 1e-6 * sF[0])
close("(27.3.2): p2^T F p1 = 0", (K2 @ p2) @ F @ (K1 @ p1), 0.0, tol=1e-12)
check("(27.3.2): F has unequal nonzero singular values in general (not essential)", abs(sF[0] - sF[1]) > 1e-3 * sF[0])

# ---------------------------------------------------------------- Proposition 27.3.1
for _ in range(10):
    Tr = rng.normal(size=3)
    Rr = rand_rot()
    s = np.linalg.svd(hat(Tr) @ Rr, compute_uv=False)
    if not np.allclose(s, [np.linalg.norm(Tr)] * 2 + [0], atol=1e-12):
        check("Prop 27.3.1: singular values of [T]x R are (|T|, |T|, 0)", False)
        break
else:
    check("Prop 27.3.1: singular values of [T]x R are (|T|, |T|, 0) (10 random)", True)
W = np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 1.0]])          # COLMAP's W = R_z(-90 deg)
close("Prop 27.3.1: W = R_z(-90 deg)", W, Rz(-np.pi / 2), tol=1e-15)
close("Prop 27.3.1: [e_z]x W = diag(1, 1, 0)", hat([0, 0, 1.0]) @ W, np.diag([1.0, 1, 0]), tol=1e-15)
U0, V0 = rand_rot(), rand_rot()
M = U0 @ np.diag([2.5, 2.5, 0]) @ V0.T
Tm = 2.5 * U0[:, 2]
Rm = U0 @ W @ V0.T
close("Prop 27.3.1: U diag(s,s,0) V^T = [s u3]x (U W V^T)", hat(Tm) @ Rm, M, tol=1e-12)
check("Prop 27.3.1: U W V^T is a rotation", np.allclose(Rm @ Rm.T, np.eye(3)) and np.isclose(np.linalg.det(Rm), 1))
Mbad = U0 @ np.diag([2.5, 1.5, 0]) @ V0.T
check("Prop 27.3.1: a rank-2 matrix with sigma1 != sigma2 is not essential (2 M M^T M - tr(M M^T) M != 0)",
      np.abs(2 * Mbad @ Mbad.T @ Mbad - np.trace(Mbad @ Mbad.T) * Mbad).max() > 1e-3)

# ---------------------------------------------------------------- Proposition 27.3.2: dimension 5
def tangent_basis_S2(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = a - (a @ n) * n
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def Emap(params, R0, T0):
    w, a, b = params[:3], params[3], params[4]
    e1, e2 = tangent_basis_S2(T0)
    Tn = T0 + a * e1 + b * e2
    Tn /= np.linalg.norm(Tn)
    return (hat(Tn) @ expso3(w) @ R0).ravel()


h = 1e-6
ranks = []
for _ in range(5):
    R0, T0 = rand_rot(), rng.normal(size=3)
    T0 /= np.linalg.norm(T0)
    Jm = np.column_stack([(Emap(h * e, R0, T0) - Emap(-h * e, R0, T0)) / (2 * h) for e in np.eye(5)])
    sv = np.linalg.svd(Jm, compute_uv=False)
    ranks.append(int(np.sum(sv > 1e-6 * sv[0])))
check(f"Prop 27.3.2: differential of SO(3) x S^2 -> R^9 has rank 5 (5 random points: {ranks})", ranks == [5] * 5)
# tangent space of {E : 2 E E^T E - tr(E E^T) E = 0, |E|^2 = 2}
R0, T0 = rand_rot(), rng.normal(size=3)
T0 /= np.linalg.norm(T0)
E0 = hat(T0) @ R0


def demazure(Ev):
    Em = Ev.reshape(3, 3)
    return np.concatenate([(2 * Em @ Em.T @ Em - np.trace(Em @ Em.T) * Em).ravel(), [np.sum(Em**2) - 2]])


check("Prop 27.3.2: E0 satisfies the Demazure equations and |E|^2 = 2", np.abs(demazure(E0.ravel())).max() < 1e-12)
Jd = np.column_stack([(demazure(E0.ravel() + h * e) - demazure(E0.ravel() - h * e)) / (2 * h) for e in np.eye(9)])
svd_ = np.linalg.svd(Jd, compute_uv=False)
rk = int(np.sum(svd_ > 1e-6 * svd_[0]))
check(f"Prop 27.3.2: Demazure + normalization have Jacobian rank 4, so the tangent space is 9 - 4 = 5 (rank {rk})", rk == 4)
Jd9 = Jd[:9]
rk9 = int(np.sum(np.linalg.svd(Jd9, compute_uv=False) > 1e-6 * svd_[0]))
check(f"Prop 27.3.2: the 9 cubic Demazure equations alone have Jacobian rank 3 (6-dim cone) (rank {rk9})", rk9 == 3)
# Tron-Daniilidis
R1q, R2q = rand_rot(), rand_rot()
Eq = R1q.T @ hat([0, 0, 1.0]) @ R2q
th = 0.7
close("Prop 27.3.2: E(Q) = R1^T [e_z]x R2 is invariant under H_z (R_z(th) R1, R_z(th) R2)",
      (Rz(th) @ R1q).T @ hat([0, 0, 1.0]) @ (Rz(th) @ R2q), Eq, tol=1e-12)
Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1.0, 0], [-np.sin(a), 0, np.cos(a)]])
Rx = lambda a: np.array([[1.0, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
signs = []
for S1, S2 in ((np.eye(3), np.eye(3)), (Ry(np.pi), Ry(np.pi)), (np.eye(3), Rz(np.pi)), (Ry(np.pi), Rx(np.pi))):
    Es = (S1 @ R1q).T @ hat([0, 0, 1.0]) @ (S2 @ R2q)
    signs.append(1 if np.allclose(Es, Eq, atol=1e-12) else (-1 if np.allclose(Es, -Eq, atol=1e-12) else 0))
check(f"Prop 27.3.2: the four elements of H_pi give E(Q) up to sign (signs {signs})", 0 not in signs)
check("Prop 27.3.2: dim SO(3) x SO(3) - dim H_z = 6 - 1 = 5", 6 - 1 == 5)
# (review) exactly two of the four H_pi elements give +E (identity and (Ry(pi), Rx(pi))), two give -E:
# so H_pi x H_z orbits are the classes of E up to sign, i.e. the quotient is E / {+-1}
check(f"Prop 27.3.2: two H_pi elements give +E and two give -E (signs {signs})", sorted(signs) == [-1, -1, 1, 1])
# (review) (Ry(pi), Rx(pi)) applied to Q is the other decomposition (R_T(pi) R, -T) with the same E
Tq, Rq_ = R1q.T @ np.array([0, 0, 1.0]), R1q.T @ R2q          # E(Q) = [T]x R with T = R1^T e_z, R = R1^T R2
Tq2, Rq2 = (Ry(np.pi) @ R1q).T @ np.array([0, 0, 1.0]), (Ry(np.pi) @ R1q).T @ (Rx(np.pi) @ R2q)
close("Prop 27.3.2: (Ry(pi), Rx(pi)) Q corresponds to (R_T(pi) R, -T)",
      np.abs(Tq2 + Tq).max() + np.abs(Rq2 - (2 * np.outer(Tq, Tq) - np.eye(3)) @ Rq_).max(), 0.0, tol=1e-12)
# (review) independent tangent-space count: (U, V) -> U diag(1,1,0) V^T has rank 5; kernel = right R_z on both
E0d = np.diag([1.0, 1, 0])
for _ in range(3):
    U0_, V0_ = rand_rot(), rand_rot()
    fUV = lambda x: (U0_ @ expso3(x[:3]) @ E0d @ (V0_ @ expso3(x[3:])).T).ravel()
    Juv = np.column_stack([(fUV(h * e) - fUV(-h * e)) / (2 * h) for e in np.eye(6)])
    _, svu, Vtu = np.linalg.svd(Juv)
    ker = Vtu[-1] / np.linalg.norm(Vtu[-1])
    ok = int(np.sum(svu > 1e-6 * svu[0])) == 5 and np.allclose(np.abs(ker), [0, 0, 1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], atol=1e-5)
    if not ok:
        break
check("Prop 27.3.2: SVD parametrization (U, V) -> U diag(1,1,0) V^T has rank 5, kernel (e_z, e_z)", ok)
# (review) the cubic equations have no solution of rank 3 and accept only (s, s, 0): check sigma_i(2 sigma_i^2 - sum) = 0
for sv in ((1.0, 1.0, 1.0), (1.0, 1.0, 0.5), (2.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
    Mt = rand_rot() @ np.diag(sv) @ rand_rot()
    res_ = np.abs(2 * Mt @ Mt.T @ Mt - np.trace(Mt @ Mt.T) * Mt).max()
    check(f"Prop 27.3.2: cubic equations reject singular values {sv}", res_ > 1e-3)

# ---------------------------------------------------------------- Proposition 27.3.3: four decompositions, cheirality
def decompose(Em):
    U, s, Vt = np.linalg.svd(Em)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    Ra, Rb = U @ W @ Vt, U @ W.T @ Vt
    t = U[:, 2]
    return [(Ra, t), (Ra, -t), (Rb, t), (Rb, -t)]


def triangulate(p1, p2, Rm, Tm):
    # least squares for depths: s1 R p1 + T = s2 p2
    A = np.column_stack([Rm @ p1, -p2])
    s, *_ = np.linalg.lstsq(A, -Tm, rcond=None)
    return s


Xtrue = np.array([1.0, 0.2, 6.0])
p1, q2 = Xtrue / Xtrue[2], R @ Xtrue + T
p2 = q2 / q2[2]
dec = decompose(E)
patterns = []
n_good = 0
for Rm, Tm in dec:
    close("Prop 27.3.3: every decomposition reproduces E up to sign",
          min(np.abs(hat(Tm) @ Rm - E).max(), np.abs(hat(Tm) @ Rm + E).max()), 0.0, tol=1e-12)
    s1, s2 = triangulate(p1, p2, Rm, Tm)
    patterns.append((np.sign(s1), np.sign(s2)))
    n_good += (s1 > 0 and s2 > 0)
check(f"Prop 27.3.3: exactly one of the four has both depths positive (patterns {patterns})", n_good == 1)
check("Prop 27.3.3: the four sign patterns are (+,+), (-,-), (+,-), (-,+) in some order",
      sorted(patterns) == sorted([(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]))
good = [d for d, pt in zip(dec, patterns) if pt == (1.0, 1.0)][0]
close("Prop 27.3.3: the cheiral decomposition is the true (R, T)", np.abs(good[0] - R).max() + np.abs(good[1] - T).max(), 0.0, tol=1e-10)
RT_pi = 2 * np.outer(T, T) - np.eye(3)
twist = [d for d in dec if not np.allclose(d[0], R, atol=1e-9)][0][0]
close("Prop 27.3.3: R' = R_T(pi) R (rotation by 180 deg about the baseline)", twist, RT_pi @ R, tol=1e-10)
close("Prop 27.3.3: [-T]x R_T(pi) R = [T]x R exactly", hat(-T) @ RT_pi @ R, E, tol=1e-12)
# (review) synthetic scene of 200 points: for every point the four decompositions give the four sign patterns once each
# (which twisted decomposition gets (+,-) can change from point to point, cf. [Tron17, (46)-(47)]), and COLMAP's vote
# (count of points with both depths > 0) selects the true pose
Xs = np.column_stack([rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200), rng.uniform(4, 15, 200)])
sds = []
for Rm, Tm in dec:
    sds.append(np.array([triangulate(x / x[2], (R @ x + T) / (R @ x + T)[2], Rm, Tm) for x in Xs]))
votes = [int(np.sum((sd[:, 0] > 0) & (sd[:, 1] > 0))) for sd in sds]
per_point = all(sorted((np.sign(sd[k, 0]), np.sign(sd[k, 1])) for sd in sds)
                == sorted([(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]) for k in range(len(Xs)))
best = dec[int(np.argmax(votes))]
check(f"Prop 27.3.3: each of 200 points gets the four sign patterns once; the vote picks the true pose (votes {votes})",
      per_point and sorted(votes) == [0, 0, 0, 200] and np.allclose(best[0], R) and np.allclose(best[1], T))
R_b_check = [d for d in dec if not np.allclose(d[0], dec[0][0])][0][0]
U_, _, Vt_ = np.linalg.svd(E)
U_ = U_ * np.sign(np.linalg.det(U_))
close("(27.3.3): R_b = U W^T V^T = R_{u3}(pi) R_a", R_b_check, (2 * np.outer(U_[:, 2], U_[:, 2]) - np.eye(3)) @ dec[0][0], tol=1e-10)

# ---------------------------------------------------------------- 5-point, 7-point, 8-point
def design(P1, P2):
    return np.array([np.kron(b, a) for a, b in zip(P1, P2)])     # row: vec(p2 p1^T) so that row . vec(F row-major) = p2^T F p1


pts1 = []
pts2 = []
for _ in range(60):
    X = np.array([rng.uniform(-3, 3), rng.uniform(-2, 2), rng.uniform(4, 12)])
    q = R @ X + T
    pts1.append(X / X[2])
    pts2.append(q / q[2])
pts1, pts2 = np.array(pts1), np.array(pts2)
A5 = design(pts1[:5], pts2[:5])
close("design matrix convention: row . vec(E) = pbar2^T E pbar1", np.abs(A5 @ E.ravel()).max(), 0.0, tol=1e-12)
sv5 = np.linalg.svd(A5, compute_uv=False)
check("5-point: the 5 x 9 system has a 4-dimensional null space", len(sv5) == 5 and sv5[-1] > 1e-8)
# 7-point with pixel coordinates
P1pix = (K1 @ pts1.T).T
P2pix = (K2 @ pts2.T).T
A7 = design(P1pix[:7], P2pix[:7])
_, _, Vt7 = np.linalg.svd(A7)
F1, F2 = Vt7[-1].reshape(3, 3), Vt7[-2].reshape(3, 3)
alphas = np.linspace(-3, 3, 7)
dets = [np.linalg.det(a * F1 + (1 - a) * F2) for a in alphas]
coef = np.polyfit(alphas, dets, 3)
close("7-point: det(a F1 + (1 - a) F2) is exactly a cubic in a", np.abs(np.polyval(coef, alphas) - dets).max() / np.abs(dets).max(), 0.0, tol=1e-9)
roots = np.roots(coef)
real = roots[np.abs(roots.imag) < 1e-9].real
Fn = F / np.linalg.norm(F)
found = min(min(np.linalg.norm((a * F1 + (1 - a) * F2) / np.linalg.norm(a * F1 + (1 - a) * F2) - sgn * Fn) for sgn in (1, -1)) for a in real)
check(f"7-point: {len(real)} real root(s), one of them is the true F", len(real) in (1, 3) and found < 1e-6)
counts = {1: 0, 3: 0}
for _ in range(300):
    idx = rng.choice(60, 7, replace=False)
    A7r = design(P1pix[idx] + rng.normal(scale=2.0, size=(7, 3)) * [1, 1, 0], P2pix[idx])
    _, _, Vr = np.linalg.svd(A7r)
    G1, G2 = Vr[-1].reshape(3, 3), Vr[-2].reshape(3, 3)
    cf = np.polyfit(alphas, [np.linalg.det(a * G1 + (1 - a) * G2) for a in alphas], 3)
    nr = int(np.sum(np.abs(np.roots(cf).imag) < 1e-7 * np.abs(np.roots(cf)).max()))
    counts[nr] = counts.get(nr, 0) + 1
check(f"7-point: over 300 noisy samples the number of real roots is always 1 or 3 ({counts})", set(counts) <= {1, 3})
# nearest essential matrix in Frobenius norm
Mn = E + 0.05 * rng.normal(size=(3, 3))
U, s, Vt = np.linalg.svd(Mn)
m = (s[0] + s[1]) / 2
Ebest = U @ np.diag([m, m, 0]) @ Vt
dbest = np.linalg.norm(Mn - Ebest)
worse = 0
for _ in range(3000):
    Rp = expso3(rng.normal(scale=0.05, size=3))
    Rq = expso3(rng.normal(scale=0.05, size=3))
    sc = m * (1 + rng.normal(scale=0.05))
    Ec = U @ Rp @ np.diag([sc, sc, 0]) @ Rq @ Vt
    worse += np.linalg.norm(Mn - Ec) < dbest - 1e-12
check("8-point projection: U diag(m, m, 0) V^T (m = (s1 + s2)/2) beats 3000 nearby essential matrices", worse == 0)


# ---------------------------------------------------------------- Hartley normalization (shared with Figure 27.3.3)
def eight_point(x1, x2, normalize):
    def nmat(x):
        c = x[:, :2].mean(0)
        d = np.mean(np.linalg.norm(x[:, :2] - c, axis=1))
        sc = np.sqrt(2) / d
        return np.array([[sc, 0, -sc * c[0]], [0, sc, -sc * c[1]], [0, 0, 1]])

    if normalize:
        N1, N2 = nmat(x1), nmat(x2)
    else:
        N1 = N2 = np.eye(3)
    y1, y2 = (N1 @ x1.T).T, (N2 @ x2.T).T
    A = design(y1, y2)
    _, sA, Vt_ = np.linalg.svd(A)
    Fh = Vt_[-1].reshape(3, 3)
    U_, s_, V_ = np.linalg.svd(Fh)
    Fh = U_ @ np.diag([s_[0], s_[1], 0]) @ V_
    Fh = N2.T @ Fh @ N1
    return Fh / np.linalg.norm(Fh), sA[0] / sA[-2]


def epi_err(Fh, x1, x2):
    l2_ = (Fh @ x1.T).T
    l1_ = (Fh.T @ x2.T).T
    r = np.abs(np.sum(x2 * l2_, axis=1))
    return 0.5 * (r / np.hypot(l2_[:, 0], l2_[:, 1]) + r / np.hypot(l1_[:, 0], l1_[:, 1]))


def experiment(seed=0, sigmas=(0.1, 0.3, 1.0, 3.0), trials=200, npts=50):
    rr = np.random.default_rng(seed)
    Kc = np.array([[1500.0, 0, 960], [0, 1500.0, 540], [0, 0, 1]])
    Rc = expso3(np.array([0.02, -0.25, 0.03]))
    Tc = np.array([-1.0, 0.05, 0.1])
    res = {"raw": [], "norm": []}
    cond = {"raw": [], "norm": []}
    for sg in sigmas:
        er = {"raw": [], "norm": []}
        for _ in range(trials):
            X = np.column_stack([rr.uniform(-3, 3, npts), rr.uniform(-2, 2, npts), rr.uniform(5, 12, npts)])
            x1 = (Kc @ X.T).T
            x1 /= x1[:, 2:]
            q = (Rc @ X.T).T + Tc
            x2 = (Kc @ q.T).T
            x2 /= x2[:, 2:]
            n1 = x1.copy()
            n2 = x2.copy()
            n1[:, :2] += rr.normal(scale=sg, size=(npts, 2))
            n2[:, :2] += rr.normal(scale=sg, size=(npts, 2))
            for key, nz in (("raw", False), ("norm", True)):
                Fh, cn = eight_point(n1, n2, nz)
                er[key].append(np.median(epi_err(Fh, x1, x2)))
                cond[key].append(cn)
        for key in er:
            res[key].append(np.median(er[key]))
    return np.array(sigmas), {k: np.array(v) for k, v in res.items()}, {k: np.median(v) for k, v in cond.items()}


sig, res, cond = experiment()
print("normalization experiment:", sig, res, cond)
check("Fig 27.3.3: normalized 8-point is more accurate at every noise level", np.all(res["norm"] < res["raw"]))
check("Fig 27.3.3: error ratio raw/normalized is about 2 (1.8-2.5) at every noise level", np.all((res["raw"] / res["norm"] > 1.8) & (res["raw"] / res["norm"] < 2.5)))
check("Fig 27.3.3: sigma_1/sigma_8 of the design matrix: raw ~ 3e5, normalized ~ 36", 2e5 < cond["raw"] < 4e5 and 30 < cond["norm"] < 42)
close("Fig 27.3.3: numbers quoted (sigma = 1 px): raw 0.75 px, normalized 0.34 px", np.array([res["raw"][2], res["norm"][2]]), np.array([0.749, 0.343]), tol=5e-3)
ratio = res["raw"] / res["norm"]
check(f"Fig 27.3.3: caption range 1.9-2.4 for raw/normalized (ratios {np.round(ratio, 2)})",
      np.all((ratio > 1.85) & (ratio < 2.45)))
# COLMAP init_max_forward_motion = 0.95 on |t_z| of the unit translation: baseline within ~18 deg of the optical axis
close("COLMAP init_max_forward_motion 0.95 <-> angle arccos(0.95) = 18.2 deg", np.degrees(np.arccos(0.95)), 18.19, tol=0.01)


# ---------------------------------------------------------------- (27.3.4) triangulation depth error
f_, z_, su = 800.0, 10.0, 0.5
def sigma_z_formula(b):
    return np.sqrt(2) * su * z_**2 / (f_ * b)


for gdeg in (1.5, 5.0, 16.0):
    b = z_ * np.tan(np.radians(gdeg))
    # Jacobian of camera 2 (center (b,0,0), R = I) applied to the unit ray direction of camera 1
    t2 = np.array([-b, 0, z_])
    J2 = f_ * np.array([[1 / t2[2], 0, -t2[0] / t2[2] ** 2], [0, 1 / t2[2], -t2[1] / t2[2] ** 2]])
    close(f"(27.3.4): |J2 r1| = f b / z^2 at gamma = {gdeg} deg", np.linalg.norm(J2 @ [0, 0, 1.0]), f_ * b / z_**2, tol=1e-9)
    # Monte Carlo with noise in both images (disparity)
    rr = np.random.default_rng(int(gdeg * 10))
    d = f_ * b / z_
    dn = d + rr.normal(scale=su, size=200000) - rr.normal(scale=su, size=200000)
    zs = f_ * b / dn
    mc = np.std(zs)
    close(f"(27.3.4): Monte Carlo std of depth vs sqrt(2) sigma_u z^2/(f b) at {gdeg} deg (rel)", mc / sigma_z_formula(b), 1.0, tol=0.03)
rel15 = sigma_z_formula(z_ * np.tan(np.radians(1.5))) / z_
rel16 = sigma_z_formula(z_ * np.tan(np.radians(16))) / z_
close("(27.3.4): relative depth error at 1.5 deg = 3.4%", rel15, 0.0337, tol=5e-4)
close("(27.3.4): relative depth error at 16 deg = 0.31%", rel16, 0.00308, tol=5e-5)
close("(27.3.4): relative error = sqrt(2) sigma_u / (f tan gamma)", rel15, np.sqrt(2) * su / (f_ * np.tan(np.radians(1.5))), tol=1e-12)

# ---------------------------------------------------------------- Exercises
# Ex 27.3.1: R = I, T = (1, 0, 0): E = [e_x]x; epipolar lines are horizontal: l2 = E (x, y, 1) = (0, -1, y)
Ex1 = hat([1.0, 0, 0])
close("Ex 27.3.1: E pbar1 = (0, -1, y)", Ex1 @ np.array([0.3, 0.2, 1.0]), np.array([0, -1.0, 0.2]), tol=1e-12)
# Ex 27.3.2: singular values of E for |T| = 3: (3, 3, 0)
check("Ex 27.3.2: [T]x R with |T| = 3 has singular values (3, 3, 0)",
      np.allclose(np.linalg.svd(hat([0, 3.0, 0]) @ rand_rot(), compute_uv=False), [3, 3, 0], atol=1e-12))
# Ex 27.3.3: dimension counts
check("Ex 27.3.3: minimal numbers 5, 7, 8 and 4", (5, 8 - 1, 8, 8 // 2) == (5, 7, 8, 4))
# Ex 27.3.4: baseline halving doubles sigma_z; z doubling quadruples
close("Ex 27.3.4: halving b doubles sigma_z", sigma_z_formula(0.5) / sigma_z_formula(1.0), 2.0, tol=1e-12)
close("Ex 27.3.4: sigma_z at f = 800, b = 0.1 m, z = 10 m, sigma_u = 0.5 px is 0.88 m",
      np.sqrt(2) * 0.5 * 100 / (800 * 0.1), 0.884, tol=1e-3)
# Ex 27.3.5: pure rotation: E = 0
check("Ex 27.3.5: T = 0 gives E = 0", np.allclose(hat([0, 0, 0.0]) @ rand_rot(), 0))
# (review) under pure rotation every [T']x R (T' in S^2) satisfies all correspondences exactly: a 2-dim family
Rpr = rand_rot()
Xp = np.column_stack([rng.uniform(-2, 2, 30), rng.uniform(-2, 2, 30), rng.uniform(3, 9, 30)])
worst_pr = 0.0
for _ in range(20):
    Tp_ = rng.normal(size=3)
    Tp_ /= np.linalg.norm(Tp_)
    for x in Xp:
        a_, b_ = x / x[2], (Rpr @ x) / (Rpr @ x)[2]
        worst_pr = max(worst_pr, abs(b_ @ hat(Tp_) @ Rpr @ a_))
close("Ex 27.3.5: pure rotation: [T']x R satisfies every match for any unit T' (20 random T')", worst_pr, 0.0, tol=1e-12)

summary()
