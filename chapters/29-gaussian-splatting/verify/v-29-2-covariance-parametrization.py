"""Verification for Section 29.2 (Parametrizing Covariances with Rotations and Scales).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/verify/v-29-2-covariance-parametrization.py``
"""

import itertools

import numpy as np
import sympy as sp

from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(292)


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expso3(w):
    th = np.linalg.norm(w)
    K = hat(w)
    if th < 1e-14:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K


def Phi(R, s):
    return R @ np.diag(np.asarray(s) ** 2) @ R.T


def quat_to_R(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    q0, v = q[0], q[1:]
    return np.eye(3) + 2 * q0 * hat(v) + 2 * hat(v) @ hat(v)


def random_rotation():
    Q, Rr = np.linalg.qr(rng.normal(size=(3, 3)))
    Q = Q @ np.diag(np.sign(np.diag(Rr)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


# ---------------------------------------------------------------- quaternion -> rotation formula
for _ in range(5):
    q = rng.normal(size=4)
    Rq = quat_to_R(q)
    close("(29.2.5): R(q) is orthogonal", Rq.T @ Rq, np.eye(3), tol=1e-12)
    close("(29.2.5): det R(q) = 1", np.linalg.det(Rq), 1.0, tol=1e-12)
    qh = q / np.linalg.norm(q)
    ang = 2 * np.arccos(np.clip(abs(qh[0]), -1, 1))
    axis = np.sign(qh[0]) * qh[1:] / np.linalg.norm(qh[1:])
    close("(29.2.5): R(q) = Rodrigues rotation by 2 arccos q0 about v/|v|", Rq, expso3(ang * axis), tol=1e-10)
    close("(29.2.5): R(-q) = R(q)", quat_to_R(-q), Rq, tol=1e-14)
    close("(29.2.5): R(cq) = R(q) for c > 0", quat_to_R(3.7 * q), Rq, tol=1e-12)

# ---------------------------------------------------------------- surjectivity (spectral theorem)
A = rng.normal(size=(3, 3))
Sig = A @ A.T + 0.1 * np.eye(3)
lam, V = np.linalg.eigh(Sig)
if np.linalg.det(V) < 0:
    V[:, 0] *= -1
close("Phi is onto: Sigma = Phi(V, sqrt(lam))", Phi(V, np.sqrt(lam)), Sig, tol=1e-10)

# ---------------------------------------------------------------- Proposition 29.2.1: fiber has 24 elements (48 quaternions)
R0 = random_rotation()
s0 = np.array([2.0, 1.0, 0.4])
Sig0 = Phi(R0, s0)
signed_perms = []
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        P = np.zeros((3, 3))
        for j, i in enumerate(perm):
            P[i, j] = signs[j]
        signed_perms.append(P)
check("signed permutation matrices: 48 in total", len(signed_perms) == 48)
rot_perms = [P for P in signed_perms if np.linalg.det(P) > 0]
check("Proposition 29.2.1: 24 of them have det +1", len(rot_perms) == 24)
fiber = []
for P in rot_perms:
    Rn = R0 @ P
    sn = np.abs(P.T @ s0)                   # permuted scales
    if np.allclose(Phi(Rn, sn), Sig0, atol=1e-12) and np.isclose(np.linalg.det(Rn), 1):
        fiber.append((Rn, sn))
check("Proposition 29.2.1: all 24 pairs (R P, P^T s) map to the same Sigma", len(fiber) == 24)
# exactness: any (R', s') in the fiber has s'^2 = eigenvalues, columns of R' = +- eigenvectors -> enumerated above
lamS, VS = np.linalg.eigh(Sig0)
cands = 0
for perm in itertools.permutations(range(3)):
    for signs in itertools.product((1, -1), repeat=3):
        Rc = VS[:, list(perm)] * np.array(signs)
        if np.linalg.det(Rc) > 0 and np.allclose(Phi(Rc, np.sqrt(lamS[list(perm)])), Sig0, atol=1e-12):
            cands += 1
check("Proposition 29.2.1: eigen-decompositions with det +1 are exactly 24", cands == 24)
# 48 unit quaternions
def R_to_quat(Rm):
    tr = np.trace(Rm)
    if tr > -0.9:
        q0 = 0.5 * np.sqrt(1 + tr)
        v = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]]) / (4 * q0)
        return np.concatenate([[q0], v])
    i = int(np.argmax(np.diag(Rm)))
    j, k = (i + 1) % 3, (i + 2) % 3
    vi = 0.5 * np.sqrt(1 + Rm[i, i] - Rm[j, j] - Rm[k, k])
    q = np.zeros(4)
    q[1 + i] = vi
    q[0] = (Rm[k, j] - Rm[j, k]) / (4 * vi)
    q[1 + j] = (Rm[j, i] + Rm[i, j]) / (4 * vi)
    q[1 + k] = (Rm[k, i] + Rm[i, k]) / (4 * vi)
    return q
quats = []
for Rn, sn in fiber:
    qq = R_to_quat(Rn)
    close("R_to_quat inverts quat_to_R", quat_to_R(qq), Rn, tol=1e-10)
    quats += [qq, -qq]
Q = np.array(quats)
D = np.abs(Q[:, None, :] - Q[None, :, :]).max(-1) + np.eye(48)
check("Proposition 29.2.1: 48 distinct unit quaternions give the same Sigma", len(quats) == 48 and D.min() > 1e-6)
check("Proposition 29.2.1: each of them maps to Sigma", all(
    np.allclose(Phi(quat_to_R(q), fiber[i // 2][1]), Sig0, atol=1e-12) for i, q in enumerate(quats)))

# Exercise 29.2.1: s = (3, 2, 1), R' = R Z with Z = 90 deg about z, s' = (2, 3, 1)
Z = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 1.0]])
close("Exercise 29.2.1: Phi(R Z, (2,3,1)) = Phi(R, (3,2,1))", Phi(R0 @ Z, [2, 3, 1]), Phi(R0, [3, 2, 1]), tol=1e-12)

# ---------------------------------------------------------------- (29.2.2): differential and rank
def dPhi_numeric(R, s, h=1e-6):
    """6x6 Jacobian of (omega, log s) -> Sigma in the Frobenius-orthonormal basis of Sym(3)."""
    basis = []
    for i in range(3):
        E = np.zeros((3, 3)); E[i, i] = 1; basis.append(E)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        E = np.zeros((3, 3)); E[i, j] = E[j, i] = 1 / np.sqrt(2); basis.append(E)
    def f(p):
        S = Phi(R @ expso3(p[:3]), s * np.exp(p[3:]))
        return np.array([np.sum(S * E) for E in basis])
    Jm = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6); e[k] = h
        Jm[:, k] = (f(e) - f(-e)) / (2 * h)
    return Jm


def sv_closed(s):
    l = np.asarray(s) ** 2
    return np.sort(np.array([2 * l[0], 2 * l[1], 2 * l[2], np.sqrt(2) * abs(l[0] - l[1]), np.sqrt(2) * abs(l[0] - l[2]),
                             np.sqrt(2) * abs(l[1] - l[2])]))


for name, s, rk in (("distinct", [2.0, 1.0, 0.4], 6), ("two equal", [2.0, 2.0, 0.5], 5), ("isotropic", [1.0, 1.0, 1.0], 3)):
    Jm = dPhi_numeric(R0, np.array(s))
    sv = np.sort(np.linalg.svd(Jm, compute_uv=False))
    close(f"(29.2.2): singular values of dPhi ({name}) = {{2 lambda_i, sqrt2 |lambda_i - lambda_j|}}", sv, sv_closed(s),
          tol=1e-6)
    check(f"Proposition 29.2.2: rank dPhi = {rk} ({name})", int(np.sum(sv > 1e-6)) == rk)

# The singular values depend on the inner products used to measure them: with the scale coordinate s itself
# (instead of log s) the three scale singular values become 2 s_i, while the rotation ones and the rank do not change.
def dPhi_numeric_s(R, s, h=1e-6):
    basis = []
    for i in range(3):
        E = np.zeros((3, 3)); E[i, i] = 1; basis.append(E)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        E = np.zeros((3, 3)); E[i, j] = E[j, i] = 1 / np.sqrt(2); basis.append(E)
    def f(p):
        S = Phi(R @ expso3(p[:3]), s + p[3:])
        return np.array([np.sum(S * E) for E in basis])
    Jm = np.zeros((6, 6))
    for k in range(6):
        e = np.zeros(6); e[k] = h
        Jm[:, k] = (f(e) - f(-e)) / (2 * h)
    return Jm


s_ = np.array([2.0, 1.0, 0.4]); l_ = s_ ** 2
sv_s = np.sort(np.linalg.svd(dPhi_numeric_s(R0, s_), compute_uv=False))
close("Proposition 29.2.2 (metric dependence): in (omega, s) the scale singular values are 2 s_i", sv_s,
      np.sort(np.r_[2 * s_, np.sqrt(2) * np.abs([l_[0] - l_[1], l_[0] - l_[2], l_[1] - l_[2]])]), tol=1e-6)

# symbolic check of (29.2.2) in the eigenframe: [Omega, Lambda] entries
w1, w2, w3, l1, l2, l3 = sp.symbols("omega1 omega2 omega3 lambda1 lambda2 lambda3", real=True)
Om = sp.Matrix([[0, -w3, w2], [w3, 0, -w1], [-w2, w1, 0]])
Lm = sp.diag(l1, l2, l3)
C = Om * Lm - Lm * Om
sym_equal("(29.2.2): [Omega, Lambda]_12 = omega3 (lambda1 - lambda2)", C[0, 1], w3 * (l1 - l2))
sym_equal("(29.2.2): [Omega, Lambda]_13 = omega2 (lambda3 - lambda1)", C[0, 2], w2 * (l3 - l1))
sym_equal("(29.2.2): [Omega, Lambda]_23 = omega1 (lambda2 - lambda3)", C[1, 2], w1 * (l2 - l3))
check("(29.2.2): diagonal of [Omega, Lambda] is zero", all(sp.simplify(C[i, i]) == 0 for i in range(3)))
# d/dt R exp(t Omega) Lambda exp(-t Omega) R^T at t = 0 is R [Omega, Lambda] R^T (numeric)
w = rng.normal(size=3)
h = 1e-6
num = (Phi(R0 @ expso3(h * w), s0) - Phi(R0 @ expso3(-h * w), s0)) / (2 * h)
close("(29.2.2): directional derivative along R exp(t[w]x) = R [Omega, Lambda] R^T", num,
      R0 @ (hat(w) @ np.diag(s0 ** 2) - np.diag(s0 ** 2) @ hat(w)) @ R0.T, tol=1e-6)
# Exercise 29.2.3: |d Sigma / d omega_3|_F = sqrt2 |s1^2 - s2^2|
d3 = R0 @ (hat([0, 0, 1.0]) @ np.diag(s0 ** 2) - np.diag(s0 ** 2) @ hat([0, 0, 1.0])) @ R0.T
close("Exercise 29.2.3: |dSigma/domega_3|_F = sqrt2 |s1^2 - s2^2|", np.linalg.norm(d3), np.sqrt(2) * abs(4 - 1), tol=1e-12)

# ---------------------------------------------------------------- (29.2.3): rotation gradient proportional to s_i^2 - s_j^2
SigT = Phi(random_rotation(), [1.5, 0.8, 0.3])


def loss(R, s):
    return 0.5 * np.sum((Phi(R, s) - SigT) ** 2)


for s in ([2.0, 1.0, 0.4], [1.3, 1.3, 0.4]):
    s = np.array(s)
    G = Phi(R0, s) - SigT
    M = R0.T @ G @ R0
    l = s ** 2
    pred = np.array([2 * (l[1] - l[2]) * M[1, 2], 2 * (l[2] - l[0]) * M[0, 2], 2 * (l[0] - l[1]) * M[0, 1]])
    fd = np.array([(loss(R0 @ expso3(h * e), s) - loss(R0 @ expso3(-h * e), s)) / (2 * h) for e in np.eye(3)])
    close(f"(29.2.3): dL/domega_k = 2 (lambda_i - lambda_j) (R^T G R)_ij, s = {s.tolist()}", fd, pred, tol=1e-6)
check("(29.2.3): with s1 = s2 the gradient about r_3 vanishes", abs(
    (loss(R0 @ expso3([0, 0, h]), [1.3, 1.3, 0.4]) - loss(R0 @ expso3([0, 0, -h]), [1.3, 1.3, 0.4])) / (2 * h)) < 1e-8)
# isotropic: Sigma does not depend on R at all
check("isotropic: Phi(R, (c,c,c)) = c^2 I for any R", np.allclose(Phi(random_rotation(), [0.7] * 3), 0.49 * np.eye(3)))
# quaternion gradient vanishes at isotropy (finite differences in q)
qa = rng.normal(size=4)
gq = np.array([(0.5 * np.sum((Phi(quat_to_R(qa + h * e), [0.7] * 3) - SigT) ** 2)
                - 0.5 * np.sum((Phi(quat_to_R(qa - h * e), [0.7] * 3) - SigT) ** 2)) / (2 * h) for e in np.eye(4)])
close("isotropic: dL/dq = 0", gq, np.zeros(4), tol=1e-8)

# ---------------------------------------------------------------- Figure 29.2.2 (a): relative change under a 15 degree rotation
for ratio, val in ((3.0, 0.323), (1.5, 0.186), (1.1, 0.049)):
    l = np.array([ratio ** 2, 1.0])
    R15 = np.array([[np.cos(np.radians(15)), -np.sin(np.radians(15))], [np.sin(np.radians(15)), np.cos(np.radians(15))]])
    d = R15 @ np.diag(l) @ R15.T - np.diag(l)
    close(f"Figure 29.2.2(a): ratio {ratio}: |dSigma|/|Sigma| = {val}", round(np.linalg.norm(d) / np.linalg.norm(l), 3), val,
          tol=0)
    close(f"Figure 29.2.2(a): ratio {ratio}: |dSigma|_F = sqrt2 |l1-l2| sin 15", np.linalg.norm(d),
          np.sqrt(2) * abs(l[0] - l[1]) * np.sin(np.radians(15)), tol=1e-12)

# ---------------------------------------------------------------- Figure 29.2.3: 2D gradient descent from an isotropic start
def R2(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def run(target, eps, eta=0.01, n=1500):
    th, u = 0.0, np.log([1.0 + eps, 1.0])
    out = []
    for _ in range(n):
        lamv = np.exp(2 * u)
        S = R2(th) @ np.diag(lamv) @ R2(th).T
        Gm = S - target
        M = R2(th).T @ Gm @ R2(th)
        out.append((np.degrees(th), np.exp(u[0] - u[1]), 0.5 * np.sum(Gm ** 2)))
        th -= eta * 2 * (lamv[0] - lamv[1]) * M[0, 1]
        u = u - eta * 2 * lamv * np.diag(M)
    return np.array(out)


T45 = np.array([[2.125, 1.875], [1.875, 2.125]])       # s = (2, 0.5) at 45 degrees, built with exact entries
close("Figure 29.2.3: T45 = R(45) diag(4, 0.25) R(45)^T", R2(np.pi / 4) @ np.diag([4, 0.25]) @ R2(np.pi / 4).T, T45, tol=1e-12)
T30 = R2(np.radians(30)) @ np.diag([4, 0.25]) @ R2(np.radians(30)).T
h0 = run(T45, 0.0)
check("Figure 29.2.3: eps = 0, target 45: angle stays exactly 0", np.all(h0[:, 0] == 0.0))
check("Figure 29.2.3: eps = 0, target 45: scales stay exactly equal", np.all(h0[:, 1] == 1.0))
close("Figure 29.2.3: eps = 0 stalls at the isotropic Sigma = 2.125 I, loss 3.516", h0[-1, 2], 0.5 * 2 * 1.875 ** 2, tol=1e-9)


def reach(h, tgt):
    return int(np.argmax(np.abs(h[:, 0] - tgt) < 1.0))


r8, r4, r1, r30 = reach(run(T45, 1e-8), 45), reach(run(T45, 1e-4), 45), reach(run(T45, 1e-1), 45), reach(run(T30, 0.0), 30)
print("iterations to reach within 1 degree:", r8, r4, r1, r30)
check("Figure 29.2.3: within 1 deg after 148 (eps 1e-8), 83 (1e-4), 34 (1e-1), 24 (target 30) iterations",
      (r8, r4, r1, r30) == (148, 83, 34, 24))
check("Figure 29.2.3: delay grows like log(1/eps): (148 - 83) ~ (83 - 34) * 4/3", abs((148 - 83) - (83 - 34) * 4 / 3) < 3)
# the target built in floating point from R(45) has diagonal entries that differ by one rounding error (~1e-16);
# that tiny asymmetry alone lets the run escape after about 270 iterations, as the log(1/eps) law predicts
T45f = R2(np.pi / 4) @ np.diag([4.0, 0.25]) @ R2(np.pi / 4).T
rf = reach(run(T45f, 0.0), 45)
print("float-built 45-degree target, eps = 0: within 1 degree after", rf)
check("Figure 29.2.3 caption: rounding error alone (eps ~ 1e-16) escapes after about 270 iterations", 255 <= rf <= 290)
check("Figure 29.2.3: all perturbed runs converge (loss < 1e-4 at the end)",
      all(run(T, e)[-1, 2] < 1e-4 for T, e in ((T45, 1e-8), (T45, 1e-4), (T45, 1e-1), (T30, 0.0))))

# ---------------------------------------------------------------- Exercise 29.2.4: naive quaternion averaging
qa1 = np.array([1.0, 0, 0, 0]); sa1 = np.array([2.0, 1.0, 1.0])
qa2 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]); sa2 = np.array([1.0, 2.0, 1.0])
close("Exercise 29.2.4: both represent diag(4, 1, 1)", [Phi(quat_to_R(qa1), sa1), Phi(quat_to_R(qa2), sa2)],
      [np.diag([4.0, 1, 1])] * 2, tol=1e-12)
qm = (qa1 + qa2) / 2
sm = (sa1 + sa2) / 2
close("Exercise 29.2.4: naive average gives diag(2.25, 2.25, 1)", Phi(quat_to_R(qm), sm), np.diag([2.25, 2.25, 1.0]),
      tol=1e-12)
close("Exercise 29.2.4: averaged quaternion is a 45 degree rotation about z", 2 * np.degrees(np.arctan2(qm[3], qm[0])), 45.0,
      tol=1e-9)
check("Exercise 29.2.4: (q + (-q))/2 = 0 is not a rotation", np.allclose((qa2 + (-qa2)) / 2, 0))

# ---------------------------------------------------------------- alternatives: Cholesky (2D symbolic) and matrix exponential
a, b, c = sp.symbols("a b c", real=True)
l11, l21, l22 = sp.sqrt(a), b / sp.sqrt(a), sp.sqrt(c - b ** 2 / a)
Lm2 = sp.Matrix([[l11, 0], [l21, l22]])
sym_equal("Exercise 29.2.5: L L^T = [[a, b], [b, c]] with the unique Cholesky factor", Lm2 * Lm2.T,
          sp.Matrix([[a, b], [b, c]]), {a: (1, 3), b: (-0.5, 0.5), c: (1, 3)})
check("Exercise 29.2.5: Cholesky of [[4,2],[2,2]] is [[2,0],[1,1]]",
      np.allclose(np.linalg.cholesky(np.array([[4.0, 2], [2, 2]])), [[2, 0], [1, 1]]))
# Cholesky is not rotation-equivariant: chol(Q S Q^T) != Q chol(S) Q^T in general
Qr = random_rotation()
Lc = np.linalg.cholesky(Sig0)
check("alternatives: Cholesky factor is not rotation-equivariant", not np.allclose(np.linalg.cholesky(Qr @ Sig0 @ Qr.T),
                                                                                    Qr @ Lc @ Qr.T, atol=1e-6))


def expm_sym(Am):
    w_, V_ = np.linalg.eigh(Am)
    return V_ @ np.diag(np.exp(w_)) @ V_.T


Asym = rng.normal(size=(3, 3)); Asym = Asym + Asym.T
close("alternatives: exp(Q A Q^T) = Q exp(A) Q^T", expm_sym(Qr @ Asym @ Qr.T), Qr @ expm_sym(Asym) @ Qr.T, tol=1e-9)
close("alternatives: exp(R diag(2 log s) R^T) = R diag(s^2) R^T (log-scale is a special case)",
      expm_sym(R0 @ np.diag(2 * np.log(s0)) @ R0.T), Sig0, tol=1e-10)

summary()
