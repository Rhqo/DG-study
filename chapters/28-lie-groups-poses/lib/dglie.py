"""Chapter 28 helper: SO(3), SE(3), Sim(3), unit quaternions, Jacobians, splines, rotation averaging.

Conventions (Section 28.1, Conventions box):
  * hat of a rotation vector:  hat3(w) = [w]_x  (Tour (0.6.6)).
  * twist xi = (v, w) in R^6, translation part FIRST (as Sophus and [Sola18]):
        hat6(xi) = [[ [w]_x, v ], [0, 0]].
  * sim(3) vector (v, w, sigma) in R^7 (as Sophus): hat7 = [[ [w]_x + sigma I, v ], [0, 0]], scale s = e^sigma.
  * Exp(xi) = expm(hat(xi)),  Log = vee(logm(.)).
  * Ad_T is the 6x6 (or 7x7) matrix with  T Exp(xi) T^{-1} = Exp(Ad_T xi).
  * Unit quaternions are Hamilton, scalar first: q = (q0, q1, q2, q3).

Used by the figure and verify scripts of Chapter 28. Plain numpy (no scipy, Appendix E).
"""

import numpy as np

# ----------------------------------------------------------------------------- basics

I3 = np.eye(3)


def hat3(w):
    w = np.asarray(w, float)
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee3(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def expm(A, terms=30):
    """Matrix exponential by scaling and squaring + Taylor series (independent check of the closed forms)."""
    A = np.asarray(A, float)
    nrm = np.linalg.norm(A, 1)
    k = max(0, int(np.ceil(np.log2(nrm))) + 4) if nrm > 0 else 0
    B = A / (2.0 ** k)
    E = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for n in range(1, terms):
        term = term @ B / n
        E = E + term
    for _ in range(k):
        E = E @ E
    return E


# ----------------------------------------------------------------------------- SO(3)

def _coefs(th):
    """sin(th)/th, (1 - cos th)/th^2, (th - sin th)/th^3 with Taylor series for small th."""
    if th < 1e-4:
        t2 = th * th
        a = 1 - t2 / 6 + t2 * t2 / 120
        b = 0.5 - t2 / 24 + t2 * t2 / 720
        c = 1.0 / 6 - t2 / 120 + t2 * t2 / 5040
    else:
        a = np.sin(th) / th
        b = (1 - np.cos(th)) / th ** 2
        c = (th - np.sin(th)) / th ** 3
    return a, b, c


def expSO3(w):
    """Rodrigues formula (Tour (0.6.7)) with Taylor coefficients near 0."""
    w = np.asarray(w, float)
    K = hat3(w)
    a, b, _ = _coefs(np.linalg.norm(w))
    return I3 + a * K + b * K @ K


def R_to_q(R):
    """Rotation matrix -> unit quaternion (Shepperd's method), returned with q0 >= 0."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    d = np.array([tr, R[0, 0], R[1, 1], R[2, 2]])
    i = int(np.argmax(d))
    if i == 0:
        q0 = 0.5 * np.sqrt(1 + tr)
        q = np.array([q0, (R[2, 1] - R[1, 2]) / (4 * q0), (R[0, 2] - R[2, 0]) / (4 * q0),
                      (R[1, 0] - R[0, 1]) / (4 * q0)])
    elif i == 1:
        q1 = 0.5 * np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / (4 * q1), q1, (R[0, 1] + R[1, 0]) / (4 * q1),
                      (R[0, 2] + R[2, 0]) / (4 * q1)])
    elif i == 2:
        q2 = 0.5 * np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / (4 * q2), (R[0, 1] + R[1, 0]) / (4 * q2), q2,
                      (R[1, 2] + R[2, 1]) / (4 * q2)])
    else:
        q3 = 0.5 * np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2])
        q = np.array([(R[1, 0] - R[0, 1]) / (4 * q3), (R[0, 2] + R[2, 0]) / (4 * q3),
                      (R[1, 2] + R[2, 1]) / (4 * q3), q3])
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def logq(q):
    """Log of a unit quaternion as a rotation vector in the ball |w| <= pi (atan2 form, as Sophus)."""
    q = np.asarray(q, float)
    if q[0] < 0:
        q = -q
    w, v = q[0], q[1:]
    n2 = v @ v
    if n2 < 1e-24:
        return (2.0 / w - (2.0 / 3.0) * n2 / w ** 3) * v
    n = np.sqrt(n2)
    return 2.0 * np.arctan2(n, w) / n * v


def logSO3(R):
    """Robust log: matrix -> quaternion -> atan2. Returns w with |w| <= pi."""
    return logq(R_to_q(R))


def logSO3_naive(R):
    """Textbook formula theta = arccos((tr R - 1)/2), w = theta/(2 sin theta) (R - R^T)^vee."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th == 0.0:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * vee3(R - R.T)


def Jr(w):
    """Right Jacobian of SO(3): Exp(w + d) ~ Exp(w) Exp(Jr(w) d)  [Forster17 (8)]."""
    K = hat3(w)
    _, b, c = _coefs(np.linalg.norm(w))
    return I3 - b * K + c * K @ K


def Jl(w):
    """Left Jacobian of SO(3) = Jr(-w) = Jr(w)^T = V(w)."""
    K = hat3(w)
    _, b, c = _coefs(np.linalg.norm(w))
    return I3 + b * K + c * K @ K


def Jr_inv(w):
    """[Forster17]: I + K/2 + (1/th^2 - (1 + cos th)/(2 th sin th)) K^2."""
    K = hat3(w)
    th = np.linalg.norm(w)
    if th < 1e-4:
        e = 1.0 / 12 + th ** 2 / 720
    else:
        e = 1 / th ** 2 - (1 + np.cos(th)) / (2 * th * np.sin(th))
    return I3 + 0.5 * K + e * K @ K


def Jl_inv(w):
    return Jr_inv(-np.asarray(w, float))


# ----------------------------------------------------------------------------- quaternions

def qmul(p, q):
    """Hamilton product, scalar first."""
    p0, pv = p[0], np.asarray(p[1:], float)
    q0, qv = q[0], np.asarray(q[1:], float)
    return np.concatenate([[p0 * q0 - pv @ qv], p0 * qv + q0 * pv + np.cross(pv, qv)])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def q_to_R(q):
    """(29.2.5): R = I + 2 q0 [v]_x + 2 [v]_x^2 for the normalized q."""
    q = np.asarray(q, float) / np.linalg.norm(q)
    K = hat3(q[1:])
    return I3 + 2 * q[0] * K + 2 * K @ K


def expq(w):
    """Rotation vector -> unit quaternion (cos(th/2), sin(th/2) w/th)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    if th < 1e-4:
        f = 0.5 - th ** 2 / 48 + th ** 4 / 3840
    else:
        f = np.sin(th / 2) / th
    return np.concatenate([[np.cos(th / 2)], f * w])


def slerp(q0, q1, s, shortest=True):
    """[Shoemake85]. With shortest=True flip q1 when <q0, q1> < 0 (as Eigen's slerp)."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    d = q0 @ q1
    if shortest and d < 0:
        q1, d = -q1, -d
    d = np.clip(d, -1.0, 1.0)
    Om = np.arccos(d)
    if Om < 1e-12:
        return q0.copy()
    return (np.sin((1 - s) * Om) * q0 + np.sin(s * Om) * q1) / np.sin(Om)


# ----------------------------------------------------------------------------- SE(3)

def hat6(xi):
    xi = np.asarray(xi, float)
    X = np.zeros((4, 4))
    X[:3, :3] = hat3(xi[3:])
    X[:3, 3] = xi[:3]
    return X


def vee6(X):
    return np.concatenate([X[:3, 3], vee3(X[:3, :3])])


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def V_SE3(w):
    """V(w) = I + (1 - cos)/th^2 [w]_x + (th - sin)/th^3 [w]_x^2 = Jl(w)."""
    return Jl(w)


def expSE3(xi):
    xi = np.asarray(xi, float)
    v, w = xi[:3], xi[3:]
    return make_T(expSO3(w), V_SE3(w) @ v)


def logSE3(T):
    w = logSO3(T[:3, :3])
    v = Jl_inv(w) @ T[:3, 3]
    return np.concatenate([v, w])


def Ad_SE3(T):
    """Ad_T in (v, w) order: [[R, [t]_x R], [0, R]] (as Sophus)."""
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = hat3(t) @ R
    A[3:, 3:] = R
    return A


# ----------------------------------------------------------------------------- Sim(3)

def hat7(z):
    z = np.asarray(z, float)
    X = np.zeros((4, 4))
    X[:3, :3] = hat3(z[3:6]) + z[6] * I3
    X[:3, 3] = z[:3]
    return X


def vee7(X):
    sig = np.trace(X[:3, :3]) / 3.0
    return np.concatenate([X[:3, 3], vee3(X[:3, :3] - sig * I3), [sig]])


def W_Sim3_formula(w, sigma):
    """W of [Strasdat10, (22)] written as in the paper (= Sophus details::calcW away from 0)."""
    w = np.asarray(w, float)
    K = hat3(w)
    th = np.linalg.norm(w)
    s = np.exp(sigma)
    a, b, c = s * np.sin(th), s * np.cos(th), (s - 1) / sigma
    A = (a * sigma + (1 - b) * th) / (th * (th ** 2 + sigma ** 2))
    B = (c - ((b - 1) * sigma + a * th) / (th ** 2 + sigma ** 2)) / th ** 2
    return A * K + B * K @ K + c * I3


def _phi(z):
    """phi(z) = (e^z - 1)/z for real or complex z, accurate near 0."""
    if abs(z) < 1e-4:
        return 1 + z / 2 + z * z / 6 + z ** 3 / 24
    if isinstance(z, complex):
        sg, th = z.real, z.imag
        em1 = np.expm1(sg) * np.cos(th) - 2 * np.sin(th / 2) ** 2 + 1j * np.exp(sg) * np.sin(th)
        return em1 / z
    return np.expm1(z) / z


def W_Sim3(w, sigma):
    """Stable evaluation of W = int_0^1 e^{sigma u} exp(u [w]_x) du (same matrix as (28.1.5)).

    On the axis of w, W acts as phi(sigma); on the plane perpendicular to w as phi(sigma + i theta).
    """
    w = np.asarray(w, float)
    K = hat3(w)
    th = np.linalg.norm(w)
    ps = _phi(float(sigma))
    if th < 1e-5:
        if abs(sigma) < 1e-3:
            d1 = 0.5 + sigma / 3 + sigma ** 2 / 8 + sigma ** 3 / 30          # phi'(sigma)
            d2h = 1.0 / 6 + sigma / 8 + sigma ** 2 / 20                       # phi''(sigma)/2
        else:
            e = np.exp(sigma)
            d1 = (sigma * e - np.expm1(sigma)) / sigma ** 2
            d2h = 0.5 * (sigma ** 2 * e - 2 * sigma * e + 2 * np.expm1(sigma)) / sigma ** 3
        # W = ps I + (Im phi/th) K + ((ps - Re phi)/th^2) (K^2 + th^2 I)   with the limits th -> 0
        return ps * I3 + d1 * K + d2h * (K @ K)
    pz = _phi(complex(sigma, th))
    # W = Re(pz) P_perp + ps P_par + Im(pz) [w/th]_x,  P_par = w w^T/th^2 = I + K^2/th^2, P_perp = -K^2/th^2
    return ps * I3 + (pz.imag / th) * K + ((ps - pz.real) / th ** 2) * (K @ K)


def expSim3(z):
    z = np.asarray(z, float)
    v, w, sig = z[:3], z[3:6], z[6]
    S = np.eye(4)
    S[:3, :3] = np.exp(sig) * expSO3(w)
    S[:3, 3] = W_Sim3(w, sig) @ v
    return S


def logSim3(S):
    sR = S[:3, :3]
    s = np.cbrt(np.linalg.det(sR))
    w = logSO3(sR / s)
    sig = np.log(s)
    v = np.linalg.solve(W_Sim3(w, sig), S[:3, 3])
    return np.concatenate([v, w, [sig]])


def Ad_Sim3(S):
    """(v, w, sigma) order, as Sophus Sim3::Adj: [[sR, [t]_x R, -t], [0, R, 0], [0, 0, 1]]."""
    sR = S[:3, :3]
    s = np.cbrt(np.linalg.det(sR))
    R, t = sR / s, S[:3, 3]
    A = np.zeros((7, 7))
    A[:3, :3] = sR
    A[:3, 3:6] = hat3(t) @ R
    A[:3, 6] = -t
    A[3:6, 3:6] = R
    A[6, 6] = 1.0
    return A


def inv_S(S):
    return np.linalg.inv(S)


# ----------------------------------------------------------------------------- random and misc

def random_rotation(rng):
    q = rng.normal(size=4)
    return q_to_R(q / np.linalg.norm(q))


def rot_angle(R):
    return float(np.linalg.norm(logSO3(R)))


def project_SO3(M):
    """Closest rotation in Frobenius norm (SVD with det correction)."""
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


# ----------------------------------------------------------------------------- rotation averaging

def mean_chordal(Rs):
    return project_SO3(np.sum(Rs, axis=0))


def mean_quat_eig(Rs):
    """Largest eigenvector of sum q q^T (sign-invariant) [Hartley13, chordal L2]."""
    A = np.zeros((4, 4))
    for R in Rs:
        q = R_to_q(R)
        A += np.outer(q, q)
    lam, V = np.linalg.eigh(A)
    return q_to_R(V[:, -1])


def mean_geodesic_L2(Rs, R0=None, tol=1e-13, maxit=200):
    """Karcher mean by R <- R Exp(mean_i Log(R^T R_i)) (Tour 0.8)."""
    R = Rs[0] if R0 is None else R0
    for it in range(maxit):
        r = np.mean([logSO3(R.T @ Ri) for Ri in Rs], axis=0)
        R = R @ expSO3(r)
        if np.linalg.norm(r) < tol:
            return R, it + 1
    return R, maxit


def mean_geodesic_L1(Rs, R0=None, tol=1e-12, maxit=2000):
    """Weiszfeld in the tangent space [Hartley13]: delta = sum v_i/|v_i| / sum 1/|v_i|."""
    R = mean_geodesic_L2(Rs)[0] if R0 is None else R0
    for it in range(maxit):
        vs = [logSO3(R.T @ Ri) for Ri in Rs]
        ns = np.array([np.linalg.norm(v) for v in vs])
        ns = np.maximum(ns, 1e-12)
        d = np.sum([v / n for v, n in zip(vs, ns)], axis=0) / np.sum(1 / ns)
        R = R @ expSO3(d)
        if np.linalg.norm(d) < tol:
            return R, it + 1
    return R, maxit


# ----------------------------------------------------------------------------- cumulative cubic B-spline

# rows j = 0..3 (lambda_j), columns powers u^0..u^3  [Sommer20 (18)-(21), Lovegrove13]
M_CUM = np.array([[6.0, 0, 0, 0], [5, 3, -3, 1], [1, 3, 3, -2], [0, 0, 0, 1]]) / 6.0


def bspline_lambdas(u):
    return M_CUM @ np.array([1.0, u, u * u, u ** 3])


def bspline_lambdas_du(u, order=1):
    if order == 1:
        return M_CUM @ np.array([0.0, 1.0, 2 * u, 3 * u * u])
    if order == 2:
        return M_CUM @ np.array([0.0, 0.0, 2.0, 6 * u])
    raise ValueError


def bspline_SE3(Ts, t):
    """Uniform cumulative cubic B-spline on SE(3) with knots at 0, 1, 2, ...; valid for 0 <= t <= len(Ts) - 3.

    X(u) = T_i prod_{j=1}^{3} Exp(lambda_j(u) d_j), d_j = Log(T_{i+j-1}^{-1} T_{i+j})  [Sommer20 (23)-(24)].
    """
    n = len(Ts)
    i = int(np.floor(t))
    i = min(max(i, 0), n - 4)
    u = t - i
    lam = bspline_lambdas(u)
    X = Ts[i].copy()
    for j in range(1, 4):
        d = logSE3(inv_T(Ts[i + j - 1]) @ Ts[i + j])
        X = X @ expSE3(lam[j] * d)
    return X


def bspline_SO3(Rs, t):
    n = len(Rs)
    i = int(np.floor(t))
    i = min(max(i, 0), n - 4)
    u = t - i
    lam = bspline_lambdas(u)
    X = Rs[i].copy()
    for j in range(1, 4):
        d = logSO3(Rs[i + j - 1].T @ Rs[i + j])
        X = X @ expSO3(lam[j] * d)
    return X
