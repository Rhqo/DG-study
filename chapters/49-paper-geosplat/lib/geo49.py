"""Shared numerics for Chapter 49 (Paper: GeoSplat [GeoSplat25]).

Everything here implements a formula of the paper (arXiv:2509.05075v3) or a check of one. Figures and verify
scripts of Chapter 49 import this module; the printed Implementation boxes are independent short versions of the
same formulas, and the verify scripts compare the two.

Conventions (Section 49.1, Conventions box):
- transpose is ``.T``; the tangent part of a vector is P v and the normal part (I - P) v (the paper's (.)^T, (.)^perp);
- principal curvatures are eigenvalues of the shape operator s(n, u1, u2) = -<D_{u1} N, u2> = <W u1, u2>
  (W = -dN, Section 6.3 of GUIDELINES), so a unit sphere with the outward normal has tau = (-1, -1);
- GeoSplat orders the two principal curvatures by size: tau_1 is the high (|tau_1| >= |tau_2|), tau_2 the low one.

Run from the project root with PYTHONPATH=tools.
"""
import math

import numpy as np

# ---------------------------------------------------------------------------
# running surfaces of the study set (GUIDELINES §7, Track 2 values) and point samples on them
# ---------------------------------------------------------------------------

TORUS_R, TORUS_r = 2.0, 0.8


def sphere_points(n, radius=1.0, seed=0, pole=True):
    """n points uniform w.r.t. area on the sphere of the given radius; the first one is the north pole."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X *= radius
    if pole:
        X[0] = (0.0, 0.0, radius)
    return X


def sphere_points_density(n, a=0.9, seed=0):
    """n points on the unit sphere with density proportional to 1 + a z (rejection sampling). First point: north pole."""
    rng = np.random.default_rng(seed)
    out = []
    while sum(len(o) for o in out) < n:
        X = rng.normal(size=(4 * n, 3))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        keep = rng.uniform(0, 1 + abs(a), size=len(X)) < 1 + a * X[:, 2]
        out.append(X[keep])
    X = np.concatenate(out)[:n]
    X[0] = (0.0, 0.0, 1.0)
    return X


def torus_xyz(u, v, R=TORUS_R, r=TORUS_r):
    """the running parametrization x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u)."""
    return np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], axis=-1)


def torus_points(n, R=TORUS_R, r=TORUS_r, seed=0):
    """n points uniform w.r.t. area on T_{R,r} (rejection with weight (R + r cos u)/(R + r)); first point: (R + r, 0, 0)."""
    rng = np.random.default_rng(seed)
    U, V = [], []
    while sum(len(x) for x in U) < n:
        u = rng.uniform(0, 2 * np.pi, 4 * n)
        v = rng.uniform(0, 2 * np.pi, 4 * n)
        keep = rng.uniform(0, 1, 4 * n) < (R + r * np.cos(u)) / (R + r)
        U.append(u[keep])
        V.append(v[keep])
    u = np.concatenate(U)[:n]
    v = np.concatenate(V)[:n]
    u[0] = v[0] = 0.0
    return torus_xyz(u, v, R, r)


def torus_distance(P, R=TORUS_R, r=TORUS_r):
    """distance from points P to the torus T_{R,r}."""
    P = np.atleast_2d(P)
    rho = np.hypot(P[:, 0], P[:, 1])
    return np.abs(np.hypot(rho - R, P[:, 2]) - r)


def saddle_points(n, half=0.6, seed=0):
    """n points uniform w.r.t. area on the saddle z = x^2 - y^2 over [-half, half]^2; first point: the origin."""
    rng = np.random.default_rng(seed)
    pts = []
    wmax = math.sqrt(1 + 8 * half * half)
    while sum(len(p) for p in pts) < n:
        xy = rng.uniform(-half, half, size=(4 * n, 2))
        w = np.sqrt(1 + 4 * xy[:, 0] ** 2 + 4 * xy[:, 1] ** 2)
        pts.append(xy[rng.uniform(0, wmax, size=len(xy)) < w])
    xy = np.concatenate(pts)[:n]
    xy[0] = 0.0
    return np.c_[xy, xy[:, 0] ** 2 - xy[:, 1] ** 2]


def cylinder_points(n, radius=1.0, half=1.0, seed=0):
    """n points uniform on the cylinder x^2 + y^2 = r^2, |z| <= half; first point: (r, 0, 0)."""
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(-half, half, n)
    th[0] = z[0] = 0.0
    return np.c_[radius * np.cos(th), radius * np.sin(th), z]



ROOM_L = (4.0, 3.0, 2.5)            # Section 49.6: synthetic room (metres), floor z = 0
ROOM_BALL_C, ROOM_BALL_R = (2.0, 1.5, 0.4), 0.4


def room_points(dens=1500, sigma=0.0, seed=0):
    """Section 49.6: points uniform on the six faces of the box [0,4]x[0,3]x[0,2.5] (dens per square metre) and on a
    ball of radius 0.4 resting on the floor at (2, 1.5); each point moved by isotropic Gaussian noise sigma.
    Returns (X noisy, X0 clean, label) with label 0 = box face, 1 = ball."""
    rng = np.random.default_rng(seed)
    L = np.array(ROOM_L)
    faces = []
    for ax in range(3):
        area = np.prod(np.delete(L, ax))
        for val in (0.0, L[ax]):
            P = rng.uniform(0, 1, (int(area * dens), 3)) * L
            P[:, ax] = val
            faces.append(P)
    c, r = np.array(ROOM_BALL_C), ROOM_BALL_R
    B = rng.normal(size=(int(4 * np.pi * r * r * dens), 3))
    B = c + r * B / np.linalg.norm(B, axis=1, keepdims=True)
    X0 = np.vstack(faces + [B])
    lab = np.r_[np.zeros(len(X0) - len(B), int), np.ones(len(B), int)]
    return X0 + sigma * rng.normal(size=X0.shape), X0, lab


def room_edge_distance(X0):
    """distance of points on the box faces to the nearest edge of the box (the line where two faces meet)."""
    L = np.array(ROOM_L)
    d = np.minimum(X0, L - X0)                 # distance to the two parallel faces in each axis
    d.sort(axis=1)                             # d[:, 0] = 0 for a face point; the edge distance is the next one
    return d[:, 1]

# ---------------------------------------------------------------------------
# Section 49.3: covariance warm-up (paper (3)-(5), Prop 3.1), MAC, truncation (7), split (10)
# ---------------------------------------------------------------------------

def warmup_scales(tau1, tau2, s_nbr, xi_min=1e-3, xi_max=None, clamp=True):
    """paper (5): s1 = (s_nbr/2) sqrt|tau1/tau2|, s2 = (s_nbr/2) sqrt|tau2/tau1|, s3 = xi_min.
    The clamp min(max(|tau|, xi_min), xi_max) is applied to |tau| (the printed clamp acts on tau; see Section 49.3)."""
    a1, a2 = abs(tau1), abs(tau2)
    if clamp:
        a1 = max(a1, xi_min)
        a2 = max(a2, xi_min)
        if xi_max is not None:
            a1, a2 = min(a1, xi_max), min(a2, xi_max)
    c = s_nbr / 2
    return np.array([c * math.sqrt(a1 / a2), c * math.sqrt(a2 / a1), xi_min])


def mac(taus):
    """mean absolute curvature (Definition 49.3.4): (sum_d |tau_d|) / D_M."""
    taus = np.asarray(taus, float)
    return np.abs(taus).sum(axis=-1) / taus.shape[-1]


def truncate_normal(g, n, xi_min=1e-3):
    """the update direction of paper (7): g^T + min(xi_min/|g^perp|, 1) g^perp (n: unit normal)."""
    gn = np.dot(g, n) * n
    gt = g - gn
    norm = np.linalg.norm(gn)
    return gt + (min(xi_min / norm, 1.0) if norm > 0 else 1.0) * gn


def split_samples_eq10(mu, w1, w2, u3, tau1, tau2, n, xi_min=1e-3, seed=0):
    """paper (10) as printed: mu + (rho2/tau2) w2 + (rho1/tau1) w1 + rho3 xi_min u3, rho ~ N(0, I)."""
    rng = np.random.default_rng(seed)
    rho = rng.normal(size=(n, 3))
    return mu + np.outer(rho[:, 1] / tau2, w2) + np.outer(rho[:, 0] / tau1, w1) + np.outer(rho[:, 2] * xi_min, u3)


# ---------------------------------------------------------------------------
# Section 49.5: the manifold estimator (App. E) with a kernel generator (paper (45)-(46))
# ---------------------------------------------------------------------------

def generator_weights(q, X, t, mode="rw", density=None):
    """weights a_j of a kernel generator L f(q) = sum_j a_j (f(x_j) - f(q)) built from k = exp(-|q - x|^2 / t).

    mode "rw":    a_j = (4/t) k_j / sum_l k_l (random-walk normalization; tends to Laplace-Beltrami for any smooth
                  positive sampling density, Remark 42.1.9).
    mode "paper": a_j = (4/t) k_j / (density * pi * t), i.e. paper (45) with the constant C fixed so that a *uniform*
                  sample with the given number of points per unit area gives Laplace-Beltrami (D_M = 2).
    q may be one point (3,) or several (m, 3); the result has shape (len(X),) or (m, len(X))."""
    q = np.asarray(q, float)
    single = q.ndim == 1
    Q = q[None] if single else q
    d2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    k = np.exp(-d2 / t)
    if mode == "rw":
        a = (4.0 / t) * k / k.sum(1, keepdims=True)
    elif mode == "paper":
        a = (4.0 / t) * k / (density * np.pi * t)
    else:
        raise ValueError(mode)
    return a[0] if single else a


def lbnz(a, fq, fX, hq, hX):
    """Lbnz_L[f, h](q) = L[f h](q) - f(q) L[h](q) - h(q) L[f](q) for L f(q) = sum_j a_j (f_j - f(q)).
    It equals sum_j a_j (f_j - f(q)) (h_j - h(q)) (Proposition 49.5.6). a: (m, N) or (N,), fq: (m,) or scalar."""
    return (a * (fX - np.asarray(fq)[..., None]) * (hX - np.asarray(hq)[..., None])).sum(-1)


def lbnz_literal(a, fq, fX, hq, hX):
    """the same quantity computed literally as L[fh] - f L[h] - h L[f] (for checking Proposition 49.5.6)."""
    L = lambda F, Fq: (a * (F - np.asarray(Fq)[..., None])).sum(-1)
    return L(fX * hX, fq * hq) - fq * L(hX, hq) - hq * L(fX, fq)


def manifold_estimate(X, q, t, mode="rw", density=None, n_ref=None, cut=3.0):
    """GeoSplat's manifold estimator at q (App. E): K (eq. 42-43), normal (Cor. E.5), shape operator (eq. 50-51).

    Returns dict with K (3x3), eigK (ascending), n (unit normal, oriented like n_ref if given), U (3x2 tangent basis),
    S3 (3x3 matrix s(n, v~_a, v~_b) on the unnormalized coordinate vectors), S (2x2), tau (eigenvalues of S ordered by
    |tau| descending), W (3x2 principal directions realigned to R^3, columns in the order of tau)."""
    q = np.asarray(q, float)
    r1 = cut * math.sqrt(t)
    dq = np.linalg.norm(X - q, axis=1)
    loc = X[dq < 2 * r1]                       # points that can enter a nested sum
    J = loc[np.linalg.norm(loc - q, axis=1) < r1]
    aq = generator_weights(q, loc, t, mode, density)
    D = loc - q
    K = 0.5 * (aq[:, None, None] * D[:, :, None] * D[:, None, :]).sum(0)        # (1/2) Lbnz[phi_a, phi_b](q)
    ev, E = np.linalg.eigh(K)
    n = E[:, 0]
    if n_ref is not None and n @ n_ref < 0:
        n = -n
    U = E[:, 1:]
    eta = (loc - q) @ n                        # height function, eta(q) = 0
    aJ = generator_weights(J, loc, t, mode, density)           # (m, N) weights at the neighbours y in J
    etaJ = (J - q) @ n
    # G_b(y) = Lbnz[eta, phi_b](y) and F_ab(y) = Lbnz[phi_a, phi_b](y) at y in J and at q
    G_J = np.einsum("mn,mn,mnb->mb", aJ, eta[None, :] - etaJ[:, None], loc[None, :, :] - J[:, None, :])
    G_q = np.einsum("n,n,nb->b", aq, eta, D)
    F_J = np.einsum("mn,mna,mnb->mab", aJ, loc[None, :, :] - J[:, None, :], loc[None, :, :] - J[:, None, :])
    F_q = 2 * K
    aqJ = generator_weights(q, J, t, mode, density)
    DJ = J - q
    # A[phi_a, eta, phi_b](q) = Lbnz[phi_a, G_b](q);  A[eta, phi_a, phi_b](q) = Lbnz[eta, F_ab](q)
    A1 = np.einsum("m,ma,mb->ab", aqJ, DJ, G_J - G_q[None, :])
    A3 = np.einsum("m,m,mab->ab", aqJ, etaJ, F_J - F_q[None, :, :])
    S3 = (A1 + A1.T - A3) / 8.0
    S = U.T @ S3 @ U
    S = 0.5 * (S + S.T)
    tau, w = np.linalg.eigh(S)
    order = np.argsort(-np.abs(tau))
    tau, w = tau[order], w[:, order]
    return {"K": K, "eigK": ev, "n": n, "U": U, "S3": S3, "S": S, "tau": tau, "W": U @ w,
            "nJ": len(J), "nloc": len(loc), "A1": A1, "A3": A3}


def local_pca_normal(X, q, k):
    """the classical local PCA normal: smallest eigenvector of the covariance of the k nearest neighbours (centered at
    their mean)."""
    d = np.linalg.norm(X - q, axis=1)
    nb = X[np.argsort(d)[:k]]
    C = np.cov((nb - nb.mean(0)).T)
    return np.linalg.eigh(C)[1][:, 0]


# ---------------------------------------------------------------------------
# Section 49.4: the varifold estimator (App. D) — [BLM22] (7.2) and GeoSplat's 2x2 closed form (12)/(38)
# ---------------------------------------------------------------------------

def rho_profile(s):
    """[BLM22, Section 7.1] profile rho(s) = exp(-1/(1 - s^2)) on [0, 1), 0 outside (unnormalized: constants cancel)."""
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    m = s < 1
    out[m] = np.exp(-1.0 / (1.0 - s[m] ** 2))
    return out


def drho_profile(s):
    s = np.asarray(s, float)
    out = np.zeros_like(s)
    m = s < 1
    sm = s[m]
    out[m] = np.exp(-1.0 / (1.0 - sm ** 2)) * (-2.0 * sm / (1.0 - sm ** 2) ** 2)
    return out


def xi_profile(s, n_amb=3):
    """natural kernel pair [BLM22, (6.7)]: n xi(s) = -s rho'(s)."""
    return -np.asarray(s, float) * drho_profile(s) / n_amb


def blm22_tensor(X, P, m, l0, eps, with_inv_eps=True, d=2):
    """[BLM22, (7.2)]: the eps-WSFF-perp tensor B^{k}_{ij}(x_l0) of a point cloud varifold sum m_l delta_(x_l, P_l).
    Returns the 3x3x3 array B[i, j, k]. with_inv_eps multiplies by the 1/eps that comes from the derivative of
    rho_eps(x) = eps^-n rho(|x|/eps) ([BLM22, (6.22)]) and that (7.2) and GeoSplat (31) leave out."""
    n_amb = X.shape[1]
    x0, P0 = X[l0], P[l0]
    diff = x0 - X
    dist = np.linalg.norm(diff, axis=1)
    idx = np.where((dist > 0) & (dist < eps))[0]
    s = dist[idx] / eps
    den = (m[idx] * xi_profile(s, n_amb)).sum()
    B = np.zeros((n_amb, n_amb, n_amb))
    for l, wl in zip(idx, m[idx] * drho_profile(s)):
        v = P[l] @ diff[l] / dist[l]
        Dl = P[l] - P0
        B += wl * 0.5 * (np.einsum("jk,i->ijk", Dl, v) + np.einsum("ik,j->ijk", Dl, v) - np.einsum("ij,k->ijk", Dl, v))
    B *= (d / n_amb) / den
    if with_inv_eps:
        B /= eps
    return B


def geosplat_B(X, normals, m, i, eps, sign=-1.0, with_inv_eps=True, Vi=None):
    """GeoSplat's 2x2 closed form (12)/(38) at point i, with the projection V~^j = I - u3^j u3^j^T (the corrected
    reading) and the second term multiplied by `sign` (printed: +1; derived from (34) and [BLM22, (7.2)]: -1).
    Vi: 3x2 tangent basis at i (default: from the normal). Kernels: Upsilon = rho, chi = xi of [BLM22] (paper: unspecified)."""
    u = normals[i]
    if Vi is None:
        Vi = tangent_basis(u)
    dlt = X[i] - X
    dist = np.linalg.norm(dlt, axis=1)
    idx = np.where((dist > 0) & (dist < eps))[0]
    s = dist[idx] / eps
    den = (m[idx] * xi_profile(s)).sum()
    num = np.zeros((2, 2))
    for j, wl in zip(idx, m[idx] * drho_profile(s)):
        Vj = np.eye(3) - np.outer(normals[j], normals[j])
        A = Vj @ Vi
        Sym = 0.5 * (np.outer(u, dlt[j]) + np.outer(dlt[j], u))
        first = 2 * A.T @ Sym @ A
        second = (dlt[j] @ Vj @ u) * (Vi.T @ Vj @ Vi - np.eye(2))
        num += wl * (first + sign * second) / (3 * dist[j])
    B = num / den
    if with_inv_eps:
        B /= eps
    return B


def geosplat_B_fast(X, normals, m, i, eps, sign=-1.0, with_inv_eps=True, Vi=None):
    """vectorized geosplat_B (same arguments and result); used by the figures of Section 49.4."""
    u = normals[i]
    V = tangent_basis(u) if Vi is None else Vi
    d = X[i] - X
    r = np.linalg.norm(d, axis=1)
    k = (r > 0) & (r < eps)
    d, r, Nj, mj = d[k], r[k], normals[k], m[k]
    P = np.eye(3) - Nj[:, :, None] * Nj[:, None, :]
    A = P @ V
    S = 0.5 * (u[None, :, None] * d[:, None, :] + d[:, :, None] * u[None, None, :])
    first = 2 * np.einsum("kai,kab,kbj->kij", A, S, A)
    c = np.einsum("ka,kab,b->k", d, P, u)
    second = c[:, None, None] * (np.einsum("ai,kab,bj->kij", V, P, V) - np.eye(2))
    w = mj * drho_profile(r / eps) / (3 * r)
    B = np.einsum("k,kij->ij", w, first + sign * second) / (mj * xi_profile(r / eps)).sum()
    return B / eps if with_inv_eps else B


def tangent_basis(n):
    """an orthonormal basis (3x2) of the plane orthogonal to the unit vector n."""
    n = np.asarray(n, float)
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u1 = a - (a @ n) * n
    u1 /= np.linalg.norm(u1)
    return np.c_[u1, np.cross(n, u1)]


def principal_from_2x2(B):
    """eigenvalues of the symmetrized 2x2 matrix ordered by |.| descending (tau_1 high, tau_2 low)."""
    ev, w = np.linalg.eigh(0.5 * (B + B.T))
    o = np.argsort(-np.abs(ev))
    return ev[o], w[:, o]


# ---------------------------------------------------------------------------
# Section 49.6: the tables of the paper (copied from the arXiv v3 PDF, Tables 1-3)
# ---------------------------------------------------------------------------

REPLICA = ["R0", "R1", "R2", "OFF0", "OFF1", "OFF2", "OFF3", "OFF4"]
TABLE1 = {  # method -> metric -> 8 values (Table 1 of [GeoSplat25]); "0.971" is printed as is
    "Vox-Fusion": {"PSNR": [22.39, 22.36, 23.92, 27.79, 29.83, 20.33, 23.47, 25.21],
                   "SSIM": [0.683, 0.751, 0.798, 0.857, 0.876, 0.794, 0.803, 0.847],
                   "LPIPS": [0.303, 0.269, 0.234, 0.241, 0.184, 0.243, 0.213, 0.199]},
    "Point-SLAM": {"PSNR": [32.40, 34.08, 35.50, 38.26, 39.16, 33.99, 33.48, 33.49],
                   "SSIM": [0.974, 0.977, 0.982, 0.983, 0.986, 0.960, 0.960, 0.979],
                   "LPIPS": [0.113, 0.116, 0.111, 0.100, 0.118, 0.156, 0.132, 0.142]},
    "GS-SLAM": {"PSNR": [34.83, 36.43, 37.49, 39.95, 42.09, 36.24, 36.70, 36.07],
                "SSIM": [0.954, 0.959, 0.965, 0.971, 0.977, 0.964, 0.963, 0.957],
                "LPIPS": [0.068, 0.076, 0.070, 0.072, 0.055, 0.078, 0.065, 0.099]},
    "GeoGaussian": {"PSNR": [35.20, 38.24, 39.14, 42.74, 42.20, 37.31, 36.66, 38.74],
                    "SSIM": [0.952, 0.979, 0.970, 0.981, 0.970, 0.970, 0.964, 0.967],
                    "LPIPS": [0.029, 0.021, 0.024, 0.016, 0.040, 0.029, 0.029, 0.031]},
    "Manifold": {"PSNR": [36.37, 39.55, 40.36, 43.38, 42.70, 37.74, 37.06, 39.31],
                 "SSIM": [0.976, 0.982, 0.985, 0.984, 0.988, 0.975, 0.969, 0.972],
                 "LPIPS": [0.024, 0.018, 0.021, 0.015, 0.037, 0.027, 0.027, 0.028]},
    "Varifold": {"PSNR": [36.35, 39.54, 40.05, 43.41, 42.64, 37.72, 37.08, 39.52],
                 "SSIM": [0.973, 0.975, 0.981, 0.985, 0.986, 0.974, 0.968, 0.981],
                 "LPIPS": [0.025, 0.019, 0.022, 0.013, 0.971, 0.028, 0.028, 0.027]},
}
ICL = ["Room-1", "Room-2", "Office-2", "Office-3"]
TABLE2 = {
    "3DGS": {"PSNR": [40.79, 39.10, 37.88, 36.04], "SSIM": [0.973, 0.974, 0.962, 0.975], "LPIPS": [0.025, 0.017, 0.024, 0.017]},
    "LightGS": {"PSNR": [41.26, 39.23, 37.99, 36.06], "SSIM": [0.974, 0.974, 0.962, 0.975], "LPIPS": [0.023, 0.017, 0.023, 0.016]},
    "GeoGaussian": {"PSNR": [41.43, 39.46, 38.54, 36.19], "SSIM": [0.976, 0.975, 0.967, 0.977], "LPIPS": [0.019, 0.018, 0.017, 0.015]},
    "Manifold": {"PSNR": [41.81, 39.97, 38.75, 36.61], "SSIM": [0.978, 0.979, 0.971, 0.978], "LPIPS": [0.016, 0.013, 0.016, 0.013]},
    "Varifold": {"PSNR": [41.92, 39.80, 38.68, 36.60], "SSIM": [0.980, 0.977, 0.969, 0.979], "LPIPS": [0.015, 0.015, 0.017, 0.012]},
}
ABLATION_ROWS = ["full", "warm-up (5)", "upsampling (6)", "MAC -> mean (SAGS)", "truncation (7)", "shape reg. (8)-(9)",
                 "densification (10)-(11)"]
TABLE3 = {  # model -> rows (ABLATION_ROWS order) -> (R0 PSNR, SSIM, LPIPS, Office-3 PSNR, SSIM, LPIPS)
    "Manifold": [(36.37, 0.976, 0.024, 36.61, 0.978, 0.013), (36.01, 0.969, 0.026, 36.43, 0.976, 0.014),
                 (36.12, 0.971, 0.025, 36.51, 0.976, 0.015), (36.25, 0.973, 0.025, 36.58, 0.977, 0.014),
                 (36.09, 0.970, 0.026, 36.47, 0.975, 0.015), (35.98, 0.965, 0.027, 36.39, 0.973, 0.016),
                 (36.05, 0.970, 0.026, 36.48, 0.975, 0.015)],
    "Varifold": [(36.35, 0.973, 0.025, 36.60, 0.979, 0.012), (36.00, 0.963, 0.027, 36.41, 0.976, 0.015),
                 (36.17, 0.968, 0.026, 36.53, 0.977, 0.013), (36.23, 0.971, 0.026, 36.57, 0.978, 0.013),
                 (36.05, 0.967, 0.028, 36.45, 0.977, 0.014), (35.93, 0.961, 0.029, 36.37, 0.975, 0.016),
                 (36.03, 0.964, 0.028, 36.43, 0.977, 0.014)],
}

# our reading of [GeoSplat25, Fig. 1] (approximate: about +-0.1 dB and +-0.002; None = hidden behind another curve)
FIG1_X = ["0.1", "1/6", "0.5", "1.0"]           # the vertices of the curves, estimated from the plot (not stated in the paper)
FIG1_READ = {
    ("R1", "PSNR"): {"LightGS": [30.7, 34.0, 37.7, 38.4], "3DGS": [30.6, 33.9, 37.4, 37.6],
                     "GeoGaussian": [31.7, 35.1, 38.0, 38.2], "Manifold": [33.2, 37.9, 39.0, 39.6],
                     "Varifold": [32.6, 37.4, 38.8, 39.5]},
    ("R2", "PSNR"): {"LightGS": [31.6, 35.9, 38.7, 39.0], "3DGS": [31.6, 35.8, 38.5, 38.7],
                     "GeoGaussian": [32.2, 36.8, 38.8, 39.1], "Manifold": [33.9, 39.1, 39.9, 40.35],
                     "Varifold": [33.3, 38.8, 39.65, 40.0]},
    ("Office-2", "LPIPS"): {"LightGS": [0.146, 0.079, 0.035, 0.023], "3DGS": [0.143, 0.079, 0.036, 0.024],
                            "GeoGaussian": [0.099, 0.056, 0.026, 0.017], "Manifold": [0.088, 0.034, 0.023, 0.016],
                            "Varifold": [0.074, 0.032, 0.022, 0.017]},
    ("Office-3", "LPIPS"): {"LightGS": [None, None, None, None], "3DGS": [0.22, 0.144, 0.044, 0.018],
                            "GeoGaussian": [0.143, 0.082, 0.024, 0.016], "Manifold": [0.136, 0.039, 0.021, 0.014],
                            "Varifold": [0.122, 0.036, 0.020, 0.012]},
}
# readings that were partly hidden by another curve (printed with "≈")
FIG1_APPROX = {("R2", "PSNR", "LightGS", 0), ("Office-2", "LPIPS", "LightGS", 1), ("Office-2", "LPIPS", "GeoGaussian", 3),
               ("Office-2", "LPIPS", "Manifold", 0), ("Office-3", "LPIPS", "3DGS", 0), ("Office-3", "LPIPS", "Manifold", 0),
               ("Office-3", "LPIPS", "Manifold", 2)}
# the cells printed in bold in [GeoSplat25, Tables 1-2]: metric -> scene -> variant
BOLD1 = {"PSNR": dict(zip(["R0", "R1", "R2", "OFF0", "OFF1", "OFF2", "OFF3", "OFF4"], "MMMVMMVV")),
         "SSIM": dict(zip(["R0", "R1", "R2", "OFF0", "OFF1", "OFF2", "OFF3", "OFF4"], "MMMVMMMV")),
         "LPIPS": dict(zip(["R0", "R1", "R2", "OFF0", "OFF1", "OFF2", "OFF3", "OFF4"], "MMMVMMMV"))}
BOLD2 = {"PSNR": dict(zip(ICL, "VMMM")), "SSIM": dict(zip(ICL, "VMMV")), "LPIPS": dict(zip(ICL, "VMMV"))}
REPLICA = ["R0", "R1", "R2", "OFF0", "OFF1", "OFF2", "OFF3", "OFF4"]


# ---------------------------------------------------------------------------
# Section 49.7: the upsampling (6) as a refinement without a mass ledger
# ---------------------------------------------------------------------------

def upsample_midpoints(X, passes, k=10):
    """paper (6) / (49.3.5): every point i with passes[i] adds the midpoints (x_i + x_j)/2 with its k nearest
    neighbours j (brute force). Returns (M, pairs, mult): the added points (one per directed pair, duplicates kept),
    the directed pairs (i, j), and for each added point the number of directed pairs that produced the same
    location (2 when j also chose i and also passes, 1 otherwise)."""
    X = np.asarray(X, float)
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(D, np.inf)
    nn = np.argsort(D, axis=1)[:, :k]
    chosen = set()
    pairs = []
    for i in np.where(passes)[0]:
        for j in nn[i]:
            pairs.append((i, j))
            chosen.add((i, j))
    pairs = np.array(pairs, int).reshape(-1, 2)
    mult = np.array([1 + ((j, i) in chosen) for i, j in pairs], int)
    M = 0.5 * (X[pairs[:, 0]] + X[pairs[:, 1]]) if len(pairs) else np.zeros((0, X.shape[1]))
    return M, pairs, mult
