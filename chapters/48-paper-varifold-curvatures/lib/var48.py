"""Helpers for Chapter 48 (Paper: Weak and Approximate Curvatures of a Measure, [BLM22] = arXiv:1904.05930v2).

Shared by the figure and verify scripts of this chapter. Conventions (Section 48.1, Conventions box):

- a plane S in G_{d,n} is stored as its orthogonal projection matrix (n x n);
- tensors are stored as arrays with the index order of the paper: A[i, j, k] = A_{ijk}
  (i = direction of differentiation, (j, k) = entry of the projection), B[i, j, k] = B^k_{ij};
- kernels are radial: rho_eps(x) = eps^-n rho(|x| / eps) with the profile rho(t) = exp(-1/(1 - t^2)) of
  [BLM22, Section 7.1] (not normalised: every ratio below is invariant under rescaling the profile) and the
  natural partner xi(t) = -t rho'(t) / n of (6.7), so that C_xi / C_rho = d / n;
- point cloud formulas ([BLM22] (6.4), (6.5), (6.9), (6.22) for V = sum_l m_l delta_{(x_l, P_l)}), with
  u_l = (x_l - x)/|x_l - x|, r_l = |x_l - x| / eps:

      beta_ijk(x) = -(C_xi/C_rho) (1/eps) sum_l m_l (P_l)_jk rho'(r_l) (P_l u_l)_i / sum_l m_l xi(r_l)

  The factor 1/eps comes from grad rho_eps = eps^(-n-1) rho'(|z|/eps) z/|z|; the printed formulas of
  [BLM22, Section 7.1] omit it (Section 48.6, Warning).
- principal curvatures ([BLM22, Section 7.1]): contract B^k_ij with a unit normal n of the plane at x and restrict to
  the plane: Wbar = Q^T (sum_k B^k_ij n_k) Q, Q an n x d orthonormal basis of the plane. For the outward normal of a
  sphere of radius R the eigenvalues approach -1/R (same sign convention as GUIDELINES §7, W = -dN).

Neighbour search: chunked brute force (pure NumPy, no scipy). Run from the project root with PYTHONPATH=tools.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import math  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# kernel profiles ([BLM22] Section 6 and Section 7.1)
# ---------------------------------------------------------------------------


def rho(t):
    """rho(t) = exp(-1/(1 - t^2)) for |t| < 1, else 0 (not normalised)."""
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = np.abs(t) < 1
    out[m] = np.exp(-1.0 / (1.0 - t[m] ** 2))
    return out


def drho(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    m = np.abs(t) < 1
    tm = t[m]
    out[m] = np.exp(-1.0 / (1.0 - tm ** 2)) * (-2.0 * tm / (1.0 - tm ** 2) ** 2)
    return out


def xi(t, n):
    """natural partner of rho: n xi(s) = -s rho'(s)  ([BLM22] (6.7))."""
    t = np.asarray(t, float)
    return -t * drho(t) / n


def radial_moment(prof, k, pieces=64, order=40):
    """int_0^1 prof(r) r^k dr by composite Gauss-Legendre."""
    x, w = np.polynomial.legendre.leggauss(order)
    x, w = 0.5 * (x + 1.0), 0.5 * w
    edges = np.linspace(0.0, 1.0, pieces + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = lo + (hi - lo) * x
        tot += (hi - lo) * float(np.sum(w * prof(t) * t ** k))
    return tot


def omega(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


def C_const(prof, d):
    """C = d omega_d int_0^1 prof(r) r^(d-1) dr  ([BLM22] (6.2))."""
    return d * omega(d) * radial_moment(prof, d - 1)


def ratio(d, n):
    """C_xi / C_rho for the natural pair, computed numerically (= d/n, Proposition 47.5.2)."""
    return C_const(lambda t: xi(t, n), d) / C_const(rho, d)


# ---------------------------------------------------------------------------
# samples and exact planes
# ---------------------------------------------------------------------------


def fib_sphere(N, R=1.0):
    """N quasi-uniform points on the sphere of radius R (Fibonacci lattice)."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = math.pi * (3.0 - math.sqrt(5.0)) * i
    s = np.sqrt(1.0 - z ** 2)
    return R * np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


def proj_from_normal(Nrm):
    """P = I - n n^T for unit normals (rows)."""
    Nrm = np.asarray(Nrm, float)
    n = Nrm.shape[-1]
    return np.eye(n) - Nrm[..., :, None] * Nrm[..., None, :]


def proj_from_tangent(T):
    """P = t t^T for unit tangents (rows) of a curve."""
    T = np.asarray(T, float)
    return T[..., :, None] * T[..., None, :]


def torus_point(u, v, R=2.0, r=0.8):
    """x(u, v) = ((R + r cos u) cos v, (R + r cos u) sin v, r sin u)  (GUIDELINES §7)."""
    return np.stack([(R + r * np.cos(u)) * np.cos(v), (R + r * np.cos(u)) * np.sin(v), r * np.sin(u)], axis=-1)


def torus_outward(u, v):
    """outward unit normal of the torus of revolution (opposite to x_u x x_v, GUIDELINES §7)."""
    return np.stack([np.cos(u) * np.cos(v), np.cos(u) * np.sin(v), np.sin(u)], axis=-1)


def torus_kappa_outward(u, R=2.0, r=0.8):
    """principal curvatures for the outward normal (W = -dN): (-1/r, -cos u/(R + r cos u))."""
    return -1.0 / r * np.ones_like(np.asarray(u, float)), -np.cos(u) / (R + r * np.cos(u))


def torus_sample(N, R=2.0, r=0.8, seed=0):
    """N points of T_{R,r}, uniform with respect to area (rejection on u with density (R + r cos u))."""
    rng = np.random.default_rng(seed)
    us = []
    while sum(len(a) for a in us) < N:
        u = rng.uniform(0, 2 * math.pi, 2 * N)
        keep = rng.uniform(0, R + r, 2 * N) < R + r * np.cos(u)
        us.append(u[keep])
    u = np.concatenate(us)[:N]
    v = rng.uniform(0, 2 * math.pi, N)
    return u, v


def cube_grid(k=61):
    """the surface of [0,1]^3 sampled on the grid of spacing 1/(k-1): 6k^2 - 12k + 8 points."""
    g = np.linspace(0.0, 1.0, k)
    pts = []
    for ax in range(3):
        for val in (0.0, 1.0):
            a, b = np.meshgrid(g, g, indexing="ij")
            P = np.zeros((k * k, 3))
            o = [c for c in range(3) if c != ax]
            P[:, ax] = val
            P[:, o[0]] = a.ravel()
            P[:, o[1]] = b.ravel()
            pts.append(P)
    X = np.concatenate(pts)
    key = np.round(X * (k - 1)).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    return X[np.sort(idx)]


# ---------------------------------------------------------------------------
# neighbours (chunked brute force)
# ---------------------------------------------------------------------------


def knn(X, k, Q=None, chunk=256):
    """indices (len(Q), k) and distances of the k nearest points of X to each query (the query itself counts
    when it belongs to X), sorted by distance."""
    X = np.asarray(X, float)
    Q = X if Q is None else np.asarray(Q, float)
    idx = np.zeros((len(Q), k), np.int64)
    dist = np.zeros((len(Q), k))
    sx = np.einsum("ij,ij->i", X, X)
    for q0 in range(0, len(Q), chunk):
        q = Q[q0:q0 + chunk]
        D2 = sx[None, :] - 2.0 * q @ X.T + np.einsum("ij,ij->i", q, q)[:, None]
        part = np.argpartition(D2, k - 1, axis=1)[:, :k]
        d2 = np.take_along_axis(D2, part, axis=1)
        o = np.argsort(d2, axis=1)
        idx[q0:q0 + chunk] = np.take_along_axis(part, o, axis=1)
        dist[q0:q0 + chunk] = np.sqrt(np.maximum(np.take_along_axis(d2, o, axis=1), 0.0))
    return idx, dist


def ball(X, x, eps):
    """indices of the points of X with |X - x| < eps."""
    D = np.asarray(X, float) - np.asarray(x, float)
    return np.nonzero(np.einsum("ij,ij->i", D, D) < eps * eps)[0]


# ---------------------------------------------------------------------------
# PCA tangent planes ([BLM22] Section 7.2)
# ---------------------------------------------------------------------------


def pca_planes(X, k, d=2, idx=None, dist=None):
    """tangent planes from the k nearest neighbours: barycenter of the neighbours, covariance weighted by
    rho(|x_j - x_i| / sigma) with sigma = distance to the k-th neighbour, top-d eigenvectors.
    Returns (P (N, n, n), normals (N, n) for codimension 1)."""
    X = np.asarray(X, float)
    if idx is None:
        idx, dist = knn(X, k)
    nb = X[idx]                                   # (N, k, n)
    bar = nb.mean(axis=1)
    sig = dist[:, -1:] * (1.0 + 1e-9)
    w = rho(dist / sig)
    Y = nb - bar[:, None, :]
    C = np.einsum("nk,nka,nkb->nab", w, Y, Y)
    evals, evecs = np.linalg.eigh(C)              # ascending
    Bas = evecs[:, :, -d:]
    P = np.einsum("nad,nbd->nab", Bas, Bas)
    return P, evecs[:, :, 0]


# ---------------------------------------------------------------------------
# epsilon-WSFF and epsilon-WSFF-perp of a point cloud varifold at one point
# ---------------------------------------------------------------------------


def local_terms(x, Xn, Pn, mn, eps, n, d):
    """(a, w, den): weights a_l = -(C_xi/C_rho) m_l rho'(r_l)/eps, vectors w_l = P_l u_l and the denominator
    sum m_l xi(r_l), so that beta_ijk(x) = sum_l a_l (P_l)_jk (w_l)_i / den."""
    D = np.asarray(Xn, float) - np.asarray(x, float)
    r = np.sqrt(np.einsum("ij,ij->i", D, D))
    s = r / eps
    u = np.zeros_like(D)
    nz = r > 0
    u[nz] = D[nz] / r[nz, None]
    w = np.einsum("lab,lb->la", Pn, u)
    a = -(d / n) * mn * drho(s) / eps
    a[~nz] = 0.0
    den = float(np.sum(mn * xi(s, n)))
    return a, w, den


def beta_tensor(x, Xn, Pn, mn, eps, n, d):
    a, w, den = local_terms(x, Xn, Pn, mn, eps, n, d)
    return np.einsum("l,li,ljk->ijk", a, w, Pn) / den


def H_from_beta(beta):
    """H_i = sum_q beta_qiq  ((6.6))."""
    return np.einsum("qiq->i", beta)


def c_eps(x, Xn, Pn, mn, eps, eta=rho):
    """regularized mean projection c^{V,eps}(x) ((6.5)) with the kernel profile eta."""
    D = np.asarray(Xn, float) - np.asarray(x, float)
    r = np.sqrt(np.einsum("ij,ij->i", D, D)) / eps
    wts = mn * eta(r)
    return np.einsum("l,lab->ab", wts, Pn) / wts.sum()


def solve_system(b, c):
    """unique solution of a_ijk + c_jk sum_q a_qiq = b_ijk ((3.4)) by the closed form (3.5)."""
    n = c.shape[0]
    h = np.einsum("qiq->i", b)
    u = np.linalg.solve(np.eye(n) + c, h)
    return b - np.einsum("jk,i->ijk", c, u)


def system_matrix(c):
    """the n^3 x n^3 matrix L of (3.4) in lexicographic order of (i, j, k)."""
    n = c.shape[0]
    N3 = n ** 3
    L = np.eye(N3)
    ind = lambda i, j, k: (i * n + j) * n + k  # noqa: E731
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for q in range(n):
                    L[ind(i, j, k), ind(q, i, q)] += c[j, k]
    return L


def B_from_A(A):
    """B^k_ij = (A_ijk + A_jik - A_kij)/2 stored as B[i, j, k]  ((3.8))."""
    return 0.5 * (A + np.swapaxes(A, -3, -2) - np.moveaxis(A, -3, -1))


def A_from_B(B):
    """A_ijk = B^k_ij + B^j_ik  ((1.3))."""
    return B + np.swapaxes(B, -2, -1)


def wsff_perp(x, P0, Xn, Pn, mn, eps, n, d):
    """A^{V,perp,eps}_ijk(x) = beta_ijk - (P0)_jk sum_q beta_qiq  ((6.22) with c^V(x) = P0)."""
    a, w, den = local_terms(x, Xn, Pn, mn, eps, n, d)
    Dl = Pn - P0[None]
    return np.einsum("l,li,ljk->ijk", a, w, Dl) / den


def wsff(x, Xn, Pn, mn, eps, n, d, eta=rho):
    """A^{V,eps}(x) ((6.9)): solve (6.8) with beta^{V,eps} and c^{V,eps}."""
    b = beta_tensor(x, Xn, Pn, mn, eps, n, d)
    c = c_eps(x, Xn, Pn, mn, eps, eta)
    return solve_system(b, c)


def shape_matrix(B, nrm, Q):
    """Wbar = Q^T (sum_k B^k_ij n_k) Q  ([BLM22] Section 7.1)."""
    Bn = np.einsum("ijk,k->ij", B, nrm)
    return Q.T @ Bn @ Q


def tangent_basis(nrm):
    """an orthonormal basis (3 x 2) of the plane orthogonal to the unit normal nrm."""
    nrm = np.asarray(nrm, float)
    a = np.array([1.0, 0.0, 0.0]) if abs(nrm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(nrm, a)
    t1 /= np.linalg.norm(t1)
    t2 = np.cross(nrm, t1)
    return np.stack([t1, t2], axis=1)


def principal_perp(x, P0, nrm, Xn, Pn, mn, eps, n=3, d=2):
    """sorted eigenvalues (kappa_min, kappa_max) of the epsilon-WSFF-perp shape matrix at x (codimension 1)."""
    A = wsff_perp(x, P0, Xn, Pn, mn, eps, n, d)
    W = shape_matrix(B_from_A(A), nrm, tangent_basis(nrm))
    return np.linalg.eigvalsh(0.5 * (W + W.T))


def shape_perp_fast(x, P0, nrm, Xn, Pn, mn, eps, n=3, d=2):
    """the same 3 x 3 contracted matrix sum_k B^k_ij n_k without forming the 27 entries:
    1/2 sum_l a_l [w (D n)^T + (D n) w^T - <w, n> D] / den with D = P_l - P0."""
    a, w, den = local_terms(x, Xn, Pn, mn, eps, n, d)
    Dl = Pn - P0[None]
    Dn = np.einsum("lab,b->la", Dl, nrm)
    wn = w @ nrm
    M = (np.einsum("l,la,lb->ab", a, w, Dn) + np.einsum("l,la,lb->ab", a, Dn, w)
         - np.einsum("l,l,lab->ab", a, wn, Dl))
    return 0.5 * M / den


def curvatures_knn(X, P, nrm, m, k, idx=None, dist=None):
    """epsilon-WSFF-perp principal curvatures at every point with eps_i = distance to the k-th nearest point
    (the reading of [BLM22] Section 7.3 adopted in Section 48.6). Returns (kappa (N, 2) ascending, eps (N,))."""
    if idx is None:
        idx, dist = knn(X, k)
    N = len(X)
    kap = np.zeros((N, 2))
    eps = dist[:, -1] * (1.0 + 1e-9)
    for i in range(N):
        nb = idx[i]
        Mx = shape_perp_fast(X[i], P[i], nrm[i], X[nb], P[nb], m[nb], eps[i])
        Q = tangent_basis(nrm[i])
        W = Q.T @ Mx @ Q
        kap[i] = np.linalg.eigvalsh(0.5 * (W + W.T))
    return kap, eps
