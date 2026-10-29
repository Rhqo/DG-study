"""Helpers for Chapter 46 (Background: Measures, Varifolds, and First Variation).

Shared by the figure and verify scripts of this chapter. Conventions (Section 46.3–46.5):

- a point of G_{d,n} (an unoriented d-plane) is stored as its orthogonal projection matrix P (n x n, P = P^T = P^2,
  tr P = d); the distance on G_{d,n} is the operator norm ||P - Q|| as in [BLM17];
- tangential divergence: div_P X = tr(P DX), DX[j, k] = dX_j / dx_k;
- a finite varifold ("atoms") is a list of (x_i, P_i, m_i) and integrates a test function phi(x, P) as
  sum_i m_i phi(x_i, P_i);
- first variation delta V(X) = int div_S X dV;  for a smooth closed curve or surface
  delta V(X) = - int <H, X> d||V||  with  H = Delta_M x  ( = kappa_s n_s for a curve, 2 H N for a surface ),
  so on a circle or sphere H points to the centre;
- kernels: radial profiles rho(r) supported in [0, 1], rho_eps(x) = eps^-n c rho(|x| / eps) normalised so that the
  integral over R^n is 1;  C_rho(d) = d omega_d int_0^1 rho(r) r^(d-1) dr  ([BLM17] (4.1)).

Run from the project root with PYTHONPATH=tools.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import math  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# quadrature
# ---------------------------------------------------------------------------

_GL_CACHE = {}


def gauss_legendre(n):
    """nodes and weights of n-point Gauss-Legendre on [0, 1]."""
    if n not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(n)
        _GL_CACHE[n] = (0.5 * (x + 1.0), 0.5 * w)
    return _GL_CACHE[n]


def integrate_1d(f, a, b, n=64, pieces=1):
    """composite Gauss-Legendre integral of a vectorised f on [a, b]."""
    x, w = gauss_legendre(n)
    edges = np.linspace(a, b, pieces + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = lo + (hi - lo) * x
        total = total + (hi - lo) * np.tensordot(w, f(t), axes=(0, 0))
    return total


# ---------------------------------------------------------------------------
# running examples (GUIDELINES 7): circle (r cos t, r sin t), ellipse (2 cos t, sin t)
# ---------------------------------------------------------------------------


def circle(t, r=1.0):
    t = np.asarray(t, float)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=-1)


def circle_tangent(t):
    t = np.asarray(t, float)
    return np.stack([-np.sin(t), np.cos(t)], axis=-1)


def ellipse(t):
    t = np.asarray(t, float)
    return np.stack([2.0 * np.cos(t), np.sin(t)], axis=-1)


def ellipse_velocity(t):
    t = np.asarray(t, float)
    return np.stack([-2.0 * np.sin(t), np.cos(t)], axis=-1)


def ellipse_speed(t):
    return np.linalg.norm(ellipse_velocity(t), axis=-1)


def ellipse_tangent(t):
    v = ellipse_velocity(t)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ellipse_kappa(t):
    """signed curvature of (2 cos t, sin t) (counter-clockwise, so kappa_s > 0)."""
    t = np.asarray(t, float)
    return 2.0 / (4.0 * np.sin(t) ** 2 + np.cos(t) ** 2) ** 1.5


def ellipse_normal(t):
    """n_s = J t (left normal); for the counter-clockwise ellipse it points inward."""
    tt = ellipse_tangent(t)
    return np.stack([-tt[..., 1], tt[..., 0]], axis=-1)


def ellipse_length(a=0.0, b=2 * np.pi):
    return float(integrate_1d(ellipse_speed, a, b, n=64, pieces=16))


ELLIPSE_LENGTH = None  # filled lazily by ellipse_length_total()


def ellipse_length_total():
    global ELLIPSE_LENGTH
    if ELLIPSE_LENGTH is None:
        ELLIPSE_LENGTH = ellipse_length()
    return ELLIPSE_LENGTH


# ---------------------------------------------------------------------------
# Grassmannian (Section 46.3)
# ---------------------------------------------------------------------------


def line_proj(theta):
    """projection onto the line in R^2 with angle theta (mod pi)."""
    theta = np.asarray(theta, float)
    c, s = np.cos(theta), np.sin(theta)
    P = np.empty(theta.shape + (2, 2))
    P[..., 0, 0] = c * c
    P[..., 0, 1] = c * s
    P[..., 1, 0] = c * s
    P[..., 1, 1] = s * s
    return P


def dir_proj(u):
    """projection onto the line spanned by u (any nonzero vector, any dimension)."""
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)
    return u[..., :, None] * u[..., None, :]


def plane_proj(normal):
    """projection onto the hyperplane with the given (unoriented) normal: I - n n^T."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    k = n.shape[-1]
    return np.eye(k) - n[..., :, None] * n[..., None, :]


def op_dist(P, Q):
    return float(np.linalg.norm(np.asarray(P) - np.asarray(Q), 2))


def fro_dist(P, Q):
    return float(np.linalg.norm(np.asarray(P) - np.asarray(Q), "fro"))


def is_projection(P, d, tol=1e-12):
    P = np.asarray(P, float)
    return (np.allclose(P, P.T, atol=tol) and np.allclose(P @ P, P, atol=tol)
            and abs(np.trace(P) - d) < 1e-9)


def proj_coords(P):
    """(P11 - P22, 2 P12): for a line projection this is (cos 2 theta, sin 2 theta) (Figure 46.3.1)."""
    P = np.asarray(P, float)
    return np.stack([P[..., 0, 0] - P[..., 1, 1], 2.0 * P[..., 0, 1]], axis=-1)


def tangential_div(P, DX):
    """div_P X = tr(P DX) with DX[j, k] = d X_j / d x_k."""
    return float(np.trace(np.asarray(P) @ np.asarray(DX)))


# ---------------------------------------------------------------------------
# kernels (Section 46.7)
# ---------------------------------------------------------------------------

OMEGA = {0: 1.0, 1: 2.0, 2: math.pi, 3: 4.0 * math.pi / 3.0}


def rho_exp(r):
    """C^infinity profile exp(-1 / (1 - r^2)) on [0, 1), 0 for r >= 1 ([BLM17] Section 8)."""
    r = np.abs(np.asarray(r, float))
    out = np.zeros_like(r)
    m = r < 1.0
    out[m] = np.exp(-1.0 / (1.0 - r[m] ** 2))
    return out


def drho_exp(r):
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    m = np.abs(r) < 1.0
    rr = r[m]
    out[m] = np.exp(-1.0 / (1.0 - rr ** 2)) * (-2.0 * rr / (1.0 - rr ** 2) ** 2)
    return out


def rho_tent(r):
    r = np.abs(np.asarray(r, float))
    return np.clip(1.0 - r, 0.0, None)


def radial_moment(profile, k, n=400):
    """int_0^1 profile(r) r^k dr."""
    return float(integrate_1d(lambda r: profile(r) * r ** k, 0.0, 1.0, n=n, pieces=8))


def kernel_constant(profile, d):
    """C_rho(d) = d omega_d int_0^1 rho(r) r^(d-1) dr = integral of rho(|y|) over R^d."""
    return d * OMEGA[d] * radial_moment(profile, d - 1)


def normalisation(profile, n):
    """c with  n omega_n c int_0^1 rho r^(n-1) dr = 1, so that c rho(|x|) integrates to 1 over R^n."""
    return 1.0 / kernel_constant(profile, n)


def kernel_eps(x, eps, profile=rho_exp, n=2, c=None):
    """rho_eps(x) = eps^-n c rho(|x| / eps) for points x of shape (..., n)."""
    if c is None:
        c = normalisation(profile, n)
    r = np.linalg.norm(np.asarray(x, float), axis=-1) / eps
    return c * profile(r) / eps ** n


def grad_kernel_eps(x, eps, dprofile=drho_exp, profile=rho_exp, n=2, c=None):
    """grad rho_eps(x) = eps^-(n+1) c rho'(|x| / eps) x / |x| (0 at x = 0)."""
    if c is None:
        c = normalisation(profile, n)
    x = np.asarray(x, float)
    nx = np.linalg.norm(x, axis=-1, keepdims=True)
    safe = np.where(nx > 0, nx, 1.0)
    return c * dprofile(nx / eps) / eps ** (n + 1) * np.where(nx > 0, x / safe, 0.0)


def bump(x, center, radius):
    """compactly supported smooth test function rho_exp(|x - c| / R) / rho_exp(0) (value 1 at the centre)."""
    r = np.linalg.norm(np.asarray(x, float) - np.asarray(center, float), axis=-1) / radius
    return rho_exp(r) / math.exp(-1.0)


# ---------------------------------------------------------------------------
# finite varifolds ("atoms") and polylines (Sections 46.4-46.5)
# ---------------------------------------------------------------------------


def atoms_integral(phi, X, P, m):
    """V(phi) = sum_i m_i phi(x_i, P_i) for phi(x, P) vectorised over the first axis."""
    return float(np.sum(np.asarray(m, float) * phi(np.asarray(X, float), np.asarray(P, float))))


def polyline_segments(V, closed):
    V = np.asarray(V, float)
    A = V
    B = np.roll(V, -1, axis=0) if closed else V[1:]
    if not closed:
        A = V[:-1]
    return A, B


def polyline_integral(phi, V, closed, nq=8):
    """int phi(x, P) dV over the polyline varifold v(polyline, 1) (each segment carries its own line P)."""
    A, B = polyline_segments(V, closed)
    x, w = gauss_legendre(nq)
    total = 0.0
    for a, b in zip(A, B):
        e = b - a
        L = np.linalg.norm(e)
        P = dir_proj(e)
        pts = a[None, :] + x[:, None] * e[None, :]
        Ps = np.broadcast_to(P, (len(x),) + P.shape)
        total += L * float(np.sum(w * phi(pts, Ps)))
    return total


def polyline_length(V, closed):
    A, B = polyline_segments(V, closed)
    return float(np.sum(np.linalg.norm(B - A, axis=1)))


def polyline_first_variation(V, closed):
    """vectors w_i with delta V(X) = sum_i <w_i, X(v_i)> for the polyline varifold.

    closed: w_i = t_{i-1} - t_i;  open: w_0 = -t_0, w_last = t_last, interior w_i = t_{i-1} - t_i.
    (-w_i = t_i - t_{i-1} is the turning vector kappa^B_i n_i of Proposition 32.3.6.)"""
    V = np.asarray(V, float)
    A, B = polyline_segments(V, closed)
    T = (B - A) / np.linalg.norm(B - A, axis=1, keepdims=True)
    W = np.zeros_like(V)
    if closed:
        W = np.roll(T, 1, axis=0) - T
    else:
        W[0] = -T[0]
        W[-1] = T[-1]
        W[1:-1] = T[:-1] - T[1:]
    return W


def polyline_first_variation_quadrature(X_field, DX_field, V, closed, nq=8):
    """delta V(X) = int div_S X dV computed segment by segment by quadrature (independent check)."""
    A, B = polyline_segments(V, closed)
    x, w = gauss_legendre(nq)
    total = 0.0
    for a, b in zip(A, B):
        e = b - a
        L = np.linalg.norm(e)
        P = dir_proj(e)
        for xi, wi in zip(x, w):
            total += L * wi * tangential_div(P, DX_field(a + xi * e))
    return total


def staircase(nsteps):
    """staircase from (0, 0) to (1, 1) with nsteps horizontal and nsteps vertical unit pieces of size 1/nsteps."""
    h = 1.0 / nsteps
    pts = [(0.0, 0.0)]
    for k in range(nsteps):
        pts.append(((k + 1) * h, k * h))
        pts.append(((k + 1) * h, (k + 1) * h))
    return np.array(pts)


P_H = line_proj(0.0)
P_V = line_proj(np.pi / 2)
P_D = line_proj(np.pi / 4)


def staircase_limit_integral(phi, nq=64):
    """V_infinity(phi) = int_D sqrt2 * (phi(x, P_h) + phi(x, P_v)) / 2 dH^1, D = diagonal from (0,0) to (1,1)."""
    x, w = gauss_legendre(nq)
    pts = np.stack([x, x], axis=1)
    L = math.sqrt(2.0)
    f = 0.5 * (phi(pts, np.broadcast_to(P_H, (len(x), 2, 2))) + phi(pts, np.broadcast_to(P_V, (len(x), 2, 2))))
    return L * math.sqrt(2.0) * float(np.sum(w * f))


def diagonal_integral(phi, nq=64):
    """v(D, 1)(phi) = int_D phi(x, P_d) dH^1."""
    x, w = gauss_legendre(nq)
    pts = np.stack([x, x], axis=1)
    return math.sqrt(2.0) * float(np.sum(w * phi(pts, np.broadcast_to(P_D, (len(x), 2, 2)))))


def hausdorff_polyline_to_segment(V, a, b):
    """Hausdorff distance between a polyline (vertices V, straight segments) and the segment [a, b]."""
    V = np.asarray(V, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    e = b - a

    def dist_to_seg(p):
        t = np.clip(np.dot(p - a, e) / np.dot(e, e), 0, 1)
        return np.linalg.norm(p - (a + t * e))

    # polyline -> segment: maximum attained at vertices (distance to a segment is convex along a segment)
    d1 = max(dist_to_seg(p) for p in V)
    # segment -> polyline: sample densely
    ts = np.linspace(0, 1, 2001)
    A, B = polyline_segments(V, closed=False)
    d2 = 0.0
    for t in ts:
        p = a + t * e
        best = np.inf
        for u, v in zip(A, B):
            f = v - u
            s = np.clip(np.dot(p - u, f) / np.dot(f, f), 0, 1)
            best = min(best, np.linalg.norm(p - (u + s * f)))
        d2 = max(d2, best)
    return max(d1, d2)


# ---------------------------------------------------------------------------
# samples on the circle (Section 46.2)
# ---------------------------------------------------------------------------


def warp(u, a=0.5):
    """nonuniform parameter t(u) = 2 pi u + a sin(2 pi u) (increasing for |a| < 1)."""
    return 2 * np.pi * u + a * np.sin(2 * np.pi * u)


def circle_samples(N, r=1.0, a=0.5):
    """N nonuniform points t_j = warp((j + 1/2) / N) on the circle, Voronoi (arc) weights and equal weights."""
    u = (np.arange(N) + 0.5) / N
    t = warp(u, a)
    tp = np.roll(t, -1)
    tp[-1] += 2 * np.pi
    tm = np.roll(t, 1)
    tm[0] -= 2 * np.pi
    m_vor = r * (tp - tm) / 2.0
    m_eq = np.full(N, 2 * np.pi * r / N)
    return t, m_vor, m_eq


# ---------------------------------------------------------------------------
# triangle meshes: hat functions and the first variation (Section 46.5)
# ---------------------------------------------------------------------------


def hat_gradients(V, F):
    """(n_f, 3, 3): gradient of the hat function of each corner on its face (lies in the face plane)."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    out = np.zeros((len(F), 3, V.shape[1]))
    for f, tri in enumerate(F):
        p = V[tri]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        A2 = np.linalg.norm(n)
        n = n / A2
        for c in range(3):
            j, k = (c + 1) % 3, (c + 2) % 3
            out[f, c] = np.cross(n, p[k] - p[j]) / A2
    return out


def hat_first_variation(V, F, i):
    """delta V(phi_i) = sum_{f at i} int_f grad^{P_f} phi_i dA  (a vector of R^3)."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    G = hat_gradients(V, F)
    total = np.zeros(V.shape[1])
    for f, tri in enumerate(F):
        for c in range(3):
            if tri[c] == i:
                p = V[tri]
                A = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
                total += A * G[f, c]
    return total


def hat_mass(V, F, i):
    """||V||(phi_i) = sum_{f at i} area(f) / 3 (barycentric area)."""
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    s = 0.0
    for tri in F:
        if i in tri:
            p = V[tri]
            s += 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])) / 3.0
    return s


def mesh_area(V, F):
    V = np.asarray(V, float)
    F = np.asarray(F, int)
    p = V[F]
    return float(0.5 * np.sum(np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)))


def cone_one_ring(k=6, h=0.4, radius=1.0):
    """apex (0, 0, h) over a regular k-gon of the given radius in z = 0; faces (apex, j, j + 1) counter-clockwise."""
    ang = 2 * np.pi * np.arange(k) / k
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang), np.zeros(k)], axis=1)
    V = np.vstack([[0.0, 0.0, h], ring])
    F = np.array([[0, 1 + j, 1 + (j + 1) % k] for j in range(k)])
    return V, F


# ---------------------------------------------------------------------------
# bounded Lipschitz distance helpers (Section 46.7)
# ---------------------------------------------------------------------------


def bl_diracs(h):
    """Delta^{1,1}(delta_0, delta_h) on R (or any metric space with d = h): min(h, 2)."""
    return min(abs(h), 2.0)


def circle_cell_bound(N, r=1.0, nq=64):
    """quadrature bound  sum_i int_{C_i} (|x - x_i| + ||T_x - P_i||) ds  for N equally spaced atoms
    (x_i = circle point at the centre of an arc cell C_i, P_i = tangent there, m_i = cell length).
    The integrand has a kink at the atom, so each half cell is integrated separately."""
    h = 2 * np.pi / N
    x, w = gauss_legendre(nq)
    s = x * (h / 2)                                # angle offset in one half cell, [0, h/2]
    pos = 2 * r * np.abs(np.sin(s / 2))           # chord length |x - x_i|
    tang = np.abs(np.sin(s))                       # ||T_x - P_i||_op = |sin(angle)|
    per_cell = 2 * r * (h / 2) * float(np.sum(w * (pos + tang)))
    return N * per_cell


def circle_cell_bound_exact(N, r=1.0):
    """closed form of circle_cell_bound: N r [8 r (1 - cos(pi / 2N)) + 2 (1 - cos(pi / N))]."""
    return N * r * (8 * r * (1 - math.cos(math.pi / (2 * N))) + 2 * (1 - math.cos(math.pi / N)))


def circle_atoms(N, r=1.0, offset=0.5):
    t = 2 * np.pi * (np.arange(N) + offset) / N
    X = circle(t, r)
    P = line_proj(t + np.pi / 2)
    m = np.full(N, 2 * np.pi * r / N)
    return t, X, P, m


def circle_varifold_integral(phi, r=1.0, nq=64, pieces=64):
    """v(C, 1)(phi) = int_0^{2 pi} phi(x(t), T(t)) r dt."""
    def f(t):
        return phi(circle(t, r), line_proj(t + np.pi / 2)) * r
    return float(integrate_1d(f, 0.0, 2 * np.pi, n=nq, pieces=pieces))


def random_lipschitz_tests(k, seed=0, n=2):
    """k random test functions phi(x, P) with |phi| <= 1 and Lipschitz constant <= 1 for the metric
    |x - y| + ||P - Q||_op:  phi = clip(a + <b, x> + <C, P>_F, -1, 1) with |b| + ||C||_* <= 1 (nuclear norm)."""
    rng = np.random.default_rng(seed)
    tests = []
    for _ in range(k):
        b = rng.normal(size=n)
        C = rng.normal(size=(n, n))
        C = 0.5 * (C + C.T)
        nuc = np.sum(np.abs(np.linalg.eigvalsh(C)))
        scale = np.linalg.norm(b) + nuc
        b, C = b / scale, C / scale
        a = rng.uniform(-0.5, 0.5)

        def phi(X, P, a=a, b=b, C=C):
            X = np.asarray(X, float)
            P = np.asarray(P, float)
            val = a + X @ b + np.einsum("...jk,jk->...", P, C)
            return np.clip(val, -1.0, 1.0)

        tests.append(phi)
    return tests
