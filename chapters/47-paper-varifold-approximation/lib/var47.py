"""Helpers for Chapter 47 (Paper: A Varifold Approach to Surface Approximation, [BLM17] = arXiv:1609.03625v1).

Shared by the figure and verify scripts of this chapter. Conventions (Section 47.1, Conventions box):

- a plane P in G_{d,n} is stored as its orthogonal projection matrix (n x n); for a curve in R^2 we often store
  the unit tangent T instead (P = T T^T);
- kernels are radial: rho_1(x) = rho(|x|), rho_eps(x) = eps^-n rho(|x| / eps), profiles supported in [0, 1].
  Every formula below is invariant under rescaling rho (or xi) by a constant, so the profiles are NOT normalised
  here; C_rho = d omega_d int_0^1 rho r^(d-1) dr is computed for the same unnormalised profile ([BLM17] (4.1));
- approximate mean curvature of a point cloud varifold sum_j m_j delta_{x_j} (x) delta_{P_j} at x
  ([BLM17] Def 4.1 with (3.4); the printed (4.3) has the opposite sign and no constant, Section 47.4):

      H(x) = - (C_xi / C_rho) * (1/eps) * sum_j m_j rho'(|x_j - x|/eps) P_j (x_j - x)/|x_j - x|
                                          / sum_j m_j xi(|x_j - x|/eps)

  and the orthogonal version projects H(x) on the normal space of the plane at x ([BLM17] Def 4.6, (8.1));
- the mean curvature vector is the sum convention of [BLM17] and Chapter 46: on a circle of radius r it points to
  the centre with length 1/r, on a sphere of radius r with length 2/r.

Neighbour search: a uniform grid of cell size eps (pure NumPy, no scipy), processed in chunks of query points.
Run from the project root with PYTHONPATH=tools.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import math  # noqa: E402

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# kernel profiles ([BLM17] Section 1 and Section 8.1.1)
# ---------------------------------------------------------------------------


def rho_exp(r):
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    m = np.abs(r) < 1
    out[m] = np.exp(-1.0 / (1.0 - r[m] ** 2))
    return out


def drho_exp(r):
    r = np.asarray(r, float)
    out = np.zeros_like(r)
    m = np.abs(r) < 1
    rm = r[m]
    out[m] = np.exp(-1.0 / (1.0 - rm ** 2)) * (-2.0 * rm / (1.0 - rm ** 2) ** 2)
    return out


def rho_tent(r):
    r = np.abs(np.asarray(r, float))
    return np.where(r < 1, 1.0 - r, 0.0)


def drho_tent(r):
    r = np.asarray(r, float)
    return np.where(np.abs(r) < 1, -np.sign(r) + (r == 0) * (-1.0), 0.0)


class Pair:
    """a kernel pair (rho, xi) given by profiles on [0, 1]; xi = None means the natural partner -s rho'(s) / n."""

    def __init__(self, name, rho, drho, xi=None, n=2):
        self.name, self.rho, self.drho, self.n = name, rho, drho, n
        self.natural = xi is None
        self.xi = (lambda s: -np.asarray(s, float) * drho(s) / n) if xi is None else xi

    def ratio(self, d):
        """C_xi / C_rho for d-dimensional varifolds ([BLM17] (4.1))."""
        return radial_moment(self.xi, d - 1) / radial_moment(self.rho, d - 1)


def pairs(n=2):
    """the four pairs of [BLM17] Section 8.1.1 for ambient dimension n."""
    return {
        "natural exp": Pair("natural exp", rho_exp, drho_exp, None, n),
        "exp": Pair("exp", rho_exp, drho_exp, rho_exp, n),
        "natural tent": Pair("natural tent", rho_tent, drho_tent, None, n),
        "tent": Pair("tent", rho_tent, drho_tent, rho_tent, n),
    }


_GL = {}


def gauss_legendre(k):
    if k not in _GL:
        x, w = np.polynomial.legendre.leggauss(k)
        _GL[k] = (0.5 * (x + 1.0), 0.5 * w)
    return _GL[k]


def radial_moment(prof, k, pieces=64, order=40):
    """int_0^1 prof(r) r^k dr by composite Gauss-Legendre (the exp profile is smooth, the tent is piecewise linear)."""
    x, w = gauss_legendre(order)
    edges = np.linspace(0.0, 1.0, pieces + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        t = lo + (hi - lo) * x
        tot += (hi - lo) * float(np.sum(w * prof(t) * t ** k))
    return tot


def omega(d):
    """volume of the unit ball of R^d."""
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


def C_const(prof, d):
    """C = d omega_d int_0^1 prof(r) r^(d-1) dr  = integral of prof(|y|) over R^d ([BLM17] (4.1))."""
    return d * omega(d) * radial_moment(prof, d - 1)


def normaliser(prof, n):
    """c with  c * n omega_n int_0^1 prof r^(n-1) dr = 1 ([BLM17] Section 1)."""
    return 1.0 / C_const(prof, n)


# ---------------------------------------------------------------------------
# neighbour search on a grid (pure NumPy)
# ---------------------------------------------------------------------------


def neighbour_pairs(X, eps, Q=None, chunk=20000):
    """yield (iq, jx, D, r) for all query points Q[iq] and data points X[jx] with 0 <= |X[jx] - Q[iq]| < eps,
    processed in chunks of queries; D = X[jx] - Q[iq], r = |D|. Q defaults to X."""
    X = np.asarray(X, float)
    Q = X if Q is None else np.asarray(Q, float)
    dim = X.shape[1]
    lo = np.minimum(X.min(0), Q.min(0)) - eps
    cx = np.floor((X - lo) / eps).astype(np.int64)
    cq = np.floor((Q - lo) / eps).astype(np.int64)
    span = int(max(cx.max(), cq.max())) + 3
    mult = span ** np.arange(dim)
    keyx = cx @ mult
    order = np.argsort(keyx, kind="stable")
    ks = keyx[order]
    offs = np.array(np.meshgrid(*[[-1, 0, 1]] * dim, indexing="ij")).reshape(dim, -1).T
    for q0 in range(0, len(Q), chunk):
        qi = np.arange(q0, min(q0 + chunk, len(Q)))
        I_list, J_list = [], []
        for o in offs:
            key = (cq[qi] + o) @ mult
            a = np.searchsorted(ks, key, "left")
            b = np.searchsorted(ks, key, "right")
            cnt = b - a
            if cnt.sum() == 0:
                continue
            rep = np.repeat(np.arange(len(qi)), cnt)
            start = np.repeat(a - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
            J_list.append(order[start + np.arange(cnt.sum())])
            I_list.append(qi[rep])
        if not I_list:
            continue
        iq = np.concatenate(I_list)
        jx = np.concatenate(J_list)
        D = X[jx] - Q[iq]
        r = np.sqrt(np.einsum("ij,ij->i", D, D))
        keep = r < eps
        yield iq[keep], jx[keep], D[keep], r[keep]


# ---------------------------------------------------------------------------
# approximate mean curvature of a point cloud varifold ([BLM17] Def 4.1, Ex 4.2, Def 4.6, (8.1))
# ---------------------------------------------------------------------------


def amc(X, P, m, eps, pair, d, Q=None, PQ=None, chunk=20000):
    """approximate mean curvature H^V_{rho,xi,eps} at the query points Q (default: the cloud points).
    X (N, n) points, P (N, n, n) projections, m (N,) masses. Returns (H, Hperp_or_None, counts):
    Hperp uses the projections PQ of the query points (normal projection I - PQ), when PQ is given
    (for Q = None, PQ defaults to P)."""
    X = np.asarray(X, float)
    N, n = X.shape
    Qp = X if Q is None else np.asarray(Q, float)
    if Q is None and PQ is None:
        PQ = P
    num = np.zeros((len(Qp), n))
    den = np.zeros(len(Qp))
    cnt = np.zeros(len(Qp), int)
    for iq, jx, D, r in neighbour_pairs(X, eps, Qp, chunk):
        s = r / eps
        u = np.zeros_like(D)
        nz = r > 0
        u[nz] = D[nz] / r[nz, None]
        Pu = np.einsum("kab,kb->ka", P[jx], u)
        w = m[jx] * pair.drho(s)
        w[~nz] = 0.0
        for a in range(n):
            num[:, a] += np.bincount(iq, w * Pu[:, a], minlength=len(Qp))
        den += np.bincount(iq, m[jx] * pair.xi(s), minlength=len(Qp))
        cnt += np.bincount(iq, minlength=len(Qp))
    with np.errstate(divide="ignore", invalid="ignore"):
        H = -pair.ratio(d) * num / (eps * den[:, None])
    Hp = None
    if PQ is not None:
        Hp = H - np.einsum("kab,kb->ka", PQ, H)
    return H, Hp, cnt


def proj_from_tangent(T):
    """P = T T^T for unit tangents T (N, n) of curves."""
    T = np.asarray(T, float)
    return np.einsum("ka,kb->kab", T, T)


def proj_from_normal(Nrm):
    """P = I - N N^T for unit normals N (N, 3) of surfaces."""
    Nrm = np.asarray(Nrm, float)
    return np.eye(Nrm.shape[1])[None] - np.einsum("ka,kb->kab", Nrm, Nrm)


def pca_planes(X, R, d, chunk=20000):
    """regression planes: for each point, the span of the top-d eigenvectors of the covariance of the points in the
    open ball of radius R, centred at their mean ([BLM17] Section 8.1.3 and 8.3). Returns projections (N, n, n)."""
    X = np.asarray(X, float)
    N, n = X.shape
    S1 = np.zeros((N, n))
    S2 = np.zeros((N, n, n))
    c = np.zeros(N)
    for iq, jx, D, r in neighbour_pairs(X, R, None, chunk):
        c += np.bincount(iq, minlength=N)
        for a in range(n):
            S1[:, a] += np.bincount(iq, D[:, a], minlength=N)
            for b in range(a, n):
                v = np.bincount(iq, D[:, a] * D[:, b], minlength=N)
                S2[:, a, b] += v
                if b != a:
                    S2[:, b, a] += v
    mu = S1 / c[:, None]
    C = S2 / c[:, None, None] - np.einsum("ka,kb->kab", mu, mu)
    w, V = np.linalg.eigh(C)
    T = V[:, :, n - d:]
    return np.einsum("kai,kbi->kab", T, T)


# ---------------------------------------------------------------------------
# the test shapes of [BLM17] Section 8 (curves in R^2): position, unit tangent, curvature vector
# ---------------------------------------------------------------------------


def _curve_from_derivs(P, d1, d2):
    sp2 = np.einsum("ka,ka->k", d1, d1)
    T = d1 / np.sqrt(sp2)[:, None]
    H = (d2 - np.einsum("ka,ka->k", d2, T)[:, None] * T) / sp2[:, None]
    return P, T, H


def circle(t, r=0.5, c=(0.0, 0.0)):
    t = np.asarray(t, float)
    P = np.stack([c[0] + r * np.cos(t), c[1] + r * np.sin(t)], 1)
    d1 = np.stack([-r * np.sin(t), r * np.cos(t)], 1)
    d2 = np.stack([-r * np.cos(t), -r * np.sin(t)], 1)
    return _curve_from_derivs(P, d1, d2)


def ellipse(t, a=1.0, b=0.5):
    t = np.asarray(t, float)
    P = np.stack([a * np.cos(t), b * np.sin(t)], 1)
    d1 = np.stack([-a * np.sin(t), b * np.cos(t)], 1)
    d2 = np.stack([-a * np.cos(t), -b * np.sin(t)], 1)
    return _curve_from_derivs(P, d1, d2)


def flower(t):
    """[BLM17] Section 8.1.1 (c): r(theta) = 0.5 (1 + 0.5 sin(6 theta + pi/2)) = 0.5 + 0.25 cos 6 theta."""
    t = np.asarray(t, float)
    r = 0.5 + 0.25 * np.cos(6 * t)
    dr = -1.5 * np.sin(6 * t)
    ddr = -9.0 * np.cos(6 * t)
    c, s = np.cos(t), np.sin(t)
    P = np.stack([r * c, r * s], 1)
    d1 = np.stack([dr * c - r * s, dr * s + r * c], 1)
    d2 = np.stack([ddr * c - 2 * dr * s - r * c, ddr * s + 2 * dr * c - r * s], 1)
    return _curve_from_derivs(P, d1, d2)


def eight(t):
    """[BLM17] Section 8.1.1 (d): (0.5 sin t (cos t + 1), 0.5 sin t (cos t - 1))."""
    t = np.asarray(t, float)
    s, c = np.sin(t), np.cos(t)
    P = np.stack([0.5 * s * (c + 1), 0.5 * s * (c - 1)], 1)
    # d/dt [s c + s] = c^2 - s^2 + c = cos 2t + cos t ;  d/dt [s c - s] = cos 2t - cos t
    d1 = 0.5 * np.stack([np.cos(2 * t) + c, np.cos(2 * t) - c], 1)
    d2 = 0.5 * np.stack([-2 * np.sin(2 * t) - s, -2 * np.sin(2 * t) + s], 1)
    return _curve_from_derivs(P, d1, d2)


def curve_cloud(shape, N, t=None):
    """point cloud varifold of a parametrised curve: points at t_j (default t_j = 2 pi j / N), exact tangents,
    masses 1 (only ratios matter in (8.2))."""
    if t is None:
        t = 2 * np.pi * np.arange(N) / N
    X, T, H = shape(t)
    return t, X, T, H


def max_abs_curvature(shape, M=2_000_001):
    t = np.linspace(0, 2 * np.pi, M)
    return float(np.linalg.norm(shape(t)[2], axis=1).max())


def mean_abs_curvature_param(shape, M=2_000_000):
    t = 2 * np.pi * np.arange(M) / M
    return float(np.linalg.norm(shape(t)[2], axis=1).mean())


def curve_length(shape, M=2_000_000):
    t = 2 * np.pi * np.arange(M + 1) / M
    X = shape(t)[0]
    return float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())


def erel_curve(shape, N, eps, pair, project=True, tangents="exact", R=None, noise=0.0, jitter=False, seed=0,
               norm="max"):
    """the relative average error (8.5) of [BLM17] Section 8.1 for a parametrised curve.
    norm = 'max' divides by ||H||_inf as printed in (8.5); norm = 'mean' divides by the mean of |H(t_j)|."""
    rng = np.random.default_rng(seed)
    t = 2 * np.pi * np.arange(N) / N
    if jitter:
        t = (np.arange(N) + rng.standard_normal(N)) * 2 * np.pi / N
    X, T, Hx = shape(t)
    if noise > 0:
        X = X + noise * rng.standard_normal(X.shape)
    if tangents == "exact":
        P = proj_from_tangent(T)
    elif tangents == "ols":
        P = ols_planes(X, R)
    else:                                   # "pca" (total least squares)
        P = pca_planes(X, R, 1)
    H, Hp, cnt = amc(X, P, np.ones(N), eps, pair, 1)
    E = Hp if project else H
    err = np.linalg.norm(E - Hx, axis=1)
    if norm == "max":
        scale = max_abs_curvature(shape)
    else:
        scale = float(np.linalg.norm(Hx, axis=1).mean())
    return float(err.mean() / scale), float(cnt.mean())


def ols_planes(X, R, chunk=20000):
    """tangent lines of a planar point cloud by ordinary least squares y = a + b x on the points of the open ball of
    radius R (our reading of the 'linear regression' of [BLM17] Section 8.1.1; the paper does not say which fit).
    Returns projections (N, 2, 2)."""
    X = np.asarray(X, float)
    N = len(X)
    S = np.zeros((N, 5))
    c = np.zeros(N)
    for iq, jx, D, r in neighbour_pairs(X, R, None, chunk):
        c += np.bincount(iq, minlength=N)
        for k, val in enumerate((D[:, 0], D[:, 1], D[:, 0] ** 2, D[:, 0] * D[:, 1])):
            S[:, k] += np.bincount(iq, val, minlength=N)
    mx, my = S[:, 0] / c, S[:, 1] / c
    sxx = S[:, 2] / c - mx * mx
    sxy = S[:, 3] / c - mx * my
    T = np.stack([sxx, sxy], 1)
    T /= np.linalg.norm(T, axis=1)[:, None]
    return proj_from_tangent(T)


def amc_single(x, Px, Y, PY, mY, eps, pair, d):
    """approximate mean curvature at one point x (plane Px) from the atoms (Y, PY, mY), including an atom at x
    itself if present in Y. Returns (H, Hperp). Fast path for Monte Carlo experiments."""
    D = np.asarray(Y, float) - np.asarray(x, float)[None, :]
    r = np.linalg.norm(D, axis=1)
    sel = r < eps
    D, r, P, m = D[sel], r[sel], PY[sel], mY[sel]
    s = r / eps
    u = np.zeros_like(D)
    nz = r > 0
    u[nz] = D[nz] / r[nz, None]
    w = m * pair.drho(s)
    w[~nz] = 0.0
    num = np.einsum("k,kab,kb->a", w, P, u)
    den = float(np.sum(m * pair.xi(s)))
    H = -pair.ratio(d) * num / (eps * den)
    return H, H - Px @ H


# ---------------------------------------------------------------------------
# singular test shapes of [BLM17] Section 8.2: two circles, the 2D standard double bubble
# ---------------------------------------------------------------------------


def two_circles(N, r=0.5, c=0.25, offset=0.0):
    """uniform sampling of the union of the circles of radius r centred at (-c, 0) and (c, 0), N/2 points each,
    starting at angle 0 (as in our reading of [BLM17, Fig. 8]). Returns X, T, H (exact tangent, curvature)."""
    M = N // 2
    t = 2 * np.pi * (np.arange(M) + offset) / M
    X1, T1, H1 = circle(t, r, (-c, 0.0))
    X2, T2, H2 = circle(t, r, (c, 0.0))
    return np.vstack([X1, X2]), np.vstack([T1, T2]), np.vstack([H1, H2])


def double_bubble_2d(r1=0.6, r2=1.0):
    """the standard double bubble in the plane with outer radii r1 < r2 ([BLM17, Fig. 9]): small circle centred at
    the origin, big circle at (dist, 0), the middle arc of radius r3 = 1/(1/r1 - 1/r2) bulging into the big bubble.
    Returns a dict with centres, radii, junction p = (px, py), and the three arcs as (centre, radius, angle range)."""
    dist = math.sqrt(r1 ** 2 + r2 ** 2 - r1 * r2)
    r3 = 1.0 / (1.0 / r1 - 1.0 / r2)
    px = (r1 ** 2 - r2 ** 2 + dist ** 2) / (2 * dist)
    py = math.sqrt(r1 ** 2 - px ** 2)
    x3 = px - math.sqrt(r3 ** 2 - py ** 2)
    c1, c2, c3 = np.array([0.0, 0.0]), np.array([dist, 0.0]), np.array([x3, 0.0])
    a1 = math.atan2(py, px)
    a2 = math.atan2(py, px - dist)
    a3 = math.atan2(py, px - x3)
    arcs = [(c1, r1, a1, 2 * math.pi - a1), (c2, r2, -a2, a2), (c3, r3, -a3, a3)]
    return dict(dist=dist, r3=r3, p=np.array([px, py]), centres=(c1, c2, c3), radii=(r1, r2, r3), arcs=arcs)


def sample_arcs(arcs, N, offset=0.5):
    """N points at equal arc-length spacing over a list of arcs (centre, radius, angle from, angle to), shifted by
    `offset` spacings from the start of the first arc; returns X, T (unit tangents), H (curvature vectors, towards
    the centre), arc index."""
    lens = np.array([r * (b - a) for c, r, a, b in arcs])
    L = lens.sum()
    s = (np.arange(N) + offset) * L / N
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    k = np.searchsorted(cum, s, side="right") - 1
    X = np.zeros((N, 2))
    T = np.zeros((N, 2))
    H = np.zeros((N, 2))
    for i, (c, r, a, b) in enumerate(arcs):
        sel = k == i
        th = a + (s[sel] - cum[i]) / r
        X[sel] = c + r * np.stack([np.cos(th), np.sin(th)], 1)
        T[sel] = np.stack([-np.sin(th), np.cos(th)], 1)
        H[sel] = (c - X[sel]) / r ** 2
    return X, T, H, k


def average_ball(X, F, R):
    """the average of the vectors F over the points in the open ball of radius R (the '2 eps averaging' of
    [BLM17] Section 8.2)."""
    N = len(X)
    A = np.zeros_like(F)
    c = np.zeros(N)
    for iq, jx, D, r in neighbour_pairs(X, R):
        c += np.bincount(iq, minlength=N)
        for a in range(F.shape[1]):
            A[:, a] += np.bincount(iq, F[jx, a], minlength=N)
    return A / c[:, None]


def junction_mean_curvature(centres, radii, p, k=1.0):
    """the average of the curvature vectors k (c - p)/r^2 of circles (k = 1) or spheres (k = 2) through p."""
    return np.mean([k * (np.asarray(c) - p) / r ** 2 for c, r in zip(centres, radii)], axis=0)


# ---------------------------------------------------------------------------
# Section 47.3: the L1 norm of the regularized first variation of a point cloud on the circle
# ---------------------------------------------------------------------------


def field_L1_circle(X, P, m, eps, R=0.5, nr=32, per=48):
    """|| delta V * rho_eps ||_{L^1(R^2)} for atoms (X, P, m) near the circle of radius R, with the exp kernel
    normalised in R^2 ([BLM17] Prop 3.2): a midpoint rule on the polar tube |r - R| < eps' (eps' = eps + the largest
    distance of an atom from the circle). Returns (L1, (Q, |F|)) with the grid points and field magnitudes."""
    c = normaliser(rho_exp, 2)
    off = float(np.abs(np.linalg.norm(X, axis=1) - R).max())
    w_ = eps + off
    nth = int(per * 2 * math.pi * R / w_)
    rr = R - w_ + (np.arange(nr) + 0.5) * 2 * w_ / nr
    th = (np.arange(nth) + 0.5) * 2 * math.pi / nth
    RR, TH = np.meshgrid(rr, th, indexing="ij")
    Q = np.stack([(RR * np.cos(TH)).ravel(), (RR * np.sin(TH)).ravel()], 1)
    wq = RR.ravel() * (2 * w_ / nr) * (2 * math.pi / nth)
    F = np.zeros_like(Q)
    for iq, jx, D, r in neighbour_pairs(X, eps, Q, chunk=100000):
        g = c * drho_exp(r / eps) / eps ** 3
        u = D / np.where(r > 0, r, 1.0)[:, None]
        Pu = np.einsum("kab,kb->ka", P[jx], u)
        for a in range(2):
            F[:, a] += np.bincount(iq, m[jx] * g * Pu[:, a], minlength=len(Q))
    mag = np.linalg.norm(F, axis=1)
    return float((mag * wq).sum()), (Q, mag)


# ---------------------------------------------------------------------------
# volumetric varifolds on a square grid (Section 47.2, [BLM17] Def 2.11, Prop 2.13, Remark 2.14)
# ---------------------------------------------------------------------------


def circle_cells(a, r=0.5, origin=(0.0, 0.0)):
    """cut the circle of radius r centred at 0 by the square grid of side a whose lines are x = origin[0] + k a,
    y = origin[1] + k a. Returns a dict {(i, j): (m_K, P_K)} with m_K = length of the circle in the cell
    K = [ox + i a, ox + (i+1) a) x [oy + j a, oy + (j+1) a) (i.e. ||V||(K)) and P_K = the unit eigenvector
    (as a projection) of the mass-weighted mean of the tangent projections of the arcs in K (a simple choice;
    any P_K is allowed in Prop 2.13)."""
    ox, oy = origin
    br = [0.0, 2 * np.pi]
    for k in range(int(np.floor((-r - ox) / a)) - 1, int(np.ceil((r - ox) / a)) + 2):
        c = (ox + k * a) / r
        if abs(c) < 1:
            th = math.acos(c)
            br += [th % (2 * np.pi), (-th) % (2 * np.pi)]
    for k in range(int(np.floor((-r - oy) / a)) - 1, int(np.ceil((r - oy) / a)) + 2):
        s = (oy + k * a) / r
        if abs(s) < 1:
            th = math.asin(s)
            br += [th % (2 * np.pi), (np.pi - th) % (2 * np.pi)]
    br = np.unique(np.array(br))
    cells = {}
    for t0, t1 in zip(br[:-1], br[1:]):
        if t1 - t0 < 1e-15:
            continue
        tm = 0.5 * (t0 + t1)
        key = (int(np.floor((r * math.cos(tm) - ox) / a)), int(np.floor((r * math.sin(tm) - oy) / a)))
        L = r * (t1 - t0)
        # integral of T T^T over the arc, T = (-sin t, cos t)
        s2 = 0.5 * ((t1 - t0) - 0.5 * (math.sin(2 * t1) - math.sin(2 * t0)))   # int sin^2
        c2 = 0.5 * ((t1 - t0) + 0.5 * (math.sin(2 * t1) - math.sin(2 * t0)))   # int cos^2
        sc = 0.5 * (math.sin(t1) ** 2 - math.sin(t0) ** 2)                      # int sin cos
        M = r * np.array([[s2, -sc], [-sc, c2]])
        m0, M0 = cells.get(key, (0.0, np.zeros((2, 2))))
        cells[key] = (m0 + L, M0 + M)
    out = {}
    for key, (m, M) in cells.items():
        w, U = np.linalg.eigh(M)
        u = U[:, -1]
        out[key] = (m, np.outer(u, u))
    return out


def volumetric_tv(cells, a):
    """total variation |delta V|(R^2) of the volumetric varifold sum_K (m_K/|K|) L^2|K (x) delta_{P_K} on a square grid
    of side a. delta V is carried by the cell edges: across an edge with unit normal nu from K to L its density is
    (D_K P_K - D_L P_L) nu with D = m/a^2 (divergence theorem in each cell, P constant), and the edge has length a."""
    D = {k: (m / a ** 2) * P for k, (m, P) in cells.items()}
    Z = np.zeros((2, 2))
    tv = 0.0
    seen = set()
    for (i, j) in D:
        for (di, dj, nu) in ((1, 0, np.array([1.0, 0.0])), (-1, 0, np.array([-1.0, 0.0])),
                             (0, 1, np.array([0.0, 1.0])), (0, -1, np.array([0.0, -1.0]))):
            nb = (i + di, j + dj)
            e = frozenset([(i, j), nb])
            if e in seen:
                continue
            seen.add(e)
            tv += a * np.linalg.norm((D[(i, j)] - D.get(nb, Z)) @ nu)
    return tv


MEAN_DIST_SQUARE = (math.sqrt(2) + math.log(1 + math.sqrt(2))) / 6   # mean distance from the centre of a unit square


def cell_planes(X, T, w, a, origin=(0.0, 0.0), maxpts=400):
    """for a curve in R^2 sampled by points X with unit tangents T and length weights w, and the square grid of side
    a with lines x = origin[0] + k a, y = origin[1] + k a: return {(i, j): (m_K, alpha_K, misfit_K)} with
    m_K = sum of weights in K (= ||V||(K)), alpha_K the angle of a minimizing line P_K and
    misfit_K = min_P sum_k w_k ||P - T_k T_k^T|| = min_alpha sum_k w_k |sin(theta_k - alpha)| (operator norm of the
    difference of two line projections). The sum is concave in alpha between its kinks alpha = theta_k (mod pi),
    so the minimum is attained at one of the theta_k ([BLM17] Lemma 6.1, the term min_P int ||P - S|| dV)."""
    key = np.floor((X - np.asarray(origin)) / a).astype(np.int64)
    theta = np.arctan2(T[:, 1], T[:, 0])
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, theta, w = key[order], theta[order], w[order]
    brk = np.flatnonzero(np.any(np.diff(key, axis=0) != 0, axis=1)) + 1
    out = {}
    for lo, hi in zip(np.concatenate([[0], brk]), np.concatenate([brk, [len(w)]])):
        th, ww = theta[lo:hi], w[lo:hi]
        cand = th if len(th) <= maxpts else th[np.linspace(0, len(th) - 1, maxpts).astype(int)]
        cost = np.abs(np.sin(th[None, :] - cand[:, None])) @ ww
        k = int(np.argmin(cost))
        out[(int(key[lo, 0]), int(key[lo, 1]))] = (float(ww.sum()), float(cand[k]), float(cost[k]))
    return out
