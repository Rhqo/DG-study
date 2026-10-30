"""Helpers for Chapter 50 (Paper: Manifold-GS, [ManifoldGS26] = arXiv:2608.00214v1).

Shared by the figure and verify scripts of this chapter. Conventions (Section 50.1, Conventions box):

- an unoriented plane of R^3 is stored as its orthogonal projection P = I - n n^T (n a unit normal, sign irrelevant);
  the paper measures plane errors in the Frobenius norm, ||P - Q||_F = sqrt(2) sin(angle between normals), which is
  sqrt(2) times the operator norm used in Chapter 46 (Proposition 46.3.4);
- an atom of a discrete varifold is (mu_i, P_i, q_i): position, plane, geometric mass; the varifold integrates a test
  function as V(phi) = sum_i q_i phi(mu_i, P_i)  ([ManifoldGS26] Definition 1, eq. (4));
- test functions are 1-Lipschitz for the sum metric |x - y| + ||P - Q||_F and bounded by 1 (eq. (7));
- the kernel varifold diagnostic is the Gaussian kernel of eq. (22),
  k((x, P), (y, Q)) = exp(-|x - y|^2 / (2 sigma^2) - ||P - Q||_F^2 / (2 tau^2)), on measures normalised to mass 1.

Run from the project root with PYTHONPATH=tools.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "4")

import math  # noqa: E402

import numpy as np  # noqa: E402

SQRT2 = math.sqrt(2.0)

# ---------------------------------------------------------------------------
# planes
# ---------------------------------------------------------------------------


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def plane_proj(normal):
    """projection onto the plane with the given (unoriented) normal: I - n n^T (works on stacks of normals)."""
    n = unit(normal)
    return np.eye(n.shape[-1]) - n[..., :, None] * n[..., None, :]


def fro_from_normals(n, m):
    """||P(n) - P(m)||_F = sqrt(2) sqrt(1 - <n, m>^2) for unit normals (broadcasting)."""
    c = np.einsum("...i,...i->...", n, m)
    return SQRT2 * np.sqrt(np.clip(1.0 - c * c, 0.0, None))


def op_from_normals(n, m):
    c = np.einsum("...i,...i->...", n, m)
    return np.sqrt(np.clip(1.0 - c * c, 0.0, None))


def gauss_legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (x + 1.0), 0.5 * w


def sph(th, ph):
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)


# ---------------------------------------------------------------------------
# Section 50.2: a latitude-longitude tiling of the unit sphere (the tiling of Example 46.8.6)
# ---------------------------------------------------------------------------


def tiling(nt):
    """cells (theta0, theta1, phi0, phi1), atoms mu_i at the parameter midpoints, exact areas A(C_i)."""
    tb, pb = np.linspace(0, np.pi, nt + 1), np.linspace(0, 2 * np.pi, 2 * nt + 1)
    cells, mus, areas = [], [], []
    for a in range(nt):
        for b in range(2 * nt):
            cells.append((tb[a], tb[a + 1], pb[b], pb[b + 1]))
            mus.append(sph(0.5 * (tb[a] + tb[a + 1]), 0.5 * (pb[b] + pb[b + 1])))
            areas.append((math.cos(tb[a]) - math.cos(tb[a + 1])) * (pb[b + 1] - pb[b]))
    return cells, np.array(mus), np.array(areas)


def cell_quadrature(cell, n=10):
    """Gauss-Legendre nodes X (n, n, 3) and area weights W (n, n) of one cell of the unit sphere."""
    t0, t1, p0, p1 = cell
    xg, wg = gauss_legendre(n)
    TH, PH = np.meshgrid(t0 + (t1 - t0) * xg, p0 + (p1 - p0) * xg, indexing="ij")
    W = np.outer(wg, wg) * (t1 - t0) * (p1 - p0) * np.sin(TH)
    return sph(TH, PH), W


def cell_sup(cell, mu, nfine=9):
    """sup over the cell of |x - mu| and of ||P(x) - P_mu||_F (on the sphere P(x) = I - x x^T) on a fine grid."""
    t0, t1, p0, p1 = cell
    TF, PF = np.meshgrid(np.linspace(t0, t1, nfine), np.linspace(p0, p1, nfine), indexing="ij")
    XF = sph(TF, PF)
    return np.linalg.norm(XF - mu, axis=-1).max(), fro_from_normals(XF, mu).max()


def smooth_tests(k, seed=0):
    """k smooth test functions phi(x, n) = 1/2 sin(<a, x> + c) + 1/4 <B, P(n)>_F with |a| = 1, ||B||_F = 1.

    Each is 1/2-Lipschitz in x and 1/4-Lipschitz in P (Frobenius), so 1-Lipschitz for the sum metric and |phi| <= 3/4.
    """
    rng = np.random.default_rng(seed)
    tests = []
    for _ in range(k):
        a = unit(rng.normal(size=3))
        c = rng.uniform(0, 2 * np.pi)
        B = rng.normal(size=(3, 3))
        B = 0.5 * (B + B.T)
        B /= np.linalg.norm(B)

        def phi(X, N, a=a, c=c, B=B):
            P = plane_proj(N)
            return 0.5 * np.sin(np.einsum("...i,i->...", X, a) + c) + 0.25 * np.einsum("...ij,ij->...", P, B)
        tests.append(phi)
    return tests


def forward_consistency(nt, qbias=0.0, nquad=10, ntests=20, lower=True):
    """numbers of Section 50.2 for the tiling(nt) of the unit sphere, planes measured in the Frobenius norm.

    qbias: masses q_i = A(C_i)(1 + qbias z_i) (a mass error that follows the height z of the atom, eps_q = 2 pi qbias).
    returns dict with h, eT (sup, Frobenius), eq, the bound (7) A(M)(h + eT) + eq, the cell-integral upper bound
    sum_i int_{C_i} (|x - mu_i| + ||P(x) - P_i||_F) dA + eq, the lower bound int min(1, min_i(...)) dA - (mass excess),
    and the largest |V_G(phi) - V_M(phi)| over smooth test functions.
    """
    cells, mus, areas = tiling(nt)
    q = areas * (1.0 + qbias * mus[:, 2])
    eq = float(np.abs(q - areas).sum())
    tests = smooth_tests(ntests, seed=7)
    VM = np.zeros(ntests)
    h = eT = upper = low = 0.0
    for cell, mu in zip(cells, mus):
        X, W = cell_quadrature(cell, nquad)
        hs, es = cell_sup(cell, mu)
        h, eT = max(h, hs), max(eT, es)
        upper += float(np.sum(W * (np.linalg.norm(X - mu, axis=-1) + fro_from_normals(X, mu))))
        if lower:
            D = np.linalg.norm(X[..., None, :] - mus[None, None, :, :], axis=-1) + fro_from_normals(X[..., None, :], mus)
            low += float(np.sum(W * np.minimum(1.0, D.min(axis=-1))))
        for k, phi in enumerate(tests):
            VM[k] += float(np.sum(W * phi(X, X)))
    VG = np.array([float(np.sum(q * phi(mus, mus))) for phi in tests])
    return {"atoms": len(areas), "h": h, "eT": eT, "eq": eq, "bound": 4 * np.pi * (h + eT) + eq,
            "upper": upper + eq, "lower": low, "smooth": float(np.abs(VG - VM).max())}


# ---------------------------------------------------------------------------
# kernel varifold diagnostic (eq. (22)) and the transport bound (eq. (23))
# ---------------------------------------------------------------------------


def kernel(XA, NA, XB, NB, sigma, tau):
    """Gram matrix k((x_a, P_a), (x_b, P_b)) of eq. (22); planes given by unit normals."""
    d2 = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(-1)
    c = NA @ NB.T
    f2 = 2.0 * (1.0 - c * c)                      # ||P - Q||_F^2
    return np.exp(-d2 / (2 * sigma ** 2) - f2 / (2 * tau ** 2))


def mmd(XA, NA, wA, XB, NB, wB, sigma, tau):
    """MMD_k between the normalised discrete varifolds sum wA delta and sum wB delta."""
    a, b = np.asarray(wA, float) / np.sum(wA), np.asarray(wB, float) / np.sum(wB)
    val = a @ kernel(XA, NA, XA, NA, sigma, tau) @ a + b @ kernel(XB, NB, XB, NB, sigma, tau) @ b \
        - 2 * a @ kernel(XA, NA, XB, NB, sigma, tau) @ b
    return math.sqrt(max(val, 0.0))


class KernelRef:
    """a fixed reference varifold (positions, unit normals, masses) with its self term cached, for repeated MMDs."""

    def __init__(self, X, N, w, sigma, tau):
        self.X, self.N, self.b = X, N, np.asarray(w, float) / np.sum(w)
        self.sigma, self.tau = sigma, tau
        self.self_term = float(self.b @ kernel(X, N, X, N, sigma, tau) @ self.b)

    def mmd(self, X, N, w):
        a = np.asarray(w, float) / np.sum(w)
        val = a @ kernel(X, N, X, N, self.sigma, self.tau) @ a + self.self_term \
            - 2 * a @ kernel(X, N, self.X, self.N, self.sigma, self.tau) @ self.b
        return math.sqrt(max(val, 0.0))


def transport_bound(XA, NA, XB, NB, pi, sigma, tau):
    """right side of (23) for a coupling matrix pi (rows: atoms of A, columns: atoms of B, total mass 1)."""
    d2 = ((XA[:, None, :] - XB[None, :, :]) ** 2).sum(-1) / sigma ** 2
    c = NA @ NB.T
    d2 = d2 + 2.0 * (1.0 - c * c) / tau ** 2
    return math.sqrt(float(np.sum(pi * d2)))


# ---------------------------------------------------------------------------
# Section 50.1: a 2D toy for floater, coverage and phantom collision (our construction)
# ---------------------------------------------------------------------------

TOY_GT = (200.0, 340.0)        # the true thin wall: arc of the unit circle, angles in degrees
TOY_CERT = (215.0, 325.0)      # the certified open patch (well-observed middle of the wall)
TOY_TOL = 0.02                 # tolerance of the toy evaluator
TOY_BOX = 1.5                  # free-space probes live in [-1.5, 1.5]^2


def arc_points(a0, a1, n):
    t = np.radians(np.linspace(a0, a1, n))
    return np.stack([np.cos(t), np.sin(t)], 1)


def dist_to_arc(P, a0, a1):
    """distance from points P (k, 2) to the arc of the unit circle with angles [a0, a1] (degrees, a0 < a1)."""
    ang = np.degrees(np.arctan2(P[:, 1], P[:, 0])) % 360.0
    r = np.linalg.norm(P, axis=1)
    inside = (ang >= a0) & (ang <= a1)
    ends = arc_points(a0, a1, 2)
    dend = np.min(np.linalg.norm(P[:, None, :] - ends[None, :, :], axis=-1), axis=1)
    return np.where(inside, np.abs(r - 1.0), dend)


def toy_metrics(n=20000, nprobe=200000, seed=3):
    """floater, coverage and phantom rates (in percent) of a watertight closure (the full unit circle as the boundary of
    a solid disk) and of the certified open arc, against the thin wall TOY_GT. Probes are uniform in the box and count
    as free space when they are farther than TOY_TOL from the wall; a probe is blocked by a watertight candidate when it
    is inside the disk and by an open candidate when it is within TOY_TOL of it."""
    gt = arc_points(*TOY_GT, n)
    full = arc_points(0.0, 360.0, n)
    cert = arc_points(*TOY_CERT, n)
    out = {}
    rng = np.random.default_rng(seed)
    probes = rng.uniform(-TOY_BOX, TOY_BOX, size=(nprobe, 2))
    free = dist_to_arc(probes, *TOY_GT) > TOY_TOL
    for name, cand, solid in (("watertight", full, True), ("certified", cert, False)):
        floater = 100.0 * np.mean(dist_to_arc(cand, *TOY_GT) > TOY_TOL)
        if name == "watertight":
            dcov = np.abs(np.linalg.norm(gt, axis=1) - 1.0)
            blocked = np.linalg.norm(probes, axis=1) <= 1.0 + TOY_TOL
        else:
            dcov = dist_to_arc(gt, *TOY_CERT)
            blocked = dist_to_arc(probes, *TOY_CERT) <= TOY_TOL
        coverage = 100.0 * np.mean(dcov <= TOY_TOL)
        phantom = 100.0 * np.mean(blocked[free])
        out[name] = (floater, coverage, phantom)
    return out


# ---------------------------------------------------------------------------
# Section 50.3: refinement schedules with a geometric mass ledger versus "opacity as area"
# ---------------------------------------------------------------------------


def tangent_frame(n):
    n = unit(n)
    t = unit(np.cross(n, [0.3, 0.5, 0.8]))
    return t, np.cross(n, t)


def schedule_run(psplit, region, steps=300, seed=0, nt=8, sigma=0.25, tau=1.0, every=10, ref_nt=32):
    """apply a random split/clone schedule to Gaussians tiling the unit sphere and track two bookkeepings.

    atoms: dict(mu, n, s (3 scales), alpha, q). split (3DGS rule, [Kerbl23]): two children at offsets drawn from the
    parent Gaussian, scales / 1.6, same opacity; the ledger recentres the offsets and gives each child q / 2 with the
    parent's plane. clone: a copy at the same place; the ledger gives each copy q / 2.  The opacity area is
    kappa * alpha * s1 * s2 ( = kappa * alpha sqrt(lambda1 lambda2) ), kappa chosen so that it starts at 4 pi.
    Returns an array of rows (step, ledger mass / 4 pi, opacity area / 4 pi, MMD(ledger, V_M), MMD(opacity, V_M)).
    """
    cells, mus, areas = tiling(nt)
    _, mR, aR = tiling(ref_nt)
    ref = KernelRef(mR, mR, aR, sigma, tau)
    rng = np.random.default_rng(seed)
    A = [dict(mu=m.copy(), n=m.copy(), s=np.array([math.sqrt(a / math.pi)] * 2 + [0.01]), alpha=0.7, q=a)
         for m, a in zip(mus, areas)]
    kappa = 4 * math.pi / sum(a["alpha"] * a["s"][0] * a["s"][1] for a in A)
    rows = []

    def record(t):
        X = np.array([a["mu"] for a in A])
        N = np.array([a["n"] for a in A])
        q = np.array([a["q"] for a in A])
        o = np.array([kappa * a["alpha"] * a["s"][0] * a["s"][1] for a in A])
        rows.append((t, q.sum() / (4 * math.pi), o.sum() / (4 * math.pi), ref.mmd(X, N, q), ref.mmd(X, N, o)))
    record(0)
    for t in range(1, steps + 1):
        idx = [j for j, a in enumerate(A) if region(a["mu"])]
        i = idx[rng.integers(len(idx))]
        a = A[i]
        if rng.random() < psplit:
            tt, bb = tangent_frame(a["n"])
            offs = [a["s"][0] * rng.normal() * tt + a["s"][1] * rng.normal() * bb for _ in range(2)]
            m = 0.5 * (offs[0] + offs[1])
            A[i:i + 1] = [dict(mu=a["mu"] + o - m, n=a["n"].copy(), s=a["s"] / 1.6, alpha=a["alpha"], q=a["q"] / 2)
                          for o in offs]
        else:
            A.insert(i, dict(mu=a["mu"].copy(), n=a["n"].copy(), s=a["s"].copy(), alpha=a["alpha"], q=a["q"] / 2))
            a["q"] /= 2
        if t % every == 0:
            record(t)
    return np.array(rows), len(A)


SCHEDULES = (("split, north", 1.0, lambda x: x[2] > 0), ("clone, north", 0.0, lambda x: x[2] > 0),
             ("mixed, everywhere", 0.5, lambda x: True))


# ---------------------------------------------------------------------------
# Section 50.4: compatibility residuals of a chart and a normal field (our formalisation)
# ---------------------------------------------------------------------------


def residuals(X, n, u, v, a=0.5, m=24):
    """RMS (area-weighted over [-a, a]^2) of the residuals of Definition 50.4.2 for a chart X(u, v) and a unit normal
    field n(u, v) (sympy expressions): disagreement ||W^s - W^c||_F, antisymmetry |h^c_12 - h^c_21| / sqrt(det g),
    normal curl |d<n, dx>| / sqrt(det g), Gauss |K_g - det h^c / det g|, Codazzi |(c1, c2)| / det g.
    The integral over [-a, a]^2 uses m x m Gauss-Legendre nodes (review: a uniform 41 x 41 grid with equal weights
    over-weighted the boundary and was off by up to 2.5 %)."""
    import sympy as sp
    import dgsym
    X, n = sp.Matrix(X), sp.Matrix(n)
    Xu, Xv = X.diff(u), X.diff(v)
    gm = sp.Matrix([[Xu.dot(Xu), Xu.dot(Xv)], [Xu.dot(Xv), Xv.dot(Xv)]])
    Ns = Xu.cross(Xv)
    Ns = Ns / sp.sqrt(Ns.dot(Ns))
    hs = sp.Matrix([[X.diff(u, 2).dot(Ns), X.diff(u, v).dot(Ns)], [X.diff(u, v).dot(Ns), X.diff(v, 2).dot(Ns)]])
    hc = sp.Matrix(2, 2, lambda i, j: -n.diff((u, v)[i]).dot((Xu, Xv)[j]))
    Gam = dgsym.christoffel(gm, [u, v])
    K = dgsym.sectional(gm, [u, v], [1, 0], [0, 1])
    hb = (hc + hc.T) / 2
    e, f, g_ = hb[0, 0], hb[0, 1], hb[1, 1]
    c1 = e.diff(v) - f.diff(u) - (e * Gam[0][0][1] + f * (Gam[1][0][1] - Gam[0][0][0]) - g_ * Gam[1][0][0])
    c2 = f.diff(v) - g_.diff(u) - (e * Gam[0][1][1] + f * (Gam[1][1][1] - Gam[0][0][1]) - g_ * Gam[1][0][1])
    curl = n.dot(Xv).diff(u) - n.dot(Xu).diff(v)
    fn = sp.lambdify((u, v), [gm, hs, hc, K, c1, c2, curl], "numpy")
    xg, wg = gauss_legendre(m)
    t, wt = a * (2 * xg - 1), wg
    acc = {k: [] for k in ("disagreement", "antisymmetry", "curl", "gauss", "codazzi")}
    wts = []
    for uu, wu in zip(t, wt):
        for vv, wv in zip(t, wt):
            gm_, hs_, hc_, K_, c1_, c2_, cu_ = fn(uu, vv)
            gm_, hs_, hc_ = (np.array(z, float) for z in (gm_, hs_, hc_))
            dg = np.linalg.det(gm_)
            gi = np.linalg.inv(gm_)
            acc["disagreement"].append(np.linalg.norm(gi @ hs_ - gi @ hc_))
            acc["antisymmetry"].append(abs(hc_[0, 1] - hc_[1, 0]) / math.sqrt(dg))
            acc["curl"].append(abs(float(cu_)) / math.sqrt(dg))
            acc["gauss"].append(abs(float(K_) - np.linalg.det(hc_) / dg))
            acc["codazzi"].append(math.hypot(float(c1_), float(c2_)) / dg)
            wts.append(wu * wv * math.sqrt(dg))
    w = np.array(wts)
    return {k: float(np.sqrt(np.sum(w * np.array(val) ** 2) / w.sum())) for k, val in acc.items()}


def residual_cases():
    """the four cases of Example 50.4.5 on the graph chart of the unit sphere over [-1/2, 1/2]^2 (sympy)."""
    import sympy as sp
    u, v = sp.symbols("u v", real=True)
    z = sp.sqrt(1 - u ** 2 - v ** 2)
    Xs = [u, v, z]
    Ns = sp.Matrix([u, v, z])
    tw = Ns + sp.Rational(3, 10) * sp.Matrix([-v, u, 0])
    gr = Ns + sp.Matrix([u, v, 0])
    return u, v, {"A": (Xs, Ns), "B": (Xs, tw / sp.sqrt(tw.dot(tw))), "C": (Xs, gr / sp.sqrt(gr.dot(gr))),
                  "D": ([u, v, 1], sp.Matrix([0, 0, 1]))}


def decomposition_atoms(m=31, a=0.5):
    """atoms (positions, unit normals, area weights) of the cases of Example 50.4.7 on a (m x m) grid of [-a, a]^2:
    the sphere cap S (true normals), B and C (sphere positions, perturbed normals), D (the plane z = 1)."""
    t = np.linspace(-a, a, m)
    U, V = (z.ravel() for z in np.meshgrid(t, t, indexing="ij"))
    Z = np.sqrt(1 - U ** 2 - V ** 2)
    Xs = np.stack([U, V, Z], 1)
    ws = 1.0 / Z
    NB = unit(Xs + 0.3 * np.stack([-V, U, 0 * U], 1))
    NC = unit(Xs + np.stack([U, V, 0 * U], 1))
    Xp = np.stack([U, V, np.ones_like(U)], 1)
    Np = np.tile([0.0, 0.0, 1.0], (len(U), 1))
    return {"S": (Xs, Xs.copy(), ws), "B": (Xs, NB, ws), "C": (Xs, NC, ws), "D": (Xp, Np, np.ones_like(U))}


# ---------------------------------------------------------------------------
# Section 50.4: a hole under a fixed-bandwidth kernel (planes all equal, so only the position kernel matters)
# ---------------------------------------------------------------------------


def _I1(sig, L=2.0):
    """int_{-1}^{1} int_{-1}^{1} exp(-(x - y)^2 / (2 sig^2)) dx dy for an interval of length L."""
    return 2 * sig ** 2 * (math.exp(-L ** 2 / (2 * sig ** 2)) - 1) + L * sig * math.sqrt(2 * math.pi) * math.erf(
        L / (sig * math.sqrt(2)))


def mmd_hole(r, sig, n=40):
    """MMD between the normalised area measures of the square [-1, 1]^2 and of the square minus the disk of radius r."""
    A, a = 4.0, math.pi * r * r
    x, w = gauss_legendre(n)
    RR, TT = np.meshgrid(r * x, 2 * math.pi * x, indexing="ij")
    W = np.outer(w * r * (r * x), w * 2 * math.pi).ravel()
    P = np.stack([RR * np.cos(TT), RR * np.sin(TT)], -1).reshape(-1, 2)
    verf = np.vectorize(math.erf)
    s2 = sig * math.sqrt(2)
    phi = np.ones(len(P))
    for c in range(2):
        phi = phi * sig * math.sqrt(math.pi / 2) * (verf((1 - P[:, c]) / s2) + verf((1 + P[:, c]) / s2))
    KSS, KSD = _I1(sig) ** 2, float(np.sum(W * phi))
    KDD = float(W @ np.exp(-((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) / (2 * sig ** 2)) @ W)
    c1, c2 = -a / (A * (A - a)), 1 / (A - a)
    return math.sqrt(max(c1 * c1 * KSS + 2 * c1 * c2 * KSD + c2 * c2 * KDD, 0.0))


def mmd_shift(d, sig):
    """MMD between the normalised area measure of [-1, 1]^2 and the same square moved by d along its normal."""
    return math.sqrt(2 * (1 - math.exp(-d * d / (2 * sig * sig)))) * _I1(sig) / 4.0


def crossing_radius(sig, target, lo=0.01, hi=0.6, it=40):
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mmd_hole(mid, sig) < target else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Section 50.4: a two-view 1D camera and the restricted rendering Jacobian (our toy)
# ---------------------------------------------------------------------------


def restricted_sigma(coef, offsets, d=2.0, npx=101):
    """cameras at (o, 0) look along +z at the line z = d; pixel x in [-0.4, 0.4] sees s = o + d x and records the
    albedo c(s) = sum_k coef[k] s^k (appearance = the polynomial coefficients, shared by the views).
    returns (|J_d|, smallest singular value of J_d after removing range(J_c))."""
    x = np.linspace(-0.4, 0.4, npx)
    p = len(coef) - 1
    Jd, Jc = [], []
    for o in offsets:
        s = o + d * x
        dc = sum(k * coef[k] * s ** (k - 1) for k in range(1, p + 1))
        Jd.append(dc * x)
        Jc.append(np.stack([s ** k for k in range(p + 1)], 1))
    Jd, Jc = np.concatenate(Jd), np.vstack(Jc)
    Q, _ = np.linalg.qr(Jc)
    r = Jd - Q @ (Q.T @ Jd)
    return float(np.linalg.norm(Jd)), float(np.linalg.norm(r))


# ---------------------------------------------------------------------------
# Section 50.6: conservative split with inherited planes versus re-estimated planes
# ---------------------------------------------------------------------------


def refined_tiling(nt, k, mode, nq=8, lower=True):
    """split every cell of tiling(nt) into 2^k x 2^k sub-cells (one child per sub-cell, q_j = A(C_j)).

    mode 'inherit': the child sits at the projection of its sub-cell centre to the parent's tangent plane, the children
    are recentred so that sum q_j mu_j = q_i mu_i, and every child keeps the parent's plane (Proposition 50.3.1 with the
    tangent moment).  mode 'reestimate': the child sits at the sub-cell centre with the true tangent plane.
    returns (number of atoms, cell-integral upper bound, lower bound) of Delta^{1,1} to the sphere (Frobenius metric)."""
    cells, mus, areas = tiling(nt)
    AX, AN, QX, QW, owner = [], [], [], [], []
    ns = 2 ** k
    for (a0, a1, b0, b1), mu in zip(cells, mus):
        ta, tb = np.linspace(a0, a1, ns + 1), np.linspace(b0, b1, ns + 1)
        kids = []
        for i in range(ns):
            for j in range(ns):
                sub = (ta[i], ta[i + 1], tb[j], tb[j + 1])
                c = sph(0.5 * (ta[i] + ta[i + 1]), 0.5 * (tb[j] + tb[j + 1]))
                q = (math.cos(ta[i]) - math.cos(ta[i + 1])) * (tb[j + 1] - tb[j])
                X, W = cell_quadrature(sub, nq)
                kids.append((c, q, X.reshape(-1, 3), W.ravel()))
        qs = np.array([kk[1] for kk in kids])
        if mode == "inherit":
            P = np.eye(3) - np.outer(mu, mu)
            pos = np.array([mu + P @ (kk[0] - mu) for kk in kids])
            pos += mu - (qs[:, None] * pos).sum(0) / qs.sum()
            nrm = np.tile(mu, (len(kids), 1))
        else:
            pos = np.array([kk[0] for kk in kids])
            nrm = pos.copy()
        for kk, p_, n_ in zip(kids, pos, nrm):
            AX.append(p_)
            AN.append(n_)
            QX.append(kk[2])
            QW.append(kk[3])
            owner.append(len(AX) - 1)
    AX, AN = np.array(AX), np.array(AN)
    upper = sum(float(np.sum(W * (np.linalg.norm(X - AX[o], axis=1) + fro_from_normals(X, AN[o]))))
                for X, W, o in zip(QX, QW, owner))
    if not lower:
        return len(AX), upper, float("nan")
    XX, WW = np.concatenate(QX), np.concatenate(QW)
    best = np.full(len(XX), np.inf)
    for s in range(0, len(AX), 64):
        D = np.linalg.norm(XX[:, None, :] - AX[None, s:s + 64, :], axis=-1) \
            + fro_from_normals(XX[:, None, :], AN[None, s:s + 64, :])
        best = np.minimum(best, D.min(1))
    return len(AX), upper, float(np.sum(WW * np.minimum(1.0, best)))


def fine_bounds(nt, m, lower=True):
    """(cell-form upper bound, lower bound) of Delta^{1,1}(V_G, V_M) for tiling(nt), Frobenius metric, with an
    equal-area midpoint grid of m x m points per cell (added in review: the integrand has a kink at the atom, so the
    10 x 10 Gauss-Legendre rule of forward_consistency is off in the 4th digit)."""
    cells, mus, areas = tiling(nt)
    XS, WS, up = [], [], 0.0
    for (t0, t1, p0, p1), mu, A in zip(cells, mus, areas):
        z = math.cos(t1) + (math.cos(t0) - math.cos(t1)) * (np.arange(m) + 0.5) / m
        ph = p0 + (p1 - p0) * (np.arange(m) + 0.5) / m
        Z, PH = np.meshgrid(z, ph, indexing="ij")
        X = sph(np.arccos(Z), PH).reshape(-1, 3)
        W = np.full(len(X), A / len(X))
        up += float(np.sum(W * (np.linalg.norm(X - mu, axis=1) + fro_from_normals(X, mu))))
        XS.append(X)
        WS.append(W)
    if not lower:
        return up, None
    X, W = np.vstack(XS), np.concatenate(WS)
    lo = 0.0
    for s0 in range(0, len(X), 4000):
        D = np.linalg.norm(X[s0:s0 + 4000, None] - mus[None], axis=-1) + fro_from_normals(X[s0:s0 + 4000, None], mus[None])
        lo += float(np.sum(W[s0:s0 + 4000] * np.minimum(1.0, D.min(1))))
    return up, lo


def parent_plane_floor(nt, m=48):
    """int over the unit sphere of min(1, min_k ||P(x) - P_k||_F) dA for the atoms of tiling(nt): a lower bound of
    Delta^{1,1} for every varifold whose planes all belong to {P_k} (test function depending on the plane only)."""
    cells, mus, areas = tiling(nt)
    tot = 0.0
    for (t0, t1, p0, p1), A in zip(cells, areas):
        z = math.cos(t1) + (math.cos(t0) - math.cos(t1)) * (np.arange(m) + 0.5) / m
        ph = p0 + (p1 - p0) * (np.arange(m) + 0.5) / m
        Z, PH = np.meshgrid(z, ph, indexing="ij")
        X = sph(np.arccos(Z), PH).reshape(-1, 3)
        tot += A * float(np.mean(np.minimum(1.0, fro_from_normals(X[:, None], mus[None]).min(1))))
    return tot


def plane_term(nt, nq=24):
    """sum_i int_{C_i} ||P(x) - P_i||_F dA for tiling(nt): the limit of the inherited model."""
    cells, mus, _ = tiling(nt)
    tot = 0.0
    for cell, mu in zip(cells, mus):
        X, W = cell_quadrature(cell, nq)
        tot += float(np.sum(W * fro_from_normals(X, mu)))
    return tot


# ---------------------------------------------------------------------------
# Section 50.5: the numbers of the paper's tables ([ManifoldGS26] v1, transcribed) and what we compute from them
# ---------------------------------------------------------------------------

TABLE1 = {  # method: (Chamfer, normal angle, kernel varifold distance, PSNR); analytic plane, 1500 steps
    "RGB": (0.05946, 64.91, 0.19335, 28.017),
    "RGB + comp.": (0.05011, 55.18, 0.15167, 27.662),
    "+ adapt. comp.": (0.04537, 55.95, 0.14614, 30.264),
    "+ depth oracle": (0.01885, 8.84, 0.06056, 30.863),
}
TABLE2 = {24: (1.7163, 1.6885, 1.62, 0.132), 65: (2.4643, 2.3577, 4.32, 0.297)}  # 3DGS, Anch., Gain %, dPSNR
TABLE3 = {24: (54.2, 18.25, 0.10, 30.11, 0.36), 65: (49.5, 0.87, 0.00, 35.34, 0.95),
          105: (61.9, 1.54, 0.00, 33.70, 0.66)}  # ID area %, floater %, phantom %, tex. dB, ext. move %
TABLE4 = {  # scan: method: (Float. %, Cov. %) at 1% bbox tolerance
    24: {"ours": (18.3, 37.1), "2DGS": (21.1, 69.9), "SuGaR": (78.7, 69.8), "Poisson": (98.5, 51.5)},
    65: {"ours": (0.9, 26.3), "2DGS": (13.7, 48.7), "SuGaR": (17.4, 46.3), "Poisson": (97.4, 49.0)},
    105: {"ours": (1.5, 41.0), "2DGS": (11.4, 51.8), "SuGaR": (7.7, 51.9), "Poisson": (54.1, 75.9)},
}
SIMPLIFIED_FLOATER = {"ours": (18.55, 0.81, 1.42), "2DGS": (18.87, 12.21, 10.12), "SuGaR": (78.89, 17.16, 7.46),
                      "Poisson": (98.51, 97.36, 52.39)}  # after ~5k-face decimation, scan 24 / 65 / 105 (Section 7.7)
FISHER = {24: (165, 19, 0, 184), 65: (146, 17, 1, 164), 105: (210, 24, 0, 234)}  # supported, weak, insuff., stated
EXTERNAL = {24: (1759, 0.274, 0.619, 0.329), 65: (1805, 0.525, 0.678, 0.700), 105: (1932, 0.525, 0.684, 0.694)}


def f1(floater, coverage):
    """F1 of precision p = 1 - floater/100 and recall r = coverage/100 (our summary; not a metric of the paper)."""
    p, r = 1 - floater / 100.0, coverage / 100.0
    return 2 * p * r / (p + r)


def iou_from_pr(p, r):
    """IoU = TP / (TP + FP + FN) = 1 / (1/p + 1/r - 1)."""
    return 1.0 / (1.0 / p + 1.0 / r - 1.0)


def random_normal_angles(n=400000, seed=5):
    """angles (degrees) between uniformly random unoriented normals and a fixed axis: density sin(theta) on [0, 90]."""
    rng = np.random.default_rng(seed)
    v = unit(rng.normal(size=(n, 3)))
    return np.degrees(np.arccos(np.abs(v[:, 2])))
