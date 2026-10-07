"""Verification for Section 29.4 (The Space of Covariances).

Run from the project root: ``PYTHONPATH=tools python3 chapters/29-gaussian-splatting/verify/v-29-4-space-of-covariances.py``
(The symbolic curvature of the Bures-Wasserstein metric takes about 10 seconds.)
"""

import numpy as np
import sympy as sp

import dgsym
from dgcheck import check, close, summary, sym_equal

rng = np.random.default_rng(294)


def fun(S, f):
    w, V = np.linalg.eigh(S)
    return V @ np.diag(f(w)) @ V.T


def sqrtm(S):
    return fun(S, np.sqrt)


def logm(S):
    return fun(S, np.log)


def expm(S):
    return fun(S, np.exp)


def rand_spd(n=3, scale=1.0):
    A = rng.normal(size=(n, n))
    return scale * (A @ A.T + 0.2 * np.eye(n))


# ---------------------------------------------------------------- the cone SPD(2) in coordinates (x, y, z)
x, y, z = sp.symbols("x y z", real=True)
S2 = sp.Matrix([[z + x, y], [y, z - x]])
sym_equal("(29.4.1): det [[z+x, y],[y, z-x]] = z^2 - x^2 - y^2", S2.det(), z ** 2 - x ** 2 - y ** 2)
ev = list(S2.eigenvals().keys())
sym_equal("(29.4.1): eigenvalues z +- sqrt(x^2 + y^2)", sp.Matrix(sorted(ev, key=sp.default_sort_key)),
          sp.Matrix(sorted([z + sp.sqrt(x ** 2 + y ** 2), z - sp.sqrt(x ** 2 + y ** 2)], key=sp.default_sort_key)))
# orientation: eigenvector of the larger eigenvalue makes angle atan2(y, x)/2 with the first axis
th = sp.symbols("theta", real=True)
r = sp.symbols("r", positive=True)
v = sp.Matrix([sp.cos(th / 2), sp.sin(th / 2)])
Sp = S2.subs({x: r * sp.cos(th), y: r * sp.sin(th)})
sym_equal("Figure 29.4.1: the major axis direction is half the angle around the cone axis", Sp * v,
          (z + r) * v)

# convex cone: convex combinations and positive multiples stay SPD
A3, B3 = rand_spd(), rand_spd()
check("SPD(3) is convex", all(np.linalg.eigvalsh((1 - t) * A3 + t * B3).min() > 0 for t in np.linspace(0, 1, 11)))
check("SPD(3) is a cone", np.linalg.eigvalsh(0.01 * A3).min() > 0)

# ---------------------------------------------------------------- the four metrics: geodesics and distances
def geo(S0, S1, t, kind):
    if kind == "E":
        return (1 - t) * S0 + t * S1
    if kind == "LE":
        return expm((1 - t) * logm(S0) + t * logm(S1))
    if kind == "AI":
        h = sqrtm(S0); hi = np.linalg.inv(h)
        return h @ fun(hi @ S1 @ hi, lambda w: w ** t) @ h
    if kind == "BW":
        h = sqrtm(S0); hi = np.linalg.inv(h)
        T = hi @ sqrtm(h @ S1 @ h) @ hi
        M = (1 - t) * np.eye(len(S0)) + t * T
        return M @ S0 @ M


def dist(S0, S1, kind):
    if kind == "E":
        return np.linalg.norm(S1 - S0)
    if kind == "LE":
        return np.linalg.norm(logm(S1) - logm(S0))
    if kind == "AI":
        hi = np.linalg.inv(sqrtm(S0))
        return np.linalg.norm(logm(hi @ S1 @ hi))
    if kind == "BW":
        h = sqrtm(S0)
        return np.sqrt(max(np.trace(S0) + np.trace(S1) - 2 * np.trace(sqrtm(h @ S1 @ h)), 0.0))


def lyap(S, U):
    """solve S L + L S = U (S symmetric positive definite)."""
    w, V = np.linalg.eigh(S)
    Ut = V.T @ U @ V
    return V @ (Ut / (w[:, None] + w[None, :])) @ V.T


def metric(S, U, kind):
    if kind == "E":
        return np.sum(U * U)
    if kind == "AI":
        Si = np.linalg.inv(S)
        return np.trace(Si @ U @ Si @ U)
    if kind == "BW":
        return 0.5 * np.trace(lyap(S, U) @ U)
    if kind == "LE":   # differential of log via finite differences
        h = 1e-6
        D = (logm(S + h * U) - logm(S - h * U)) / (2 * h)
        return np.sum(D * D)


def length(S0, S1, kind, n=400):
    ts = np.linspace(0, 1, n + 1)
    Ls = 0.0
    for a, b in zip(ts[:-1], ts[1:]):
        m = 0.5 * (a + b)
        h = 1e-5
        U = (geo(S0, S1, m + h, kind) - geo(S0, S1, m - h, kind)) / (2 * h)
        Ls += np.sqrt(metric(geo(S0, S1, m, kind), U, kind)) * (b - a)
    return Ls


S0, S1 = rand_spd(), rand_spd()
for kind in ("E", "LE", "AI", "BW"):
    close(f"(29.4.2)-(29.4.5) {kind}: geodesic starts at Sigma_0", geo(S0, S1, 0, kind), S0, tol=1e-9)
    close(f"(29.4.2)-(29.4.5) {kind}: geodesic ends at Sigma_1", geo(S0, S1, 1, kind), S1, tol=1e-8)
    close(f"(29.4.2)-(29.4.5) {kind}: length of the geodesic = closed-form distance", length(S0, S1, kind),
          dist(S0, S1, kind), tol=2e-4 * max(1, dist(S0, S1, kind)))
    # constant speed
    sp_ = []
    for m in (0.1, 0.5, 0.9):
        h = 1e-5
        U = (geo(S0, S1, m + h, kind) - geo(S0, S1, m - h, kind)) / (2 * h)
        sp_.append(np.sqrt(metric(geo(S0, S1, m, kind), U, kind)))
    close(f"{kind}: the geodesic has constant speed", np.array(sp_), np.full(3, sp_[0]), tol=1e-4 * sp_[0])
# BW distance = W2 between centered Gaussians; check with the commuting closed form
Da, Db = np.diag([4.0, 1.0, 0.25]), np.diag([1.0, 9.0, 0.04])
close("BW: commuting case d^2 = sum (sqrt a_i - sqrt b_i)^2", dist(Da, Db, "BW") ** 2,
      np.sum((np.sqrt(np.diag(Da)) - np.sqrt(np.diag(Db))) ** 2), tol=1e-10)
# other short-path comparisons: perturbed path is longer
for kind in ("LE", "AI", "BW"):
    def pert(t, S0=S0, S1=S1, kind=kind):
        P = geo(S0, S1, t, kind)
        return P + 0.05 * np.sin(np.pi * t) * np.eye(3)
    ts = np.linspace(0, 1, 401)
    Lp = 0.0
    for a_, b_ in zip(ts[:-1], ts[1:]):
        m = 0.5 * (a_ + b_)
        U = (pert(m + 1e-5) - pert(m - 1e-5)) / 2e-5
        Lp += np.sqrt(metric(pert(m), U, kind)) * (b_ - a_)
    check(f"{kind}: a perturbed path is longer than the geodesic", Lp > dist(S0, S1, kind))

# ---------------------------------------------------------------- invariances
Am = rng.normal(size=(3, 3)) + 2 * np.eye(3)
Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
close("AI is affine-invariant: d(A S0 A^T, A S1 A^T) = d(S0, S1)", dist(Am @ S0 @ Am.T, Am @ S1 @ Am.T, "AI"),
      dist(S0, S1, "AI"), tol=1e-8)
for kind in ("E", "LE", "BW"):
    close(f"{kind} is rotation-invariant", dist(Q @ S0 @ Q.T, Q @ S1 @ Q.T, kind), dist(S0, S1, kind), tol=1e-8)
close("LE is invariant under uniform scaling", dist(4 * S0, 4 * S1, "LE"), dist(S0, S1, "LE"), tol=1e-9)
check("LE is not affine-invariant (general A)", abs(dist(Am @ S0 @ Am.T, Am @ S1 @ Am.T, "LE") - dist(S0, S1, "LE")) > 1e-3)
close("BW scales like a length: d(c^2 S0, c^2 S1) = c d", dist(4 * S0, 4 * S1, "BW"), 2 * dist(S0, S1, "BW"), tol=1e-9)
close("E scales like an area: d(c^2 S0, c^2 S1) = c^2 d", dist(4 * S0, 4 * S1, "E"), 4 * dist(S0, S1, "E"), tol=1e-9)

# ---------------------------------------------------------------- Figure 29.4.3: interpolating two flat ellipses (70 degrees apart)
def R2(d):
    a = np.radians(d)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


F0 = np.diag([1.0, 0.15 ** 2])
F1 = R2(70) @ F0 @ R2(70).T
expect = {"E": (0.2334, 1.40), "LE": (0.0225, 1.91), "AI": (0.0225, 1.40), "BW": (0.0604, 2.61)}
for kind, (dmid, ratio) in expect.items():
    M = geo(F0, F1, 0.5, kind)
    w = np.linalg.eigvalsh(M)
    close(f"Figure 29.4.3 {kind}: det at t = 1/2 is {dmid}", round(np.linalg.det(M), 4), dmid, tol=1e-9)
    close(f"Figure 29.4.3 {kind}: aspect ratio at t = 1/2 is {ratio}", round(np.sqrt(w[1] / w[0]), 2), ratio, tol=1e-9)
    w_ = np.linalg.eigh(M)[1][:, 1]
    close(f"Figure 29.4.3 {kind}: midpoint major axis at 35 degrees", np.degrees(np.arctan2(w_[1], w_[0])) % 180, 35.0,
          tol=1e-6)
close("Figure 29.4.3: endpoint aspect ratio 1/0.15 = 6.67", round(1 / 0.15, 2), 6.67, tol=1e-9)
for kind in ("LE", "AI"):
    dets = [np.linalg.det(geo(F0, F1, t, kind)) for t in np.linspace(0, 1, 11)]
    close(f"(29.4.6): {kind} det Sigma(t) = det0^(1-t) det1^t (constant 0.0225 here)", np.array(dets), np.full(11, 0.0225),
          tol=1e-12)
G0, G1 = rand_spd(), rand_spd()
for kind in ("LE", "AI"):
    t = 0.3
    close(f"(29.4.6): {kind} geometric interpolation of det (random pair)", np.linalg.det(geo(G0, G1, t, kind)),
          np.linalg.det(G0) ** (1 - t) * np.linalg.det(G1) ** t, tol=1e-9 * np.linalg.det(G0))
check("swelling: E det exceeds both endpoint dets at t = 1/2 (example)", np.linalg.det(geo(F0, F1, 0.5, "E")) > 0.0225)
check("swelling: det^(1/n) is concave along E (Minkowski), so E det >= geometric mean",
      all(np.linalg.det(geo(G0, G1, t, "E")) >= np.linalg.det(G0) ** (1 - t) * np.linalg.det(G1) ** t - 1e-12
          for t in np.linspace(0, 1, 11)))
# commuting (90 degrees) case: LE = AI midpoint diag(sqrt eps, sqrt eps); E (1+eps)/2 I; BW ((1+sqrt eps)/2)^2 I
eps = 0.15 ** 2
F2 = np.diag([eps, 1.0])
close("90 degrees: E midpoint (1+eps)/2 I", geo(F0, F2, 0.5, "E"), (1 + eps) / 2 * np.eye(2), tol=1e-12)
close("90 degrees: LE = AI midpoint sqrt(eps) I", [geo(F0, F2, 0.5, "LE"), geo(F0, F2, 0.5, "AI")],
      [np.sqrt(eps) * np.eye(2)] * 2, tol=1e-12)
close("90 degrees: BW midpoint ((1 + sqrt eps)/2)^2 I", geo(F0, F2, 0.5, "BW"), ((1 + np.sqrt(eps)) / 2) ** 2 * np.eye(2),
      tol=1e-12)

# ---------------------------------------------------------------- Figure 29.4.2: distance from I to diag(1, eps)
for e_ in (1e-1, 1e-3, 1e-6):
    De = np.diag([1.0, e_])
    close(f"Figure 29.4.2: d_E(I, diag(1, {e_})) = 1 - eps", dist(np.eye(2), De, "E"), 1 - e_, tol=1e-12)
    close(f"Figure 29.4.2: d_LE = d_AI = |log eps| for eps = {e_}", [dist(np.eye(2), De, "LE"), dist(np.eye(2), De, "AI")],
          [abs(np.log(e_))] * 2, tol=1e-9)
    close(f"Figure 29.4.2: d_BW = 1 - sqrt(eps) for eps = {e_}", dist(np.eye(2), De, "BW"), 1 - np.sqrt(e_), tol=1e-9)
close("Figure 29.4.2: at eps = 1e-6, d_LE = d_AI = 13.8", round(abs(np.log(1e-6)), 1), 13.8, tol=1e-9)

# ---------------------------------------------------------------- curvature of SPD(2) (symbolic, coordinates (a, b, c))
a, b, c = sp.symbols("a b c", real=True)
coords = [a, b, c]
Sab = sp.Matrix([[a, b], [b, c]])
Ebas = [sp.Matrix([[1, 0], [0, 0]]), sp.Matrix([[0, 1], [1, 0]]), sp.Matrix([[0, 0], [0, 1]])]
Si = Sab.inv()
gAI = sp.Matrix(3, 3, lambda i, j: sp.simplify((Si * Ebas[i] * Si * Ebas[j]).trace()))


def lyap_sym(U):
    l1, l2, l3 = sp.symbols("l1 l2 l3")
    L = sp.Matrix([[l1, l2], [l2, l3]])
    sol = sp.solve(list(Sab * L + L * Sab - U), [l1, l2, l3], dict=True)[0]
    return L.subs(sol)


Ls = [lyap_sym(Ei) for Ei in Ebas]
gBW = sp.Matrix(3, 3, lambda i, j: sp.simplify(sp.Rational(1, 2) * (Ls[i] * Ebas[j]).trace()))
planes = [((1, 0, 0), (0, 1, 0)), ((1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)), ((1, 0, -1), (0, 1, 0)),
          ((1, 0, 1), (1, 0, -1)), ((1, 0, 1), (0, 1, 0))]
for name, g, ok in (("AI", gAI, lambda K: -sp.Rational(1, 2) <= K <= 0), ("BW", gBW, lambda K: K >= 0)):
    Gam = dgsym.christoffel(g, coords)
    Rm = dgsym.riemann(g, coords, Gamma=Gam)
    for pt in ({a: 1, b: 0, c: 1}, {a: 2, b: sp.Rational(1, 2), c: 1}, {a: 3, b: -1, c: sp.Rational(1, 2)}):
        gp = g.subs(pt)
        Rp = [[[[Rm[i][j][k][l].subs(pt) for l in range(3)] for k in range(3)] for j in range(3)] for i in range(3)]
        Ks = []
        for vv, ww in planes:
            num = sum(gp[l, m] * Rp[i][j][k][m] * vv[i] * ww[j] * ww[k] * vv[l]
                      for i in range(3) for j in range(3) for k in range(3) for l in range(3) for m in range(3))
            ip = lambda p_, q_: sum(gp[i, j] * p_[i] * q_[j] for i in range(3) for j in range(3))
            Ks.append(sp.nsimplify(sp.simplify(num / (ip(vv, vv) * ip(ww, ww) - ip(vv, ww) ** 2))))
        check(f"curvature: {name} sectional curvatures at {pt} lie in the stated range: {Ks}", all(ok(K) for K in Ks))
        if name == "AI" and pt == {a: 1, b: 0, c: 1}:
            check("curvature: AI at I has K = -1/2 on the traceless plane", Ks[3] == -sp.Rational(1, 2))
            check("curvature: AI at I has K = 0 on planes containing the identity direction", Ks[4] == 0 and Ks[5] == 0)

# AI: planes tangent to a level set {det = const} have K = -1/2 (the det-level slice is a hyperbolic plane)
GamAI = dgsym.christoffel(gAI, coords)
RmAI = dgsym.riemann(gAI, coords, Gamma=GamAI)
for pt in ({a: 2, b: 0, c: sp.Rational(1, 2)}, {a: 2, b: 1, c: 1}, {a: 3, b: -1, c: 2}):
    Spt = Sab.subs(pt)
    row = sp.Matrix([[(Spt.inv() * Ei).trace() for Ei in Ebas]])
    ns = row.nullspace()
    vv, ww = list(ns[0]), list(ns[1])
    gp = gAI.subs(pt)
    num = sum(gp[l, m] * RmAI[i][j][k][m].subs(pt) * vv[i] * ww[j] * ww[k] * vv[l]
              for i in range(3) for j in range(3) for k in range(3) for l in range(3) for m in range(3))
    ip = lambda p_, q_: sum(gp[i, j] * p_[i] * q_[j] for i in range(3) for j in range(3))
    check(f"curvature: AI, plane tangent to the det-level set at {pt} has K = -1/2",
          sp.simplify(num / (ip(vv, vv) * ip(ww, ww) - ip(vv, ww) ** 2)) == -sp.Rational(1, 2))

# ---------------------------------------------------------------- Fréchet means (AI Karcher mean, BW barycenter)
Ss = [rand_spd() for _ in range(4)]
wts = np.array([0.1, 0.2, 0.3, 0.4])
M = sum(w_ * S_ for w_, S_ in zip(wts, Ss))
for _ in range(100):
    h = sqrtm(M); hi = np.linalg.inv(h)
    Tm = sum(w_ * logm(hi @ S_ @ hi) for w_, S_ in zip(wts, Ss))
    M = h @ expm(Tm) @ h
h = sqrtm(M); hi = np.linalg.inv(h)
close("means: AI Karcher mean satisfies sum w_i log(M^-1/2 S_i M^-1/2) = 0",
      sum(w_ * logm(hi @ S_ @ hi) for w_, S_ in zip(wts, Ss)), np.zeros((3, 3)), tol=1e-10)
close("means: det of the AI mean = prod det(S_i)^w_i", np.linalg.det(M), np.prod([np.linalg.det(S_) ** w_ for w_, S_ in
                                                                                    zip(wts, Ss)]), tol=1e-8)
Bm = np.eye(3)
for _ in range(300):
    hb = sqrtm(Bm)
    Bm = sum(w_ * sqrtm(hb @ S_ @ hb) for w_, S_ in zip(wts, Ss))
    Bm = np.linalg.inv(hb) @ Bm @ Bm @ np.linalg.inv(hb)
hb = sqrtm(Bm)
close("means: BW barycenter fixed point Sigma = sum w_i (Sigma^1/2 S_i Sigma^1/2)^1/2 (squared form)",
      sum(w_ * sqrtm(hb @ S_ @ hb) for w_, S_ in zip(wts, Ss)), Bm, tol=1e-8)

# ---------------------------------------------------------------- (29.4.7): moment matching minimizes KL(p || q)
mus = [np.array([-1.2, 0.0]), np.array([1.2, 0.3])]
Sg = [R2(20) @ np.diag([0.64, 0.0225]) @ R2(20).T, R2(-30) @ np.diag([0.64, 0.0225]) @ R2(-30).T]
wv = np.array([0.5, 0.5])
mu_mm = sum(w_ * m_ for w_, m_ in zip(wv, mus))
S_mm = sum(w_ * (S_ + np.outer(m_ - mu_mm, m_ - mu_mm)) for w_, m_, S_ in zip(wv, mus, Sg))


def cross_entropy(mu_q, S_q):
    """E_p[-log q] for the mixture p, closed form (depends on p only through its mean and covariance)."""
    Sqi = np.linalg.inv(S_q)
    val = 0.0
    for w_, m_, S_ in zip(wv, mus, Sg):
        d = m_ - mu_q
        val += w_ * 0.5 * (np.trace(Sqi @ S_) + d @ Sqi @ d)
    return val + 0.5 * np.log(np.linalg.det(2 * np.pi * S_q))


base = cross_entropy(mu_mm, S_mm)
worse = []
for _ in range(200):
    dm = 0.05 * rng.normal(size=2)
    dS = 0.05 * rng.normal(size=(2, 2)); dS = dS + dS.T
    if np.linalg.eigvalsh(S_mm + dS).min() > 0:
        worse.append(cross_entropy(mu_mm + dm, S_mm + dS) >= base)
check("(29.4.7): perturbing the moment-matched (mu, Sigma) never lowers E_p[-log q] (so KL(p||q))", all(worse))
# numerical KL on a grid for Figure 29.4.4(a)
gx = np.linspace(-6, 6, 601)
GX, GY = np.meshgrid(gx, gx, indexing="ij")
P = np.stack([GX, GY], -1)
dA = (gx[1] - gx[0]) ** 2


def npdf(m_, S_):
    d = P - m_
    return np.exp(-0.5 * np.einsum("...i,ij,...j->...", d, np.linalg.inv(S_), d)) / (2 * np.pi * np.sqrt(np.linalg.det(S_)))


pmix = sum(w_ * npdf(m_, S_) for w_, m_, S_ in zip(wv, mus, Sg))
h = sqrtm(Sg[0]); hi = np.linalg.inv(h)
S_ai = h @ fun(hi @ Sg[1] @ hi, np.sqrt) @ h               # AI midpoint = AI mean of two
KL = lambda q: np.sum(pmix * (np.log(pmix + 1e-300) - np.log(q + 1e-300))) * dA
kl_mm, kl_ai = KL(npdf(mu_mm, S_mm)), KL(npdf(mu_mm, S_ai))
print(f"KL(p||q_mm) = {kl_mm:.3f}, KL(p||q_AI) = {kl_ai:.3f}")
close("Figure 29.4.4(a): KL(p || q_moment) = 0.86", kl_mm, 0.86, tol=0.006)
close("Figure 29.4.4(a): KL(p || q_AI mean) = 4.0", kl_ai, 4.0, tol=0.03)
close("Figure 29.4.4(a): grid check, mixture integrates to 1", pmix.sum() * dA, 1.0, tol=1e-6)
# (b) same centers: moment matching = Euclidean mean of covariances
S_mm_b = sum(w_ * S_ for w_, S_ in zip(wv, Sg))
close("Figure 29.4.4(b): with equal centers moment matching is the Euclidean mean", S_mm_b, geo(Sg[0], Sg[1], 0.5, "E"),
      tol=1e-15)
print("(b) det: inputs %.4f, moment %.4f, AI %.4f" % (np.linalg.det(Sg[0]), np.linalg.det(S_mm_b), np.linalg.det(S_ai)))
close("Figure 29.4.4(b): det of the inputs 0.0144, moment-matched 0.0703, AI mean 0.0144",
      np.round([np.linalg.det(Sg[0]), np.linalg.det(S_mm_b), np.linalg.det(S_ai)], 4), [0.0144, 0.0703, 0.0144], tol=1e-9)

# ---------------------------------------------------------------- split: expected moment of the two children
Ssplit = rand_spd()
musplit = np.zeros(3)
Lc = np.linalg.cholesky(Ssplit)
acc = np.zeros((3, 3))
nrep = 100000
X1 = rng.normal(size=(nrep, 3)) @ Lc.T
X2 = rng.normal(size=(nrep, 3)) @ Lc.T
mid = 0.5 * (X1 + X2)
spread = 0.5 * (np.einsum("ni,nj->ij", X1 - mid, X1 - mid) + np.einsum("ni,nj->ij", X2 - mid, X2 - mid)) / nrep
Emix = Ssplit / 1.6 ** 2 + spread
close("split: E[moment of the two children] = (1/2.56 + 1/2) Sigma = 0.89 Sigma (MC, rel tol 1%)",
      Emix / np.linalg.norm(Ssplit), 0.890625 * Ssplit / np.linalg.norm(Ssplit), tol=0.01)
close("split: 1/1.6^2 + 1/2 = 0.890625", 1 / 1.6 ** 2 + 0.5, 0.890625, tol=1e-15)
# the ray {c Sigma} is a geodesic of all four metrics: every point of the geodesic Sigma -> Sigma/2.56 is a multiple of Sigma
for kind in ("E", "LE", "AI", "BW"):
    on_ray = []
    for t in np.linspace(0, 1, 9):
        G = geo(Ssplit, Ssplit / 2.56, t, kind)
        c = np.trace(G) / np.trace(Ssplit)
        on_ray.append(np.allclose(G, c * Ssplit, atol=1e-10))
    check(f"Example 29.4.4: the geodesic from Sigma to Sigma/2.56 stays on the ray c Sigma ({kind})", all(on_ray))
close("Example 29.4.4: AI and LE parametrize the ray as e^t Sigma (midpoint Sigma/1.6)",
      [geo(Ssplit, Ssplit / 2.56, 0.5, k) for k in ("AI", "LE")], [Ssplit / 1.6] * 2, tol=1e-10)

# ---------------------------------------------------------------- R2 review: extra independent checks
# AI midpoint = geometric mean: the unique SPD solution of G A^{-1} G = B (Riccati equation)
A_, B_ = rand_spd(), rand_spd()
G_ = geo(A_, B_, 0.5, "AI")
close("AI midpoint solves the Riccati equation G A^-1 G = B", G_ @ np.linalg.inv(A_) @ G_, B_, tol=1e-9)
# BW between commuting matrices = linear interpolation of standard deviations (text after (29.4.6))
Da, Db = np.diag([4.0, 0.3, 1.0]), np.diag([0.5, 2.0, 1.0])
close("BW geodesic of commuting covariances interpolates the standard deviations",
      geo(Da, Db, 0.3, "BW"), np.diag((0.7 * np.sqrt(np.diag(Da)) + 0.3 * np.sqrt(np.diag(Db))) ** 2), tol=1e-12)
# det = 1 slice of SPD(2) in the chart Sigma = (1/v) [[1, u], [u, u^2 + v^2]]: AI metric 2 (du^2 + dv^2)/v^2, K = -1/2;
# with the Fisher normalization (1/2) tr(...) the curvature is -1
uu, vv = sp.symbols("u v", real=True)
vv = sp.Symbol("v", positive=True)
Sg2 = sp.Matrix([[1 / vv, uu / vv], [uu / vv, (uu ** 2 + vv ** 2) / vv]])
Sgi = Sg2.inv()
gAI = lambda X, Y: sp.simplify((Sgi * X * Sgi * Y).trace())
Eu, Fu, Gu = gAI(Sg2.diff(uu), Sg2.diff(uu)), gAI(Sg2.diff(uu), Sg2.diff(vv)), gAI(Sg2.diff(vv), Sg2.diff(vv))
sym_equal("Looking back box: on det = 1 the AI metric is 2 (du^2 + dv^2) / v^2", sp.Matrix([Eu, Fu, Gu]),
          sp.Matrix([2 / vv ** 2, 0, 2 / vv ** 2]))
Wu = sp.sqrt(Eu * Gu)
Kslice = sp.simplify(-1 / (2 * Wu) * (sp.diff(Gu.diff(uu) / Wu, uu) + sp.diff(Eu.diff(vv) / Wu, vv)))
sym_equal("curvature text: K = -1/2 on the det = 1 slice for g = tr(S^-1 U S^-1 V)", Kslice, sp.Rational(-1, 2))
Eh, Gh = Eu / 2, Gu / 2
Wh = sp.sqrt(Eh * Gh)
Khalf = sp.simplify(-1 / (2 * Wh) * (sp.diff(Gh.diff(uu) / Wh, uu) + sp.diff(Eh.diff(vv) / Wh, vv)))
sym_equal("curvature text: with the Fisher normalization (1/2) tr(...) the slice has K = -1", Khalf, -1)
# Fisher information of N(0, Sigma) in the direction U is (1/2) tr(S^-1 U S^-1 U) (Monte Carlo variance of the score)
Sf = rand_spd(2); Uf = np.array([[0.7, -0.4], [-0.4, 0.2]]); Sfi = np.linalg.inv(Sf)
Xf = rng.normal(size=(400000, 2)) @ np.linalg.cholesky(Sf).T
score = -0.5 * np.trace(Sfi @ Uf) + 0.5 * np.einsum("ni,ij,nj->n", Xf, Sfi @ Uf @ Sfi, Xf)
close("curvature text: Fisher metric of N(0, Sigma) is (1/2) tr(S^-1 U S^-1 U) (MC, rel tol 2%)",
      np.var(score) / (0.5 * np.trace(Sfi @ Uf @ Sfi @ Uf)), 1.0, tol=0.02)
# Alexandrov comparison (independent of Christoffel symbols): AI is CAT(0), BW satisfies the K >= 0 inequality
gap_ai, gap_bw = [], []
for _ in range(300):
    a_, b_, p_ = rand_spd(), rand_spd(), rand_spd()
    for kind, gaps in (("AI", gap_ai), ("BW", gap_bw)):
        m_ = geo(a_, b_, 0.5, kind)
        gaps.append(dist(p_, m_, kind) ** 2 - 0.5 * dist(p_, a_, kind) ** 2 - 0.5 * dist(p_, b_, kind) ** 2
                    + 0.25 * dist(a_, b_, kind) ** 2)
check("curvature sign: AI satisfies the CAT(0) midpoint inequality (300 random triangles)", max(gap_ai) <= 1e-9)
check("curvature sign: BW satisfies the reverse (K >= 0) midpoint inequality (300 random triangles)", min(gap_bw) >= -1e-9)

# ---------------------------------------------------------------- Exercises
Ea, Eb = np.diag([4.0, 1.0]), np.diag([1.0, 4.0])
close("Exercise 29.4.1: E midpoint 2.5 I", geo(Ea, Eb, 0.5, "E"), 2.5 * np.eye(2), tol=1e-12)
close("Exercise 29.4.1: LE = AI midpoint 2 I", [geo(Ea, Eb, 0.5, "LE"), geo(Ea, Eb, 0.5, "AI")], [2 * np.eye(2)] * 2, tol=1e-12)
close("Exercise 29.4.1: BW midpoint 2.25 I", geo(Ea, Eb, 0.5, "BW"), 2.25 * np.eye(2), tol=1e-12)
close("Exercise 29.4.1: dets 6.25, 4, 4, 5.0625 vs endpoints 4", [np.linalg.det(geo(Ea, Eb, 0.5, k)) for k in ("E", "LE", "AI", "BW")],
      [6.25, 4.0, 4.0, 5.0625], tol=1e-12)
close("Exercise 29.4.2: d_AI(diag(1,1,1e-4), diag(1,1,1e-6)) = ln 100 = 4.61",
      dist(np.diag([1, 1, 1e-4]), np.diag([1, 1, 1e-6]), "AI"), np.log(100), tol=1e-9)
close("Exercise 29.4.2: d_E between them is about 1e-4", dist(np.diag([1, 1, 1e-4]), np.diag([1, 1, 1e-6]), "E"), 9.9e-5,
      tol=1e-12)
m1, m2 = np.array([0.0, 0]), np.array([2.0, 0])
Sm = sum(0.5 * (np.eye(2) * 0.25 + np.outer(m_ - np.array([1.0, 0]), m_ - np.array([1.0, 0]))) for m_ in (m1, m2))
close("Exercise 29.4.3: moment matching of two s = 0.5 circles 2 apart: diag(1.25, 0.25)", Sm, np.diag([1.25, 0.25]), tol=1e-12)

summary()
