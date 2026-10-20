"""Verification for Section 42.6 (Poisson Problems and Implicit Mean Curvature Flow).

Run from the project root:
``PYTHONPATH=tools python3 chapters/42-laplace-beltrami/verify/v-42-6-implicit-mcf.py``
"""

import pathlib
import re
import sys
import warnings

import dgdec as dd
import dgmesh as dm
import numpy as np
import sympy as sp

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "lib"))
import ch42  # noqa: E402

from dgcheck import check, summary  # noqa: E402

warnings.simplefilter("ignore")
PAGE = HERE / "42-6-implicit-mcf.html"


def near(a, b, tol=1e-9):
    return bool(np.all(np.abs(np.asarray(a, float) - np.asarray(b, float)) <= tol))


rng = np.random.default_rng(0)

# ---------------------------------------------------------------- Proposition 42.6.1 on two closed meshes
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)), ("torus", dm.torus_mesh(2.0, 0.8, 24, 12))):
    L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
    A = -L + np.outer(m, m)
    check(f"Proposition 42.6.1(c): -L + a a^T is symmetric positive definite on the {name}",
          np.allclose(A, A.T) and np.linalg.eigvalsh(A).min() > 1e-8)
    rho = rng.normal(size=len(V))
    rho_bar = (m @ rho) / m.sum()
    b = m * (rho - rho_bar)
    check(f"Proposition 42.6.1(b): 1^T M (rho - rho_bar 1) = 0 on the {name}", abs(b.sum()) < 1e-12)
    u = np.linalg.solve(A, -b)
    check(f"Proposition 42.6.1(d): the solution has a^T u = 0 and L u = b on the {name}",
          abs(m @ u) < 1e-11 and np.abs(L @ u - b).max() < 1e-11)
    u2, rb2 = ch42.solve_closed_poisson(L, m, rho)
    check(f"ch42.solve_closed_poisson agrees on the {name}", np.abs(u2 - u).max() < 1e-12 and near(rb2, rho_bar))
    bad = m * rho                                         # 1^T b != 0 in general
    x = np.linalg.lstsq(L, bad, rcond=None)[0]
    check(f"Proposition 42.6.1(a): L u = M rho with nonzero total has no solution (least-squares residual > 0) on the {name}",
          abs(bad.sum()) > 1e-3 and np.linalg.norm(L @ x - bad) > 1e-3)
    ev = np.linalg.eigvalsh(-L)
    check(f"Proposition 42.6.1(a): ker L is one-dimensional on the {name}", abs(ev[0]) < 1e-10 and ev[1] > 1e-6)
    # pinning one vertex instead: the reduced matrix is positive definite (as in Proposition 42.5.7)
    keep = np.arange(1, len(V))
    check(f"Section 42.6 text: deleting the row and column of one vertex leaves -L positive definite on the {name}",
          np.linalg.eigvalsh(-L[np.ix_(keep, keep)]).min() > 1e-8)
    up = np.zeros(len(V))
    up[keep] = np.linalg.solve(L[np.ix_(keep, keep)], b[keep])
    check(f"Section 42.6 text: the pinned solution differs from the mean-zero one by a constant on the {name}",
          np.ptp(up - u) < 1e-9 and np.abs(L @ up - b).max() < 1e-9)

# ---------------------------------------------------------------- Example 42.6.2: Green's function on the unit sphere
th, eps = sp.symbols("theta epsilon", positive=True)
G = sp.log(1 - sp.cos(th)) / (4 * sp.pi)
lapG = sp.simplify(sp.diff(sp.sin(th) * sp.diff(G, th), th) / sp.sin(th))
check("Example 42.6.2: Delta G = -1/(4 pi) for theta > 0 (the axially symmetric coordinate formula)",
      sp.simplify(lapG + 1 / (4 * sp.pi)) == 0)
check("Example 42.6.2: (1/sin) d_theta (sin d_theta G) = d_theta(1 + cos theta) / (4 pi sin theta)",
      sp.simplify(sp.sin(th) * sp.diff(G, th) - (1 + sp.cos(th)) / (4 * sp.pi)) == 0)
check("Example 42.6.2: G - (1/2pi) log theta has a finite limit at theta = 0 (the planar singularity)",
      sp.limit(G - sp.log(th) / (2 * sp.pi), th, 0) == -sp.log(2) / (4 * sp.pi))
res = {}
for level in (3, 4):
    V, F = dm.icosphere(level)
    L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
    s = int(np.argmax(V[:, 2]))
    rho = np.zeros(len(V))
    rho[s] = 1.0 / m[s]
    u, rb = ch42.solve_closed_poisson(L, m, rho)
    t = np.arccos(np.clip(V[:, 2], -1, 1))
    far = t > 0.3
    Gf = np.log(1 - np.cos(t[far])) / (4 * np.pi)
    c = np.mean(u[far] - Gf)
    res[level] = (len(V), rb, np.abs(u[far] - Gf - c).max(), u[s], m.sum())
n4, rb4, err4, us4, area4 = res[4]
check("Example 42.6.2: icosphere(4) has 2562 vertices and a^T rho = 1", n4 == 2562)
check("Example 42.6.2: rho_bar = 0.07967 on icosphere(4) (1/(4 pi) = 0.07958; rho_bar = 1 / mesh area)",
      near(rb4, 0.07967, tol=5e-6) and near(1 / (4 * np.pi), 0.07958, tol=5e-6) and near(rb4, 1 / area4))
check("Example 42.6.2, Figure 42.6.1: max |u - G - c| = 4.4e-4 on icosphere(4) for theta > 0.3", near(err4, 4.4e-4, tol=5e-6))
check("Example 42.6.2: the same error is 1.2e-3 on icosphere(3)", near(res[3][2], 1.2e-3, tol=5e-5))
check("Figure 42.6.1: u at the source is -0.731 on icosphere(4)", near(us4, -0.731, tol=5e-4))

# ---------------------------------------------------------------- Implementation code-42-6-1
ns, same, out = ch42.run_printed(PAGE, "code-42-6-1", "out-42-6-1")
check("code-42-6-1 runs and prints exactly out-42-6-1", same)
V3, _ = dm.icosphere(3)
check("code-42-6-1: the two sources are vertices 25 (north pole) and 21 (-x direction)",
      int(np.argmax(V3[:, 2])) == 25 and int(np.argmin(V3[:, 0])) == 21 and near(V3[25, 2], 1.0) and near(V3[21, 0], -1.0))
C = ns["C"]
check("code-42-6-1: chol_solve agrees with numpy.linalg.solve", np.abs(
    ns["chol_solve"](C, np.arange(len(C), dtype=float)) - np.linalg.solve(C @ C.T, np.arange(len(C), dtype=float))).max() < 1e-10)

# ---------------------------------------------------------------- Remark 42.6.3
check("Remark 42.6.3: icosphere(5) has 10242 vertices (10 4^k + 2)", 10 * 4 ** 5 + 2 == 10242 and 10 * 4 ** 4 + 2 == 2562)
check("Remark 42.6.3: 8 n0^2 bytes is about 52 MB (n0 = 2562) and 840 MB (n0 = 10242)",
      round(8 * 2562 ** 2 / 1e6) == 53 and abs(8 * 2562 ** 2 / 1e6 - 52) < 1 and abs(8 * 10242 ** 2 / 1e6 - 840) < 2)
check("Remark 42.6.3: n0^3 / 3 = 5.6e9 for n0 = 2562", near(2562 ** 3 / 3, 5.6e9, tol=0.05e9))

# ---------------------------------------------------------------- Proposition 42.6.4 on icosphere(2) and a noisy sphere
for name, (V, F) in (("icosphere(2)", dm.icosphere(2)),
                     ("noisy icosphere(2)", (dm.icosphere(2)[0] * (1 + 0.05 * rng.normal(size=(162, 1))), dm.icosphere(2)[1]))):
    L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
    M = np.diag(m)
    s = 1 / np.sqrt(m)
    lam, Psi = np.linalg.eigh(-(s[:, None] * L * s[None, :]))
    Phi = s[:, None] * Psi                                     # M-orthonormal generalized eigenvectors
    check(f"Proposition 42.6.4: Phi^T M Phi = I and -L Phi = M Phi diag(lambda) on the {name}",
          np.abs(Phi.T @ M @ Phi - np.eye(len(m))).max() < 1e-10 and np.abs(-L @ Phi - M @ Phi * lam).max() < 1e-9)
    ok_a = ok_b = ok_c = ok_d = True
    u0 = rng.normal(size=len(m))
    for h in (1e-3, 0.1, 10.0, 1e3):
        ok_a &= np.linalg.eigvalsh(M - h * L).min() > 0
        u1 = np.linalg.solve(M - h * L, M @ u0)
        ok_b &= abs(m @ u1 - m @ u0) < 1e-10 * (1 + abs(m @ u0))
        c0, c1 = Phi.T @ M @ u0, Phi.T @ M @ u1
        ok_c &= np.abs(c1 - c0 / (1 + h * lam)).max() < 1e-8 and u1 @ M @ u1 <= u0 @ M @ u0 + 1e-12 \
            and -u1 @ L @ u1 <= -u0 @ L @ u0 + 1e-12
        ue = u0 + h * (L @ u0) / m
        ok_d &= np.abs(Phi.T @ M @ ue - (1 - h * lam) * c0).max() < 1e-6 * max(1, h * lam[-1])
    check(f"Proposition 42.6.4(a): M - h L is positive definite for h = 1e-3 ... 1e3 on the {name}", ok_a)
    check(f"Proposition 42.6.4(b): the implicit step conserves 1^T M u on the {name}", ok_b)
    check(f"Proposition 42.6.4(c): implicit factors 1/(1 + h lambda_j); M-norm and -u^T L u do not increase ({name})", ok_c)
    check(f"Proposition 42.6.4(d): explicit factors 1 - h lambda_j on the {name}", ok_d)
    hc = 2 / lam[-1]
    pert = Phi[:, -1]
    def grow(h):
        w = np.linalg.matrix_power(np.eye(len(m)) + h * L / m[:, None], 200) @ pert
        return np.sqrt(w @ (m * w))                            # pert has M-norm 1
    check(f"Proposition 42.6.4(d): explicit steps stay bounded at h = h_c and grow at 1.05 h_c ({name})",
          grow(hc) <= 1.0 + 1e-6 and grow(1.05 * hc) > 1e3)

# ---------------------------------------------------------------- Example 42.6.5, Exercise 42.6.4
ns, same, out = ch42.run_printed(PAGE, "code-ex-42-6-4", "out-ex-42-6-4")
check("code-ex-42-6-4 runs and prints exactly out-ex-42-6-4", same)
check("Example 42.6.5: lambda_max l^2 = 6.16, 6.61, 6.99 and h_c / l^2 = 0.325, 0.303, 0.286 (as printed)",
      all(f"lambda_max l^2 = {a}" in out and f"h_c / l^2 = {b}" in out
          for a, b in (("6.16", "0.325"), ("6.61", "0.303"), ("6.99", "0.286"))))
check("Exercise 42.6.4: |1 - 0.99 * 2| = 0.98 and 1.02^400 is about 2.8e3",
      near(abs(1 - 0.99 * 2), 0.98) and near(1.02 ** 400, 2.75e3, tol=0.06e3))
V3, F3e = dm.icosphere(3)
L3, m3 = dm.cotan_laplacian(V3, F3e), dm.mass_voronoi(V3, F3e)
s3 = 1 / np.sqrt(m3)
lam3 = np.linalg.eigvalsh(-(s3[:, None] * L3 * s3[None, :]))
h3 = 0.99 * 2 / lam3[-1]
check("Exercise 42.6.4: at 0.99 h_c on icosphere(3) lambda_1 is about 2 and its factor 0.987 is the largest for j >= 1",
      near(lam3[1], 2.0, tol=1e-4) and near(1 - h3 * lam3[1], 0.987, tol=5e-4)
      and np.isclose(np.abs(1 - h3 * lam3[1:]).max(), 1 - h3 * lam3[1]))
check("Example 42.6.5: growth at 1.01 h_c exceeds 400 at every level (as printed)",
      all(float(re.search(r", ([0-9.e+-]+) \(1\.01 h_c\)", ln).group(1)) > 400 for ln in out.strip().splitlines()))

# ---------------------------------------------------------------- Theorem 42.6.6 (area never increases) and its proof
V0, F3 = dm.icosphere(3)
noisy = V0 * (1 + 0.05 * np.random.default_rng(5).normal(size=(len(V0), 1)))
sv = 1 - 0.5 * np.exp(-(V0[:, 2] / 0.4) ** 2)
peanut = np.c_[V0[:, 0] * sv, V0[:, 1] * sv, 1.5 * V0[:, 2]]
Vt, Ft = dm.torus_mesh(2.0, 0.8, 24, 12)
ok = True
for X, F in ((noisy, F3), (peanut, F3), (Vt, Ft)):
    for h in (1e-3, 0.05, 1.0, 100.0):
        L, m = dm.cotan_laplacian(X, F), dm.mass_voronoi(X, F)
        Y = np.linalg.solve(np.diag(m) - h * L, m[:, None] * X)
        ok &= dm.face_areas(Y, F).sum() <= dm.face_areas(X, F).sum() + 1e-12
check("Theorem 42.6.6: one semi-implicit step never increases area (noisy sphere, peanut, torus; h = 1e-3 ... 100)", ok)


def frozen_energy(X, F, Y):
    L = dm.cotan_laplacian(X, F)
    return -0.5 * np.sum(Y * (L @ Y))


def sv_terms(X, F, Y):
    """sum_f A_f(X) (1/2 |J_f|^2) and sum_f A_f(X) sigma1 sigma2, J_f the derivative of the affine map X_f -> Y_f."""
    e_half, e_prod = 0.0, 0.0
    for tri in F:
        P, Q = X[tri], Y[tri]
        u1 = P[1] - P[0]
        nrm = np.cross(u1, P[2] - P[0])
        a = 0.5 * np.linalg.norm(nrm)
        e1 = u1 / np.linalg.norm(u1)
        e2 = np.cross(nrm / np.linalg.norm(nrm), e1)
        Bp = np.array([[e1 @ (P[1] - P[0]), e1 @ (P[2] - P[0])], [e2 @ (P[1] - P[0]), e2 @ (P[2] - P[0])]])
        J = np.c_[Q[1] - Q[0], Q[2] - Q[0]] @ np.linalg.inv(Bp)     # 3 x 2
        sg = np.linalg.svd(J, compute_uv=False)
        e_half += a * 0.5 * np.sum(J ** 2)
        e_prod += a * sg[0] * sg[1]
    return e_half, e_prod


X, F = noisy, F3
check("Theorem 42.6.6 proof (i): E_k(x^k) = Area(x^k)", near(frozen_energy(X, F, X), dm.face_areas(X, F).sum(), tol=1e-10))
ok_ii = ok_sv = True
for _ in range(5):
    Y = X + 0.2 * rng.normal(size=X.shape)
    eh, ep = sv_terms(X, F, Y)
    ok_sv &= near(eh, frozen_energy(X, F, Y), tol=1e-9) and near(ep, dm.face_areas(Y, F).sum(), tol=1e-9)
    ok_ii &= frozen_energy(X, F, Y) >= dm.face_areas(Y, F).sum()
check("Theorem 42.6.6 proof: E_k(y) = sum A_f |J_f|^2 / 2 and sum A_f sigma1 sigma2 = Area(y) (random y)", ok_sv)
check("Theorem 42.6.6 proof (ii): E_k(y) >= Area(y) for random y", ok_ii)
L, m = dm.cotan_laplacian(X, F), dm.mass_voronoi(X, F)
h = 0.05
Y = np.linalg.solve(np.diag(m) - h * L, m[:, None] * X)
Phi_ = lambda Z: frozen_energy(X, F, Z) + np.sum(m[:, None] * (Z - X) ** 2) / (2 * h)  # noqa: E731
check("Theorem 42.6.6 proof (iii): the step minimizes Phi (random perturbations increase it)",
      all(Phi_(Y + 1e-3 * rng.normal(size=Y.shape)) > Phi_(Y) for _ in range(5)))
check("Theorem 42.6.6 proof: Area(x^{k+1}) <= E_k(x^{k+1}) <= Phi(x^{k+1}) <= Phi(x^k) = Area(x^k)",
      dm.face_areas(Y, F).sum() <= frozen_energy(X, F, Y) <= Phi_(Y) <= Phi_(X)
      and near(Phi_(X), dm.face_areas(X, F).sum(), tol=1e-10))

# ---------------------------------------------------------------- Example 42.6.7, Figure 42.6.2, Exercise 42.6.2
L, m = dm.cotan_laplacian(V0, F3), dm.mass_voronoi(V0, F3)
Hn = (L @ V0) / m[:, None]
rad = np.sum(Hn * V0, 1)
check("Example 42.6.7: icosphere(3) has no obtuse triangle; mixed = circumcentric areas",
      np.degrees(dm.corner_angles(V0, F3)).max() < 90 and np.abs(m - dd.dual_areas(V0, F3, "circumcentric")).max() < 1e-14)
check("Example 42.6.7: the radial component of M^{-1} L x is exactly -2/r (r = 1); the tangential part is small",
      np.abs(rad + 2).max() < 1e-12 and np.linalg.norm(Hn - rad[:, None] * V0, axis=1).max() < 0.02)
runs = {}
for h in (0.01, 0.05, 0.2):
    V = V0.copy()
    r2, pred = [1.0], [1.0]
    for k in range(int(round(0.6 / h))):
        L, m = dm.cotan_laplacian(V, F3), dm.mass_voronoi(V, F3)
        V = np.linalg.solve(np.diag(m) - h * L, m[:, None] * V)
        r2.append(np.mean(np.sum((V - V.mean(0)) ** 2, 1)))
        rp = np.sqrt(pred[-1])
        pred.append((rp / (1 + 2 * h / rp ** 2)) ** 2)
        if r2[-1] < 1e-3:
            break
    runs[h] = (h * np.arange(len(r2)), np.array(r2), np.array(pred))
t2, r2, pred = runs[0.2]
check("Example 42.6.7: h = 0.2 gives r^2 = 0.5102, 0.1603, 0.0131 at t = 0.2, 0.4, 0.6",
      np.allclose(r2[1:4], [0.5102, 0.1603, 0.0131], atol=5e-5))
check("Example 42.6.7, Figure 42.6.2: mesh r^2 agrees with the recurrence to 2e-3 and lies above 1 - 4t (all h)",
      all(np.abs(r - p).max() < 2e-3 and np.all(r >= 1 - 4 * t - 1e-12) and np.all(r > 0) for t, r, p in runs.values()))
check("Example 42.6.7: r^2(0.2) = 0.343 for h = 0.05 and 0.241 for h = 0.01",
      near(runs[0.05][1][4], 0.343, tol=5e-4) and near(runs[0.01][1][20], 0.241, tol=5e-4))
check("Figure 42.6.2 alt text: h = 0.01 reaches r^2 < 1e-3 at t = 0.29; h = 0.05 at t = 0.4",
      near(runs[0.01][0][-1], 0.29) and runs[0.01][1][-1] < 1e-3 and near(runs[0.05][0][-1], 0.4))
hs, r = sp.symbols("h r", positive=True)
nxt = r / (1 + 2 * hs / r ** 2)
check("Exercise 42.6.2: r_{k+1}^2 = r_k^2 - 4h + O(h^2)",
      sp.simplify(sp.series(nxt ** 2, hs, 0, 2).removeO() - (r ** 2 - 4 * hs)) == 0)
check("Exercise 42.6.2: r_{k+1} / (r_k^3 / (2h)) -> 1 as r_k -> 0", sp.limit(nxt / (r ** 3 / (2 * hs)), r, 0) == 1)
import mpmath  # noqa: E402  (arbitrary exponent range: r_k decays like r^3 / (2h) and underflows in floats)

seq = [mpmath.mpf(1)]
for _ in range(30):
    seq.append(seq[-1] / (1 + mpmath.mpf("0.4") / seq[-1] ** 2))
check("Exercise 42.6.2: 0 < r_{k+1} < r_k (h = 0.2, 30 steps from r = 1, arbitrary-precision floats)",
      all(0 < b < a for a, b in zip(seq[:-1], seq[1:])))

# ---------------------------------------------------------------- Implementation code-42-6-2
ns, same, out = ch42.run_printed(PAGE, "code-42-6-2", "out-42-6-2")
check("code-42-6-2 runs and prints exactly out-42-6-2", same)
check("code-42-6-2: roughness 0.050 -> 0.014 in 10 steps of h = 0.002 (t = 0.02)",
      near(ns["rough"][0], 0.0497, tol=5e-5) and near(ns["rough"][-1], 0.0139, tol=5e-5) and len(ns["rough"]) == 11)

# ---------------------------------------------------------------- Example 42.6.8, Figure 42.6.3
L, m = dm.cotan_laplacian(noisy, F3), dm.mass_voronoi(noisy, F3)
lam = np.linalg.eigvalsh(-(L / np.sqrt(m)[:, None] / np.sqrt(m)[None, :]))
hc = 2 / lam[-1]
ell = dd.primal_edge_lengths(noisy, F3).mean()
check("Example 42.6.8: h_c = 0.00656 = 0.24 l^2 on the noisy sphere", near(hc, 0.00656, tol=5e-6) and near(hc / ell ** 2, 0.24, tol=0.005))
check("Example 42.6.8: h = 0.02 is about 3 h_c", near(0.02 / hc, 3.0, tol=0.1))


def rough(V):
    d = np.linalg.norm(V - V.mean(0), axis=1)
    return d.std() / d.mean()


stab = {}
for kind, h in (("e", 0.5 * hc), ("e", 0.9 * hc), ("e", 1.5 * hc), ("i", 0.002), ("i", 0.02)):
    V = noisy.copy()
    R, Ar = [rough(V)], [dm.face_areas(V, F3).sum()]
    for k in range(int(0.1 / h)):
        L, m = dm.cotan_laplacian(V, F3), dm.mass_voronoi(V, F3)
        V = V + h * (L @ V) / m[:, None] if kind == "e" else np.linalg.solve(np.diag(m) - h * L, m[:, None] * V)
        R.append(rough(V))
        Ar.append(dm.face_areas(V, F3).sum())
    stab[(kind, round(h / hc, 1) if kind == "e" else h)] = (np.array(R), np.array(Ar), h * (len(R) - 1))
R, _, T = stab[("e", 0.5)]
check("Example 42.6.8: explicit 0.5 h_c ends at roughness 0.0057 (t = 0.098)", near(R[-1], 0.0057, tol=5e-5) and near(T, 0.098, tol=1e-3))
R, _, T = stab[("e", 0.9)]
check("Example 42.6.8: explicit 0.9 h_c reaches 0.0080 and ends at 0.0188 (t = 0.094)",
      near(R.min(), 0.0080, tol=5e-5) and near(R[-1], 0.0188, tol=5e-5) and near(T, 0.094, tol=1e-3))
check("Figure 42.6.3: explicit 0.9 h_c starts rising again near t = 0.06",
      0.05 < 0.9 * hc * int(np.argmin(R)) < 0.065)
R, _, T = stab[("e", 1.5)]
check("Example 42.6.8: explicit 1.5 h_c ends rougher (0.0664) than it started (0.0497), minimum 0.0274",
      near(R[-1], 0.0664, tol=5e-5) and near(R[0], 0.0497, tol=5e-5) and near(R.min(), 0.0274, tol=5e-5))
check("Example 42.6.8: semi-implicit h = 0.002 and 0.02 decrease roughness and area at every step (ends 0.0058, 0.0070)",
      all(np.all(np.diff(stab[("i", h)][0]) < 0) and np.all(np.diff(stab[("i", h)][1]) <= 1e-12) for h in (0.002, 0.02))
      and near(stab[("i", 0.002)][0][-1], 0.0058, tol=5e-5) and near(stab[("i", 0.02)][0][-1], 0.0070, tol=5e-5))
V = noisy.copy()
for k in range(int(0.06 / (0.9 * hc))):
    L, m = dm.cotan_laplacian(V, F3), dm.mass_voronoi(V, F3)
    V = V + 0.9 * hc * (L @ V) / m[:, None]
L, m = dm.cotan_laplacian(V, F3), dm.mass_voronoi(V, F3)
lam_mid = np.linalg.eigvalsh(-(L / np.sqrt(m)[:, None] / np.sqrt(m)[None, :]))[-1]
check("Example 42.6.8: by t = 0.06 the shrunken mesh has 2 / lambda_max < 0.9 h_c (the limit has narrowed)",
      2 / lam_mid < 0.9 * hc)

# ---------------------------------------------------------------- Example 42.6.9, Figure 42.6.4
A0 = dm.face_areas(peanut, F3).sum()
L0 = dm.cotan_laplacian(peanut, F3)


def flow(fixed, steps, h=0.02):
    X, out = peanut.copy(), {0: peanut.copy()}
    for k in range(1, steps + 1):
        L = L0 if fixed else dm.cotan_laplacian(X, F3)
        m = dm.mass_voronoi(X, F3)
        if not (np.all(np.isfinite(L)) and np.all(np.isfinite(m)) and m.min() > 0):
            return out, k, (np.all(np.isfinite(L)), np.all(np.isfinite(m)))
        X = np.linalg.solve(np.diag(m) - h * L, m[:, None] * X)
        c = X.mean(0)
        X = c + (X - c) * np.sqrt(A0 / dm.face_areas(X, F3).sum())
        out[k] = X.copy()
    return out, None, None


minang = lambda X: np.degrees(dm.corner_angles(X, F3).min())  # noqa: E731
mcf, broke, why = flow(False, 60)
cm, _, _ = flow(True, 120)
check("Example 42.6.9: MCF z extent 3.0, 3.9, 5.1 at steps 0, 20, 35",
      np.allclose([np.ptp(mcf[k][:, 2]) for k in (0, 20, 35)], [3.0, 3.86, 5.1], atol=0.02))
check("Example 42.6.9: MCF smallest angle 19.1, 1.9, 0.0 degrees at steps 0, 35, 40",
      np.allclose([minang(mcf[k]) for k in (0, 35, 40)], [19.1, 1.9, 0.0], atol=0.05))
check("Example 42.6.9: MCF cannot continue at step 54 because the cotan weights are not finite (M is still finite)",
      broke == 54 and why == (False, True))
check("Example 42.6.9: the fixed-L_0 flow keeps every angle above 10.6 degrees and ends with std/mean 0.054",
      near(min(minang(cm[k]) for k in cm), 10.6, tol=0.05)
      and near(np.std(np.linalg.norm(cm[120] - cm[120].mean(0), axis=1)) / np.mean(np.linalg.norm(cm[120] - cm[120].mean(0), axis=1)), 0.054, tol=5e-4))
check("Figure 42.6.4: the fixed-L_0 flow has z extent 3.0, 2.88, 2.32 at steps 0, 40, 120",
      np.allclose([np.ptp(cm[k][:, 2]) for k in (0, 40, 120)], [3.0, 2.88, 2.32], atol=0.01))
check("Figure 42.6.4: every step keeps the initial area", all(near(dm.face_areas(X, F3).sum(), A0, tol=1e-9) for X in (mcf[35], cm[120])))
El, _ = dd.edge_list(F3)
ratio = lambda X: (lambda l: l.max() / l.min())(np.linalg.norm(X[El[:, 1]] - X[El[:, 0]], axis=1))  # noqa: E731
check("Section 42.6 text: longest / shortest edge is 3.8 initially and 460 at MCF step 35",
      near(ratio(mcf[0]), 3.8, tol=0.05) and abs(ratio(mcf[35]) - 460) < 5)

# ---------------------------------------------------------------- Exercise 42.6.1
V, F = dm.icosphere(2)
L, m = dm.cotan_laplacian(V, F), dm.mass_voronoi(V, F)
u0 = rng.normal(size=len(V))
u = u0.copy()
step = np.linalg.inv(np.diag(m) - 0.5 * L) @ np.diag(m)
for _ in range(200):
    u = step @ u
check("Exercise 42.6.1: repeated implicit heat steps converge to the area-weighted mean",
      np.abs(u - (m @ u0) / m.sum()).max() < 1e-10)

# ---------------------------------------------------------------- Exercise 42.6.3
flux = sp.simplify((sp.diff(G, th) * 2 * sp.pi * sp.sin(th)).subs(th, eps))
cap = sp.integrate(2 * sp.pi * sp.sin(th), (th, 0, eps))
check("Exercise 42.6.3: the flux through theta = eps is (1 + cos eps)/2 = 1 - (1 - cos eps)/2",
      sp.simplify(flux - (1 + sp.cos(eps)) / 2) == 0)
check("Exercise 42.6.3: the cap area is 2 pi (1 - cos eps) and the flux is 1 - cap / (4 pi); limit 1 as eps -> 0",
      sp.simplify(cap - 2 * sp.pi * (1 - sp.cos(eps))) == 0 and sp.simplify(flux - (1 - cap / (4 * sp.pi))) == 0
      and sp.limit(flux, eps, 0) == 1)

# ---------------------------------------------------------------- Exercise 42.6.5
ns, same, out = ch42.run_printed(PAGE, "code-ex-42-6-5", "out-ex-42-6-5")
check("code-ex-42-6-5 runs and prints exactly out-ex-42-6-5", same)
V4 = ns["V"]
check("code-ex-42-6-5: p is the north pole, q the vertex (1, 0, 0)",
      near(V4[ns["p"], 2], 1.0) and near(V4[ns["q"], 0], 1.0))
check("Exercise 42.6.5: ch42.chol_solve handles several right-hand sides at once",
      np.abs(ch42.chol_solve(ns["C"], np.c_[ns["m"], ns["m"] ** 2])[:, 1]
             - ch42.chol_solve(ns["C"], ns["m"] ** 2)).max() < 1e-12)

# ---------------------------------------------------------------- Exercise 42.6.6
ok = True
for _ in range(50):
    J = rng.normal(size=(3, 2))
    s1, s2 = np.linalg.svd(J, compute_uv=False)
    ok &= near(0.5 * np.sum(J ** 2) - s1 * s2, 0.5 * (s1 - s2) ** 2, tol=1e-12)
Q, _ = np.linalg.qr(rng.normal(size=(3, 2)))
Jc = 1.7 * Q @ np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
s1, s2 = np.linalg.svd(Jc, compute_uv=False)
check("Exercise 42.6.6(a): |J|_F^2 / 2 - sigma1 sigma2 = (sigma1 - sigma2)^2 / 2 (50 random J)", ok)
check("Exercise 42.6.6(a): a similarity (scaled isometry) has sigma1 = sigma2 and equality",
      near(s1, s2, tol=1e-12) and near(0.5 * np.sum(Jc ** 2), s1 * s2, tol=1e-12))
X = noisy
Ysim = 1.3 * X @ np.linalg.qr(rng.normal(size=(3, 3)))[0].T + 0.5
check("Exercise 42.6.6(b): E_k(y) = Area(y) when y is a similarity image of x^k",
      near(frozen_energy(X, F3, Ysim), dm.face_areas(Ysim, F3).sum(), tol=1e-9))

summary()
